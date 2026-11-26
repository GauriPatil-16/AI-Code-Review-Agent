"""Fixed-host GitHub REST operations. Every mutation rechecks durable approval."""

import asyncio
import base64
import hashlib
import re
from urllib.parse import quote

import httpx

from app.core.config import get_settings
from app.services.groq_service import ServiceError


def safe_path(value: str) -> str:
    parts = value.split("/")
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if (
        not value
        or len(value) > 500
        or "\\" in value
        or ":" in value
        or any(p in ("", ".", "..") or p.startswith(".") for p in parts)
        or any(ord(c) < 32 for c in value)
        or any(
            p.endswith((" ", ".")) or p.split(".")[0].upper() in reserved for p in parts
        )
    ):
        raise ServiceError("Unsafe or unsupported repository file path.", 422)
    return value


def validate_source(source: dict):
    if (
        not source
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source.get("repository") or ""
        )
        or not source.get("branch")
        or not re.fullmatch(r"[a-f0-9]{40}", source.get("sha", ""))
    ):
        raise ServiceError(
            "The reviewed source branch is unavailable. Run a new GitHub review.", 409
        )
    branch = source["branch"]
    if any(p in (".", "..") for p in source["repository"].split("/")):
        raise ServiceError("Unsupported source repository.", 422)
    if (
        any(c in branch for c in "\x00\r\n?\\")
        or ".." in branch
        or branch.startswith("/")
    ):
        raise ServiceError("Unsupported source branch.", 422)


class GitHubFixClient:
    def __init__(self, source: dict):
        validate_source(source)
        self.source = source
        self.prefix = "/repos/" + source["repository"]
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if get_settings().github_token:
            headers["Authorization"] = "Bearer " + get_settings().github_token
        self.client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
            timeout=25,
            follow_redirects=False,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def request(self, method, path, *, optional=False, **kwargs):
        try:
            response = await self.client.request(method, path, **kwargs)
            if optional and response.status_code == 404:
                return None
            if response.status_code in (401, 403):
                raise ServiceError(
                    "GitHub authentication or repository permission is insufficient. Check token access.",
                    403,
                )
            if response.status_code == 404:
                raise ServiceError(
                    "The source repository, branch or file is inaccessible. No other branch will be used.",
                    409,
                )
            if response.status_code in (409, 422):
                raise ServiceError(
                    "GitHub reported a conflict or rejected the branch/PR. Refresh the review before retrying.",
                    409,
                )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            raise ServiceError(
                "GitHub request failed. The workflow can be retried without duplicating its branch or PR.",
                502,
            ) from None

    async def current_source(self):
        ref = await self.request(
            "GET",
            self.prefix + "/git/ref/heads/" + quote(self.source["branch"], safe=""),
        )
        if ref["object"]["sha"] != self.source["sha"]:
            raise ServiceError(
                "The source branch changed after review. Run a new review and approve its new changes.",
                409,
            )
        return ref

    async def preflight(self, files: list[dict]):
        if not get_settings().github_token:
            raise ServiceError(
                "Configure GITHUB_TOKEN in backend/.env with Contents and Pull requests write access, then restart.",
                503,
            )
        user = await self.request("GET", "/user")
        repo = await self.request("GET", self.prefix)
        if not repo.get("permissions", {}).get("push"):
            raise ServiceError(
                "The GitHub account cannot push to the reviewed source repository. No other repository will be used.",
                403,
            )
        await self.current_source()
        tree = await self.request(
            "GET",
            self.prefix + "/git/trees/" + self.source["sha"],
            params={"recursive": "1"},
        )
        if tree.get("truncated"):
            raise ServiceError("Repository tree is too large to verify safely.", 422)
        entries = {e["path"]: e for e in tree["tree"]}
        modes = {}
        for file in files:
            entry = entries.get(safe_path(file["path"]))
            raw = file["original"].encode("utf-8")
            blob_sha = hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest()
            if (
                not entry
                or entry.get("type") != "blob"
                or entry.get("mode") not in ("100644", "100755")
                or entry["sha"] != blob_sha
            ):
                raise ServiceError(
                    "Reviewed source does not match a regular file at the approved commit. Run a new review.",
                    409,
                )
            modes[file["path"]] = entry["mode"]
        return user, modes

    async def validation_snapshot(self):
        """A bounded Python source/test snapshot at the immutable reviewed SHA."""
        tree = await self.request(
            "GET",
            self.prefix + "/git/trees/" + self.source["sha"],
            params={"recursive": "1"},
        )
        if tree.get("truncated"):
            return None, "Repository test snapshot exceeds supported limits."
        entries = [
            e
            for e in tree.get("tree", [])
            if e.get("type") == "blob"
            and (
                e["path"].endswith(".py")
                or e["path"].split("/")[-1]
                in ("pytest.ini", "pyproject.toml", "setup.cfg")
            )
        ]
        if len(entries) > 100 or sum(e.get("size", 0) for e in entries) > 500000:
            return (
                None,
                "Repository test snapshot exceeds 100 files or 500 KB; tests were not run.",
            )
        files = {}
        for entry in entries:
            try:
                name = safe_path(entry["path"])
            except ServiceError:
                return (
                    None,
                    "Repository test snapshot has unsupported paths; tests were not run.",
                )
            if entry.get("mode") not in ("100644", "100755"):
                return (
                    None,
                    "Repository tests depend on unsupported symlinks; tests were not run.",
                )
            blob = await self.request("GET", self.prefix + "/git/blobs/" + entry["sha"])
            try:
                if blob.get("encoding") != "base64":
                    return None, "Unsupported repository test source encoding."
                raw = base64.b64decode(blob["content"], validate=False)
                files[name] = raw.decode("utf-8")
                if sum(len(v.encode("utf-8")) for v in files.values()) > 500000:
                    return None, "Repository test snapshot exceeds the size limit."
            except (ValueError, UnicodeError):
                return None, "Repository test source could not be decoded."
        return files, None

    async def mutate(self, workflow_id, method, path, **kwargs):
        from app.services.fix_service import assert_approved

        await asyncio.to_thread(assert_approved, workflow_id, self.lease_owner)
        await self.current_source()
        await asyncio.to_thread(assert_approved, workflow_id, self.lease_owner)
        return await self.request(method, path, **kwargs)

    async def publish(self, workflow_id: str, stage: str = "all"):
        from app.services.fix_service import assert_approved, update_workflow

        row = await asyncio.to_thread(assert_approved, workflow_id)
        proposal = row.proposal
        branch = row.branch or f"ai-fix/review-{row.review_id[:8]}-{row.id[:12]}"
        if (
            not branch.startswith("ai-fix/review-")
            or branch in ("main", "master")
            or branch == self.source["branch"]
        ):
            raise ServiceError(
                "Invalid fix branch; refusing to modify the source branch.", 409
            )
        await asyncio.to_thread(update_workflow, workflow_id, branch=branch)
        marker = f"<!-- ai-fix:{workflow_id}:{row.proposal_hash} -->"
        params = {
            "state": "all",
            "head": self.source["repository"].split("/")[0] + ":" + branch,
            "base": self.source["branch"],
            "per_page": 100,
        }

        async def find_pr():
            prs = await self.request("GET", self.prefix + "/pulls", params=params)
            for pr in prs:
                if (
                    marker in (pr.get("body") or "")
                    and pr.get("head", {}).get("ref") == branch
                ):
                    if (pr.get("head", {}).get("repo") or {}).get(
                        "full_name", ""
                    ).lower() == self.source["repository"].lower():
                        return pr
            return None

        existing = await find_pr()
        if existing:
            return existing["html_url"]
        user, modes = await self.preflight(proposal["files"])
        ref_path = self.prefix + "/git/ref/heads/" + quote(branch, safe="")
        ref = await self.request("GET", ref_path, optional=True)
        if ref is None:
            await self.mutate(
                workflow_id,
                "POST",
                self.prefix + "/git/refs",
                json={"ref": "refs/heads/" + branch, "sha": self.source["sha"]},
            )
        elif ref["object"]["sha"] not in (self.source["sha"], row.commit_sha):
            raise ServiceError(
                "The fix branch was modified externally. Refusing to overwrite it.", 409
            )
        if stage == "branch":
            return None
        if not row.commit_sha:
            commit = await self.request(
                "GET", self.prefix + "/git/commits/" + self.source["sha"]
            )
            tree = await self.mutate(
                workflow_id,
                "POST",
                self.prefix + "/git/trees",
                json={
                    "base_tree": commit["tree"]["sha"],
                    "tree": [
                        {
                            "path": f["path"],
                            "mode": modes[f["path"]],
                            "type": "blob",
                            "content": f["corrected"],
                        }
                        for f in proposal["files"]
                    ],
                },
            )
            identity = row.git_identity or {
                "name": user["login"],
                "email": user.get("email")
                or f"{user['id']}+{user['login']}@users.noreply.github.com",
                "date": row.approved_at,
            }
            await asyncio.to_thread(update_workflow, workflow_id, git_identity=identity)
            commit = await self.mutate(
                workflow_id,
                "POST",
                self.prefix + "/git/commits",
                json={
                    "message": f"Fix reviewed issues ({workflow_id})",
                    "tree": tree["sha"],
                    "parents": [self.source["sha"]],
                    "author": identity,
                    "committer": identity,
                },
            )
            row = await asyncio.to_thread(
                update_workflow, workflow_id, commit_sha=commit["sha"]
            )
        if ref is None or ref["object"]["sha"] != row.commit_sha:
            await self.mutate(
                workflow_id,
                "PATCH",
                self.prefix + "/git/refs/heads/" + quote(branch, safe=""),
                json={"sha": row.commit_sha, "force": False},
            )
        if stage == "commit":
            return None
        issue_titles = [i["title"] for i in proposal["addressed_issues"]]
        title = ("Fix: " + "; ".join(issue_titles))[:240]
        body = "\n".join(
            [
                marker,
                "## Reviewed fixes",
                *["- " + t for t in issue_titles],
                "",
                "## Files",
                *[f"- `{f['path']}`: {f['explanation']}" for f in proposal["files"]],
                "",
                "## Validation",
                *[
                    f"- {v['name']}: {v['status']} — {v['detail']}"
                    for v in proposal["validation"]
                ],
                "",
                "## Remaining issues",
                *["- " + i["title"] for i in proposal["remaining_issues"]],
                "",
                "Source commit: `" + self.source["sha"] + "`",
                "Human approval recorded for proposal `" + row.proposal_hash + "`.",
                "AI verification is not proof of functional correctness. This PR is not automatically merged.",
            ]
        )
        if len(body) > 60000:
            body = (
                body[:59000]
                + "\n\nAdditional details are available in the approved proposal in AI Code Review Agent."
            )
        try:
            pr = await self.mutate(
                workflow_id,
                "POST",
                self.prefix + "/pulls",
                json={
                    "title": title,
                    "body": body,
                    "head": branch,
                    "base": self.source["branch"],
                    "draft": False,
                },
            )
        except ServiceError:
            # Recover a response lost after GitHub accepted the request.
            pr = await find_pr()
            if not pr:
                raise
        url = pr["html_url"]
        if not re.fullmatch(
            r"https://github\.com/"
            + re.escape(self.source["repository"])
            + r"/pull/[1-9][0-9]*",
            url,
            re.IGNORECASE,
        ):
            raise ServiceError("GitHub returned an unexpected pull request URL.", 502)
        return url
