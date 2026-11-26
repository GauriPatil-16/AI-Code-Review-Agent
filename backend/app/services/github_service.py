import base64
from pathlib import PurePosixPath

import httpx

from app.core.config import get_settings
from app.schemas.schemas import PRRequest
from app.services.groq_service import ServiceError

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "c++",
    ".cc": "c++",
    ".h": "c++",
}
TEXT_EXTENSIONS = set(LANGUAGES) | {
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".css",
    ".html",
    ".sql",
    ".sh",
    ".go",
    ".rs",
}


async def fetch_pull_request(url: str) -> dict:
    request = PRRequest(url=url)
    owner, repo, _, number = request.url.split("/")[3:]
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if get_settings().github_token:
        headers["Authorization"] = "Bearer " + get_settings().github_token
    async with httpx.AsyncClient(
        base_url="https://api.github.com",
        headers=headers,
        timeout=20,
        follow_redirects=False,
    ) as client:

        async def get(path, params=None):
            try:
                response = await client.get(path, params=params)
                if response.status_code == 404:
                    raise ServiceError(
                        "GitHub PR or file not found, or token lacks access.", 404
                    )
                if response.status_code in (401, 403, 429):
                    raise ServiceError(
                        "GitHub access denied or rate limit reached. Check token permissions.",
                        503,
                    )
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                raise ServiceError("GitHub request timed out.", 504) from None
            except (httpx.HTTPError, ValueError):
                raise ServiceError(
                    "GitHub request failed. Check the URL and network access."
                ) from None

        prefix = f"/repos/{owner}/{repo}"
        pr = await get(f"{prefix}/pulls/{number}")
        source_repo = (pr.get("head", {}).get("repo") or {}).get("full_name")
        # A PR's source commit may live in a fork. Read it from that repository,
        # never silently substitute a different branch or the base repository.
        import re
        source_prefix = prefix
        if source_repo:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source_repo):
                raise ServiceError("GitHub returned an unsupported source repository.", 422)
            source_prefix = "/repos/" + source_repo
        changed = await get(
            f"{prefix}/pulls/{number}/files", {"per_page": 100, "page": 1}
        )
        warnings, files, documents = [], [], []
        if pr.get("changed_files", len(changed)) > 100:
            warnings.append(
                "Only the first 100 changed-file metadata entries were fetched."
            )
        if sum(f.get("additions", 0) + f.get("deletions", 0) for f in changed) > 1000:
            warnings.append(
                "Large PR (over 1,000 changed lines). Only bounded context is reviewed."
            )
        remaining = 60000
        for item in changed:
            filename = item["filename"]
            extension = PurePosixPath(filename).suffix.lower()
            if (
                item["status"] == "removed"
                or extension not in TEXT_EXTENSIONS
                or "patch" not in item
            ):
                warnings.append(
                    f"{filename}: removed, binary, unsupported or missing patch; not reviewed."
                )
                continue
            if len(files) >= 8 or remaining <= 0:
                warnings.append(
                    "Review limited to 8 text files and 60,000 source characters; remaining files omitted."
                )
                break
            if item.get("changes", 0) > 1000:
                warnings.append(
                    f"{filename}: large change; GitHub patch may be truncated."
                )
            # The API host is fixed. No raw_url from the repository is followed.
            from urllib.parse import quote

            try:
                blob = await get(
                    f"{source_prefix}/contents/{quote(filename, safe='/')}",
                    {"ref": pr["head"]["sha"]},
                )
                if (
                    not isinstance(blob, dict)
                    or blob.get("size", 0) > 200000
                    or blob.get("encoding") != "base64"
                ):
                    warnings.append(
                        f"{filename}: source too large or unsupported encoding; omitted."
                    )
                    continue
                content = base64.b64decode(blob["content"]).decode("utf-8")
            except (UnicodeError, ValueError, ServiceError):
                warnings.append(
                    f"{filename}: full source unavailable; omitted rather than scanning a diff as source."
                )
                continue
            if len(content) > min(20000, remaining):
                warnings.append(f"{filename}: source exceeds context budget; omitted.")
                continue
            if any(
                marker in content
                for marker in ["<<<<<<< ", "\n=======\n", "\n>>>>>>> "]
            ):
                warnings.append(f"{filename}: possible merge conflict markers.")
            files.append(
                {
                    "path": filename,
                    "code": content,
                    "language": LANGUAGES.get(extension, "other"),
                    "diff": item["patch"][:15000],
                    "blob_sha": blob.get("sha"),
                }
            )
            remaining -= len(content)
        if not files:
            raise ServiceError("No reviewable text source found within PR limits.", 422)
        # Best-effort repository guidelines at the reviewed commit, with the same fixed API host.
        try:
            guideline = await get(
                f"{source_prefix}/contents/CONTRIBUTING.md", {"ref": pr["head"]["sha"]}
            )
            if (
                isinstance(guideline, dict)
                and guideline.get("size", 0) <= 20000
                and guideline.get("encoding") == "base64"
            ):
                documents.append(
                    {
                        "path": "CONTRIBUTING.md",
                        "content": base64.b64decode(guideline["content"]).decode(
                            "utf-8"
                        ),
                    }
                )
        except (ServiceError, UnicodeError, ValueError):
            pass
        return {
            "title": pr["title"][:250],
            "repository_id": f"{owner}/{repo}@{pr['head']['sha']}",
            "source_reference": request.url,
            "language": files[0]["language"]
            if len({f["language"] for f in files}) == 1
            else "multiple",
            "files": files,
            "documents": documents,
            "warnings": warnings,
            "repository": f"{owner}/{repo}",
            "source": {
                "repository": (pr.get("head", {}).get("repo") or {}).get("full_name"),
                "branch": pr.get("head", {}).get("ref"),
                "sha": pr["head"]["sha"],
            },
            "changed_files": pr.get("changed_files", len(changed)),
            "reviewed_files": len(files),
        }
