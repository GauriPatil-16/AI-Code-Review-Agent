"""Nonexecuting validation plus optional, restricted Docker-only test execution."""

import ast
import asyncio
import difflib
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.services.analysis_service import scan_code
from app.services.github_fix_service import GitHubFixClient, safe_path
from app.services.groq_service import GroqService, ServiceError


def result(name, status, detail):
    return {"name": name, "status": status, "detail": detail}


async def docker_command(*args, timeout=15):
    process = await asyncio.create_subprocess_exec(
        shutil.which("docker") or "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    async def collect():
        output, total = bytearray(), 0
        while chunk := await process.stdout.read(4096):
            total += len(chunk)
            if total > 1_000_000:
                raise RuntimeError("Sandbox output limit exceeded")
            output.extend(chunk[: max(0, 12000 - len(output))])
        await process.wait()
        return process.returncode, output.decode("utf-8", errors="replace")

    try:
        return await asyncio.wait_for(collect(), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise


async def sandbox_tests(snapshot, corrected):
    if not snapshot.get("source"):
        return [
            result(
                "Existing tests",
                "skipped",
                "Pasted snippets have no repository test suite attached.",
            )
        ]
    if any(f["language"] != "python" for f in corrected):
        return [
            result(
                "Existing tests",
                "skipped",
                "The isolated test runner currently supports Python repositories only.",
            )
        ]
    if not shutil.which("docker"):
        return [
            result(
                "Existing tests",
                "skipped",
                "Docker is unavailable. Repository code was not executed on the host.",
            )
        ]
    try:
        code, _ = await docker_command(
            "image", "inspect", get_settings().fix_sandbox_image
        )
        if code:
            return [
                result(
                    "Existing tests",
                    "skipped",
                    "The trusted local test image is unavailable; no image was pulled or repository command run.",
                )
            ]
        async with GitHubFixClient(snapshot["source"]) as github:
            files, warning = await github.validation_snapshot()
        if files is None:
            return [result("Existing tests", "skipped", warning)]
        tests = [
            p
            for p in files
            if Path(p).name.startswith("test_") or Path(p).name.endswith("_test.py")
        ]
        if not tests:
            return [
                result(
                    "Existing tests",
                    "skipped",
                    "No existing Python tests were found in the bounded repository snapshot.",
                )
            ]
        statuses = []
        for label, changes in (("Baseline tests", []), ("Corrected tests", corrected)):
            with tempfile.TemporaryDirectory(prefix="fix-tests-") as temp:
                for name, content in files.items():
                    path = Path(temp) / safe_path(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8", newline="")
                for file in changes:
                    path = Path(temp) / safe_path(file["path"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(file["corrected"], encoding="utf-8", newline="")
                container = "review-fix-tests-" + uuid4().hex
                try:
                    code, output = await docker_command(
                        "run",
                        "--rm",
                        "--name",
                        container,
                        "--pull=never",
                        "--network=none",
                        "--read-only",
                        "--cap-drop=ALL",
                        "--security-opt=no-new-privileges",
                        "--pids-limit=64",
                        "--memory=256m",
                        "--cpus=1",
                        "--user=65534:65534",
                        "--tmpfs",
                        "/tmp:rw,nosuid,nodev,size=64m",
                        "--workdir=/workspace",
                        "--mount",
                        f"type=bind,source={temp},target=/workspace,readonly",
                        "--env=HOME=/tmp",
                        "--env=PYTHONDONTWRITEBYTECODE=1",
                        "--env=PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                        "--entrypoint=python",
                        get_settings().fix_sandbox_image,
                        "-m",
                        "pytest",
                        "-q",
                        "--disable-warnings",
                        "-p",
                        "no:cacheprovider",
                        *tests,
                        timeout=get_settings().fix_test_timeout,
                    )
                    statuses.append(
                        result(
                            label,
                            "passed" if code == 0 else "failed",
                            output[-12000:] or f"pytest exited {code}.",
                        )
                    )
                except (TimeoutError, OSError, RuntimeError):
                    statuses.append(
                        result(
                            label,
                            "failed",
                            "The isolated test run failed, timed out or exceeded its output limit.",
                        )
                    )
                finally:
                    # This generated container name cannot identify a user's other container.
                    try:
                        await docker_command("rm", "--force", container)
                    except (OSError, TimeoutError, RuntimeError):
                        pass
        return statuses
    except (ServiceError, OSError, TimeoutError, RuntimeError):
        return [
            result(
                "Existing tests",
                "skipped",
                "The isolated test environment or bounded repository snapshot was unavailable. Tests were not run.",
            )
        ]


def finding_key(f):
    return f.source, f.title, f.description


def python_interfaces(code):
    """Structural guard, not an execution or a semantic-equivalence claim."""
    tree = ast.parse(code)
    signatures = {}

    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                signatures[name] = (
                    type(node).__name__,
                    ast.dump(node.args, include_attributes=False)
                    if hasattr(node, "args")
                    else "class",
                )
                visit(node.body, name + ".")

    visit(tree.body)
    return signatures


async def validate_plan(snapshot, findings, plan):
    originals = {f["path"]: f for f in snapshot["files"]}
    files, checks, errors, claimed = [], [], [], set()
    seen = set()
    with tempfile.TemporaryDirectory(prefix="review-fixes-") as temp:
        for fixed in plan.files:
            safe_path(fixed.path)
            if "\x00" in fixed.code:
                raise ServiceError("Generated source contains null bytes.", 422)
            if fixed.path in seen or fixed.path not in originals:
                raise ServiceError(
                    "Fix response contains a duplicate or unknown file.", 422
                )
            seen.add(fixed.path)
            original = originals[fixed.path]
            if fixed.code == original["code"]:
                errors.append(f"{fixed.path}: no source changes were generated.")
                continue
            ids = set(fixed.issue_ids)
            if any(
                i < 0
                or i >= len(findings)
                or findings[i].get("file") != fixed.path
                or not findings[i].get("line")
                for i in ids
            ):
                raise ServiceError(
                    "Fix references unsupported, unlocated or unrelated issues.", 422
                )
            if Path(fixed.path).suffix and Path(fixed.path).suffix.lower() not in (
                ".py",
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".java",
                ".cpp",
                ".cc",
                ".h",
            ):
                raise ServiceError(
                    "Automatic fixes are limited to reviewed source files, not configuration or workflows.",
                    422,
                )
            claimed.update(ids)
            # Isolated scratch files are never imported or executed here.
            path = Path(temp) / fixed.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(fixed.code, encoding="utf-8", newline="")
            if original["language"] == "python":
                try:
                    ast.parse(fixed.code, filename=fixed.path)
                    try:
                        before_interfaces = python_interfaces(original["code"])
                    except (SyntaxError, ValueError, RecursionError):
                        before_interfaces = {}
                    after_interfaces = python_interfaces(fixed.code)
                    if any(
                        after_interfaces.get(name) != signature
                        for name, signature in before_interfaces.items()
                    ):
                        errors.append(
                            f"{fixed.path}: existing function/class interfaces were removed or changed."
                        )
                    checks.append(
                        result(
                            f"{fixed.path}: Python syntax",
                            "passed",
                            "Parsed without executing the source.",
                        )
                    )
                except (SyntaxError, ValueError, RecursionError):
                    errors.append(
                        f"{fixed.path}: corrected Python source has invalid syntax."
                    )
                    checks.append(
                        result(
                            f"{fixed.path}: Python syntax",
                            "failed",
                            "The corrected source could not be parsed.",
                        )
                    )
            else:
                checks.append(
                    result(
                        f"{fixed.path}: syntax",
                        "skipped",
                        "No standalone syntax validator is configured for this language.",
                    )
                )
            before, before_warnings = await scan_code(
                original["code"], original["language"], fixed.path
            )
            after, after_warnings = await scan_code(
                fixed.code, original["language"], fixed.path
            )
            introduced = Counter(map(finding_key, after)) - Counter(
                map(finding_key, before)
            )
            if introduced:
                errors.append(f"{fixed.path}: static analysis detected new findings.")
            if any(
                "failed" in w.lower() or "parse or scan errors" in w.lower()
                for w in before_warnings + after_warnings
            ):
                errors.append(
                    f"{fixed.path}: a configured scanner failed; validation is incomplete."
                )
            for i in ids:
                f = findings[i]
                if f["source"] in ("ruff", "semgrep"):
                    original_count = sum(
                        a.source == f["source"] and a.title == f["title"]
                        for a in before
                    )
                    corrected_count = sum(
                        a.source == f["source"] and a.title == f["title"] for a in after
                    )
                    if original_count == 0 or corrected_count >= original_count:
                        errors.append(
                            f"Issue {i}: static evidence does not confirm resolution."
                        )
            checks.append(
                result(
                    f"{fixed.path}: static analysis",
                    "failed"
                    if introduced
                    else "partial"
                    if before_warnings or after_warnings
                    else "passed",
                    "\n".join(dict.fromkeys(before_warnings + after_warnings))
                    or "No additional static findings were detected.",
                )
            )
            diff = list(
                difflib.unified_diff(
                    original["code"].splitlines(keepends=True),
                    fixed.code.splitlines(keepends=True),
                    fromfile="a/" + fixed.path,
                    tofile="b/" + fixed.path,
                )
            )
            files.append(
                {
                    "path": fixed.path,
                    "language": original["language"],
                    "original": original["code"],
                    "corrected": fixed.code,
                    "issue_ids": sorted(ids),
                    "explanation": fixed.explanation,
                    "diff": "".join(
                        line
                        if line.endswith("\n")
                        else line + "\n\\ No newline at end of file\n"
                        for line in diff
                    ),
                    "added": sum(
                        line.startswith("+") and not line.startswith("+++")
                        for line in diff
                    ),
                    "removed": sum(
                        line.startswith("-") and not line.startswith("---")
                        for line in diff
                    ),
                }
            )
    if not files:
        errors.append("No supported, actionable source changes were generated.")
    for skipped in plan.skipped_issues:
        if skipped.issue_id >= len(findings) or skipped.issue_id in claimed:
            errors.append("Fix response contradicts its skipped-issue list.")
    verification = None
    if files and not errors:
        verification = await GroqService().verify_fixes(
            {
                "files": files,
                "issues": [{"issue_id": i, **findings[i]} for i in sorted(claimed)],
                "context": snapshot.get("context", []),
                "instructions": snapshot.get("instructions", ""),
            }
        )
        verdicts = {i.issue_id: i for i in verification.issues}
        if len(verdicts) != len(verification.issues) or set(verdicts) != claimed:
            errors.append(
                "Independent verification did not assess exactly the proposed issues."
            )
        if not verification.behavior_preserved or verification.unintended_changes:
            errors.append(
                "Independent verification detected unintended behavior changes."
            )
        if any(
            verdicts.get(i) is None or verdicts[i].verdict != "fixed" for i in claimed
        ):
            errors.append(
                "Some proposed fixes remain uncertain, unsupported or unresolved."
            )
        checks.append(
            result(
                "Independent AI verification",
                "failed" if errors else "passed",
                "AI assessment only; this does not prove functional correctness.",
            )
        )
    else:
        checks.append(
            result(
                "Independent AI verification",
                "skipped",
                "Source checks must pass first.",
            )
        )
    if files and not errors:
        tests = await sandbox_tests(snapshot, files)
        checks.extend(tests)
        if any(
            t["name"] == "Corrected tests" and t["status"] == "failed" for t in tests
        ):
            errors.append("Existing tests failed for the corrected code.")
    else:
        checks.append(
            result(
                "Existing tests",
                "skipped",
                "Source validation failed; repository tests were not executed.",
            )
        )
    remaining = [
        {"issue_id": i, **finding}
        for i, finding in enumerate(findings)
        if i not in claimed
    ]
    proposal = {
        "files": files,
        "source": snapshot.get("source"),
        "validation": checks,
        "addressed_issues": [{"issue_id": i, **findings[i]} for i in sorted(claimed)],
        "remaining_issues": remaining,
        "skipped_issues": [s.model_dump() for s in plan.skipped_issues],
        "verification": verification.model_dump() if verification else None,
        "valid": not errors,
        "errors": errors,
    }
    return proposal
