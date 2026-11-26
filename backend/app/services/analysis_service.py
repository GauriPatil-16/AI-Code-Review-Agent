import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.schemas.schemas import Finding

EXTENSIONS = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "java": ".java",
    "c++": ".cpp",
}
SEMGREP_RULES = """rules:
  - id: python-eval
    languages: [python]
    message: eval can execute arbitrary code; validate the trust boundary.
    severity: WARNING
    pattern: eval(...)
  - id: python-shell
    languages: [python]
    message: Shell execution needs a strict trust boundary; prefer an argument list without shell=True.
    severity: WARNING
    pattern: subprocess.$FUNC(..., shell=True, ...)
  - id: javascript-eval
    languages: [javascript, typescript]
    message: eval can execute arbitrary code; prefer structured parsing.
    severity: WARNING
    pattern: eval(...)
  - id: java-runtime-exec
    languages: [java]
    message: Process execution needs strict input validation and argument separation.
    severity: WARNING
    pattern: Runtime.getRuntime().exec(...)
  - id: cpp-gets
    languages: [cpp]
    message: gets has no buffer bound; replace with a bounded input function.
    severity: ERROR
    pattern: gets(...)
"""


async def run_scanner(args: list[str], cwd: str):
    process = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(), get_settings().scanner_timeout
        )
        if process.returncode not in (0, 1):
            raise RuntimeError("Scanner failed")
        return json.loads(stdout)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.communicate()
        raise


async def scan_code(
    code: str, language: str, filename: str = "snippet"
) -> tuple[list[Finding], list[str]]:
    findings, warnings = [], []
    if language not in EXTENSIONS:
        return [], [f"No configured static scanner for {language}. AI analysis only."]
    with tempfile.TemporaryDirectory(prefix="review-scan-") as temp:
        path = Path(temp) / ("snippet" + EXTENSIONS[language])
        path.write_text(code, encoding="utf-8")
        if language == "python":
            if not shutil.which("ruff"):
                warnings.append("Ruff unavailable; Python lint scan skipped.")
            else:
                try:
                    result = await run_scanner(
                        [
                            "ruff",
                            "check",
                            "--isolated",
                            "--no-cache",
                            "--output-format",
                            "json",
                            str(path),
                        ],
                        temp,
                    )
                    for item in result[:100]:
                        findings.append(
                            Finding(
                                title=f"{item['code']}: {item['message']}",
                                severity="medium"
                                if item["code"].startswith(("F", "E9"))
                                else "low",
                                category="quality",
                                description=item["message"],
                                suggestion=(item.get("fix") or {}).get("message")
                                or "Resolve the reported lint issue and rerun Ruff.",
                                file=filename,
                                line=item["location"]["row"],
                                source="ruff",
                            )
                        )
                except (TimeoutError, OSError, RuntimeError, ValueError, KeyError):
                    warnings.append(
                        "Ruff failed or timed out; lint results are incomplete."
                    )
        if not shutil.which("semgrep"):
            warnings.append(
                "Semgrep unavailable; security scan skipped. On Windows use Docker or WSL."
            )
        else:
            config = Path(temp) / "rules.yaml"
            config.write_text(SEMGREP_RULES, encoding="utf-8")
            try:
                result = await run_scanner(
                    [
                        "semgrep",
                        "scan",
                        "--config",
                        str(config),
                        "--json",
                        "--metrics=off",
                        "--disable-version-check",
                        "--no-git-ignore",
                        "--timeout",
                        "10",
                        str(path),
                    ],
                    temp,
                )
                for item in result.get("results", [])[:100]:
                    findings.append(
                        Finding(
                            title=item["check_id"].split(".")[-1],
                            severity="high"
                            if item["extra"]["severity"] == "ERROR"
                            else "medium",
                            category="security",
                            description=item["extra"]["message"],
                            suggestion="Check whether untrusted data can reach this operation. Replace dangerous parsing or process execution with a bounded, validated alternative.",
                            file=filename,
                            line=item["start"]["line"],
                            source="semgrep",
                        )
                    )
                if result.get("errors"):
                    warnings.append(
                        "Semgrep reported parse or scan errors; coverage is incomplete."
                    )
                warnings.append(
                    "Semgrep uses a small bundled rule set, not a comprehensive security audit."
                )
            except (TimeoutError, OSError, RuntimeError, ValueError, KeyError):
                warnings.append(
                    "Semgrep failed or timed out; security results are incomplete."
                )
    return findings, warnings
