"""Provenance header of a result file: the command that wrote it, the code version, the time.

Every result file a script writes starts with these lines, so a number can always be traced back
to the exact command line and commit that produced it:

    # command: python scripts/... --seeds 50
    # git HEAD: <full hash> (working tree clean)
    # date (UTC): 2026-10-08T12:00:00Z

"working tree dirty" means tracked files differed from HEAD when the file was written (untracked
files are ignored); "unknown" means git could not be asked.
"""

import shlex
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNKNOWN = "unknown"


def git_state(root: Path = ROOT, run: Callable = subprocess.run) -> tuple[str, str]:
    """HEAD hash and "clean" or "dirty" for the repository at root.

    Args:
        root: Directory inside the repository.
        run: subprocess.run or a stand-in with the same signature (for tests).

    Returns:
        (hash, state); ("unknown", "unknown") if git is missing or the directory is no repository.
    """
    options = {"cwd": root, "capture_output": True, "text": True, "check": True}
    try:
        head = run(["git", "rev-parse", "HEAD"], **options).stdout.strip()
        status = run(["git", "status", "--porcelain", "--untracked-files=no"], **options).stdout
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN, UNKNOWN
    if not head:
        return UNKNOWN, UNKNOWN
    return head, "dirty" if status.strip() else "clean"


def provenance_lines(
    argv: Sequence[str],
    root: Path = ROOT,
    now: datetime | None = None,
    run: Callable = subprocess.run,
) -> list[str]:
    """The header lines of a result file, ending with an empty line.

    Args:
        argv: The command line of the script, as sys.argv (script path first).
        root: Directory inside the repository.
        now: The time to record (timezone-aware); the current time if None.
        run: subprocess.run or a stand-in (for tests).
    """
    moment = datetime.now(UTC) if now is None else now.astimezone(UTC)
    head, state = git_state(root, run)
    return [
        f"# command: python {shlex.join(argv)}",
        f"# git HEAD: {head} (working tree {state})",
        f"# date (UTC): {moment.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
    ]
