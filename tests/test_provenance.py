"""Tests of the provenance header of result files.

Each test docstring names the defect that makes it fail.
"""

import re
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

from fusion.provenance import ROOT, git_state, provenance_lines

HASH = "0123456789abcdef0123456789abcdef01234567"
NOON = datetime(2026, 10, 8, 12, 0, 5, tzinfo=UTC)


def fake_git(status: str = "", head: str = HASH):
    """A stand-in for subprocess.run that answers the two git queries."""
    calls = []

    def run(args, **kwargs):
        calls.append((tuple(args), kwargs))
        out = head + "\n" if args[1] == "rev-parse" else status
        return SimpleNamespace(stdout=out)

    run.calls = calls
    return run


def failing_git(error):
    def run(args, **kwargs):
        raise error

    return run


def test_header_lists_command_head_state_and_utc_time_then_a_blank_line():
    """Fails if a header line is missing, reordered or formatted differently."""
    lines = provenance_lines(["scripts/x.py", "--seeds", "50"], now=NOON, run=fake_git())
    assert lines == [
        "# command: python scripts/x.py --seeds 50",
        f"# git HEAD: {HASH} (working tree clean)",
        "# date (UTC): 2026-10-08T12:00:05Z",
        "",
    ]


def test_a_modified_tracked_file_marks_the_tree_dirty():
    """Fails if uncommitted changes to tracked files are not reported."""
    assert git_state(run=fake_git(status=" M src/fusion/x.py\n")) == (HASH, "dirty")
    assert git_state(run=fake_git(status="")) == (HASH, "clean")


def test_untracked_files_are_excluded_from_the_status_query():
    """Fails if new, uncommitted result or scratch files would mark the tree dirty."""
    run = fake_git()
    git_state(run=run)
    status_args = [args for args, _ in run.calls if args[1] == "status"]
    assert status_args == [("git", "status", "--porcelain", "--untracked-files=no")]
    assert all(kwargs["cwd"] == ROOT for _, kwargs in run.calls)


def test_missing_git_or_no_repository_gives_unknown_instead_of_raising():
    """Fails if a script crashes on a machine without git or outside a repository."""
    assert git_state(run=failing_git(FileNotFoundError("git"))) == ("unknown", "unknown")
    error = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
    assert git_state(run=failing_git(error)) == ("unknown", "unknown")
    assert git_state(run=fake_git(head="")) == ("unknown", "unknown")


def test_local_times_are_converted_to_utc():
    """Fails if a timezone-aware local time is written without conversion to UTC."""
    local = datetime(2026, 10, 8, 15, 0, 5, tzinfo=timezone(timedelta(hours=3)))
    line = provenance_lines(["s.py"], now=local, run=fake_git())[2]
    assert line == "# date (UTC): 2026-10-08T12:00:05Z"


def test_arguments_with_spaces_are_quoted_so_the_command_can_be_rerun():
    """Fails if an argument with a space is written unquoted (the command would split it)."""
    line = provenance_lines(["s.py", "--out", "a b.txt"], now=NOON, run=fake_git())[0]
    assert line == "# command: python s.py --out 'a b.txt'"


def test_the_real_repository_gives_a_full_hash():
    """Fails if the default root does not point at this repository."""
    head, state = git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    assert state in ("clean", "dirty")
