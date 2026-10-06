"""Guard of the mutation script: its mutations must still apply, and name tests that exist.

The script itself (scripts/run_mutation_checks.py) runs the tests in a scratch copy and is slow;
this test only keeps its table from rotting when the code it mutates is edited.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mutation_checks.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_mutation_checks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_mutation_applies_exactly_once_to_the_current_source():
    """Fails if an edit of the code leaves a mutation without its target text (or with two)."""
    module = load_script()
    assert len(module.MUTATIONS) >= 15
    for mutation in module.MUTATIONS:
        text = (ROOT / mutation.path).read_text(encoding="utf-8")
        for old, new in mutation.edits:
            assert text.count(old) == 1, mutation.name
            assert old != new, mutation.name


def test_every_mutation_names_tests_that_exist():
    """Fails if a mutation names a test file or function that is not in the repository."""
    module = load_script()
    for mutation in module.MUTATIONS:
        assert mutation.tests, mutation.name
        for node in mutation.tests:
            path, name = node.split("::")
            source = (ROOT / path).read_text(encoding="utf-8")
            assert re.search(rf"^def {re.escape(name)}\(", source, re.MULTILINE), node


def test_the_mutation_names_are_distinct():
    """Fails if two mutations share a name (the report would be ambiguous)."""
    module = load_script()
    names = [m.name for m in module.MUTATIONS]
    assert len(names) == len(set(names))
