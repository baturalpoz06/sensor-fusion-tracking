"""Sanity checks for package installation and metadata."""

from importlib.metadata import version

import fusion


def test_version_matches_metadata():
    # __version__ in __init__.py must match the version declared in pyproject.toml
    assert fusion.__version__ == version("sensor-fusion-tracking")


def test_subpackages_importable():
    import fusion.association  # noqa: F401
    import fusion.filters  # noqa: F401
    import fusion.sensors  # noqa: F401
    import fusion.tracker  # noqa: F401