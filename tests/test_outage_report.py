"""Tests for the outage tables.

Each test docstring names the condition that makes it fail without the feature.
"""

import numpy as np
import pytest

from fusion.mtt_experiment import SweepResult
from fusion.mtt_report import format_cell, format_median
from fusion.outage_experiment import OutageConfig
from fusion.outage_metrics import OutageMetrics
from fusion.outage_report import (
    LABEL_WIDTH,
    OUTAGE_COLUMNS,
    format_outage_table,
    format_window_table,
    outage_caption,
    select_rows,
)

ROWS = ["unaware 4 s", "aware 4 s"]


def make_result() -> SweepResult:
    """Two rows by three seeds; every score NaN unless set below."""
    metrics = {name: np.full((2, 3), np.nan) for name in OutageMetrics._fields}
    metrics["during_position_rmse"] = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 7.0]])
    metrics["during_missed_rate"] = np.array([[0.0, 0.1, 0.2], [0.5, 0.5, 0.5]])
    metrics["during_ghost_rate"] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    metrics["after_position_rmse"] = np.array([[9.0, 9.0, 9.0], [1.0, 1.0, 1.0]])
    metrics["reacquisition_time"] = np.array([[0.0, 1.0, 2.0], [0.5, np.nan, np.nan]])
    metrics["nees_outage"] = np.array([[3.5, 4.0, 4.5], [np.nan, np.nan, np.nan]])
    return SweepResult("outage [s]", np.array([4.0, 4.0]), metrics)


def test_window_table_has_one_row_per_point_and_the_cells_of_that_window_only():
    """Fails if a row shows another window's scores or the cell format changes."""
    result = make_result()
    lines = format_window_table(result, "during", ROWS, value_label="policy", caption="settings")
    assert lines[0] == "settings"
    header, rule, first, second = lines[1:]
    assert len(set(map(len, [header, rule, first, second]))) == 1  # aligned columns
    assert header.startswith(f"{'policy':>{LABEL_WIDTH}}") and "rmse [m]" in header
    for text in ("ghost/step", "false/step", "missed", "id sw", "births/scan", "med rmse [m]"):
        assert text in header
    assert first.startswith(f"{ROWS[0]:>{LABEL_WIDTH}}")
    assert format_cell(result.metrics["during_position_rmse"][0], 3).strip() in first
    assert format_cell(result.metrics["during_ghost_rate"][1], 3).strip() in second
    assert format_median(result.metrics["during_position_rmse"][0]) in first.split()
    assert "9" not in first.split()[1:3]  # the after-window rmse (9) is not in the during table


def test_window_table_counts_the_valid_seeds_and_marks_undefined_scores():
    """Fails if undefined scores print a number, or the valid-seed count includes NaN seeds."""
    lines = format_window_table(make_result(), "during", ROWS)
    assert lines[-2].split()[-1] == "3/3" and lines[-1].split()[-1] == "2/3"
    assert "n/a" in lines[-1]  # the false-track rate is undefined for every seed

    empty = format_window_table(make_result(), "before", ROWS)
    assert all(line.split()[-1] == "0/0" for line in empty[-2:])


def test_outage_table_lists_every_outage_score_with_its_valid_counts():
    """Fails if an outage score is missing from the table or its undefined seeds are counted."""
    lines = format_outage_table(make_result(), ROWS)
    header = lines[0]
    for text, _ in OUTAGE_COLUMNS.values():
        assert text in header
    assert "med reacq [s]" in header and "med NEES" in header
    assert lines[-2].split()[-1] == "3/3" and lines[-1].split()[-1] == "1/0"


def test_row_labels_must_match_the_points_and_the_window_must_exist():
    """Fails if a label list of the wrong length is silently truncated, or a typo is accepted."""
    with pytest.raises(ValueError, match="row labels"):
        format_window_table(make_result(), "during", ["only one"])
    with pytest.raises(ValueError, match="row labels"):
        format_outage_table(make_result(), ROWS + ["extra"])
    with pytest.raises(ValueError, match="window"):
        format_window_table(make_result(), "while", ROWS)


def test_select_rows_keeps_the_requested_rows_in_order():
    """Fails if the rows of a plotted group are mixed up with another group's."""
    result = make_result()
    part = select_rows(result, [1])
    assert part.parameter == result.parameter and part.values.shape == (1,)
    np.testing.assert_array_equal(
        part.metrics["during_position_rmse"], result.metrics["during_position_rmse"][[1]]
    )
    swapped = select_rows(result, [1, 0])
    np.testing.assert_array_equal(
        swapped.metrics["reacquisition_time"][0], result.metrics["reacquisition_time"][1]
    )


def test_caption_gives_the_shared_settings_but_not_the_per_row_ones():
    """Fails if the caption claims the outage or policy of one row for the whole table."""
    config = OutageConfig(after_window=25.0, max_coast_time=12.0, aware_tentatives="freeze")
    caption = outage_caption(config)
    assert "after window 25 s" in caption and "max coast 12 s" in caption
    assert "freeze" in caption and "unless a row names them" in caption
    assert "match distance" in caption  # the Phase 6 settings are included
    assert "unaware" not in caption and "span" not in caption
