"""Tests for the outage tables.

Each test docstring names the condition that makes it fail without the feature.
"""

from dataclasses import replace

import numpy as np
import pytest

from fusion.dropout import MarkovBursts, PeriodicFlicker, SingleOutage
from fusion.mtt_experiment import SweepResult
from fusion.mtt_report import COLUMN_WIDTH, format_cell, format_median
from fusion.outage_experiment import OutageConfig
from fusion.outage_metrics import OutageMetrics
from fusion.outage_report import (
    LABEL_WIDTH,
    OUTAGE_COLUMNS,
    format_coast_check,
    format_outage_table,
    format_window_table,
    longest_outage,
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
    first_cell = first[LABEL_WIDTH : LABEL_WIDTH + COLUMN_WIDTH]
    assert "9" not in first_cell  # the after-window rmse (9) is not in the during table


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
    assert "med reacq [s]" in header and "med NEES" in header and "med NEES bef" in header
    assert "after len [s]" in header
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
    assert "after len column" in caption
    assert "freeze" in caption and "unless a row names them" in caption
    assert "match distance" in caption  # the Phase 6 settings are included
    assert "unaware" not in caption and "span" not in caption


# --- known limitation: coast deletions, crossing caveat ------------------------------


def test_the_crossing_caveat_is_in_the_caption_of_crossing_tables_only():
    """Fails if a table of the crossing scenario hides that its targets cross inside outages."""
    config = OutageConfig()
    crossing = outage_caption(config, "crossing")
    assert "targets 0 and 1 cross at about 30 s" in crossing
    assert "whose after window starts before it" in crossing
    assert "read identity results from the separated scenario" in crossing
    for other in ("separated", "vanishing", None):
        assert "cross at about" not in outage_caption(config, other)


def test_the_longest_continuous_outage_of_each_specification():
    """Fails if the threshold of the coast check uses the span of a flicker, not its off time."""
    radar = ("radar",)
    base = OutageConfig()
    assert longest_outage(base) == 0.0
    assert longest_outage(replace(base, dropout=SingleOutage(radar, 20.0, 6.0))) == 6.0
    flicker = PeriodicFlicker(radar, 20.0, 30.0, 10.0, 3.0)
    assert longest_outage(replace(base, dropout=flicker)) == 3.0
    bursts = MarkovBursts(radar, 20.0, 30.0, 0.2, 2.0)
    assert longest_outage(replace(base, dropout=bursts)) == 30.0


def test_the_coast_check_flags_deletions_only_where_none_are_planned():
    """Fails if a deletion in a row without a planned one goes unreported, or if rows where
    deletions are the point (limit below the outage) or unaware rows are flagged."""
    radar = ("radar",)
    base = OutageConfig(max_coast_time=15.0)
    aware = replace(base, outage_policy="aware")
    configs = [
        replace(aware, dropout=SingleOutage(radar, 20.0, 4.0)),  # checked
        replace(aware, dropout=SingleOutage(radar, 20.0, 20.0)),  # limit below the outage
        replace(base, dropout=SingleOutage(radar, 20.0, 4.0)),  # unaware
        replace(aware, dropout=SingleOutage(radar, 20.0, 8.0)),  # checked
    ]
    labels = ["aware 4 s", "aware 20 s", "unaware 4 s", "aware 8 s"]
    counts = np.array([[0, 0, 0], [3, 3, 3], [0, 0, 0], [0, 1, 0]], dtype=float)
    result = SweepResult("x", np.arange(4.0), {"coast_deletions": counts})

    clean = format_coast_check(configs[:3], result, labels[:3])
    assert len(clean) == 1 and "1 aware rows" in clean[0] and clean[0].endswith("all 0")

    flagged = format_coast_check(configs, result, labels)
    assert "2 aware rows" in flagged[0] and "UNEXPECTED" in flagged[0]
    assert flagged[1:] == ["  aware 8 s: 1 deletions"]
