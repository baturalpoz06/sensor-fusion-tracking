"""Tests for the robustness tables.

Each test docstring names the condition that makes it fail without the feature.
"""

import numpy as np
import pytest

from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import PairedDifference, paired_difference
from fusion.mtt_report import COLUMN_WIDTH, format_cell, format_median
from fusion.robustness_experiment import RobustnessConfig, TrackerBelief
from fusion.robustness_metrics import FIELDS
from fusion.robustness_report import (
    DIAGNOSTIC_COLUMNS,
    LABEL_WIDTH,
    LEGEND,
    PAIRED_METRICS,
    PILOT_BANNER,
    SCORE_COLUMNS,
    WINDOW_COLUMNS,
    format_diagnostic_table,
    format_paired_cell,
    format_paired_table,
    format_score_table,
    format_window_table,
    has_window_scores,
    robustness_caption,
)

LABELS = ["neutral", "bias 1 deg", "offset 50 ms"]


def make_result() -> SweepResult:
    """Three rows by four seeds; every score NaN unless set below."""
    metrics = {name: np.full((3, 4), np.nan) for name in FIELDS}
    metrics["run_position_rmse"] = np.array(
        [[2.0, 2.0, 2.0, 2.0], [5.0, 6.0, np.nan, 7.0], [2.0, 3.0, 4.0, 3.0]]
    )
    metrics["run_missed_rate"] = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 1.0, 0.5], [0.0, 0.1, 0.0, 0.1]]
    )
    metrics["cross_rms"] = np.array(
        [[1.0, 1.0, 1.0, 1.0], [10.0, 12.0, np.nan, 14.0], [2.0, 2.0, 2.0, 2.0]]
    )
    metrics["nees_mean"] = np.array([[4.0, 4.0, 4.0, 4.0], [20.0, 30.0, 40.0, 50.0], [5.0] * 4])
    metrics["camera_true_accept_rate"] = np.array(
        [[0.9, 0.9, 0.9, 0.9], [0.3, 0.4, 0.5, 0.2], [0.9, 0.8, 0.9, 0.9]]
    )
    return SweepResult("disturbance", np.array([0.0, 1.0, 50.0]), metrics)


def cells(line: str) -> list[str]:
    """Split a table line into the first column and fixed-width cells."""
    return [line[:LABEL_WIDTH].strip()] + [
        line[LABEL_WIDTH + i : LABEL_WIDTH + i + COLUMN_WIDTH].strip()
        for i in range(0, len(line) - LABEL_WIDTH, COLUMN_WIDTH)
    ]


# --- scores and diagnostics -------------------------------------------------------------


def test_the_score_table_has_one_row_per_label_with_mean_interval_and_median():
    """Fails if rows are mixed up, a column is dropped, or a cell is not mean +- half-width."""
    result = make_result()
    lines = format_score_table(result, LABELS, value_label="disturbance", caption="CAPTION")
    assert lines[0] == "CAPTION"
    assert lines[1].lstrip().startswith("disturbance") and "rmse [m]" in lines[1]
    for header, _ in SCORE_COLUMNS.values():
        assert header in lines[1], header
    assert len(lines) == 2 + 1 + 3  # caption, header, rule, three rows
    body = lines[3:]
    assert [row[:LABEL_WIDTH].strip() for row in body] == LABELS
    expected = format_cell(result.metrics["run_position_rmse"][1], 3)
    assert expected in body[1]
    assert format_median(result.metrics["run_position_rmse"][1]) in body[1]  # the median column


def test_the_valid_seed_counts_are_for_the_rmse_and_the_missed_rate():
    """Fails if the last column counts the wrong metrics (a seed without a match is not valid)."""
    lines = format_score_table(make_result(), LABELS)
    assert lines[2].rstrip().endswith("4/4")
    assert lines[3].rstrip().endswith("3/4")  # row 1: rmse has a NaN seed, missed has none


def test_the_diagnostic_table_lists_the_error_and_camera_columns():
    """Fails if an error or camera diagnostic column is missing from the table."""
    lines = format_diagnostic_table(make_result(), LABELS)
    for header, _ in DIAGNOSTIC_COLUMNS.values():
        assert header in lines[0], header
    assert lines[2].rstrip().endswith("4/4")  # cross rms and NEES valid in the neutral row
    assert lines[3].rstrip().endswith("3/4")  # row 1: no cross-range error in one seed


def test_a_table_refuses_labels_that_do_not_match_the_rows():
    """Fails if a table is printed with labels shifted against its rows."""
    with pytest.raises(ValueError, match="row labels"):
        format_score_table(make_result(), LABELS[:2])


def test_a_table_refuses_a_result_that_lacks_its_metrics():
    """Fails if a table silently prints without a metric (a KeyError would mask the cause)."""
    result = make_result()
    del result.metrics["along_rms"]
    with pytest.raises(ValueError, match="lacks the metrics"):
        format_diagnostic_table(result, LABELS)


def test_window_scores_are_detected_and_printed_only_where_a_window_exists():
    """Fails if a window table is offered for a sweep without focus windows."""
    result = make_result()
    assert not has_window_scores(result)
    result.metrics["window_position_rmse"][1] = [8.0, 9.0, 10.0, 11.0]
    assert has_window_scores(result)
    lines = format_window_table(result, LABELS)
    for header, _ in WINDOW_COLUMNS.values():
        assert header in lines[0]
    assert "n/a" in lines[2 + 1]  # the neutral row has no window


# --- paired table ----------------------------------------------------------------------


def test_the_paired_table_compares_every_row_with_the_reference_seed_by_seed():
    """Fails if rows are compared by their means instead of per seed, or the sign is reversed.

    Row 1 has a NaN position error in seed 2: that seed is dropped from the rmse pair (and
    from the cross-range pair), and the table says so.
    """
    result = make_result()
    lines = format_paired_table(result, 0, LABELS, caption="PAIRED")
    assert lines[0] == "PAIRED" and len(lines) == 2 + 1 + 2  # the reference row is not listed
    assert [row[:LABEL_WIDTH].strip() for row in lines[3:]] == LABELS[1:]
    difference = paired_difference(
        result.metrics["run_position_rmse"][1], result.metrics["run_position_rmse"][0]
    )
    assert difference.mean == pytest.approx((3.0 + 4.0 + 5.0) / 3)
    assert format_paired_cell(difference) in lines[3]
    assert f"{difference.median:+.3g}" in lines[3]
    assert lines[3].rstrip().endswith("1/0/1/0/0")  # dropped seeds per metric, in column order
    assert lines[4].rstrip().endswith("0/0/0/0/0")
    assert lines[3].split()[0:2] == ["bias", "1"]


def test_the_paired_table_lists_one_header_per_metric_and_its_median():
    """Fails if a metric loses its mean-difference or median column."""
    header = format_paired_table(make_result(), 0, LABELS)[0]
    for text in PAIRED_METRICS.values():
        assert text in header and f"med {text}" in header
    assert "dropped" in header


def test_a_reference_in_the_middle_is_skipped_and_the_signs_follow_the_reference():
    """Fails if the reference is not the one subtracted: row - reference, whatever its index."""
    result = make_result()
    lines = format_paired_table(result, 2, LABELS)
    assert [row[:LABEL_WIDTH].strip() for row in lines[2:]] == ["neutral", "bias 1 deg"]
    neutral_row = lines[2]
    # neutral minus the offset row: rmse 2 - (2, 3, 4, 3) = (0, -1, -2, -1), mean -1
    assert "-1 +- 1.3" in neutral_row


@pytest.mark.parametrize("reference", [-1, 3])
def test_a_reference_outside_the_rows_is_refused(reference):
    """Fails if a bad reference index wraps around (negative indexing) instead of failing."""
    with pytest.raises(ValueError, match="reference"):
        format_paired_table(make_result(), reference, LABELS)


def test_the_paired_cell_formats_the_special_cases():
    """Fails if an undefined or single-seed difference prints a number it does not have."""
    nothing = PairedDifference(np.nan, np.nan, np.nan, np.nan, 0, 4)
    single = PairedDifference(2.0, np.nan, np.nan, 2.0, 1, 3)
    full = PairedDifference(1.5, 1.0, 2.0, 1.4, 4, 0)
    assert format_paired_cell(nothing) == "n/a"
    assert format_paired_cell(single) == "+2 +- ?"
    assert format_paired_cell(full) == "+1.5 +- 0.5"


# --- caption and legend ----------------------------------------------------------------


def test_the_caption_names_the_scene_and_both_match_distances():
    """Fails if a table cannot be read without the source: the scene or the distances are lost."""
    config = RobustnessConfig(belief=TrackerBelief(), diagnostic_match_distance=150.0)
    caption = robustness_caption(config, "crossing")
    assert caption.startswith("scenario crossing: ")
    assert "Phase 6 scores match within 50 m, diagnostics within 150 m" in caption
    assert not robustness_caption(config).startswith("scenario")


def test_the_legend_warns_about_the_reading_traps_and_the_pilot_banner_says_no_conclusions():
    """Fails if the legend drops the notes that keep a table from being misread."""
    text = "\n".join(LEGEND)
    for note in (
        "rmse is taken over matched",
        "not adjusted for the many",
        "not the difference of the medians",
        "conditioned on a match",
        "worst seeds",
        "counter-clockwise positive",
    ):
        assert note in text, note
    assert "No conclusions" in PILOT_BANNER
