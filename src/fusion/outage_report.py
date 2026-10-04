"""Text tables of an outage sweep: mean, confidence half-width and median per row.

One table per window (before, during, after the outage) with the Phase 6 scores, and one
table with the outage scores (reacquisition, identity, NEES, ghost lifetime, counters).
This module only formats numbers; it contains no prose about the results.

Known limitation (see format_coast_check): an aware tracker's coast clock counts steps since
the last measurement update of any sensor, and the camera update of a track is skipped when
its bearing gate overlaps the gate of another confirmed track (camera ambiguity). Two real
targets with close bearings can then coast without any update, and in a radar outage longer
than max_coast_time both are deleted although they exist. The coast del column of the outage
table counts these deletions; the experiments keep max_coast_time above their radar-only
outages, so it must read 0 in those rows.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from fusion.dropout import MarkovBursts, PeriodicFlicker, SingleOutage
from fusion.mtt_experiment import SweepResult
from fusion.mtt_metrics import seed_confidence_interval
from fusion.mtt_report import (
    COLUMN_WIDTH,
    COLUMNS,
    MEDIAN_WIDTH,
    VALID_WIDTH,
    format_cell,
    format_median,
    settings_caption,
)
from fusion.outage_experiment import OutageConfig
from fusion.outage_metrics import WINDOW_METRICS, WINDOWS

LABEL_WIDTH = 32

# Outage-score name -> (column header, significant digits of the mean).
OUTAGE_COLUMNS = {
    "reacquisition_time": ("reacq [s]", 3),
    "reacquired_fraction": ("reacquired", 3),
    "reacquisition_time_censored": ("reacq cens [s]", 3),
    "identity_kept": ("id kept", 3),
    "nees_outage": ("NEES outage", 3),
    "nees_before": ("NEES before", 3),
    "nees_outlier_fraction": ("NEES outliers", 3),
    "nees_samples": ("NEES samples", 4),
    "ghost_lifetime": ("ghost life [s]", 3),
    "ghost_censored": ("ghost cens", 3),
    "coast_deletions": ("coast del", 3),
    "tentative_drops": ("tent drops", 3),
}
# Scores that also get a median column, for heavy-tailed values: name -> column header.
WINDOW_MEDIANS = {
    "position_rmse": "med rmse [m]",
    "ghost_rate": "med ghost",
    "missed_rate": "med missed",
    "id_switches": "med id sw",
}
OUTAGE_MEDIANS = {
    "reacquisition_time": "med reacq [s]",
    "nees_outage": "med NEES",
    "ghost_lifetime": "med ghost [s]",
}


CROSSING_CAVEAT = (
    "scenario crossing: targets 0 and 1 cross at about 30 s, so identity and reacquisition "
    "scores of outages that cover that time mix coasting with the crossing (read identity "
    "results from the separated scenario)"
)


def outage_caption(config: OutageConfig, scenario: str | None = None) -> str:
    """One line with the settings shared by all rows of an outage table.

    The outage itself and the tracker policy differ between rows and are named in the row
    labels; the line gives the settings that do not (and the defaults that a row may
    override). For the crossing scenario it adds the caveat that the targets cross inside
    long outages.

    Args:
        config: Configuration of any row (for the shared settings).
        scenario: Name of the scenario of the table.
    """
    caption = (
        f"{settings_caption(config)}; after window {config.after_window:g} s (clipped at the "
        f"end of the run); aware tentatives {config.aware_tentatives} and max coast "
        f"{config.max_coast_time:g} s unless a row names them"
    )
    if scenario == "crossing":
        caption += f"; {CROSSING_CAVEAT}"
    return caption


def longest_outage(config: OutageConfig) -> float:
    """Longest continuous outage in seconds that a configuration can contain.

    A single outage: its duration; periodic flicker: the off time; random bursts: the whole
    span (a burst can in principle last that long).
    """
    spec = config.dropout
    if spec is None:
        return 0.0
    if isinstance(spec, SingleOutage):
        return spec.duration
    if isinstance(spec, PeriodicFlicker):
        return spec.off_time
    if isinstance(spec, MarkovBursts):
        return spec.span_length
    raise ValueError(f"unknown dropout specification {spec!r}")


def format_coast_check(
    configs: Sequence[OutageConfig], result: SweepResult, row_labels: Sequence[str]
) -> list[str]:
    """Report lines checking the coast deletion counter where no deletion is planned.

    An aware row whose max coast time is at least its longest continuous outage should
    delete nothing for coasting: tracks in a radar-only outage keep being updated by the
    camera, and in a blackout none coasts longer than the outage. A deletion there would be
    the known limitation described in the module docstring. Rows with a shorter limit, where
    deletions are the point of the experiment, are not checked.

    Args:
        configs: Configuration of each row.
        result: Sweep result of the rows.
        row_labels: Text of each row.

    Returns:
        One summary line, followed by one line per row with unexpected deletions.
    """
    checked = [
        i
        for i, config in enumerate(configs)
        if config.outage_policy == "aware" and config.max_coast_time >= longest_outage(config)
    ]
    totals = {i: float(np.nansum(result.metrics["coast_deletions"][i])) for i in checked}
    unexpected = [i for i, total in totals.items() if total > 0.0]
    n_seeds = result.metrics["coast_deletions"].shape[1]
    verdict = "all 0" if not unexpected else "UNEXPECTED deletions"
    lines = [
        f"coast deletion check: {len(checked)} aware rows with max coast >= longest outage "
        f"(no deletion planned), summed over {n_seeds} seeds: {verdict}"
    ]
    lines += [f"  {row_labels[i]}: {totals[i]:g} deletions" for i in unexpected]
    return lines


def _format_table(
    result: SweepResult,
    columns: Mapping[str, tuple[str, int]],
    medians: Mapping[str, str],
    valid_names: tuple[str, str],
    row_labels: Sequence[str],
    value_label: str,
    caption: str | None,
    confidence: float,
) -> list[str]:
    if len(row_labels) != len(result.values):
        raise ValueError(f"need {len(result.values)} row labels, got {len(row_labels)}")
    header = f"{value_label:>{LABEL_WIDTH}}" + "".join(
        f"{text:>{COLUMN_WIDTH}}" for text, _ in columns.values()
    )
    header += "".join(f"{text:>{MEDIAN_WIDTH}}" for text in medians.values())
    header += f"{'valid seeds':>{VALID_WIDTH}}"
    lines = [caption] if caption else []
    lines += [header, "-" * len(header)]
    for i, label in enumerate(row_labels):
        cells = "".join(
            f"{format_cell(result.metrics[name][i], digits, confidence):>{COLUMN_WIDTH}}"
            for name, (_, digits) in columns.items()
        )
        median_cells = "".join(
            f"{format_median(result.metrics[name][i]):>{MEDIAN_WIDTH}}" for name in medians
        )
        counts = "/".join(
            str(seed_confidence_interval(result.metrics[name][i], confidence).n_valid)
            for name in valid_names
        )
        lines.append(f"{label:>{LABEL_WIDTH}}{cells}{median_cells}{counts:>{VALID_WIDTH}}")
    return lines


def format_window_table(
    result: SweepResult,
    window: str,
    row_labels: Sequence[str],
    value_label: str = "row",
    caption: str | None = None,
    confidence: float = 0.95,
) -> list[str]:
    """Table of the Phase 6 scores in one window: one row per sweep point.

    The last column counts the seeds with a defined position error and missed rate.

    Args:
        result: Sweep result from fusion.outage_experiment.outage_sweep.
        window: "before", "during" or "after".
        row_labels: Text of the first column, one per point.
        value_label: Header of the first column.
        caption: Optional first line.
        confidence: Coverage of the Student t interval across seeds.

    Raises:
        ValueError: If the window is unknown or row_labels does not match the points.
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")
    columns = {f"{window}_{m}": COLUMNS[m] for m in WINDOW_METRICS}
    medians = {f"{window}_{m}": text for m, text in WINDOW_MEDIANS.items()}
    valid = (f"{window}_position_rmse", f"{window}_missed_rate")
    return _format_table(
        result, columns, medians, valid, row_labels, value_label, caption, confidence
    )


def format_outage_table(
    result: SweepResult,
    row_labels: Sequence[str],
    value_label: str = "row",
    caption: str | None = None,
    confidence: float = 0.95,
) -> list[str]:
    """Table of the outage scores: one row per sweep point.

    The last column counts the seeds with a defined reacquisition time and outage NEES.
    Averages are over the seeds where the score is defined; read reacquired fraction and
    the NEES sample count with them.

    Args:
        result: Sweep result from fusion.outage_experiment.outage_sweep.
        row_labels: Text of the first column, one per point.
        value_label: Header of the first column.
        caption: Optional first line.
        confidence: Coverage of the Student t interval across seeds.

    Raises:
        ValueError: If row_labels does not match the points.
    """
    valid = ("reacquisition_time", "nees_outage")
    return _format_table(
        result, OUTAGE_COLUMNS, OUTAGE_MEDIANS, valid, row_labels, value_label, caption, confidence
    )


def select_rows(result: SweepResult, rows: Sequence[int]) -> SweepResult:
    """The sweep result restricted to the given point indices, in that order."""
    index = np.asarray(rows, dtype=int)
    return SweepResult(
        result.parameter,
        result.values[index],
        {name: values[index] for name, values in result.metrics.items()},
    )
