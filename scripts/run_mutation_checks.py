"""Mutation checks: deliberately break the code in a scratch copy and require a named test to fail.

Usage:
    python scripts/run_mutation_checks.py [--only SUBSTRING] [--keep]

The repository is copied (without .git, .venv, results and caches) into a temporary directory.
For every mutation the exact text of one source file is replaced (the text must occur exactly
once), the tests named for that mutation are run in the copy, and the mutation counts as caught
only if one of them fails. The copy is put first on the import path and the script checks that
`fusion` is imported from it, so the original repository is never tested by mistake and never
modified. Exit status 1 if any mutation is missed or does not apply.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
IGNORED = shutil.ignore_patterns(
    ".git", ".venv", "results", "__pycache__", ".pytest_cache", ".ruff_cache", ".claude",
    "*.egg-info", "dist", "build",
)  # fmt: skip


class Mutation(NamedTuple):
    """One deliberate defect.

    Attributes:
        name: What is broken.
        path: Source file, relative to the repository.
        edits: (old text, new text) replacements; each old text must occur exactly once.
        tests: pytest node ids (file::test, parametrized variants included) that must catch it.
    """

    name: str
    path: str
    edits: tuple[tuple[str, str], ...]
    tests: tuple[str, ...]


IMM = "src/fusion/filters/imm.py"
TRACKER = "src/fusion/tracker/multi_target.py"
BIAS = "src/fusion/tracker/camera_bias.py"
SENS = "src/fusion/filters/sensitivity.py"
T_IMM = "tests/test_imm.py"
T_IMM_TRACKER = "tests/test_imm_tracker.py"
T_BIAS = "tests/test_camera_bias.py"
T_BIAS_TRACKER = "tests/test_bias_tracker.py"
T_SENS = "tests/test_sensitivity.py"
MIXING = f"{T_IMM}::test_mixing_uses_the_transition_matrix_the_right_way_round_with_the_spread_term"
PREDICT_MIXES = f"{T_IMM}::test_predict_starts_every_mode_from_the_mixed_estimate"
FD = f"{T_SENS}::test_the_sensitivity_matches_a_finite_difference_of_the_filter"
IMM_FD = f"{T_SENS}::test_the_imm_sensitivity_matches_a_finite_difference_with_the_mode_probabilities_fixed"  # noqa: E501
SINGLE_MODE = f"{T_IMM_TRACKER}::test_a_single_mode_imm_tracker_reproduces_the_default_tracker_bitwise"  # noqa: E501
GOLDEN = (
    "tests/test_phase6_regression.py::test_every_recorded_trial_is_reproduced",
    "tests/test_phase7_regression.py::test_every_recorded_outage_trial_is_reproduced",
)

MUTATIONS = (
    Mutation(
        "mixing uses Pi instead of Pi transposed",
        IMM,
        (("    c = transition.T @ mu\n", "    c = transition @ mu\n"),),
        (MIXING, PREDICT_MIXES),
    ),
    Mutation(
        "mixing drops the spread term",
        IMM,
        (('    P0 = np.einsum("ij,iab->jab", weights, P_modes) + spread  # noqa: N806\n',
          '    P0 = np.einsum("ij,iab->jab", weights, P_modes)  # noqa: N806\n'),),
        (MIXING, PREDICT_MIXES),
    ),
    Mutation(
        "combination drops the spread term",
        IMM,
        (('    P = np.einsum("j,jab->ab", mu, P_modes) + spread  # noqa: N806\n',
          '    P = np.einsum("j,jab->ab", mu, P_modes)  # noqa: N806\n'),),
        (f"{T_IMM}::test_combination_includes_the_spread_of_the_means",),
    ),
    Mutation(
        "predict skips the mixing",
        IMM,
        (("            c, x0, P0 = mix_modes(self._pi, self._mu, self._x_modes, self._P_modes)  # noqa: N806\n",  # noqa: E501
          "            c = self._pi.T @ self._mu\n            x0, P0 = self._x_modes.copy(), self._P_modes.copy()  # noqa: N806\n"),),  # noqa: E501
        (PREDICT_MIXES,),
    ),
    Mutation(
        "likelihoods are normalized in the linear domain",
        IMM,
        (("        total = logsumexp(log_post)\n",
          "        total = float(np.log(np.sum(np.exp(log_post))))\n"),),
        (f"{T_IMM}::test_probability_update_survives_log_likelihoods_that_underflow_in_the_linear_domain",),
    ),
    Mutation(
        "coordinated turn to the wrong side",
        IMM,
        (("            [0.0, 0.0, cos_theta, -sin_theta],\n            [0.0, 0.0, sin_theta, cos_theta],\n",  # noqa: E501
          "            [0.0, 0.0, cos_theta, sin_theta],\n            [0.0, 0.0, -sin_theta, cos_theta],\n"),),  # noqa: E501
        (f"{T_IMM}::test_the_turn_matrix_is_the_exact_motion_of_the_simulator",
         f"{T_IMM}::test_a_zero_turn_rate_is_constant_velocity_and_positive_turns_left"),
    ),
    Mutation(
        "the per-scan stay probability is used per step",
        IMM,
        (("    lam_step = lam ** (1.0 / steps_per_scan)\n", "    lam_step = lam\n"),),
        (f"{T_IMM}::test_the_step_matrix_raised_to_the_scan_length_is_the_scan_matrix",
         f"{T_IMM}::test_coasting_moves_the_probabilities_with_the_step_matrix"),
    ),
    Mutation(
        "covariance propagated as F^T P F",
        IMM,
        (("            self._P_modes[0] = mode.F @ self._P_modes[0] @ mode.F.T + mode.Q\n",
          "            self._P_modes[0] = mode.F.T @ self._P_modes[0] @ mode.F + mode.Q\n"),),
        (f"{T_IMM}::test_the_mode_covariance_is_propagated_as_F_P_Ft",),
    ),
    Mutation(
        "gating and association use the first mode instead of the combined estimate",
        IMM,
        (("            self._x, self._P, self.spread_trace = combine_modes(\n"
          "                self._mu, self._x_modes, self._P_modes\n            )\n",
          "            self._x, self._P, self.spread_trace = (\n"
          "                self._x_modes[0], self._P_modes[0], 0.0\n            )\n"),),
        (f"{T_IMM_TRACKER}::test_gating_and_association_receive_the_combined_estimate",),
    ),
    Mutation(
        "bias taken from the post-update track residual instead of the radar-camera difference",
        TRACKER,
        (("                deltas.append(float(camera[camera_rows[track_id], 0] - radar[row, 1]))\n",  # noqa: E501
          "                track = next(t for t in self._tracks if t.track_id == track_id)\n"
          "                own = float(np.arctan2(track.filter.x[1], track.filter.x[0]))\n"
          "                deltas.append(float(camera[camera_rows[track_id], 0]) - own)\n"),),
        (f"{T_BIAS_TRACKER}::test_a_real_bias_is_learned_from_the_pairs",
         f"{T_BIAS_TRACKER}::test_a_turn_is_not_read_as_a_bias"),
    ),
    Mutation(
        "tentative tracks form bias pairs",
        TRACKER,
        (("            if status is TrackStatus.CONFIRMED:\n"
          "                self._radar_rows = dict(group_pairs)\n",
          "            if True:\n"
          "                self._radar_rows.update(group_pairs)\n"),),
        (f"{T_BIAS_TRACKER}::test_a_track_confirmed_by_this_very_scan_forms_no_pair",),
    ),
    Mutation(
        "pairs with a radar measurement in two gates are used",
        TRACKER,
        (("                if row in self._radar_shared:\n", "                if False:\n"),),
        (f"{T_BIAS_TRACKER}::test_a_radar_measurement_in_the_gate_of_two_confirmed_tracks_forms_no_pair",),
    ),
    Mutation(
        "outlier gate centered on the current estimate",
        BIAS,
        (("            if not math.isfinite(delta) or abs(delta) > self._gate_half_width:\n",
          "            if not math.isfinite(delta) or abs(delta - self._b1) > self._gate_half_width:\n"),),  # noqa: E501
        (f"{T_BIAS}::test_the_gate_is_centered_on_zero_so_a_correct_pair_is_not_lost_with_a_wrong_estimate",),
    ),
    Mutation(
        "output correction adds V b instead of subtracting",
        SENS,
        (("    return x - V * bias, P + bias_variance * np.outer(V, V)\n",
          "    return x + V * bias, P + bias_variance * np.outer(V, V)\n"),),
        (f"{T_SENS}::test_the_output_correction_subtracts_the_scaled_sensitivity_and_adds_the_bias_variance",
         f"{T_BIAS_TRACKER}::test_the_oracle_bias_removes_the_cross_range_error_of_a_one_degree_bias"),
    ),
    Mutation(
        "sensitivity update without the K J_b term",
        SENS,
        (("        self.V = (np.eye(4) - step.K @ H) @ self.V + step.K @ bias_jacobian(model)\n",
          "        self.V = (np.eye(4) - step.K @ H) @ self.V\n"),),
        (FD, f"{T_SENS}::test_one_camera_update_gives_the_gain_as_sensitivity"),
    ),
    Mutation(
        "sensitivities are not mixed across modes",
        IMM,
        (("                v0 = mixing_weights(self._pi, self._mu)[1].T @ self._V_modes\n",
          "                v0 = self._V_modes\n"),),
        (f"{T_SENS}::test_the_sensitivities_are_mixed_with_the_mixing_weights",),
    ),
    Mutation(
        "the IMM path is on by default",
        TRACKER,
        (("    motion: MotionConfig | None = None\n",
          "    motion: MotionConfig | None = MotionConfig((ModeSpec(), ModeSpec(accel_std=3.0)))\n"),  # noqa: E501
         ("from fusion.filters.imm import IMMFilter, MotionConfig\n",
          "from fusion.filters.imm import IMMFilter, ModeSpec, MotionConfig\n")),
        GOLDEN,
    ),
    Mutation(
        "the bias estimate is on by default",
        TRACKER,
        (("    camera_bias: CameraBiasConfig | None = None\n",
          "    camera_bias: CameraBiasConfig | None = CameraBiasConfig()\n"),),
        GOLDEN,
    ),
    Mutation(
        "a random stream is inserted before the existing ones",
        "src/fusion/mtt_simulation.py",
        (('RNG_STREAMS = (\n    "trajectory",\n', 'RNG_STREAMS = (\n    "extra",\n    "trajectory",\n'),),  # noqa: E501
        ("tests/test_phase7_regression.py::test_phase_7_random_streams_keep_their_draws_when_streams_are_appended",
         "tests/test_improvement_experiment.py::test_the_random_streams_of_the_simulation_are_unchanged_and_nothing_is_appended"),
    ),
    Mutation(
        "a single-mode IMM takes a different path from the EKF",
        IMM,
        (("            self._x_modes[0] = mode.F @ self._x_modes[0]\n",
          "            self._x_modes[0] = mode.F @ self._x_modes[0] + 1e-12\n"),),
        (SINGLE_MODE, f"{T_IMM}::test_a_single_mode_imm_is_the_ekf_bit_for_bit"),
    ),
)


def copy_repository(destination: Path) -> Path:
    target = destination / "repo"
    shutil.copytree(ROOT, target, ignore=IGNORED)
    return target


def environment(copy: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(copy / "src"), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def check_isolation(copy: Path, env: dict[str, str]) -> None:
    """Raise unless `fusion` is imported from the copy when run in it."""
    out = subprocess.run(
        [sys.executable, "-c", "import fusion; print(fusion.__file__)"],
        cwd=copy, env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    if not Path(out).resolve().is_relative_to(copy.resolve()):
        raise RuntimeError(f"fusion is imported from {out}, not from the scratch copy {copy}")


def run_tests(copy: Path, env: dict[str, str], tests: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Run the named tests; return (all passed, ids of the failed ones)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-rf", *tests],  # noqa: E501
        cwd=copy, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    failed = [line.split(" ")[1].split(" - ")[0] for line in result.stdout.splitlines()
              if line.startswith("FAILED ")]  # fmt: skip
    return result.returncode == 0, failed


def apply(mutation: Mutation, copy: Path) -> str:
    """Write the mutated file into the copy and return its original text."""
    path = copy / mutation.path
    original = path.read_text(encoding="utf-8")
    text = original
    for old, new in mutation.edits:
        if text.count(old) != 1:
            raise ValueError(f"{mutation.name}: text occurs {text.count(old)} times in {mutation.path}")  # noqa: E501
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    return original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="", help="run only mutations whose name contains this")
    parser.add_argument("--keep", action="store_true", help="keep the scratch copy")
    args = parser.parse_args()
    chosen = [m for m in MUTATIONS if args.only in m.name]
    scratch = Path(tempfile.mkdtemp(prefix="mutation_checks_"))
    problems = 0
    try:
        copy = copy_repository(scratch)
        env = environment(copy)
        check_isolation(copy, env)
        print(f"scratch copy: {copy} ({len(chosen)} mutations)")
        for mutation in chosen:
            try:
                original = apply(mutation, copy)
            except ValueError as error:
                print(f"NOT APPLIED  {error}")
                problems += 1
                continue
            try:
                passed, failed = run_tests(copy, env, mutation.tests)
            finally:
                (copy / mutation.path).write_text(original, encoding="utf-8")
            named = {t.split("::")[-1] for t in mutation.tests}
            caught_by = [f for f in failed if f.split("::")[-1].split("[")[0] in named]
            status = "CAUGHT" if caught_by else "MISSED"
            problems += status == "MISSED"
            print(f"{status:<8} {mutation.name}")
            for test in caught_by or mutation.tests:
                print(f"           {'by' if caught_by else 'expected'} {test}")
    finally:
        if args.keep:
            print(f"kept {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    print(f"{len(chosen) - problems} of {len(chosen)} mutations caught")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
