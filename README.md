# sensor-fusion-tracking

![CI](https://github.com/baturalpoz06/sensor-fusion-tracking/actions/workflows/ci.yml/badge.svg)

Radar and camera target tracking: an extended Kalman filter with a constant-velocity model,
gated global association, track lifecycle management, sensor outages, and a simulator with
clutter, missed detections and disturbances.

## Robustness measurement (Phase 8a)

Phase 8a measures how the current tracker degrades when its beliefs about the world are wrong.
It changes no tracker logic: the truth side (the simulator) and the belief side (the tracker
configuration) are separate objects, so the real world and the tracker's assumptions can differ.

- Truth (`fusion.maneuvers.TruthDisturbance`): coordinated turns, sudden accelerations and
  random turns of the targets, a constant camera bearing bias, a camera time offset, and a
  camera that sits away from the radar (lever arm).
- Belief (`fusion.robustness_experiment.TrackerBelief`): scale factors on the noise levels the
  tracker assumes (radar range, radar bearing, camera bearing, process noise) and the camera
  position it assumes.
- Every knob defaults to neutral; with the defaults the results reproduce the Phase 6 and
  Phase 7 regression fixtures bit for bit.
- Diagnostics (counters only, no change of tracker behavior): the camera accept rate, and an
  along-range / cross-range split of the position error relative to the line of sight from the
  radar, plus NEES.

Run the measurement (results go to the git-ignored `results/` folder):

```
python scripts/run_robustness_experiment.py --scenario all --seeds 50 --workers 8
python scripts/run_robustness_experiment.py --scenario noise --layouts separated
python scripts/run_robustness_experiment.py --pilot --workers 8
```

`--scenario` is one of `noise`, `maneuver`, `camera`, `lever-arm`, `geometry` or `all`.
Every sweep runs the same seeds at every row and contains its neutral row, so the report
compares each row with the neutral one seed by seed (mean difference with a 95% t interval,
the median of the differences, and the number of seeds dropped for undefined values).
`--pilot` runs 5 seeds into `results/pilot` only to check that everything runs.

## IMM motion model and camera-bias estimate (Phase 8b)

Phase 8b adds two optional tracker features, both off by default (with the defaults every
earlier regression fixture is still reproduced bit for bit), and a before / after comparison.

- IMM (`fusion.filters.imm`, `TrackerConfig.motion`): every track runs several linear motion
  modes of the same state: constant velocity with a low and a high process noise, or constant
  velocity plus coordinated turns at fixed known rates to the left and right. Modes are mixed
  with a transition matrix given per radar scan, and the mode probabilities are updated in the
  log domain. Gating, the association cost and the camera ambiguity rule use the combined
  estimate of each track. A single-mode IMM is the plain EKF, bit for bit.
- Camera bias (`fusion.tracker.camera_bias`, `TrackerConfig.camera_bias`): one scalar estimate
  of a constant camera bearing bias, shared by all tracks and learned from simultaneous
  radar-camera bearing differences of confirmed tracks (these do not depend on any track
  estimate, and target motion cancels in them). The tracks run on the raw camera data; each
  carries the sensitivity of its estimate to a constant bias (`fusion.filters.sensitivity`) and
  the reported snapshots are corrected by the shared estimate, with its uncertainty added to the
  covariance.
- `TrackerConfig.record_diagnostics` switches the per-step logs (camera, mode and bias logs) off;
  nothing in the tracker reads them.

The comparison runs several tracker variants ("arms") on the same simulated data and compares
each with the Phase 8a tracker seed by seed. Parameters are chosen on a separate tuning seed
set (1000-1019) by fixed rules and frozen before the evaluation seeds (0-49) are used:

```
python scripts/run_improvement_experiment.py --stage tune --workers 8
python scripts/run_improvement_experiment.py --stage timing
python scripts/run_improvement_experiment.py --stage eval --pilot --workers 8
python scripts/run_improvement_experiment.py --stage eval --workers 8
```

The last command refuses to run until the tuned values are set in
`fusion.improvement_experiment.FROZEN` at a commit that is an ancestor of `HEAD` with no
uncommitted changes to tracked files. It writes the tables, a numbers-only headline file and
`criteria_8b.txt`, the mechanical verdicts of the success criteria in
`fusion.improvement_criteria`, to the git-ignored `results/` folder.
