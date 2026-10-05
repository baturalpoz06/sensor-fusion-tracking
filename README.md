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
