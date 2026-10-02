"""Multi-target tracking experiment: scenarios, trials and parameter sweeps."""

import numpy as np

# Initial states [x, y, vx, vy] of the multi-target scenarios. All targets stay
# inside the default field of view and more than 250 m from the sensor for 60 s.
SCENARIOS = {
    # Two targets on crossing courses (the stress case for association and ID
    # switches), one crossing the -x axis (bearing wraps through +-pi), one far away.
    "crossing": np.array(
        [
            [-450.0, 1000.0, 15.0, 0.0],
            [450.0, 1010.0, -15.0, 0.0],
            [-1200.0, 400.0, 0.0, -15.0],
            [-1500.0, -2000.0, 10.0, -3.0],
        ]
    ),
    # Three targets far apart on non-crossing courses: the easy case.
    "separated": np.array(
        [
            [-450.0, 600.0, 15.0, 0.0],
            [400.0, -1400.0, 0.0, 12.0],
            [-1800.0, 1800.0, 10.0, -8.0],
        ]
    ),
}
