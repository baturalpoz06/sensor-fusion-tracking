# Theory: the mathematics behind the tracker

This document explains the mathematics that the code in `src/fusion` implements: how the filters,
the association logic, the track management, the statistics and the two optional additions (an IMM
motion model and a camera-bias estimate) work, and exactly where each one stops being true. It is
written so that a reader who knows linear algebra and basic probability can derive each step and
see where its assumptions stop holding. It does not repeat the results; [RESULTS.md](RESULTS.md) holds those and
is linked wherever a claim rests on a measurement.

Contents: 0 How to read this · 1 Prerequisites · 2 State space and the constant-velocity model ·
3 Sensors and track initialisation · 4 The linear Kalman filter · 5 The extended Kalman filter ·
6 Gating · 7 Data association and the camera ambiguity rule · 8 Track management and outage-aware
coasting · 9 Evaluation statistics and criteria · 10 The IMM filter · 11 Camera bias ·
12 Truth-side disturbances · 13 Assumptions, approximations and failure modes ·
14 Equation sheet.

## 0. How to read this

### Shape of every section

Each of sections 2 to 12 has the same parts, in this order: a **key idea** in a few lines, a small
**worked example**, the **equations as implemented** with `file:line` references, an **optional
derivation** in a collapsed block (skip it on a first read), and a table of **assumptions and
limits**. Section 13 is an index of what the repository does and does not show, and section 14 is a
one-page sheet of the equations.

### Four kinds of statement

Every claim in the assumptions tables carries one of these tags, and the text uses them the same
way.

| Tag | Meaning | Example |
|---|---|---|
| **Identity** | Follows from definitions and algebra; true for any data once the stated objects are defined. | The Joseph form equals (I − KH)P when K = P Hᵀ S⁻¹. |
| **Assumption** | A modelling choice of the simulator or the tracker. The world may not obey it. | Target acceleration is white, zero-mean and Gaussian. |
| **Approximation** | A controlled error: linearisation, moment matching, a heuristic rule. | The EKF replaces h by its first-order Taylor expansion. |
| **Empirical** | A measurement made in this repository, valid only for the scene, seeds and match distance named with it. | RESULTS.md Table D. |

Where the tables below write "Empirical (negative)", "Empirical (cost)", "Empirical (limitation, not
triggered)", "Design choice" or "Hypothesis", they are Empirical or Assumption statements with the
extra word saying what kind of result it is; "Hypothesis" marks a reading that no experiment in the
repository tests.

### Conventions and default parameters

- State vector `[x, y, vx, vy]` in metres and metres per second. Bearing is measured
  counter-clockwise from the +x axis, in radians. The model functions return values in [−π, π].
  Only the bearing component of an innovation is wrapped to that range before it is used; the radar
  range residual is never wrapped. Simulated measurements are not wrapped.
- In quoted results, `±` is the half-width of a 95 % Student t interval over the seeds (50 unless
  stated), as in RESULTS.md section 0. The one exception is the single-target table of RESULTS.md
  section 1, where `±` is a standard deviation and is labelled SD here.
- Radar and camera sit at the origin unless a section says otherwise; the camera model assumes the
  origin unless it is given another position.
- Time step Δt = 0.1 s. The radar scans every 10 steps (1 Hz), the camera at every step (10 Hz).
- Radar range noise σ_r = 5 m, radar bearing noise σ_β = 2° (0.03491 rad), camera bearing noise
  σ_c = 0.1° (0.001745 rad), target acceleration σ_a = 0.5 m/s², prior velocity std σ_v = 20 m/s
  (`src/fusion/experiment.py:45-53`, `src/fusion/mtt_simulation.py:121-129`).
- Field of view: full circle between 50 m and 3000 m (`src/fusion/mtt_simulation.py:52-55`).
- Tracker: 99 % gates, 3-of-5 confirmation, deletion after more than 5 consecutive misses, minimum
  range 50 m, maximum coasting time 15 s (`src/fusion/tracker/multi_target.py:70-82`,
  `src/fusion/tracker/track.py:33-35`).

Display equations are written as math fences (code fences whose language is `math`) rather than as
`$$` blocks. Rendering this file through GitHub's Markdown renderer showed that its Markdown pass
removes the backslash from sequences such as `\\`, `\,` and `\{` inside `$$` blocks, which breaks
matrix rows and spacing, while a math fence passes the equation through unchanged.

### Examples: exact or illustrative

Every worked example is tagged. **Exact** means it was recomputed with the repository function named
in the example, at commit a7dc570, and is rounded as shown. **Illustrative** means a simplification
was made on purpose, and the example says which one. No example uses a number that was not
recomputed or copied from a named source.

### Source references

`path:line` references are those of commit a7dc570; the tests named next to an equation pin its
behaviour if the lines move. Results are cited as RESULTS.md sections. Two kinds of result are not in
RESULTS.md and are cited by file: the pair-level rows of `results/improvement_bias.txt` and the
per-seed values in `results/improvement_results.pkl`. Both are git-ignored and are produced by
`python scripts/run_improvement_experiment.py --stage eval --workers 14` (RESULTS.md section 9).

Reproducibility of those two files. The evaluation run states commit 16185d2 and seeds 0 to 49 in the
first line of `results/criteria_8b.txt`; `improvement_bias.txt` has no header and is tied to that run
only by its file date (2026-10-07), and the pickle by the same date. The source of the tracker, the
estimators and the scores is unchanged between 16185d2 and a7dc570 (the only change to
`improvement_experiment.py` is a docstring), but the evaluation was not re-run for this document. The
numbers taken from these two files were recomputed from the local copies; they cannot be
reproduced from a clean checkout without running the evaluation stage again (about 90 minutes with 14
workers, RESULTS.md section 9), and the files are not part of the repository.

### Notation

| Symbol | Meaning | Unit |
|---|---|---|
| x, x̂ | true state, state estimate | m, m/s |
| x̂⁻, P⁻ | predicted estimate and covariance (before the update) | |
| x̂⁺, P⁺ | updated estimate and covariance (after the update) | |
| F, G, Q | transition matrix, noise input matrix, process noise covariance | |
| σ_a | std of the random acceleration | m/s² |
| z, h(x), R | measurement, measurement function, measurement noise covariance | |
| H | Jacobian of h at the predicted state | |
| ν, S, K | innovation, innovation covariance, Kalman gain | |
| d² | squared Mahalanobis distance νᵀ S⁻¹ ν | |
| γ_m | χ² gate threshold for m degrees of freedom | |
| M, N, K | confirmation hits, window, tolerated consecutive misses (3, 5, 5); not the Kalman gain K, which is a matrix | scans |
| μ_j, Π, c_j | IMM mode probability, mode transition matrix, predicted mode probability | |
| V, b, b̂ | sensitivity dx̂/db, true camera bias, estimated bias | m/rad, rad |
| e, ε | estimation error, NEES | |

### Reading orders

- Minimum path for the single-target filter and the tracker: 2 → 3 → 4 → 5 → 6 → 7 → 8.
- Evaluation and statistics only: 9 (needs section 5 for NEES).
- The two additions: 10 (IMM) and 11 (camera bias) are independent of each other.

## 1. Prerequisites

| Topic | What you need | Used in |
|---|---|---|
| Matrices | Product, transpose, inverse, solving S y = r instead of inverting, symmetric positive definite matrices, Cholesky factor S = L Lᵀ | 2 to 11 |
| Gaussians | Mean and covariance, linear maps, sums of independent variables, conditioning | 4, 5, 10, 11 |
| χ² and t distributions | Sum of squared standard normals; the t interval of a mean | 6, 9 |
| Calculus | Partial derivatives, Jacobian, first-order Taylor expansion | 3, 5, 11 |
| Trigonometry | atan2, angle wrapping, rotation matrices | 3, 5, 10, 11 |
| Optimisation | Assignment problem, cost matrix | 7 |
| Statistics | Confidence interval, paired difference, hypothesis margins | 9 |

The ideas that cause most difficulty, in the order you meet them:

1. P is the filter's belief about its own error, not the error. Whether the belief is right is a
   separate question (consistency, section 9).
2. The innovation covariance S contains the measurement noise R. A measurement can disagree with a
   prediction because either one is noisy (section 5).
3. The EKF is the linear filter applied to a first-order copy of a nonlinear model (section 5).
4. A bearing-only sensor constrains one direction only (sections 3 and 5).
5. IMM mixing averages beliefs, not measurements (section 10).
6. A constant bearing bias and a rotated trajectory look the same to a camera (section 11).

<details>
<summary>Gaussian facts used throughout (optional)</summary>

Let x ~ N(m, P).

- Linear map: y = A x + c gives y ~ N(A m + c, A P Aᵀ).
- Independent sum: covariances add.
- Conditioning: if (x, z) is jointly Gaussian with Cov(x, z) = C and Var(z) = S, then
  x given z ~ N(m_x + C S⁻¹ (z − m_z), P − C S⁻¹ Cᵀ).
- Quadratic form: if ν ~ N(0, S) in m dimensions, then νᵀ S⁻¹ ν ~ χ²_m. The sum of n independent
  such variables is χ² with n m degrees of freedom.
- t statistic: for n independent normal values with sample mean x̄ and sample std s,
  (x̄ − μ) / (s / √n) follows a t distribution with n − 1 degrees of freedom.
- Cholesky: S = L Lᵀ with L lower triangular; y = L⁻¹ ν has identity covariance, so
  νᵀ S⁻¹ ν = yᵀ y.

</details>

## 2. State space and the constant-velocity model

**Key idea.** The tracker believes that a target moves in a straight line at constant velocity and
that the only disturbance is a small random acceleration applied at every step. The state holds
position and velocity; the covariance P describes how unsure the filter is about them. Prediction
moves the state with F, transports P with F and adds Q (the target may have been pushed), which
need not increase every variance; an update removes uncertainty along the measured directions.

**Example 1 (exact; `cv_process_noise`, `ExtendedKalmanFilter.predict`).** For Δt = 0.1 s and
σ_a = 0.5 m/s²:

- Q has the entries Q_pp = σ_a² Δt⁴/4 = 6.25e-6 m², Q_pv = σ_a² Δt³/2 = 1.25e-4 m²/s and
  Q_vv = σ_a² Δt² = 2.5e-3 m²/s², on the x and y blocks separately.
- Start from the radar-initialised covariance of example 2 (section 3), diag(25, 1218.47, 400, 400),
  and predict once. The x variance becomes 25 + Δt² · 400 + Q_pp = 29.00000625 m², the x–vx
  covariance becomes Δt · 400 + Q_pv = 40.000125, and the vx variance becomes 400.0025. A
  position–velocity correlation has appeared that was not there before.
- Predicting for 10 s with no update (100 steps) turns the x std from 5 m into 200.08 m and the y std
  from 34.9 m into 203.04 m, almost entirely through the 20 m/s velocity prior (20 m/s × 10 s).

### Equations as implemented

Motion model, with a_k the random acceleration held constant during step k:

```math
x_{k+1}=F\,x_k+G\,a_k,\qquad a_k\sim\mathcal N(0,\sigma_a^2 I_2)
```

```math
F=\begin{pmatrix}I_2&\Delta t\,I_2\\0&I_2\end{pmatrix},\qquad
G=\begin{pmatrix}\tfrac12\Delta t^2\,I_2\\ \Delta t\,I_2\end{pmatrix},\qquad
Q=\sigma_a^2\,G G^{\top}=\sigma_a^2\begin{pmatrix}\tfrac14\Delta t^4 I_2&\tfrac12\Delta t^3 I_2\\ \tfrac12\Delta t^3 I_2&\Delta t^2 I_2\end{pmatrix}
```

Prediction:

```math
\hat x^-=F\hat x,\qquad P^-=F P F^{\top}+Q
```

- F: `src/fusion/filters/base.py:8-17`. G and Q: `base.py:20-34`. Prediction: `base.py:96-99`.
- The simulator draws the same acceleration and applies the same update to the true state:
  `src/fusion/scenario.py:34-42`. The acceleration std is shared by truth and filter
  (`src/fusion/mtt_simulation.py:101-102`, `src/fusion/experiment.py:30-31`).

<details>
<summary>Optional derivation: where G and Q come from</summary>

Hold the acceleration a constant over one step. Integrating twice gives
p(Δt) = p + v Δt + ½ a Δt² and v(Δt) = v + a Δt. The first two terms are F x; the last is G a with
the G above. Because a has covariance σ_a² I₂, the noise term G a has covariance
σ_a² G Gᵀ. Multiplying out the 4 × 2 matrix G by its transpose gives the block form of Q.

Q has rank 2, not 4: only two independent random numbers (a_x and a_y) enter the 4-dimensional state
at each step. P still stays positive definite, because P⁻ = F P Fᵀ + Q is positive definite whenever P
is, and the initial P₀ is.

This is the discrete white-noise acceleration model: the acceleration is independent from step to
step. It is not the same as the continuous-time white-noise model, whose Q differs in the entries.
The choice only matters because the simulator and the filter use the same one.

</details>

| Kind | Statement | Reference |
|---|---|---|
| Assumption | Acceleration is white (independent per step), zero-mean, Gaussian, with the same std in x and y | `base.py:20-34`, `scenario.py:36` |
| Assumption | In the neutral runs the filter is given the σ_a that generated the data | README Limitations, `mtt_simulation.py:101-102`, `experiment.py:30-31` |
| Approximation | Acceleration is constant within a step, so the model is exact only at the step boundaries | `base.py:21-24` |
| Identity | With a linear model and Gaussian noise, prediction is exact: P⁻ is the covariance of x_{k+1} given the past | `base.py:96-99` |
| Empirical | A maneuvering target (turn, sustained acceleration) violates the zero-mean white assumption; the plain EKF falls behind its targets (a missed target at 50 m is mostly a track that is alive but more than 50 m away): turn 5 deg/s, scene separated, 50 m, d missed +0.0751 ± 0.0099, ID switches 3.3 ± 0.33 | RESULTS.md [section 5.2](RESULTS.md#52-maneuvers-scene-separated-and-crossing) |
| Empirical | Raising σ_a lowers the window missed rate on the maneuver rows (reduction 0.0263 ± 0.0036, 0.0677 ± 0.0037 and 0.0767 ± 0.0039 at σ_a = 1, 3 and 5; mean over the 30 winnable row-blocks) and costs RMSE on the neutral row, where nothing maneuvers (0.143 ± 0.038 m, 0.804 ± 0.097 m and 1.24 ± 0.14 m; mean over four blocks, 50 m) | RESULTS.md [Table D](RESULTS.md#table-d-values-plotted-in-the-hero-figure-4-block-means) |

## 3. Sensors and track initialisation

**Key idea.** The radar measures range and bearing: it is good at range (5 m) and poor across the
line of sight (a 2° bearing error is 35 m at 1000 m). The camera measures bearing only, but twenty
times more precisely (0.1°, 1.7 m at 1000 m), and says nothing about range. The two are
complementary, which is why fusing them helps most across the line of sight. A new track is started
from one radar measurement; its covariance is that measurement's noise converted to Cartesian
coordinates.

**Example 2 (exact; `radar_initial_estimate`).** First radar measurement r = 1000 m, β = 0:

- x₀ = (1000, 0, 0, 0) and diag(P₀) = (25, 1218.47, 400, 400). The y entry is
  r² σ_β² = 10⁶ × 0.00121847 = 1218.47 m², a std of 34.9 m.
- A camera bearing at the same range has a cross-range std of 1000 × 0.001745 = 1.745 m: twenty
  times smaller. At 2000 m the radar cross-range std is 69.8 m and the camera's is 3.49 m.

### Equations as implemented

Radar (`src/fusion/sensors/radar.py`) and camera (`src/fusion/sensors/camera.py`):

```math
z_r=\begin{pmatrix}\sqrt{p_x^2+p_y^2}\\ \operatorname{atan2}(p_y,p_x)\end{pmatrix}+v_r,\quad v_r\sim\mathcal N\big(0,\operatorname{diag}(\sigma_r^2,\sigma_\beta^2)\big)
```

```math
z_c=\operatorname{atan2}(p_y-c_y,\;p_x-c_x)+b+v_c,\quad v_c\sim\mathcal N(0,\sigma_c^2)
```

- Radar model h and R: `radar.py:75-79`; simulated measurement: `radar.py:32-41`.
- Camera model h and R: `camera.py:69-74`; the simulator adds the constant bias b at `camera.py:43`.
  The tracker's camera model has no b: the tracker does not know it.
- A model is rejected when the target is closer than 10⁻⁶ m to the sensor, where the bearing is
  undefined (`src/fusion/sensors/base.py:9`, `base.py:33-51`).

Initialisation from the first radar measurement (r, β) (`radar.py:100-135`):

```math
\hat x_0=(r\cos\beta,\;r\sin\beta,\;0,\;0),\qquad
J=\begin{pmatrix}\cos\beta&-r\sin\beta\\ \sin\beta&r\cos\beta\end{pmatrix},\qquad
P_0=\begin{pmatrix}J\,\operatorname{diag}(\sigma_r^2,\sigma_\beta^2)\,J^{\top}&0\\0&\sigma_v^2 I_2\end{pmatrix}
```

The detection model of the multi-target simulation, by default: each target inside the field of view
is detected with probability 0.9 by each sensor, and each scan also holds a Poisson number (mean 5)
of clutter points (many experiments set the clutter rate to 0), uniform in range and bearing (`src/fusion/mtt_simulation.py:121-131`, `:191-223`).

<details>
<summary>Optional derivation: why P₀ looks like that</summary>

The position is the nonlinear function (r cos β, r sin β) of the measurement (r, β). To first order,
a small error (δr, δβ) moves the position by J (δr, δβ), so the position covariance is
J diag(σ_r², σ_β²) Jᵀ. For β = 0 this is diag(σ_r², r² σ_β²): the range error moves x, the bearing
error moves y by r δβ.

The velocity is unknown, so it starts at zero with a large variance σ_v². Zero is the mean of a
symmetric prior over directions; σ_v = 20 m/s is comparable to or larger than the target speeds in
the scenes (about 10 to 15 m/s).

The first-order picture is not exact. The true distribution of the converted position is a short arc,
and its mean is pulled towards the radar: for β = 0 the mean of x is r · exp(−σ_β²/2), about
0.6 m closer than r at 1000 m. The code does not correct this (analytic remark; the repository does
not measure it).

</details>

| Kind | Statement | Reference |
|---|---|---|
| Assumption | Radar noise is Gaussian, range and bearing independent, with the std the tracker believes | `radar.py:75` |
| Assumption | The camera is at the position the tracker assumes (default origin), shares the radar's clock, and has no bias | `camera.py:62-70` |
| Assumption | Detections are independent across scans and sensors; clutter is Poisson and uniform in range and bearing | `mtt_simulation.py:191-223` |
| Approximation | Position covariance of a new track is a first-order propagation of the polar noise | `radar.py:125-134` |
| Approximation | Initial velocity is zero with std σ_v, not estimated | `radar.py:123`, `:134` |
| Empirical | Single target, far scene, RMSE on radar steps: radar-only EKF 26.31 ± 7.21 m, fusion EKF 2.56 ± 0.60 m (mean ± SD, 50 seeds) | RESULTS.md [section 1](RESULTS.md#1-radar-vs-fusion-single-target) |
| Empirical | A constant camera bias of 1° adds +22.4 ± 0.16 m RMSE; an unknown 50 m lever arm adds +37.9 ± 0.63 m (scene separated, 50 m match) | RESULTS.md [section 5.1](RESULTS.md#51-camera-bearing-bias-time-offset-and-lever-arm-scene-separated) |

## 4. The linear Kalman filter

**Key idea.** A Kalman update blends a prediction with a measurement, weighting each by how certain
it is. The weight is the gain K. The same equations hold for the radar and the camera in the EKF;
this section shows them in the one case where nothing is approximated. The repository's linear
filter (`src/fusion/filters/kalman.py`) measures position directly and is used only in tests, as a
reference for the EKF.

**Example 3 (illustrative: one scalar quantity instead of the four-dimensional state).** A prior
variance P = 4 and a measurement variance R = 1, with H = 1. Then S = P + R = 5, K = P / S = 0.8, and
the posterior variance is (1 − K)² · 4 + K² · 1 = 0.16 + 0.64 = 0.8, equal to (1 − K) P = 0.8. The
posterior is more certain than either input (0.8 is less than both 4 and 1). With R → ∞ the gain
goes to 0 and the measurement is ignored; with R → 0 the gain goes to 1 and the measurement is
trusted completely.

### Equations as implemented

```math
\nu=z-H\hat x^-,\qquad S=HP^-H^{\top}+R,\qquad K=P^-H^{\top}S^{-1}
```

```math
\hat x^+=\hat x^-+K\nu,\qquad P^+=(I-KH)\,P^-\,(I-KH)^{\top}+K R K^{\top}
```

- `src/fusion/filters/kalman.py:52-68`: innovation line 59, S line 60, K as a linear solve line 62,
  state line 64, Joseph form lines 67-68. H and R: `kalman.py:44-50`.
- The EKF uses the same code path with H replaced by a Jacobian: `src/fusion/filters/ekf.py:53-70`.

<details>
<summary>Optional derivation: Kalman gain from Gaussian conditioning, and the Joseph form</summary>

The prediction says x ~ N(x̂⁻, P⁻). The measurement is z = H x + v with v ~ N(0, R) independent of x.
Then (x, z) is jointly Gaussian with Cov(x, z) = P⁻ Hᵀ and Var(z) = H P⁻ Hᵀ + R = S. By the
conditioning rule of section 1, x given z has mean x̂⁻ + P⁻ Hᵀ S⁻¹ (z − H x̂⁻), which is x̂⁺ with
K = P⁻ Hᵀ S⁻¹, and covariance P⁻ − P⁻ Hᵀ S⁻¹ H P⁻ = (I − K H) P⁻.

The Joseph form (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ equals (I − K H) P⁻ when K = P⁻ Hᵀ S⁻¹:
expanding gives P⁻ − K H P⁻ − P⁻ Hᵀ Kᵀ + K S Kᵀ, and K S = P⁻ Hᵀ makes the last two terms cancel.
**Identity.** The Joseph form is also valid for any K, and in exact arithmetic it is a sum of two
positive semi-definite terms. In floating point it is numerically more robust than the short form
(I − K H) P⁻, which can lose symmetry and definiteness through cancellation, but it gives no
universal guarantee: rounding can still leave a small asymmetry or, for badly conditioned matrices,
a slightly indefinite result. That robustness is why the code uses it (`ekf.py:67-69`).

The gain is also optimal. The trace of the Joseph-form covariance is quadratic and convex in K with
gradient 2 (K S − P⁻ Hᵀ), which vanishes at K = P⁻ Hᵀ S⁻¹: no other gain gives a smaller total
posterior variance.

</details>

| Kind | Statement | Reference |
|---|---|---|
| Identity | With a linear model, Gaussian noise and known F, Q, H, R, the update gives the exact posterior mean and covariance, and K minimises the posterior covariance | derivation above |
| Assumption | The model matrices are known and the noises are white, Gaussian and independent of each other | section 2 |
| Identity | The posterior covariance does not depend on the measurement values, only on H, R and the prior | `kalman.py:60-68` |

## 5. The extended Kalman filter

**Key idea.** The radar and camera measure range and bearing, which are nonlinear in position. The
EKF replaces the measurement function h by its tangent at the predicted state, then applies the
Kalman update of section 4 to that tangent. Two details decide whether it works in practice: the
bearing difference must be wrapped to [−π, π], and the camera, which measures bearing only, corrects
mainly the direction across the line of sight.

**Example 4 (exact; `ekf_update`, `SensitivityEKF`).** Take the track of example 2,
x̂ = (1000, 0, 0, 0) with P₀ = diag(25, 1218.47, 400, 400), and a camera bearing of 1° = 0.017453 rad
applied immediately, with no prediction step between (the measurement value is chosen for the
example):

- H = (0, 0.001, 0, 0), because ∂β/∂y = x / r² = 1/1000 at (1000, 0).
- S = H P Hᵀ + R = 1.2185e-3 + 3.046e-6 = 1.2215e-3, so the predicted bearing std is
  √S = 0.034950 rad = 2.0025°. The 2° inherited from the radar dominates the camera's 0.1°.
- K = (0, 997.51, 0, 0): the gain is almost 1/H = 1000 m/rad, that is, the camera is trusted in y.
- x̂⁺ = (1000, 17.41, 0, 0). The y std falls from 34.9 m to 1.743 m; the x std stays 5 m and the
  velocities do not move, because P₀ has no position–velocity correlation yet.
- After one prediction step, P has x–vx and y–vy covariances of 40.000125. The same measurement then
  gives K = (0, 997.51, 0, 32.64): the update also moves vy by 0.570 m/s and reduces its std from
  20.0001 to 19.967 m/s.

This ordering (radar initialisation, then a camera update, with no prediction between) is step 0 of
the single-target experiment (`src/fusion/tracking.py:117-125`). In the multi-target tracker it does
not occur: only confirmed tracks receive camera updates (`multi_target.py:517-522`).

**Example 5 (exact; `wrap_angle`).** The predicted bearing is 179°, the measured one is −179°. The
raw difference is −358°; wrapping gives +2°, which is the true angular distance. Without wrapping
the update would see an innovation of about 6.25 rad and pull the track far across the plane.

### Equations as implemented

For a measurement z with model h, Jacobian H = ∂h/∂x evaluated at the predicted state x̂⁻:

```math
\nu=z-h(\hat x^-)\ \text{with each bearing component replaced by}\ \operatorname{wrap}(\nu_i)=\operatorname{atan2}(\sin\nu_i,\cos\nu_i)\in[-\pi,\pi]
```

```math
S=HP^-H^{\top}+R,\qquad K=P^-H^{\top}S^{-1},\qquad \hat x^+=\hat x^-+K\nu,\qquad P^+=(I-KH)P^-(I-KH)^{\top}+KRK^{\top}
```

Jacobians, with r = √(x² + y²) for the radar, and Δx, Δy, ρ² = Δx² + Δy² relative to the camera:

```math
H_r=\begin{pmatrix}x/r&y/r&0&0\\-y/r^2&x/r^2&0&0\end{pmatrix},\qquad
H_c=\begin{pmatrix}-\Delta y/\rho^2&\Delta x/\rho^2&0&0\end{pmatrix}
```

- Update: `src/fusion/filters/ekf.py:30-70` (H line 53, innovation lines 55-59, S line 61, K line 63
  as `np.linalg.solve`, state line 65, Joseph form lines 68-69). Wrapping: `src/fusion/angles.py:15`.
- Jacobians: `radar.py:81-97`, `camera.py:76-88`. Velocity columns are zero because neither sensor
  measures velocity.
- A model that cannot be linearised (the target is within 10⁻⁶ m of the sensor) raises `ValueError`
  before anything is changed (`ekf.py:52-53`, `ekf.py:94-98`).
- Order inside one tracker step: predict every track, then the radar scan (association, update,
  lifecycle, births), then the camera scan (`multi_target.py:336-341`). In the single-target loop the
  radar update comes before the camera update (`tracking.py:118-123`).
- Tests: `test_camera_update_only_corrects_along_the_bearing_direction` (`tests/test_ekf.py:89`),
  `test_update_wraps_innovation_when_prediction_and_measurement_straddle_the_cut`
  (`tests/test_ekf.py:104`).

<details>
<summary>Optional derivation: the Jacobians, the tangent model, and why updates are sequential</summary>

With r = √(x² + y²): ∂r/∂x = x/r, ∂r/∂y = y/r. With β = atan2(y, x): ∂β/∂x = −y/(x² + y²) and
∂β/∂y = x/(x² + y²). These are the rows above. For the camera the same bearing derivative is taken
relative to the camera position.

Linearise: h(x) ≈ h(x̂⁻) + H (x − x̂⁻). Then z − h(x̂⁻) ≈ H (x − x̂⁻) + v, which is a linear measurement
of the error with noise R, and section 4 applies with this H.

Bearing differences are meaningful only modulo 2π, so the bearing component of the innovation is
wrapped before it is used by the update and by the gate (section 6); a range difference is an
ordinary length and is left as it is. The wrapped bearing is the same angle seen the short way round.

The radar and camera noises are independent, so processing the radar vector and then the camera
scalar gives the same result as one joint update **when h is linear**. With a nonlinear h the camera
Jacobian is evaluated at the state already corrected by the radar, which is slightly different from
a joint linearisation (an approximation: the already corrected state is used as the linearisation
point).

The camera Jacobian is H_c = (1/ρ)(−sin β, cos β, 0, 0): it points perpendicular to the line of
sight. The position part of the correction is P_pp H_cᵀ ν / S. If the position covariance has no
correlation between range and cross-range directions (as in example 4, where P_pp is diagonal), the
correction is purely cross-range. If it has such a correlation, the camera also moves the range
estimate through it. The test above uses a diagonal P at a point on the x axis, where the condition
holds (its comment calls this "a wide prior").

</details>

| Kind | Statement | Reference |
|---|---|---|
| Approximation | h is replaced by its first-order Taylor expansion at the predicted state, once per update (no iteration) | `ekf.py:53-55` |
| Approximation | The posterior is treated as Gaussian with covariance P⁺ | `ekf.py:68-69` |
| Assumption | Measurement noises are independent between sensors and across scans | `tracking.py:118-123` |
| Identity | In exact arithmetic the Joseph form keeps P⁺ symmetric and positive semi-definite for any K; in floating point it is more robust than the short form, with no universal guarantee | derivation in section 4 |
| Assumption | The geometry is planar: no height, no earth curvature | README Limitations |
| Empirical | Neutral single-target runs are consistent: fusion EKF near, state, all steps, mean NEES 3.87, interval [3.62, 4.12] around the expected 4 (50 seeds, no association) | RESULTS.md [section 2](RESULTS.md#2-filter-consistency) |
| Empirical (negative) | Not every part passes: radar-only EKF near, velocity, all steps: mean 1.86, interval [1.72, 1.99], below the expected 2 ("underconfident") | RESULTS.md [section 2](RESULTS.md#2-filter-consistency) |
| Empirical (negative) | When the model is wrong the covariance is no longer a good description of the error: camera bias 1° gives d NEES +2.68e+03 ± 8.3 against 4.09 neutral (scene separated, 200 m diagnostic distance) | RESULTS.md [section 5.1](RESULTS.md#51-camera-bearing-bias-time-offset-and-lever-arm-scene-separated) |

## 6. Gating

**Key idea.** A gate decides which measurements a track may claim. A measurement is accepted when
its innovation, measured in units of the innovation covariance S, is small enough that a correct
measurement would fall that close with 99 % probability. The distance that does this is the
Mahalanobis distance; the threshold comes from the χ² distribution.

**Example 6 (exact thresholds; illustrative innovations).**
- Exact: `gate_threshold(2, 0.99)` = 9.210 for the radar (two components) and
  `gate_threshold(1, 0.99)` = 6.635 for the camera (one component).
- Illustrative: a radar innovation covariance S = diag(36 m², 0.0009 rad²) (std 6 m and 0.03 rad).
  An innovation ν = (6 m, 0.06 rad) has d² = 6²/36 + 0.06²/0.0009 = 1 + 4 = 5, inside the gate
  (5 ≤ 9.21). An innovation (15 m, 0.06 rad) has d² = 6.25 + 4 = 10.25 > 9.21, outside, although
  15 m is only 2.5 standard deviations and 0.06 rad only 2 standard deviations: the gate is an
  ellipse in the combined space, not a box.
- Illustrative cost comparison: a vague track with S_B = diag(3600, 0.09) and a tight track with
  S_A = diag(36, 0.0009) see the same innovation (6 m, 0.06 rad). Their d² values are 0.05 and 5, so
  d² alone favours the vague track. The costs d² + ln det S are 5.83 for the vague track and 1.57
  for the tight one: the tight track wins, as a likelihood comparison would say.

### Equations as implemented

With S = L Lᵀ (Cholesky) and ν the innovation (its bearing component wrapped, its range component
not):

```math
d^2=\nu^{\top}S^{-1}\nu=\lVert L^{-1}\nu\rVert^2,\qquad \text{in gate}\iff d^2\le\gamma_m=F^{-1}_{\chi^2_m}(0.99)
```

```math
c=d^2+\ln\det S=d^2+2\sum_i\ln L_{ii}
```

- Threshold: `src/fusion/association/gating.py:14-32`. The tracker uses m = 2 for the radar and m = 1
  for the camera (`multi_target.py:218-223`).
- Symmetrised innovation covariance: `gating.py:47-59`. Distance by Cholesky solve:
  `gating.py:89-101`. Costs, gate and the range guard: `gating.py:118-179` (guard defined at line 161
  and applied at line 166, cost line 177, gate line 178).
- A track whose predicted position is closer than max(min_range, 10⁻⁶) m to the origin, where the
  radar is, is out of the gate for every measurement (`gating.py:136-138`, `:161-167`), because the
  bearing Jacobian explodes there. The guard measures the distance from the origin, not from the
  camera model's position; for the camera the tracker applies its own guard before gating
  (`multi_target.py:517-522`).
- Tests: `test_gate_threshold_matches_chi_square_table` (`tests/test_gating.py:35`),
  `test_bearing_residual_is_wrapped_across_the_minus_x_axis` (`tests/test_gating.py:90`).

<details>
<summary>Optional derivation: why χ², and why d² + ln det S</summary>

If the track is the true source and the filter is correct, ν ~ N(0, S). With S = L Lᵀ, y = L⁻¹ ν has
identity covariance, so d² = yᵀ y is a sum of m squared standard normals, which is χ² with m degrees
of freedom. The probability that a correct measurement falls inside the ellipse d² ≤ γ is therefore
F_χ²(γ) = 0.99 when γ is the 99 % quantile.

The Gaussian density of ν is (2π)^(−m/2) (det S)^(−1/2) exp(−d²/2). Its negative logarithm times two
is d² + ln det S + m ln 2π. The last term is the same for every track, so ranking by likelihood is
ranking by d² + ln det S. The ln det S term is a penalty for vagueness: a track with a large S spreads
its probability over a large region and assigns little density to any point in it.

</details>

| Kind | Statement | Reference |
|---|---|---|
| Assumption | The innovation of the true measurement is N(0, S) | derivation above |
| Approximation | This holds only if the filter is consistent, the linearisation is good, and the measurement really comes from this target | sections 5 and 9 |
| Identity | 1 % of correct measurements fall outside a 99 % gate if the assumption holds; those become misses in section 8 | derivation above |
| Approximation | S is computed from the predicted estimate; a track pulled off its target gates around the wrong place | `gating.py:168` |
| Empirical (negative) | When the assumed radar range std is four times too small (scale ×0.25 everywhere in the tracker, so the gates are one of several things that change) tracking degrades: scene crossing, clutter 0, 50 m, d RMSE +6.4 ± 0.57 m, ghost rate 1.6 ± 0.11 per step, 18.8 ± 2.1 ID switches | RESULTS.md [section 5.3](RESULTS.md#53-tracker-noise-assumptions-scene-crossing-clutter-0) |

## 7. Data association and the camera ambiguity rule

**Key idea.** Gating says which measurements a track may claim; association decides who claims what
when several tracks and several measurements compete. The tracker solves one global problem per
group of tracks and sensor, not one measurement at a time: it chooses the set of in-gate pairs that
has the most pairs and, among those, the lowest total cost. Camera measurements add a complication: a bearing alone cannot tell two
nearby tracks apart, so the tracker withholds the camera when it cannot.

**Example 7 (illustrative cost matrix; exact function call).** Two tracks, two measurements, cost
matrix [[1, 2], [1.5, 10]], and a gate that accepts costs below 5.
- Greedy nearest neighbour takes the cheapest pair first, (track 0, measurement 0) at cost 1. Track 1
  is left with measurement 1 at cost 10, outside the gate: one pair, total cost 1, one track missed.
- The global assignment takes (track 0, measurement 1) and (track 1, measurement 0): two pairs, total
  cost 2 + 1.5 = 3.5. `gated_assignment` returns exactly `[(0, 1), (1, 0)]`.

**Example 8 (illustrative geometry; computed with `_ambiguous_camera_tracks`).** Confirmed tracks at
(1000, 0), (1000, 100) and (1000, 150) m, each with P = diag(400, 400, 1, 1) (a 20 m position std).
For the first track the predicted bearing std is s = √(0.02² + 0.001745²) = 0.020076 rad = 1.150°,
so its gate half-width is √6.635 × s = 2.5758 × 0.020076 = 0.05171 rad = 2.963°; the other two are
within a few per cent of that (2.948° and 2.930°). Two gates overlap when the bearing gap is below
about 5.9° (about 104 m at 1000 m). The first two tracks are 5.711° apart and the second and third
2.820° apart, so with all three present the rule flags all three: the third is flagged through the
second. The first and third alone are 8.531° apart, and with only those two present neither is
flagged. The result depends on who else is present, because the rule looks at every pair.

### Equations as implemented

Assignment problem for one sensor and one group of tracks, with the cost matrix c and gate of
section 6: among all one-to-one choices of in-gate pairs,

```math
\text{maximise}\ \sum_{i,j}X_{ij},\ \text{then, among those, minimise}\ \sum_{i,j}c_{ij}X_{ij}\quad\text{s.t.}\quad X_{ij}\in\{0,1\},\ \sum_jX_{ij}\le1,\ \sum_iX_{ij}\le1,\ X_{ij}=0\ \text{outside the gate}
```

(Minimising the total cost alone would choose the empty assignment once the costs are shifted to be
non-negative; the sentinel below is what turns the Hungarian algorithm into this lexicographic
problem.)

Implementation (`src/fusion/association/assignment.py:23-71`): costs of usable (in-gate, finite)
pairs are shifted by their minimum so they are non-negative (line 59); every other pair gets the
sentinel B = (n + 1) · max(shifted) + 1 with n = min(rows, columns) (line 61); SciPy's
`linear_sum_assignment` solves the problem (line 62); any pair that still lands on the sentinel is
dropped (line 63).

Camera ambiguity rule (`src/fusion/tracker/multi_target.py:493-510`). For each confirmed track i that
is at least max(min_range, 10⁻⁶) m from the assumed camera position (`:512-522`):

```math
\theta_i=h_c(\hat x_i),\qquad s_i^2=H_{c,i}P_iH_{c,i}^{\top}+\sigma_c^2,\qquad w_i=\sqrt{\gamma_1}\,s_i
```

```math
i\ \text{is ambiguous}\iff\exists\,j\neq i:\ \lvert\operatorname{wrap}(\theta_i-\theta_j)\rvert<w_i+w_j
```

Here s_i is the std of the track's predicted bearing innovation, including the camera noise
(`innovation_covariance`, `gating.py:47-59`), and γ₁ = 6.635. Ambiguous tracks are counted in
`camera_skipped` and receive no camera update at this scan; the others are associated with the camera
scan by the same gated assignment with threshold γ₁ (`:523-535`).

Structure of one radar scan (`multi_target.py:461-491`): confirmed tracks are associated first, then
tentative tracks claim what is left; every track's lifecycle advances (section 8); each radar
measurement still unclaimed, finite and at least min_range away starts a new tentative track from
`radar_initial_estimate`. The camera never starts a track and never counts as a hit.

- Tests: `test_global_optimum_beats_greedy_nearest_neighbour` (`tests/test_assignment.py:39`).

<details>
<summary>Optional derivation: why the sentinel maximises the number of pairs first</summary>

Let s be the number of sentinel entries in an assignment. Its cost is at least s B. An assignment with
one sentinel fewer costs at most (n − s + 1) · max(shifted) + (s − 1) B. Since
B = (n + 1) · max(shifted) + 1 exceeds n · max(shifted), the second is always smaller than the first.
So the solver first minimises the number of sentinel pairs, that is, maximises the number of in-gate
pairs, and only then minimises the total in-gate cost. Dropping the rolled-back sentinel pairs leaves
a valid gated assignment.

The ambiguity test is the statement that the two intervals [θ_i ± w_i] and [θ_j ± w_j] overlap, which
for intervals on a circle is |θ_i − θ_j| < w_i + w_j after wrapping the difference.

</details>

| Kind | Statement | Reference |
|---|---|---|
| Assumption | Each target produces at most one measurement per scan, and each measurement comes from at most one target (one-to-one) | `assignment.py:23-31` |
| Approximation | Association is staged: confirmed tracks get priority and tentative tracks only see the leftovers, so the result is not the global optimum over all tracks | `multi_target.py:466-474` |
| Approximation | Unresolved targets (two targets merged into one measurement) are not modelled; there is no probabilistic data association | section 13 |
| Approximation | The ambiguity rule is a heuristic: it withholds the camera when the gates overlap, that is when targets are close, so the camera information is not used in that situation | `multi_target.py:493-510` |
| Empirical (limitation, not triggered) | Skipping the camera can let two close targets both reach the coast limit in a long radar outage; in the 50-seed scenes measured (radar outage 20 s to 40 s, no clutter, aware tracker, 50 m) the coast deletions were `0 ± 0` in every row, and in the crossing control run the longest stretch without any update of a confirmed track was 0.668 ± 0.019 s | RESULTS.md [section 4.3](RESULTS.md#43-measured-limitations) |

## 8. Track management and outage-aware coasting

**Key idea.** A measurement the tracker cannot explain might be a new target or clutter. A new track
is therefore tentative; it becomes confirmed after enough hits, and is deleted after too many misses.
During a radar outage the tracker can either treat the missing scans as misses (unaware) or know
that nothing was measured and keep the tracks alive for a limited time (aware).

**Example 9 (exact; `Lifecycle.born`, `advance`).** With M = 3, N = 5, K = 5, and the birth counting
as the first hit of scan 1:
- hit at birth, miss, hit, hit: hits and scans are 1/1, 1/2, 2/3, then 3/4, so the track is confirmed
  at scan 4.
- hit at birth, miss, miss, miss: 1/1, 1/2, 1/3, then 1/4 with scans − hits = 3 > N − M = 2, so the
  track is deleted at scan 4. One miss never deletes; the third does.
- hit at birth, miss, miss, hit, miss: after scan 5 there are 2 hits and 3 misses, so the track is
  deleted at scan 5. A tentative track is never tentative after its N-th scan.
- A confirmed track with repeated misses: the consecutive-miss counter reaches 5 and the track is
  still confirmed; the sixth consecutive miss deletes it (misses > K).

**Example 10 (exact; `TrackerConfig.max_coast_steps`).** With T_max = 15 s and Δt = 0.1 s the limit is
150 steps. A track is deleted at the first outage step at which more than 150 steps (15.1 s) have
passed since its last update. The tolerance matters: in floating point 0.3 / 0.1 = 2.9999999999999996,
so a plain floor would give 2 steps for 0.3 s instead of 3; adding 10⁻⁹ before the floor gives 3.

### Equations as implemented

Lifecycle after each radar scan, with hits h, scans n and consecutive misses m
(`src/fusion/tracker/track.py:72-103`):

```math
\text{tentative:}\ \begin{cases}\text{confirmed}&h\ge M\\ \text{deleted}&n-h>N-M\\ \text{tentative}&\text{otherwise}\end{cases}\qquad
\text{confirmed:}\ \text{deleted}\iff m>K
```

A hit is a radar association only (`multi_target.py:476-478`); a newborn track has h = n = 1
(`track.py:64-69`).

Outage-aware coasting (`multi_target.py`): every track's `coast_steps` increases by one at each
predict (`:338`) and returns to zero at any measurement update, radar or camera (`:435`). In an aware
outage step the radar scan is not processed (`:339`), so lifecycle counters are frozen and nothing is
born; after the camera update (`:341`), `_end_of_outage_step` (`:441-459`) first drops tentative
tracks (policy "drop") and then deletes every track with

```math
\text{coast\_steps}>\left\lfloor T_{\max}/\Delta t+10^{-9}\right\rfloor
```

The limit is `max_coast_steps` (`:84-87`); the tolerance is `STEP_TOLERANCE` (`src/fusion/dropout.py:24`).
These deletions happen only on steps flagged `radar_down`. An unaware tracker processes the empty
scans and counts each as a miss.

- Tests: `test_every_outcome_sequence_matches_the_closed_form` (`tests/test_track.py:96`),
  `test_max_coast_time_converts_to_whole_steps_without_float_errors`
  (`tests/test_outage_tracker.py:81`).

<details>
<summary>Optional derivation: reading M-of-N as a counting rule</summary>

For a tentative track with birth counted, after n scans with h hits the rule asks two questions in
order: have I reached M hits? have I used up the allowed N − M misses? The second can be written
n − h > N − M. Since h ≤ n, the track can stay tentative only while n − h ≤ N − M, which for
n = N means h ≥ M, so a track that has not confirmed by scan N is certainly deleted. This is why the
rule is a window of N scans without storing the window: the counts are sufficient.

The rule is not a likelihood test. It does not know how likely a hit is for clutter versus a real
target, and it does not use the quality of the association (d²). It is a deliberately simple
counting heuristic.

</details>

| Kind | Statement | Reference |
|---|---|---|
| Design choice | The measurement that starts a track counts as its first hit. This does not assume it is a true detection: clutter births are expected and are removed by the confirmation rule | `track.py:64-69` |
| Approximation | M-of-N and K are counting rules that ignore the quality of each association | `track.py:72-103` |
| Assumption | The aware tracker is told that the radar is down (the flag `radar_down`); a real system needs a health monitor for this | `multi_target.py:315-316` |
| Empirical | Radar outage, scene separated, 50 m: the share of identities kept ("ID kept") is 0 ± 0 for the unaware tracker after outages of 6 s and 12 s, and 1 ± 0 for the aware tracker | RESULTS.md [section 4.1](RESULTS.md#41-radar-outage-of-growing-duration-experiment-a) |
| Empirical (cost) | Coasting costs accuracy: aware position error during a 12 s outage is 5.53 ± 0.7 m against 3.41 ± 0.35 m before it | RESULTS.md [section 4.1](RESULTS.md#41-radar-outage-of-growing-duration-experiment-a) |
| Empirical (negative) | A vanished target leaves a ghost that lives about as long as the coast limit: 4.98, 9.98 and 15 s for limits of 5, 10 and 15 s | RESULTS.md [section 4.2](RESULTS.md#42-ghost-lifetime-against-maximum-coasting-time-experiment-f) |
| Empirical (negative) | Camera clutter resets the ghost's coast clock: with 5 camera clutter points per scan the ghost lives 6.99 s at a 5 s limit and 20.8 s at a 10 s limit | RESULTS.md [section 4.3](RESULTS.md#43-measured-limitations) |

## 9. Evaluation statistics and criteria

**Key idea.** Numbers from a simulation are only as meaningful as the way they are compared. This
section covers how tracks are scored against the truth, how a mean over seeds gets an interval, why
two settings are compared seed by seed, how filter consistency is tested, and how the pre-registered
"improves / worsens / non-inferior" rules turn intervals into verdicts.

### 9.1 Scores

Matching, per step (`src/fusion/mtt_metrics.py:70-127`): confirmed tracks are matched one-to-one to
true targets by the gated assignment of section 7 with the squared position distance as cost and a
gate of 50 m (200 m for the error diagnostics); it first maximises the number of pairs, then
minimises the total squared distance. Tentative tracks are ignored.

```math
\text{RMSE}=\sqrt{\frac1{|\mathcal M|}\sum_{(i,k)\in\mathcal M}\lVert p_{i,k}-\hat p_{j(i,k),k}\rVert^2}
```

The sum runs over all matched (target, step) pairs after the 5 s burn-in, and the divisor is their
number (`mtt_metrics.py:198-199`).

- missed: share of in-view (target, step) pairs without a matched confirmed track (`:203-204`).
- ghost rate: mean over steps of (confirmed tracks − matched tracks), which includes real tracks
  pulled off their target (`:200-202`).
- false-track rate: mean number of confirmed tracks that are never matched during the whole run
  (`:243-245`).
- ID switches: changes in the id of the track matched to a target; gaps are ignored, so
  1 → gap → 2 is a switch (`:151-164`).

**Approximation / pitfall.** The RMSE is taken over matched pairs only. A tracker that loses its
tracks has fewer pairs and can show a smaller RMSE: in RESULTS.md [section 4.1](RESULTS.md#41-radar-outage-of-growing-duration-experiment-a)
the unaware tracker's RMSE during outages of 6 s and longer stays at 3.69 m while its missed rate
rises to 0.595 (scene separated, radar outage, 50 m; the "RMSE during" column). Read RMSE together with missed.

### 9.2 Seeds, standard deviations and confidence intervals

For n seeds with sample mean x̄ and sample standard deviation s (`mtt_metrics.py:309-337`):

```math
\bar x\pm t_{0.975,\,n-1}\,\frac{s}{\sqrt n}
```

Seeds with an undefined score (for example no target matched) are dropped and counted in `n_valid`.

**Example 11 (exact; `scipy.stats.t`, `nees_band`).**
- For n = 50, t₀.₉₇₅,₄₉ = 2.0096 and t/√n = 0.2842. A standard deviation of 0.60 m, as printed for the
  far-scene fusion EKF in RESULTS.md [section 1](RESULTS.md#1-radar-vs-fusion-single-target)
  (2.56 ± 0.60 m, an SD), corresponds to a 95 % half-width of 0.60 × 0.2842 = 0.17 m. That table is
  labelled SD for this reason; RESULTS.md section 0 states that its other cells are mean ± 95 %
half-width.
- The NEES acceptance band for 50 independent runs and a 4-dimensional error is
  [3.255, 4.821] (`nees_band(4, 50)`), derived in 9.4.

### 9.3 Paired differences and common random numbers

What the seeds share. Within one row, `run_improvement_trial` simulates the scene once per seed and
then runs every arm on that single simulated object (`improvement_experiment.py:580-591`), so the
arms of a row are scored on bit-identical data: truth, noise, detections and clutter. Across rows
(for example a maneuver row against the neutral row, or one clutter rate against another) the
streams are separate per source and keyed by the seed (`mtt_simulation.py:18-35`, `:184-188`), so a
given stream gives the same draws, but the realisations are not identical: a maneuver changes the
truth, and the clutter count shifts the clutter stream. RESULTS.md section 0 words this more
broadly; this document uses the within-row statement for the arm comparisons below.

For two settings with per-seed scores a_s and b_s the interval is built on d_s = a_s − b_s
(`mtt_metrics.py:362-391`). Two statements are needed and they are different. First, an algebraic
identity of the sample variances and the sample covariance s_ab of the two score columns:

```math
s_d^2=s_a^2+s_b^2-2\,s_{ab}
```

Second, the estimated variance of the sample mean of the differences, and the interval built on it:

```math
\widehat{\operatorname{Var}}(\bar d)=\frac{s_d^2}{n},\qquad \bar d\pm t_{0.975,\,n-1}\,\frac{s_d}{\sqrt n}
```

**Identity.** The first line is exact for sample quantities. Together with the second it shows what
pairing can and cannot do: for the same s_a and s_b, the paired interval is narrower than one built
from s_a² + s_b² alone exactly when s_ab > 0, that is, when the two arms tend to do well and badly on
the same seeds. If s_ab = 0 the two widths coincide, and if s_ab < 0 pairing would make the interval
wider. Whether s_ab is positive is a property of each comparison, not a guarantee of the design. The
paired interval is the right interval for paired data whatever s_ab; only its width depends on it.

**Example 12 (exact; real per-seed data, one condition).** Scene crossing, clutter 0, neutral row
(no maneuver, no bias), run RMSE at the 50 m match distance, seeds 0 to 49, from
`results/improvement_results.pkl` (not reproducible from a clean checkout without re-running the
evaluation; see the reproducibility note in section 0). The paired columns reproduce RESULTS.md [Table B](RESULTS.md#table-b-side-effect-on-the-neutral-row)
(0.118 ± 0.038 and 0.502 ± 0.077). The EKF's own RMSE is 2.858 ± 0.108 m (SD 0.3791).

| Arm − EKF | Paired mean d [m] | Paired 95 % half-width [m] | Common-critical-value reference half-width [m] | Correlation of a and b |
|---|---|---|---|---|
| IMM-A | 0.1182 | 0.0379 | 0.1452 | 0.937 |
| IMM-B | 0.5017 | 0.0771 | 0.1441 | 0.719 |

For IMM-A: s_a = 0.3424, s_b = 0.3791, s_ab = 0.12157, so
s_d² = 0.11723 + 0.14372 − 2 × 0.12157 = 0.01781 (the direct sample variance of d_s is 0.017808),
s_d = 0.1334, s_d²/n = 0.000356, and the half-width is 2.0096 × 0.1334 / √50 = 0.0379.

The fourth column is a reference calculation on the same 50 pairs: t₄₉ √((s_a² + s_b²)/n), which uses
the same critical value as the paired interval and drops the term −2 s_ab. It is not a standard
independent-samples interval (that would estimate the two variances from two separate samples and
use its own degrees of freedom) and it is not a measurement of what separate seeds would have
produced, since the arms here ran on identical data. It only isolates the contribution of s_ab to the
width. With this reference the paired half-width is 3.8 and 1.9 times smaller for IMM-A and IMM-B;
the smaller factor goes with the smaller correlation. This is the width of these two intervals on
this condition, not a general rule for other rows, metrics or arms.

### 9.4 NEES and consistency

The error of a track is e = x − x̂ and the filter claims it has covariance P. The normalised
estimation error squared is (`src/fusion/consistency.py:17-39`):

```math
\varepsilon=e^{\top}P^{-1}e
```

**Identity under consistency.** If e ~ N(0, P) then ε ~ χ² with d degrees of freedom (d = 4 for the
state, 2 for position or velocity), so E[ε] = d. For N independent runs at the same step the sum of
the ε values is χ² with N d degrees of freedom, so their mean lies in
[χ²₀.₀₂₅(N d) / N, χ²₀.₉₇₅(N d) / N] with probability 95 % (`consistency.py:66-87`).

Steps of one run are correlated, so counting how many steps leave the band is not a test. The
repository's verdict is at the seed level (`consistency.py:157-215`): each seed's NEES is averaged
over the selected steps, a Student t interval is built over the seeds, and the verdict is
"overconfident" if the interval lies above d, "underconfident" if it lies below d, and "consistent"
if it contains d. RESULTS.md [section 2](RESULTS.md#2-filter-consistency) lists the verdicts;
the step-band columns there are for reference only.

NEES answers a different question from the RMSE. The RMSE says how large the error is; the NEES says
whether P honestly describes it. Gating (section 6) and the bias covariance correction (section 11)
both rely on P being honest.

What a mean-NEES interval can and cannot show. An interval that contains d means that this one test,
the mean NEES of the selected samples, did not reject the value d. It does not establish that the
covariance is calibrated in every direction (a mean can be right while the shape is wrong), and when a
method adds a term to P it does not validate that term separately from the rest of the covariance.
The verdict names above are the literal strings of `seed_mean_nees`, which the single-target
experiments use.

The multi-target NEES is a different statistic. `nees_mean` of the robustness and improvement results
is the mean, over the matched (target, step) pairs after the burn-in, of the state NEES of the track
matched to the target at the 200 m distance (`robustness_metrics.py:239-264`, `:336-337`); each run
gives one such mean, and the tables give a t interval across the seeds with `seed_confidence_interval`,
not the `seed_mean_nees` verdict. The pairs are chosen by a global assignment that uses the truth
(minimum total squared distance within 200 m), pooled over targets and steps, and tracks that were
pulled off a target but stay within 200 m are included. Selecting the track that is closest to the
target is not independent of the error, so the expected value d is only a nominal reference for this
statistic, not an exact expectation; the direction and size of the selection effect are not measured
here. Statements below about such NEES values therefore say "above" or "contains" the nominal value,
and do not use the verdict names.

On real data the error e needs reference truth, which is usually not available. Without truth the
innovation statistic d² = νᵀ S⁻¹ ν of section 6 (the NIS, which is χ² with m degrees of freedom when
the filter is consistent) can be averaged in the same way as an alternative check; it tests the
predicted measurement covariance S, not the state error directly.

### 9.5 Improvement criteria

The criteria of `src/fusion/improvement_criteria.py` are fixed in code before the evaluation. All
scores are lower-is-better, and all comparisons are paired differences arm − reference.

| Rule | Definition | Code |
|---|---|---|
| judge(d, τ) | "improves" if upper(d) < 0 and mean(d) ≤ −τ; "worsens" if lower(d) > 0 and mean(d) ≥ τ; "undefined" with fewer than 2 valid seeds; else "neither" | `:107-115` |
| Practical threshold τ | max(floor, fraction × max(D, 0)); D is the baseline's degradation (row minus neutral row) on the data being judged. Missed: floor 0.01, fraction 0.25. RMSE and cross-range: 1.0 m, 0.25. Ghost: 0.02 per step. ID switches: 0.5 per run | `:99-104`, `:42-47` |
| non_inferior(d, m) | PASS if upper ≤ m, that is mean + half-width ≤ m; UNDERPOWERED if mean ≤ m and the half-width exceeds m (or fewer than 2 seeds); else FAIL | `:134-143` |
| equivalence(d, m) | PASS if lower ≥ −m and upper ≤ m; UNDERPOWERED if \|mean\| ≤ m and the half-width exceeds m (or fewer than 2 seeds); else FAIL | `:122-131` |
| Aggregate | All PASS gives PASS; any FAIL gives FAIL; otherwise UNDERPOWERED | `:484-489` |
| Margins | RMSE 0.3 m, missed 0.005, ghost 0.01 per step, ID switches 0.2 per run | `:35-40` |
| Winnable set | Maneuver row-blocks in which the baseline's window missed rate exceeds the neutral row's by more than the threshold on the tuning seeds 1000 to 1019; with the floor of 0.01 this reduces to "more than 0.01". Stored in `FROZEN.winnable` | `:215-231` |
| A1 | With W winnable row-blocks, PASS if the window missed rate "improves" in at least ⌈(2/3) W⌉ of them and no row-block (winnable or not) "worsens" in window missed, ghost rate or ID switches | `:365-429` |
| CTRL | A1 with each high-Q control as reference. PASS only against the strongest control (lowest mean window missed over the winnable rows); otherwise "HOLDS ONLY VS WEAKER CONTROL" or FAIL | `:432-481` |
| A2 | non_inferior of arm − EKF on the neutral row of every block, for every margin | `:492-506` |
| H-drag | Missed rate summed over the 5 s bins covering 25 to 45 s, separated layout, turn 5 deg/s; SUPPORTED if the high-Q interval excludes 0 with a reduction of at least 0.5 in every block; also reported on the seeds from 10 on, which did not generate the hypothesis | `:509-567` |
| B1 | "improves" for run RMSE and cross-range at 0.5° and 1° in every block, with τ from the baseline's degradation against the bias-0 row | `:570-588` |
| B2 | equivalence of EKF+bias − EKF on the bias-0 row for every margin | `:591-606` |
| B3 | non_inferior of the shift of the run-mean \|b̂ − b\| across rows against 0.05°, plus equivalence of every margin, on every maneuver row | `:609-627` |

PASS, FAIL and UNDERPOWERED are this repository's classifications, not standard statistical terms.
Non-inferiority is established exactly when mean + half-width ≤ m (upper ≤ m). The code then splits
the cases where it is not established: UNDERPOWERED when the mean is within the margin but the
half-width exceeds it (a rule of thumb for "the interval is wide"), and FAIL otherwise. FAIL means
non-inferiority was not established under the implemented rule; it does not mean that inferiority
was demonstrated, which would need the lower bound above m.

**Example 13 (exact; `non_inferior` re-run on the printed values).** The A2 lines of IMM-A in the
scene crossing with clutter 5 (RESULTS.md [section 6.3](RESULTS.md#63-neutral-non-inferiority-a2-the-cost-side)),
with the half-width computed from the printed interval:

| Metric | d [interval] | Half-width | Margin | Classification and reason |
|---|---|---|---|---|
| run position RMSE | +0.1229 [−0.1316, +0.3774] | 0.2545 | 0.3 | FAIL: mean + half-width = 0.3774 > 0.3, so non-inferiority is not established; the half-width is below the margin, so the code does not call it UNDERPOWERED. Inferiority is not demonstrated either: the interval still contains values below 0.3 |
| run missed rate | +0.0003176 [−0.004943, +0.005578] | 0.00526 | 0.005 | UNDERPOWERED: mean + half-width = 0.005578 > 0.005, so non-inferiority is not established; the mean is inside the margin and the half-width exceeds it. A more negative mean at this width could have passed |
| run ID switches | +0.1 [−0.08382, +0.2838] | 0.1838 | 0.2 | FAIL: mean + half-width = 0.2838 > 0.2, so non-inferiority is not established; the half-width is below the margin. Inferiority is not demonstrated |

**Example 14 (exact classification; thresholds reconstructed).** B1 for scene crossing, clutter 0
(RESULTS.md [section 6.6](RESULTS.md#66-camera-bias-estimate) and [Table C](RESULTS.md#table-c-camera-bias-estimate)).
The criteria file prints the verdicts, not τ, so τ is rebuilt from the rounded EKF RMSE values of
Table C (2.86 m at 0°, 13.7 m at 0.5°, 27 m at 1°):
- 0.5°: D ≈ 13.7 − 2.86 = 10.84 m, so τ = max(1.0, 0.25 × 10.84) = 2.71 m. The paired difference is
  −1.271 [−1.636, −0.9069]: the interval is below zero but the mean (−1.271) is not at or below −2.71,
  so `judge` returns "neither". The estimate removes about 12 % of the degradation; the rule asks
  for 25 %.
- 1°: D ≈ 27 − 2.86 = 24.14 m, τ = 6.035 m. The difference is −11.43 [−12.69, −10.18], below −6.035:
  "improves". It removes about 47 % of the degradation.

Because B1 requires "improves" in every listed line, one "neither" makes B1 FAIL.

<details>
<summary>Optional: why the success criteria are written as intervals against a margin</summary>

A confidence interval for a difference answers "which differences are compatible with the data".
"Improves" asks that all compatible differences are improvements and that the central one is large
enough to matter (the threshold τ). "Non-inferior" asks that no compatible difference exceeds a
tolerated loss m: mean + half-width ≤ m. It is established only if the mean is far enough below m for
the half-width to fit, so a wide interval needs a mean well inside the margin; a mean inside the margin
with a wide interval is what the code labels UNDERPOWERED (its own classification, not a standard
term). "Equivalent" asks that the whole interval lies within ±m. The three share one structure:
compare the interval, not only the mean, with a pre-stated number.

The criteria are code, frozen before the evaluation seeds were used (RESULTS.md
[section 0](RESULTS.md#0-how-to-read-this)). Many of them are evaluated side by side ("family of 108
checks, unadjusted"), with no correction for multiple comparisons; at a 95 % level some intervals
will exclude the truth by chance.

</details>

| Kind | Statement | Reference |
|---|---|---|
| Assumption | Per-seed scores are independent across seeds and close enough to normal for a t interval at n = 50 | `mtt_metrics.py:309-337` |
| Approximation | Rates such as the missed rate are bounded and skewed; the t interval is an approximation for them. Medians are listed in the result files | RESULTS.md section 0 |
| Assumption | "Found" means within the match distance, and the meaning of missed and ghost depends on it: for the 5 deg/s turn the 50 m missed shares in the bins 25 to 30, 30 to 35 and 35 to 40 s are 0.2440, 0.3627 and 0.2533, while the 200 m shares are 0.0000 in every bin after the first (10 seeds, scene separated, clutter 0) | RESULTS.md [section 5.2](RESULTS.md#52-maneuvers-scene-separated-and-crossing) |
| Empirical (negative) | The frozen parameters were chosen by the fallback rule: no candidate met the neutral margins on the 20 tuning seeds | RESULTS.md [section 6.1](RESULTS.md#61-limitations-of-the-frozen-parameters-verbatim) |
| Empirical (negative) | The ID-switch margin is flagged WIDER in the tuning output and may be UNDERPOWERED even at 50 seeds; intervals are not adjusted for multiple comparisons | RESULTS.md [section 6.1](RESULTS.md#61-limitations-of-the-frozen-parameters-verbatim), [section 8](RESULTS.md#8-general-limitations) |
| Empirical | The evaluation seeds 0 to 49 were also used by the baseline in earlier sections, and seeds 0 to 9 produced the drag hypothesis | RESULTS.md [section 0](RESULTS.md#0-how-to-read-this) |

## 10. The IMM filter

**Key idea.** One constant-velocity filter has to choose its process noise: low noise is accurate on
straight targets and slow to follow a turn, high noise follows turns and is noisier on straight
targets. The IMM (interacting multiple model) filter keeps several models running side by side, each
with its own estimate. It tracks how probable each model is from how well it predicts the incoming
measurements, and at each cycle it restarts every model from a probability-weighted mixture of all
the estimates, so a model that has just become the right one is not stuck with a stale estimate.

**Example 15 (exact; `uniform_transition`).** The mode-switching matrix is specified per radar scan
(stay probability 0.95) and used per filter step (10 steps per scan). For M = 2 modes:
λ = (M · 0.95 − 1) / (M − 1) = 0.9, λ_step = 0.9^(1/10) = 0.98952, and the per-step stay probability
is (1 + (M − 1) λ_step) / M = 0.99476. For M = 3, λ = 0.925, and the per-step stay probability is
0.99482. In both cases the step matrix raised to the 10th power returns 0.95 on the diagonal.

**Example 16 (exact; `ct_transition`).** A coordinated turn at 7.5 deg/s over Δt = 0.1 s turns the
velocity by θ = 0.75°. The entries are sin θ / ω = 0.0999971 (against Δt = 0.1 for straight motion),
(1 − cos θ) / ω = 6.545e-4 and cos θ = 0.999914. Over one step the turn is a small correction; over
10 s it changes the heading by 75°.

**Example 17 (illustrative: two modes, a per-step stay probability of 0.9 that the code does not
use, mode estimates that differ only in vy).** Mode probabilities μ = (0.8, 0.2), Π = [[0.9, 0.1],
[0.1, 0.9]]. The predicted probabilities are c = (0.74, 0.26). Mode 1 started from mode 1 (weight
0.973) or mode 2 (0.027); mode 2 started from mode 1 (0.308) or mode 2 (0.692). Mode estimates with
vy = 0 and vy = 2 and unit covariances mix to vy = 0.054 for mode 1 and vy = 1.385 for mode 2, and the
mixed variance of mode 2 is 1 + (0.308 × 1.385² + 0.692 × 0.615²) = 1.852: the spread term adds the
disagreement between the two estimates to the variance.

**Example 18 (exact; `update_mode_probabilities`).** Pre-update probabilities μ⁻ = (0.5, 0.5) (for
the first measurement of a step these are the predicted c) and mode log-likelihoods (−2, −4) give
μ⁺ ∝ (0.5 e⁻², 0.5 e⁻⁴), that is μ⁺ = (0.8808, 0.1192): a log-likelihood advantage of 2 is a
factor e² = 7.39 in odds. A second measurement in the same step would start from this μ⁺, not from c.

### Equations as implemented

All modes share the state [x, y, vx, vy]. Mode j has a transition F_j and process noise Q_j. The CV
mode uses the matrices of section 2. The coordinated-turn (CT) mode at the fixed rate ω, with
θ = ω Δt, uses (`src/fusion/filters/imm.py:48-67`)

```math
F_{\mathrm{CT}}=\begin{pmatrix}I_2&\tfrac1\omega\begin{pmatrix}\sin\theta&-(1-\cos\theta)\\1-\cos\theta&\sin\theta\end{pmatrix}\\0&\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}\end{pmatrix}
```

(positive ω turns counter-clockwise, and ω → 0 gives the CV matrix). The CT mode keeps the CV noise
matrix: the random acceleration is added after the rotation (`imm.py:70-80`).

Mode transition (`imm.py:83-116`). With M modes, per-scan stay probability a and n steps per scan:

```math
\lambda=\frac{Ma-1}{M-1},\qquad \lambda_s=\lambda^{1/n},\qquad \Pi_{ii}=\frac{1+(M-1)\lambda_s}{M},\qquad \Pi_{ij}=\frac{1-\Pi_{ii}}{M-1}\ (i\ne j)
```

One IMM cycle (`imm.py:402-452`), with μ the mode probabilities:

```math
c_j=\sum_i\Pi_{ij}\mu_i,\qquad w_{ij}=\frac{\Pi_{ij}\mu_i}{c_j}
```

```math
x_{0j}=\sum_iw_{ij}x_i,\qquad P_{0j}=\sum_iw_{ij}\big[P_i+(x_i-x_{0j})(x_i-x_{0j})^{\top}\big]\qquad(\text{mixing})
```

```math
x_j^-=F_jx_{0j},\qquad P_j^-=F_jP_{0j}F_j^{\top}+Q_j,\qquad \mu\leftarrow c\qquad(\text{predict})
```

For each measurement in turn, every mode is updated by the EKF update of section 5, and the mode
probabilities are reweighted in the log domain, starting from the current pre-update probabilities
μ⁻:

```math
\ln L_j=-\tfrac12\big(d_j^2+\ln\det S_j+m\ln2\pi\big),\qquad \mu_j^{+}=\frac{\mu_j^{-}L_j}{\sum_l\mu_l^{-}L_l}
```

The pre-update probability μ⁻ is c from the prediction for the first measurement of a step, and the
posterior μ⁺ of the previous measurement for the next one: `predict` sets μ to c (`imm.py:419`) and
`update` passes the current μ to `update_mode_probabilities` (`imm.py:446`). A step with a radar
scan and a camera scan therefore uses c for the radar and the radar posterior for the camera; a step
with only a camera scan uses c for the camera.

The reported estimate is the moment-matched combination (`imm.py:159-177`):

```math
\hat x=\sum_j\mu_j\hat x_j,\qquad P=\sum_j\mu_j\big[P_j+(\hat x_j-\hat x)(\hat x_j-\hat x)^{\top}\big]
```

- Code: mixing `imm.py:129-156`, predict `imm.py:416-419`, mode log-likelihood and probability
  update `imm.py:180-219`.
- The tracker gates, associates and applies the camera ambiguity rule with the combined estimate
  (`multi_target.py:58-60`). The mode probabilities are updated by each measurement in turn, so the
  posterior after the radar update is the pre-update probability for the camera update; no new
  mixing or prediction happens between the two (`imm.py:422-452`).
- The arms of the evaluation: IMM-A has two CV modes (σ_a = 0.5, the tracker's own, and 2.0); IMM-B
  has a CV mode with the tracker's own σ_a = 0.5 and two CT modes at ±7.5 deg/s with σ_a = 2.0 (`src/fusion/improvement_experiment.py:194-202`,
  `:111-116`).
- With a single mode the IMM is the EKF bit for bit: mixing and combination are skipped
  (`imm.py:11-12`, `:384-385`, `:404-409`, `:440-441`).
- Tests: `test_the_step_matrix_raised_to_the_scan_length_is_the_scan_matrix`
  (`tests/test_imm.py:49`), `test_mixing_uses_the_transition_matrix_the_right_way_round_with_the_spread_term`
  (`:69`), `test_probability_update_matches_hand_numbers` (`:112`).

<details>
<summary>Optional derivation: the n-th root, mixing and the spread term</summary>

**Per-step transition.** The per-scan matrix has the form (a − off) I + off · u uᵀ, where u is the
vector of ones and off = (1 − a)/(M − 1). Such matrices commute. Its eigenvalue on the vector of ones is 1 and on every
vector orthogonal to it is (a − off) = (M a − 1)/(M − 1) = λ. Taking the n-th root keeps the eigenvalue 1 and
replaces λ by λ^(1/n); the matrix with those eigenvalues and non-negative entries is the Π above
(off_s = (1 − λ_s)/M, stay_s = λ_s + off_s). A real root needs λ ≥ 0, that is a ≥ 1/M; the code requires a > 1/M strictly
(`imm.py:108-111`).

**Mixing.** By Bayes' rule, the probability that the filter was in mode i at the previous step given
that it is in mode j now is Π_ij μ_i / c_j = w_ij. The mixture of the previous estimates, weighted by
w_ij, is a Gaussian mixture; replacing it by one Gaussian with the same mean and covariance gives the
mixed mean x₀ and covariance P₀ of mode j. The covariance of a mixture is the average covariance plus the covariance of the means, which
is the spread term: modes that disagree make the mixed estimate less certain.

**Mode probabilities.** Under mode j the innovation is N(0, S_j), so its likelihood is
L_j = exp(−d_j²/2) / √((2π)^m det S_j). The posterior is proportional to prior times likelihood,
where the prior is the current pre-update probability μ⁻_j.
The log domain avoids underflow when a likelihood is tiny; if all likelihoods are −∞ the predicted
probabilities are kept and a counter is increased (`imm.py:200-219`).

</details>

| Kind | Statement | Reference |
|---|---|---|
| Assumption | Mode switching is a Markov chain with equal probability to every other mode, the same for all tracks | `imm.py:83-116` |
| Assumption | The CT rate is fixed and known (±7.5 deg/s); the robustness rows turn at 5, 10 and 20 deg/s, so no CT mode is exact | `improvement_experiment.py:199-202`, RESULTS.md section 5.2 |
| Approximation | Moment matching replaces a Gaussian mixture by one Gaussian at every cycle | `imm.py:129-156`, `:159-177` |
| Approximation | Each mode uses an EKF update, with the first-order approximation of section 5 | `imm.py:429-430` |
| Empirical | Primary criterion on the maneuver rows: IMM-A PASS (improved 20 of 30, need 20) and IMM-B PASS (23 of 30); on the held-out rows IMM-A FAIL (7 of 12, need 8) and IMM-B PASS (10 of 12) | RESULTS.md [section 6.2](RESULTS.md#62-verdicts-heading-lines-verbatim-from-criteria_8btxt) |
| Empirical (negative) | Neutral non-inferiority A2: FAIL for both arms (non-inferiority not established under the implemented rule). IMM-A: run RMSE upper bound 0.3774 against 0.3 and ID switches upper bound 0.2838 against 0.2 (crossing, clutter 5). IMM-B: run RMSE +0.5017 [+0.4247, +0.5788] against 0.3 (crossing, clutter 0) | RESULTS.md [section 6.3](RESULTS.md#63-neutral-non-inferiority-a2-the-cost-side) |
| Empirical (negative) | Neither arm beats the strongest control, the EKF with σ_a = 5: improved 0 of 30 winnable row-blocks, with a worsening in 29 (IMM-A) and 15 (IMM-B) row-blocks. On the window missed rate the IMM arms (0.0337, 0.0248) lose to the EKF with σ_a = 3 and 5 (0.0135, 0.00443), which in turn cost more neutral RMSE (Table D) | RESULTS.md [section 6.4](RESULTS.md#64-comparison-with-the-ekf-high-q-controls-ctrl), [Table A](RESULTS.md#table-a-benefit-on-the-maneuver-rows) |
| Empirical (cost) | Runtime per trial: IMM-A ×3.29 and IMM-B ×3.85 of the EKF | RESULTS.md [section 6.7](RESULTS.md#67-runtime) |

## 11. Camera bias

**Key idea.** A camera whose mounting is rotated by a small angle b adds that same angle to every
bearing it reports. At 1000 m, 1° is 17.45 m of cross-range error, ten times the camera's own noise.
Four points make up the treatment in this repository. The camera alone cannot reveal b (a precise
statement is given below). The radar can, because its bearing is unbiased: the difference between
the camera bearing and the radar bearing of the same target at the same instant is b plus noise. A
small Bayesian filter turns those differences into an estimate. The estimate is then applied to the
reported tracks through a precomputed sensitivity, not fed back into the track filters.

**Link to example 4.** After the camera update of example 4, `SensitivityEKF` holds
V = K J_b = (0, 997.51, 0, 0) m/rad. A bias of 1° = 0.017453 rad therefore moves the estimate by
17.41 m in y, the same number as the 1° innovation of that example: right after one update, a bias
and an innovation of the same size are indistinguishable.

**Example 19 (illustrative: the evidence an estimator can collect).** Assume n pairs whose sample
mean equals the true bias exactly, no drift, the outlier gate ignored, a pair std of 2.0025° and a
slab std of 1°. With no drift the sample mean is sufficient (derivation below), so the posterior of
H1 follows from the Bayes factor BF = N(δ̄; 0, σ_s² + σ_p²/n) / N(δ̄; 0, σ_p²/n).

| True bias | Pairs n | ln BF | P(H1) at prior 0.5 | P(H1) at prior 0.01 | Applied estimate at prior 0.01 |
|---|---|---|---|---|---|
| 0.5° | 25 | −0.318 | 0.421 | 0.007 | 0.003° |
| 0.5° | 100 | 1.369 | 0.797 | 0.038 | 0.018° |
| 0.5° | 240 | 5.304 | 0.995 | 0.670 | 0.330° |
| 0.5° | 400 | 10.039 | 1.000 | 0.996 | 0.493° |
| 1° | 25 | 1.697 | 0.845 | 0.052 | 0.045° |
| 1° | 100 | 10.360 | 1.000 | 0.997 | 0.958° |
| 1° | 240 | 27.379 | 1.000 | 1.000 | 0.984° |

Cross-check with `CameraBiasEstimator`: 100 identical pairs of 0.5° at prior 0.01 give P(H1) = 0.0382
and 0.0184°; 400 pairs give 0.9957 and 0.4929°. A 60 s run offers at most one pair per confirmed
track per radar scan: with one confirmed track per target, at most about 240 pairs in the crossing
scene (4 targets) and 180 in the separated scene (3 targets), fewer after missed detections, gate
rejections and exclusions (clutter-born confirmed tracks could add pairs).

### 11.1 What the camera alone cannot tell, and what can

Let T_b rotate the plane by b about the camera position, acting on position and velocity. A rotated
state has camera bearing β_c(T_b x) = β_c(x) + b (modulo 2π), so a camera with bias b observing the
state x reports exactly what an unbiased camera reports for the state T_b x, up to noise.

**Statement (symmetry, camera data only).** Suppose (i) the detection and clutter process does not
depend on the bearing (a full-circle field of view, as in the repository default); (ii) the motion
model commutes with rotations, T_b F_j = F_j T_b and T_b Q T_bᵀ = Q; and (iii) the prior over
trajectories is rotation-invariant. Then the likelihood of the camera data is the same for the pair
(trajectory x, bias b) and the pair (trajectory T_b x, bias 0), and the prior probabilities of x and
T_b x are equal. Hence the marginal likelihood of the camera data does not depend on b: the camera
data and the kinematics alone leave b at its prior.

- **Identity** given (i) to (iii). Condition (ii) holds for the matrices of this repository: the CV
  matrix and the CT matrix have blocks of the form a I + c J (J the 90° rotation), which commute with
  rotations (about any centre, because the dynamics do not depend on position), and Q = σ_a² G Gᵀ is
  a block scalar matrix. A bearing sector narrower than the full circle (`FieldOfView` with
  `bearing_half_width` below π) would violate (i). Condition (iii) is a property of the prior chosen
  for a track; the radar-initialised prior of section 3 does not satisfy it, and that is precisely
  where the radar enters.
- The radar range is rotation-invariant (‖T_b x‖ = ‖x‖) for a radar at the centre of rotation, so it
  does not separate x from T_b x either. The radar bearing does: it reports β(x) + noise, not
  β(x) + b. The radar bearing is the only measured quantity in this model that breaks the symmetry.
- The information can be used in two ways. (1) Simultaneously: for a target seen by both sensors at
  the same instant, δ = z_c − z_r = b + v_c − v_r, because the target's own bearing cancels. (2)
  Through the motion model over time: a radar bearing at one time predicts the camera bearing at the
  next. The implemented estimator uses only (1). An augmented-state filter over (x, b) could use
  both; this is reasoning, and the repository has no such filter to measure how much is left unused.
- The camera-only statement does not need the camera to be at the radar position or on the radar's
  clock; a rotation about the camera position works for any camera position and any time stamps.
  What a lever arm or a time offset spoils is the pair of route (1) and the tracker's model: with a
  lever arm the two sensors see different bearings of the same target (parallax), so δ is no longer
  b plus noise; with a time offset δ also contains the change of the target's bearing during the
  offset. Kinematics carry information on b only if (i), (ii) or (iii) fails: for example a motion
  model that is not rotation-invariant, or a prior (such as the radar-initialised one) that is not.
  The robustness rows of RESULTS.md section 5.1 test the lever arm and the time offset as
  disturbances of the world, not as sources of information.
- The module docstring (`src/fusion/tracker/camera_bias.py:3-7`) says, with the parenthetical "for a
  camera at the radar and no time offset", that target kinematics carry no information on the bias,
  and that "the only information" is the simultaneous difference. The second sentence is to be read
  as "the only information this estimator uses".

### 11.2 The estimator

Pair, outlier gate and hypotheses (`src/fusion/tracker/camera_bias.py:96-186`). The tracker builds
pairs in `multi_target.py:373-393`: a track that was confirmed at this step's radar association, took
a radar measurement and a camera measurement in the same step, and whose radar measurement lies in the
gate of exactly one confirmed track. The raw pair value is z_c − z_r(bearing); `update` wraps it.

```math
\sigma_p^2=\sigma_\beta^2+\sigma_c^2\ (\text{believed values}),\qquad \lvert\delta\rvert\le\sqrt{\gamma_1\,(\sigma_p^2+\sigma_s^2)}\ \ (=5.77^\circ\text{ with the defaults})
```

H₀: b = 0. H₁: b ~ N(0, σ_s²) with σ_s = 1°, plus a random walk of variance q Δt per tracker step,
q = (0.1°)²/600 rad²/s (`camera_bias.py:34-71`, `:158-162`). The random walk is the estimator's
model, which lets its H₁ estimate follow a slowly drifting mounting angle. The simulated bias is a
constant that `camera_measure` adds at every step (`camera.py:43`), so the experiments never exercise
the drift; the two are different things. For each accepted δ in order
(`camera_bias.py:171-186`), with the H₁ filter state (b₁, v₁) and logit ℓ of P(H₁):

```math
S_1=v_1+\sigma_p^2,\qquad \ell\leftarrow\ell+\Big[-\tfrac12\Big(\tfrac{(\delta-b_1)^2}{S_1}+\ln S_1\Big)\Big]-\Big[-\tfrac12\Big(\tfrac{\delta^2}{\sigma_p^2}+\ln\sigma_p^2\Big)\Big]
```

```math
k=\frac{v_1}{S_1},\qquad b_1\leftarrow b_1+k(\delta-b_1),\qquad v_1\leftarrow\max\big((1-k)v_1,\ 10^{-18}\big)
```

The ℓ line runs only in the "spike_slab" mode; the filter line runs in both "spike_slab" and "always".
Applied estimate and its model-based uncertainty (`camera_bias.py:144-156`), with
P₁ = 1/(1 + e^(−ℓ)). The applied estimate is the posterior mean b̂ = E[b | D] of the model, and the
uncertainty is the posterior variance s_b² = Var(b | D), where D is the pairs seen so far:

```math
\hat b=P_1b_1,\qquad s_b^2=\operatorname{Var}(b\mid D)=P_1v_1+P_1(1-P_1)\,b_1^2\qquad(\text{spike\_slab});\qquad (b_1,v_1)\ (\text{always});\qquad (b_{\text{true}},0)\ (\text{oracle})
```

Three different variances must be kept apart. s_b² is model-based: it is computed from D and from the
model (the prior, the pair variance, the random walk) and is what the code reports. The
repeated-sampling variance of the estimator, Var(b̂) over data sets generated with a given b, is not
computed anywhere. For a fixed true b the mean squared error of b̂ is that variance plus the squared
bias of b̂, and b̂ is shrunk towards zero, so for a fixed b ≠ 0 it is biased. s_b² equals none of these
in general; it describes the expected squared error only if b is actually drawn from the model's prior.

<details>
<summary>Optional derivation: the Bayes factor, the Kalman recursion and the mixture variance</summary>

With q = 0 and n accepted pairs, under H₀ the pairs are independent N(0, σ_p²) and the sample mean δ̄
is N(0, σ_p²/n); under H₁ it is N(0, σ_s² + σ_p²/n). The sample mean is sufficient for b, so the
Bayes factor is BF = N(δ̄; 0, σ_s² + σ_p²/n) / N(δ̄; 0, σ_p²/n), and the posterior log-odds are the
prior log-odds plus ln BF (`test_the_posterior_equals_the_closed_form_bayes_factor_without_drift`,
`tests/test_camera_bias.py:58`). The sequential update above gives the same number: the product of
the one-step predictive densities N(δ_i; b₁, v₁ + σ_p²) is the marginal likelihood of the whole
sequence, by the chain rule of probability.

Under H₁ the posterior of b is N(b₁, v₁) with v₁ = 1/(1/σ_s² + n/σ_p²) and
b₁ = v₁ n δ̄ / σ_p²; the scalar Kalman recursion reproduces it, and
`test_a_known_bias_is_recovered_with_the_closed_form_posterior_std` (`:35`) checks the variance.

The applied estimate is the mean of the two-component mixture (point mass at 0 with weight 1 − P₁,
N(b₁, v₁) with weight P₁). By the law of total variance the variance of that mixture is
P₁ (v₁ + b₁²) − (P₁ b₁)² = P₁ v₁ + P₁ (1 − P₁) b₁², which is s_b². **Identity.** s_b² is the variance of
the mixture, that is Var(b | D) under the model; it describes the uncertainty about b only if the
model, including the prior P(H₁), is right.

</details>

### 11.3 Applying the estimate: sensitivity and output correction

The track filters run on the raw camera data. Alongside, each track carries V = ∂x̂/∂b, the
derivative of its estimate with respect to a constant bias added to every camera bearing. With
J_b = 1 for the camera and 0 for the radar (`src/fusion/filters/sensitivity.py:23-26`):

```math
V\leftarrow FV\ (\text{predict}),\qquad V\leftarrow(I-KH)\,V+K\,J_b\ (\text{update, with the }K,H\text{ of that update})
```

The reported snapshot is corrected by the shared estimate (`sensitivity.py:29-45`,
`multi_target.py:364-371`):

```math
\hat x_c=\hat x-V\hat b,\qquad P_c=P+s_b^2\,VV^{\top}
```

The IMM keeps one V_j per mode, mixes them with the mixing weights like the states, and combines them
with the current mode probabilities (`imm.py:383-395`, `:412-415`, `:431-439`).
Tests: `test_the_sensitivity_matches_a_finite_difference_of_the_filter` (`tests/test_sensitivity.py:85`),
`test_with_a_model_mismatch_the_sensitivity_is_the_first_order_derivative` (`:103`),
`test_the_imm_sensitivity_matches_a_finite_difference_with_the_mode_probabilities_fixed` (`:133`).

<details>
<summary>Optional derivation and the assumptions behind P + s_b² V Vᵀ</summary>

Write x̂(b) for the output the filter would give if every camera bearing were shifted by b. To first
order x̂(b) ≈ x̂(0) + V b, with V computed by the recursion above: differentiating
x⁺ = x⁻ + K (z − h(x⁻)) with respect to b, with z shifted by J_b b, gives
∂x⁺/∂b = (I − K H) ∂x⁻/∂b + K J_b, if K and H are held fixed. The track actually holds
x̂_raw = x̂(b_true) ≈ x̂(0) + V b_true. Subtracting V b̂ leaves x̂_c ≈ x̂(0) + V (b_true − b̂), whose error is

```math
x_{\text{true}}-\hat x_c=u-V\,w,\qquad u=x_{\text{true}}-\hat x(0),\quad w=b_{\text{true}}-\hat b
```

```math
\operatorname{Cov}(u-Vw)=\operatorname{Cov}(u)+V\operatorname{Var}(w)V^{\top}-\operatorname{Cov}(u,w)V^{\top}-V\operatorname{Cov}(w,u)
```

The code reports the first two terms, with Cov(u) = P and s_b² in the place of Var(w). The
substitution of s_b² for Var(w) is itself an assumption (item 3). The result is correct to the extent
that:

1. **First-order sensitivity holds.** The gains K and the Jacobians H depend on the estimate, and the
   exact derivative contains terms (∂K/∂b) ν, the derivative of the gain times an innovation ν, which
   V omits. Such a term vanishes where the relevant innovation is zero. Noise-free data with an exact
   model give zero innovations; noise-free data alone do not, because a model mismatch (the 8 deg/s
   turn against a constant-velocity filter in the second test) produces nonzero innovations. In that
   test V stays within 1 % of a finite difference. Both tests use noise-free measurements, so the term
   is checked only with model-mismatch innovations, not with measurement-noise innovations; that is a
   check on two trajectories, not a bound.
2. **The two errors are uncorrelated**: Cov(u, w) = 0. Independence, or zero covariance, has not been
   established in this system. The pairs that estimate b and the tracks are built from overlapping
   radar and camera measurements, and shared measurements can produce a nonzero cross-covariance;
   they do not prove that it is nonzero (the pair value δ cancels the target's own motion, and how
   much of the track error u it still shares was not analysed). The omitted terms can matter, and
   their sign and size are not measured in this repository; the reported P_c could be too small or
   too large because of them.
3. **s_b² describes the error of b̂.** s_b² is a model-based posterior variance, Var(b | D). The error
   w = b_true − b̂ of a fixed experiment (a constant 0.5° bias) has a different mean squared value,
   which includes the shrinkage of b̂ towards zero; s_b² matches it only if the model, including the
   prior P(H₁), describes how the bias was generated. A two-moment summary also discards the shape of
   the two-component posterior.
4. **P is honest for an unbiased camera.** Section 9.4.
5. For the IMM, V is combined with the current mode probabilities and the dependence of those
   probabilities on b is ignored (`imm.py:300-303`).
6. One scalar b̂ is shared by all tracks, so the corrections of different tracks are fully
   correlated; each track's P_c is a marginal covariance and does not carry that correlation.

"No double counting" holds in one precise sense: b̂ is never fed back into the track filters or the
association, so no measurement enters a track update twice. It does not establish that the errors of
the track and of b̂ are independent.

</details>

### 11.4 What the repository measured

Source: `results/improvement_bias.txt`, block "bias, crossing, clutter 0", seeds 0 to 49, mean ±
half-width of the 95 % t interval (all 24 cells of the table were also recomputed from
`results/improvement_results.pkl`; both files are local, git-ignored artifacts, see the
reproducibility note in section 0). RMSE is the run position error at the 50 m match distance; NEES
is the whole-state NEES at the 200 m diagnostic distance, pooled over matched pairs as described in
9.4 (nominal reference value 4, not an exact expectation). These rows are not
reproduced in RESULTS.md, which reports the EKF and EKF+bias columns of RMSE in
[Table C](RESULTS.md#table-c-camera-bias-estimate). The arms differ only in how the same pairs are
turned into an applied estimate: EKF+bias is the spike-and-slab estimate with prior 0.01 (the frozen
value), EKF+always applies the H₁ estimate with full weight, EKF+oracle applies the true bias with
zero variance.

| Arm | RMSE b = 0 | RMSE 0.5° | RMSE 1° | NEES b = 0 | NEES 0.5° | NEES 1° |
|---|---|---|---|---|---|---|
| EKF | 2.86 ± 0.11 | 13.7 ± 0.043 | 27 ± 0.05 | 5.53 ± 2.2 | 715 ± 2.6 | 2.85e+03 ± 8.4 |
| EKF+bias | 2.86 ± 0.11 | 12.4 ± 0.37 | 15.5 ± 1.3 | 4.71 ± 1.9 | 129 ± 34 | 61.7 ± 29 |
| EKF+always | 6.73 ± 0.64 | 7.04 ± 0.78 | 7.8 ± 0.9 | 4.59 ± 1.1 | 4.74 ± 1.1 | 5.55 ± 1.6 |
| EKF+oracle | 2.86 ± 0.11 | 2.86 ± 0.11 | 2.9 ± 0.11 | 5.53 ± 2.2 | 5.53 ± 2.2 | 7.16 ± 3.9 |

Run-mean absolute error of the applied estimate, in degrees: EKF+bias 0.00169 ± 0.00086 (b = 0),
0.441 ± 0.018 (0.5°), 0.449 ± 0.056 (1°); EKF+always 0.183 ± 0.026, 0.194 ± 0.031, 0.225 ± 0.036.
Final error at the end of the run: EKF+bias 0.337 ± 0.045° at 0.5° and 0.136 ± 0.032° at 1°;
EKF+always 0.114 ± 0.029° and 0.133 ± 0.03°.

What these rows show, and what they do not:
- **Empirical.** With the true bias and zero variance the RMSE returns to 2.86 m at 0.5° and 2.9 m at
  1°, and the NEES to 5.53 ± 2.2 and 7.16 ± 3.9. Both intervals contain 4, which means only that this
  mean-NEES test did not reject 4; intervals this wide cannot separate a mean near 4 from one about
  twice as large, and the b = 0 EKF already has 5.53. They do not show that the covariance is
  calibrated, and with var = 0 for the oracle they say nothing about the added variance term. In this
  scene the first-order mean correction works for biases up to 1°. This says nothing about larger
  biases or other scenes.
- **Empirical.** For the always-on estimate the NEES interval contains 4 at b = 0 ([3.5213, 5.6652])
  and at 0.5° ([3.6175, 5.8587]) and lies above 4 at 1°: mean 5.5530, interval [4.0021, 7.1038],
  recomputed from `results/improvement_results.pkl` with the project's t interval, so the lower
  endpoint is above 4 by 0.0021 (the rounded cell 5.55 ± 1.6 would suggest 3.95). With the pooled,
  truth-selected pairs of 9.4 the value 4 is only nominal, so this is a statement that the interval
  is just above the nominal value, not a verdict. The test does not validate the added variance term
  separately. The cost of the always-on estimate is at b = 0: RMSE 6.73 m against 2.86 m. The noise
  of the estimate (run-mean error 0.183°, which is 3.2 m at 1000 m) is carried through V into the
  track estimates. The paired difference is +3.87 ± 0.65 m (same file).
- **Empirical (negative).** The spike-and-slab estimate is neutral at b = 0 (RMSE 2.86 m, criterion
  B2 PASS) but removes only part of the error at 0.5° (12.4 m against 13.7 m) and its NEES is 129 ± 34,
  far above the nominal 4 (interval [94.9, 163.3]). Criterion B1 FAILS because at 0.5° the improvement stays
  below the practical threshold (example 14); at 1° it improves (15.5 m against 27 m) and still
  leaves a residual of 15.4 to 18.5 m of cross-range error in the four 1° rows (RESULTS.md
  [section 6.6](RESULTS.md#66-camera-bias-estimate), [Table C](RESULTS.md#table-c-camera-bias-estimate)).
- **Empirical (code fact).** EKF+bias and EKF+always update the same (b₁, v₁) from the same pairs and
  differ only in the weight P₁, so the difference between their results is due to that weight.
- **Hypothesis, not tested.** Why the weight hurts at 0.5°: a plausible reading is example 19. With
  prior 0.01 and at most a few hundred pairs, P₁ can still be well below 1 for a 0.5° bias, so the
  applied estimate P₁ b₁ is shrunk towards zero. Whether the reported s_b² is calibrated under this
  fixed-bias condition is unverified: s_b² is a posterior variance under the model's prior, and one
  experiment with a fixed 0.5° bias neither supports nor contradicts the prior probability 0.01 of a
  biased camera. The repository has no experiment that varies only the prior at 0.5° or that checks
  the calibration of s_b² (for example over a population of biases drawn from the prior).

| Kind | Statement | Reference |
|---|---|---|
| Assumption | The radar bearing is unbiased; the simulated camera bias is a constant added at every step | `camera.py:43` |
| Assumption | The estimator's H₁ model lets the bias drift as a random walk with variance q per second, which the simulated constant bias never uses | `camera_bias.py:34-71`, `:158-162` |
| Assumption | The believed σ_β and σ_c are the true ones; otherwise the pair variance and the gate are wrong | `camera_bias.py:115` |
| Assumption | A radar row and a camera row assigned to the same track belong to the same target; wrong matches inside the 5.77° gate corrupt a pair | `multi_target.py:373-393` |
| Identity | Under conditions (i) to (iii) of section 11.1, camera data and kinematics leave b at its prior | section 11.1 |
| Assumption | Pairs are independent and Gaussian with the believed variance; the gate truncates their tails and the evidence calculation ignores that | `camera_bias.py:171-186` |
| Approximation | V is a first-order, frozen-gain derivative; for the IMM the dependence of the mode probabilities on b is ignored | `sensitivity.py:81`, `imm.py:300-303` |
| Approximation | P + s_b² V Vᵀ neglects the covariance between the track error and the bias error (independence has not been established), uses the model-based s_b² = Var(b given D) in the place of the error variance of b̂, and summarises a two-component posterior by two moments | section 11.3 |
| Design choice | Only simultaneous radar–camera pairs at radar scans are used; no feedback of b̂ into association or filters | `multi_target.py:373-393`, `:364-371` |
| Empirical (negative) | B1 FAIL at 0.5°; a residual of 15.4 to 18.5 m cross-range error at 1°; EKF+bias NEES 129 ± 34 at 0.5° | table above, RESULTS.md section 6.6 |
| Empirical | B2 PASS (neutral world) and B3 PASS (maneuvers are not read as bias) | RESULTS.md [section 6.6](RESULTS.md#66-camera-bias-estimate) |

## 12. Truth-side disturbances

**Key idea.** The simulator and the tracker are separate objects. The tracker has beliefs (noise
levels, a motion model, the camera position) and the world follows a truth that can differ from them
in one controlled way at a time: a turn, a sustained acceleration, a camera bias, a time offset, a
lever arm, or wrongly scaled noise. In neutral runs the beliefs equal the truth; this is the
simplest case for the tracker and the case in which a good result proves least.

**Example 20 (exact; `maneuvers._advance` and `ExtendedKalmanFilter.predict`).** A target at 15 m/s
along +x turns left at 5 deg/s for 10 s. It ends at (131.67, 61.40) m with heading 50°, which is
64.1 m from the straight-line position (150, 0). The constant-velocity model of section 2, started
from zero covariance and predicted for the same 10 s with σ_a = 0.5 m/s², has a position std of
2.89 m per axis. The unaided prediction is far outside what the model expects; only the measurements
keep the filter on target, and in the maneuver rows of RESULTS.md section 5.2 they do not always.

### Equations as implemented

During step k, at time τ after its start, the true state moves with the total acceleration a of the
step and the turn rate ω (`src/fusion/maneuvers.py:1-22`, `:193-221`):

```math
v(\tau)=\mathcal R(\omega\tau)\,v_k+a\,\tau,\qquad p(\tau)=p_k+C(\omega,\tau)\,v_k+\tfrac12a\tau^2
```

with C the same position–velocity coupling as in the CT matrix of section 10. The IMM's CT mode uses
the same deterministic part and, like the simulator, adds the random acceleration after the rotation
(`imm.py:70-80`); it differs from the truth in that its turn rate is fixed and known, whereas the
true rate is whatever the scenario schedules (5, 10 or 20 deg/s, or random). The camera bias is added to the camera's bearing, the time offset makes the camera measure
the truth at t − delay (`maneuvers.py:21`), and the lever arm moves the true camera position while the
tracker's `CameraModel` keeps the position it believes (`camera.py:62-70`).

| Kind | Statement | Reference |
|---|---|---|
| Assumption | In the neutral runs the tracker is given the noise levels and σ_a that generated the data (the "inverse crime" of simulation studies), so neutral results are optimistic for any real system | README Limitations (states the content, not the term) |
| Approximation | A turn and an acceleration together differ from the exact solution by a splitting error of about 10⁻³ m/s per step at 20 deg/s | `maneuvers.py:19-21` |
| Empirical | Knowing the geometry removes the error that not knowing it creates: an unknown 20 m lever arm at 0° adds +14.7 ± 0.15 m of RMSE, a known one adds +0.00034 ± 0.0025 m | RESULTS.md [section 5.1](RESULTS.md#51-camera-bearing-bias-time-offset-and-lever-arm-scene-separated) |
| Empirical | A camera time offset of 200 ms adds +0.471 ± 0.057 m, much less than a 1° bias (+22.4 ± 0.16 m) | RESULTS.md [section 5.1](RESULTS.md#51-camera-bearing-bias-time-offset-and-lever-arm-scene-separated) |

## 13. Assumptions, approximations and failure modes

Each section above ends with its own table. This section is an index: which claims are supported,
which were tested and failed, and what the repository cannot say. Numbers are in the sections linked.

### 13.1 Claims that might be made, and what the repository shows

| Claim | Verdict here | Where |
|---|---|---|
| The IMM beats a high-Q EKF on maneuvers | Not against the stronger controls. Against the descriptive, untuned controls (σ_a = 3 and 5) CTRL fails on the maneuver rows and holds only against the weaker control on the held-out rows; against the frozen σ_a = 1 EKF both IMM arms fail on the maneuver rows (11 and 15 of 30) and pass on the held-out rows (8 and 10 of 12) | section 10, RESULTS.md [section 6.4](RESULTS.md#64-comparison-with-the-ekf-high-q-controls-ctrl) |
| The IMM costs nothing on straight targets | No. A2 fails for both arms | sections 9.5 and 10, RESULTS.md [section 6.3](RESULTS.md#63-neutral-non-inferiority-a2-the-cost-side) |
| IMM-A generalises to unseen maneuvers | Not shown. Held-out A1 fails for IMM-A, 7 of 12 winnable row-blocks where 8 are needed (IMM-B passes with 10 of 12) | section 10 |
| The bias estimate fixes a camera bias | Partly. B1 fails at 0.5°; a residual remains at 1° | section 11.4 |
| The corrected covariance is calibrated | Not shown. In one scene the mean NEES interval contains the nominal 4 for the oracle arm and, at b = 0 and 0.5°, for the always-on arm (just above 4 for always-on at 1°, lower endpoint 4.0021); that is one mean test on truth-selected pairs, not a calibration check. The EKF+bias interval is far above 4 at 0.5° and 1° | section 11.4 |
| Outage-aware coasting is free | No. Position error grows during the outage, ghosts live as long as the coast limit, and camera clutter prolongs them | section 8 |
| Two close targets survive a long radar outage | Not established either way. The failure path exists in the code and was not triggered in the measured scenes | section 7 |
| The filters are consistent | In the neutral single-target runs mostly yes; radar-only EKF near, velocity is underconfident | sections 5 and 9.4 |
| The parameters were optimally tuned | No. They were chosen by a fallback rule; no candidate met the neutral margins on the tuning seeds | section 9 |
| The intervals are corrected for multiple comparisons | No. They are unadjusted | section 9 |
| The results hold for real sensors | Not tested. Everything is simulated | section 13.3 |

### 13.2 Failure modes: the first symptom to look for

| Disturbance | Mechanism | First symptom | Where |
|---|---|---|---|
| Maneuver or acceleration | Violates zero-mean white acceleration | Innovations outside the gate, track pulled off, ID switches, NEES far above the dimension | sections 2 and 6, RESULTS.md section 5.2 |
| Camera bias or lever arm | Camera bearings are systematically wrong | Cross-range error and NEES, with missed rate and ID switches nearly unchanged | sections 3 and 11 |
| Noise assumed too small | Covariances too small | Gates shrink, ghosts and ID switches rise | section 6, RESULTS.md section 5.3 |
| Radar outage | Missing scans counted as misses | Tracks deleted and identities lost (unaware); ghosts (aware) | section 8 |
| Closely spaced targets | Gates overlap; camera withheld | Association swaps, ID switches | section 7 |
| Camera clutter during an outage | A clutter point is accepted as an update | Coast clock reset, ghost life extended | section 8 |
| Target near the sensor | Bearing Jacobian explodes | Guard at 50 m; `ValueError` below 10⁻⁶ m | sections 5 and 6 |

### 13.3 What the repository cannot say

- Simulation only. All sensors, targets and disturbances are generated by `src/fusion`; no real data
  was used, and the neutral runs give the tracker the true noise levels (RESULTS.md
  [section 8](RESULTS.md#8-general-limitations)).
- Planar geometry, one radar and one camera at or near the origin, a constant-velocity baseline.
- The robustness rows change one factor at a time; combined disturbances appear only in the "hard"
  group of the improvement experiment.
- The 5-seed pilot runs are not evidence. The descriptive controls (EKF with σ_a = 3 and 5) were not
  tuned and are not criteria.

## 14. Equation sheet

A reference of the equations of the sections above, in the form the code uses them.

Motion and prediction (section 2):

```math
x_{k+1}=Fx_k+Ga_k,\quad Q=\sigma_a^2GG^{\top},\quad \hat x^-=F\hat x,\quad P^-=FPF^{\top}+Q
```

Sensors and initialisation (section 3):

```math
z_r=\big(\lVert p\rVert,\ \operatorname{atan2}(p_y,p_x)\big)+v_r,\qquad z_c=\operatorname{atan2}(p_y-c_y,p_x-c_x)+b+v_c
```

```math
P_0=\operatorname{blkdiag}\big(J\operatorname{diag}(\sigma_r^2,\sigma_\beta^2)J^{\top},\ \sigma_v^2I_2\big)
```

Update (sections 4 and 5):

```math
\nu=z-h(\hat x^-),\quad \nu_\beta\leftarrow\operatorname{wrap}(\nu_\beta)\ \text{(bearing components only; the range residual is not wrapped)}
```

```math
S=HP^-H^{\top}+R,\quad K=P^-H^{\top}S^{-1},\quad \hat x^+=\hat x^-+K\nu
```

```math
P^+=(I-KH)P^-(I-KH)^{\top}+KRK^{\top}
```

Gating and association (sections 6 and 7):

```math
d^2=\nu^{\top}S^{-1}\nu\le\gamma_m,\qquad c=d^2+\ln\det S
```

```math
\text{ambiguous}\iff\lvert\operatorname{wrap}(\theta_i-\theta_j)\rvert<\sqrt{\gamma_1}\,(s_i+s_j)
```

Lifecycle and coasting (section 8):

```math
\text{confirm}\iff h\ge M,\qquad \text{delete tentative}\iff n-h>N-M,\qquad \text{delete confirmed}\iff m>K
```

```math
\text{delete coasting}\iff \text{steps}>\lfloor T_{\max}/\Delta t+10^{-9}\rfloor
```

IMM (section 10):

```math
c_j=\sum_i\Pi_{ij}\mu_i,\qquad w_{ij}=\tfrac{\Pi_{ij}\mu_i}{c_j},\qquad \Pi_{ii}=\tfrac{1+(M-1)\lambda^{1/n}}{M},\quad \lambda=\tfrac{Ma-1}{M-1}
```

```math
x_{0j}=\sum_iw_{ij}x_i,\qquad P_{0j}=\sum_iw_{ij}\big[P_i+(x_i-x_{0j})(x_i-x_{0j})^{\top}\big]
```

```math
\mu_j^{+}=\frac{\mu_j^{-}L_j}{\sum_l\mu_l^{-}L_l},\qquad \mu^{-}=c\ \text{(first measurement of a step)},\quad \mu^{-}=\mu^{+}\ \text{of the previous measurement otherwise}
```

Camera bias (section 11):

```math
\delta=z_c-z_r=b+v_c-v_r,\qquad \hat b=P_1b_1=E[b\mid D],\qquad s_b^2=\operatorname{Var}(b\mid D)=P_1v_1+P_1(1-P_1)\,b_1^2
```

```math
V\leftarrow FV,\qquad V\leftarrow(I-KH)V+KJ_b,\qquad \hat x_c=\hat x-V\hat b,\qquad P_c=P+s_b^2\,VV^{\top}
```

Statistics (section 9):

```math
\bar x\pm t_{0.975,n-1}\tfrac{s}{\sqrt n},\qquad s_d^2=s_a^2+s_b^2-2s_{ab},\qquad \widehat{\operatorname{Var}}(\bar d)=\tfrac{s_d^2}{n}
```

```math
\varepsilon=e^{\top}P^{-1}e,\qquad \Big[\tfrac{\chi^2_{0.025}(Nd)}{N},\ \tfrac{\chi^2_{0.975}(Nd)}{N}\Big]
```
