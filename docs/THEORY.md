# Tracking theory and implementation reference

This reference covers the planar radar–camera tracker, optional IMM and camera-bias correction,
and evaluation limits. Results and reproduction commands are in [RESULTS.md](RESULTS.md).

Contents:

1. [Conventions and assumptions](#1-conventions-and-assumptions)
2. [State space and sensors](#2-state-space-and-sensors)
3. [Kalman filter and EKF](#3-kalman-filter-and-ekf)
4. [Gating and data association](#4-gating-and-data-association)
5. [Track lifecycle and radar outages](#5-track-lifecycle-and-radar-outages)
6. [Evaluation and statistical criteria](#6-evaluation-and-statistical-criteria)
7. [IMM motion model](#7-imm-motion-model)
8. [Camera-bias estimation and correction](#8-camera-bias-estimation-and-correction)
9. [Limitations and failure modes](#9-limitations-and-failure-modes)
10. [Implementation and reproducibility](#10-implementation-and-reproducibility)

## 1. Conventions and assumptions

The state is x = [p_x, p_y, v_x, v_y]ᵀ, with position in metres and velocity in m/s.
P is an estimated error covariance, not a measurement of the actual error. Superscripts − and +
denote prediction and update; z, h, H, R, ν, S and K denote measurement, measurement model,
Jacobian, measurement-noise covariance, innovation, innovation covariance and Kalman gain.
Bearings are counter-clockwise from +x, in radians; model outputs are in [−π, π].

**Implemented defaults:** Δt = 0.1 s; radar 1 Hz; camera 10 Hz; σ_r = 5 m; σ_β = 2°;
σ_c = 0.1°; σ_a = 0.5 m/s²; initial velocity std σ_v = 20 m/s. Radar and camera are at the
origin unless configured otherwise. The default field of view is full-circle, 50–3000 m;
detection probability is 0.9 per target and sensor, with mean five Poisson clutter points per
scan, uniform in range and bearing. Experiment rows can override these defaults.

Statement labels distinguish **implemented facts**, mathematical **identities** under stated
conditions, modelling **assumptions**, **approximations**, and scene-specific **empirical** findings.
Neutral simulations give the tracker the true noise parameters; this is an optimistic assumption,
not evidence that real sensor uncertainty is known. Coordinates and dynamics are planar.

Results use mean ± 95% Student t half-width across seeds, except the single-target table in
[RESULTS §1](RESULTS.md#1-radar-vs-fusion-single-target), where ± is a standard deviation (SD).
Source line references below refer to commit `a7dc570`; named functions help if lines move.

## 2. State space and sensors

### 2.1 Constant-velocity process

**Assumption:** independent, zero-mean Gaussian acceleration a_k is held constant within each
step, with equal variance on both axes. This is discrete white-noise acceleration, not the
continuous-time white-noise model, whose Q has different time-step scaling.

```math
x_{k+1}=Fx_k+Ga_k,\qquad a_k\sim\mathcal N(0,\sigma_a^2 I_2)
```

```math
F=\begin{pmatrix}I_2&\Delta t I_2\\0&I_2\end{pmatrix},\qquad
G=\begin{pmatrix}\tfrac12\Delta t^2 I_2\\\Delta t I_2\end{pmatrix}
```

```math
Q=\sigma_a^2 GG^{\top}
=\sigma_a^2\begin{pmatrix}\tfrac14\Delta t^4 I_2&\tfrac12\Delta t^3 I_2\\\tfrac12\Delta t^3 I_2&\Delta t^2 I_2\end{pmatrix}
```

Q has rank two; positive-definite P remains positive definite under prediction. Covariance
transport need not increase every variance. Neutral truth and filter use the same model and σ_a.
Implementation: [base.py](../src/fusion/filters/base.py), `cv_transition`, `cv_process_noise` (`:8-34`, `:96-99`); [scenario.py](../src/fusion/scenario.py), `:34-42`.

### 2.2 Measurement models and initialisation

The radar model below assumes the origin. The camera uses its assumed position c = (c_x, c_y).
The simulator adds a true constant camera bearing bias b; the raw tracker's camera model omits b.

```math
z_r=\begin{pmatrix}\sqrt{p_x^2+p_y^2}\\\mathrm{atan2}(p_y,p_x)\end{pmatrix}+v_r,
\quad v_r\sim\mathcal N\big(0,\mathrm{diag}(\sigma_r^2,\sigma_\beta^2)\big)
```

```math
z_c=\mathrm{atan2}(p_y-c_y,p_x-c_x)+b+v_c,
\quad v_c\sim\mathcal N(0,\sigma_c^2)
```

**Assumptions:** range and bearing noise are independent; measurement noise is independent
across sensors and scans; detections are independent. The camera constrains bearing, not range.
Wrong mounting position, timing or bias invalidates the assumed measurement model.

An unclaimed radar measurement (r, β) starts a track with unknown velocity:

```math
\begin{aligned}
&\hat x_0=(r\cos\beta,\ r\sin\beta,\ 0,\ 0)^{\top},\\
&J=\begin{pmatrix}\cos\beta&-r\sin\beta\\\sin\beta&r\cos\beta\end{pmatrix},\\
&P_0=\begin{pmatrix}J\,\mathrm{diag}(\sigma_r^2,\sigma_\beta^2)J^{\top}&0\\0&\sigma_v^2 I_2\end{pmatrix}
\end{aligned}
```

**Approximation:** P₀ is first-order polar-to-Cartesian covariance propagation, not the exact
covariance of the converted nonlinear measurement. Velocity starts at zero with the stated prior.
Measurement models reject targets within 10⁻⁶ m of their sensor, where bearing is undefined.
Implementation: [radar.py](../src/fusion/sensors/radar.py), `:75-135`; [camera.py](../src/fusion/sensors/camera.py), `:43`, `:62-88`; [sensors/base.py](../src/fusion/sensors/base.py), `:9`, `:33-51`.

## 3. Kalman filter and EKF

Prediction for each track or IMM mode:

```math
\hat x^-=F\hat x,\qquad P^-=FPF^{\top}+Q
```

For a linear sensor h(x) = Hx, the Kalman update is exact under a linear-Gaussian model with
known covariances and independent process and measurement noise. The EKF uses the same update
with h evaluated at x̂⁻ and H its first-order Jacobian there:

```math
\nu=z-h(\hat x^-),\qquad S=HP^-H^{\top}+R
```

Replace **each bearing component only** by the wrapped residual before using ν:

```math
\nu_\beta\leftarrow\mathrm{atan2}(\sin\nu_\beta,\cos\nu_\beta)\in[-\pi,\pi]
```

The radar range residual is never wrapped. Simulated measurements themselves are not wrapped.

```math
K=P^-H^{\top}S^{-1},\qquad \hat x^+=\hat x^-+K\nu
```

```math
P^+=(I-KH)P^-(I-KH)^{\top}+KRK^{\top}
```

The implementation solves linear systems instead of forming S⁻¹. The Joseph covariance form
is symmetric positive semidefinite in exact arithmetic for any K when P⁻ and R are; floating-point
arithmetic gives no universal guarantee. For the optimal gain it equals the short update form.
**EKF approximations:** first-order linearisation once per update, no iteration, and a Gaussian
posterior approximation. Model mismatch can invalidate the resulting uncertainty estimate.

With r² = p_x² + p_y², Δp = p − c and ρ² = Δp_x² + Δp_y²:

```math
\begin{aligned}
&H_r=\begin{pmatrix}p_x/r&p_y/r&0&0\\-p_y/r^2&p_x/r^2&0&0\end{pmatrix},\\
&H_c=\begin{pmatrix}-\Delta p_y/\rho^2&\Delta p_x/\rho^2&0&0\end{pmatrix}
\end{aligned}
```

Neither sensor directly measures velocity. Position–velocity covariance transfers information
to velocity during updates. Near the sensor, the bearing Jacobian becomes ill-conditioned.
Each tracker step predicts all tracks, then processes radar association, updates, lifecycle and
births, then camera association and updates. Only confirmed tracks receive camera updates.

Implementation: [kalman.py](../src/fusion/filters/kalman.py), `:52-68`; [ekf.py](../src/fusion/filters/ekf.py), `:30-70`; [angles.py](../src/fusion/angles.py), `:15`;
[multi_target.py](../src/fusion/tracker/multi_target.py), `:336-341`, `:517-522`.

## 4. Gating and data association

### 4.1 Innovation gate and assignment cost

The tracker symmetrises S, uses a Cholesky factor S = LLᵀ, and accepts a measurement when:

```math
d^2=\nu^{\top}S^{-1}\nu=\lVert L^{-1}\nu\rVert^2\le\gamma_m,
\qquad \gamma_m=F^{-1}_{\chi_m^2}(0.99)
```

Defaults are γ₂ = 9.210 for radar and γ₁ = 6.635 for camera. **Identity under an assumption:**
if the true innovation is N(0, S), d² is χ²_m and 1% of correct measurements fall outside the
99% gate. This is not guaranteed under poor linearisation, wrong associations or wrong noise.
Association uses a Gaussian likelihood cost, up to terms constant for a given sensor:

```math
c_{ij}=d_{ij}^2+\ln\det S_i
=d_{ij}^2+2\sum_q\ln L_{i,qq}
```

The determinant term penalises vague predictions that d² alone would favour. Gating rejects
predicted positions within max(min_range, 10⁻⁶) of the origin; min_range defaults to 50 m.
For cameras, the tracker also checks distance from the camera model's assumed position.
Implementation: [gating.py](../src/fusion/association/gating.py), `:14-32`, `:47-59`, `:89-179`.

### 4.2 Global assignment within each group

**Implemented objective:** maximise the number of usable one-to-one in-gate pairs, then minimise
their total cost. Costs are shifted by their minimum. Unusable entries receive a sentinel
B = (n + 1) max(shifted costs) + 1, with n = min(rows, columns); SciPy's Hungarian solver is run
and sentinel pairs are discarded. This enforces the lexicographic objective, not unrestricted
minimum total cost, which could prefer no pairs.

Confirmed tracks are associated first; tentative tracks claim remaining radar measurements.
Thus assignment is global within each group, **not** globally optimal over all tracks together.
Unclaimed finite radar measurements outside min_range create tentative tracks. The camera
never creates tracks or supplies lifecycle hits. One-to-one measurement generation is assumed;
merged detections and probabilistic data association are not modelled.
Implementation: [assignment.py](../src/fusion/association/assignment.py), `:23-71`; [multi_target.py](../src/fusion/tracker/multi_target.py), `:461-491`.

### 4.3 Camera ambiguity rule

For confirmed tracks with valid camera geometry:

```math
\theta_i=h_c(\hat x_i),\quad
s_i^2=H_{c,i}P_iH_{c,i}^{\top}+\sigma_c^2,\quad w_i=\sqrt{\gamma_1}\,s_i
```

```math
i\text{ is ambiguous}\iff\exists j\ne i:
\lvert\mathrm{wrap}(\theta_i-\theta_j)\rvert<w_i+w_j
```

**Heuristic:** withhold camera updates when predicted bearing gates overlap; other tracks use
the ordinary gated assignment. Camera noise is included in s_i. The rule depends on every
pair present, can withhold useful information, and can leave close tracks without updates
during a radar outage. That failure path was not triggered in the measured outage scenes;
see [RESULTS §4.3](RESULTS.md#43-measured-limitations).
Implementation: [multi_target.py](../src/fusion/tracker/multi_target.py), `:493-535`.

## 5. Track lifecycle and radar outages

Let h be radar hits, n radar scans since birth and m consecutive radar misses. Defaults are
M = 3, N = 5 and K_miss = 5 (distinct from the Kalman gain K). Birth counts as h = n = 1.
After each radar scan the counting rules are:

```math
\begin{aligned}
&\text{confirm tentative}\iff h\ge M,\\
&\text{delete tentative}\iff n-h>N-M\quad\text{if not confirmed},\\
&\text{delete confirmed}\iff m>K_{\mathrm{miss}}
\end{aligned}
```

Confirmation is checked first; an unconfirmed track cannot remain tentative after scan N.
The sixth consecutive miss deletes a confirmed track. These are counting heuristics, not
likelihood tests; clutter can create tracks and association quality does not enter the counts.
Implementation: [track.py](../src/fusion/tracker/track.py), `:64-103`.

**Outage-aware policy:** the tracker must be told `radar_down` by a sensor-health mechanism.
It skips the radar scan, freezes lifecycle counters and prevents births. Camera updates still
run. After them, the default "drop" policy removes tentative tracks, then deletes tracks with:

```math
\mathrm{coast\_steps}>\left\lfloor T_{\max}/\Delta t+10^{-9}\right\rfloor
```

Every prediction increments coast_steps; any radar or camera update resets it to zero.
T_max = 15 s gives 150 steps; deletion requires **more than** 150 since the last update.
The 10⁻⁹ tolerance avoids rounding a whole-step duration down. Coasting deletions occur only
on steps marked radar_down. An unaware tracker instead processes empty radar scans as misses.

**Empirical tradeoff:** awareness preserves identities in the reported separated-scene outages,
but error grows and vanished targets leave ghosts. Accepted camera clutter resets a ghost's
clock and can extend its lifetime beyond the configured limit. See [RESULTS §4.1](RESULTS.md#41-radar-outage-of-growing-duration-experiment-a),
[§4.2](RESULTS.md#42-ghost-lifetime-against-maximum-coasting-time-experiment-f) and [§4.3](RESULTS.md#43-measured-limitations).
Implementation: [multi_target.py](../src/fusion/tracker/multi_target.py), `:315-316`, `:338-341`,
`:435`, `:441-459`; [dropout.py](../src/fusion/dropout.py), `STEP_TOLERANCE` (`:24`).

## 6. Evaluation and statistical criteria

### 6.1 Truth matching and scores

Confirmed tracks are matched one-to-one to true targets using squared position distance:
maximise pairs within 50 m, then minimise their total squared distance. Error diagnostics use
200 m; tentative tracks are excluded. Scores use matched target–step pairs after a 5 s burn-in.

```math
\mathrm{RMSE}=\sqrt{\frac1{|\mathcal M|}
\sum_{(i,k)\in\mathcal M}\lVert p_{i,k}-\hat p_{j(i,k),k}\rVert^2}
```

Missed rate is the share of in-view target–step pairs without a matched confirmed track.
Ghost rate is the mean confirmed-minus-matched count per step, including real tracks pulled
away from truth. False-track rate is the mean number of confirmed tracks never matched during the run.
ID switches count changes in a target's matched track ID across gaps; 1 → gap → 2 is a switch.
**Qualification:** matched-only RMSE can improve when a tracker loses difficult targets.
Always read it alongside missed rate; match-distance changes alter missed and ghost meanings.
Implementation: [mtt_metrics.py](../src/fusion/mtt_metrics.py), `:70-127`, `:151-245`.

### 6.2 Seed intervals and paired comparisons

For valid per-seed scores, use a sample mean x̄, sample SD s and the Student t interval:

```math
\bar x\pm t_{0.975,n-1}\frac{s}{\sqrt n}
```

Undefined scores are dropped and counted in n_valid. Seed independence and an adequately
normal sampling distribution are assumptions; bounded/skewed rates make t intervals approximate.
Intervals are **unadjusted for multiple comparisons**, including the family of 108 checks.

Within an improvement row, every arm runs on one bit-identical simulated object per seed.
Across rows, source-specific random streams share seed keys, but maneuvers or changed clutter
produce different realisations. Comparisons use per-seed differences d_s = a_s − b_s:

```math
\begin{aligned}
&s_d^2=s_a^2+s_b^2-2s_{ab},\\
&\widehat{\mathrm{Var}}(\bar d)=s_d^2/n,\qquad
\bar d\pm t_{0.975,n-1}s_d/\sqrt n
\end{aligned}
```

The variance identity is algebraic. Pairing narrows the interval relative to dropping the
covariance term only when s_ab > 0; common random numbers do not guarantee that sign.
Implementation: [mtt_metrics.py](../src/fusion/mtt_metrics.py), `:309-391`; [improvement_experiment.py](../src/fusion/improvement_experiment.py), `:580-591`;
[mtt_simulation.py](../src/fusion/mtt_simulation.py), `:18-35`, `:184-188`.

### 6.3 NEES and consistency qualifications

For e = x − x̂ and reported covariance P:

```math
\varepsilon=e^{\top}P^{-1}e,\qquad
\left[\frac{\chi^2_{0.025}(Nd)}N,\frac{\chi^2_{0.975}(Nd)}N\right]
```

**Identity under consistency:** if e ~ N(0, P), ε ~ χ²_d and E[ε] = d. The displayed band is
for mean NEES from N **independent runs at the same step**; d = 4 for the full state and 2
for position or velocity. Steps within a run are correlated; counting band violations over
steps is not a test. Single-target verdicts instead use a t interval across seed-level time
means: above d is labelled "overconfident", below d "underconfident", containing d "consistent".

Containing d means only that this mean test did not reject d. It establishes neither covariance
calibration in every direction nor the validity of an added covariance term separately.
Multi-target NEES is pooled over truth-selected matched pairs at 200 m, then summarised across
seeds. Truth-based assignment selects on the error itself, so d = 4 is a **nominal reference**,
not an exact expectation; the direction and size of the selection effect were not measured.
These results do not use the single-target verdict names. Without reference truth, NIS
νᵀS⁻¹ν can check measurement-prediction covariance, not state error directly.
Implementation: [consistency.py](../src/fusion/consistency.py), `:17-39`, `:66-87`, `:157-215`; [robustness_metrics.py](../src/fusion/robustness_metrics.py), `:239-264`, `:336-337`.

### 6.4 Practical improvement and non-inferiority

All scores are lower-is-better; differences are arm − reference. Criteria were fixed in code
before evaluation. Let L, U and h_w denote the interval bounds and half-width.

| Rule | Implemented condition |
|---|---|
| improves | U < 0 and mean ≤ −τ; fewer than two valid seeds gives "undefined" |
| worsens | L > 0 and mean ≥ τ; otherwise judge returns "neither" |
| non-inferior | PASS if U ≤ m; otherwise UNDERPOWERED if mean ≤ m and h_w > m; else FAIL |
| equivalent | PASS if L ≥ −m and U ≤ m; otherwise UNDERPOWERED if absolute mean ≤ m and h_w > m; else FAIL |
| insufficient data | Non-inferiority/equivalence return UNDERPOWERED with fewer than two valid seeds |
| aggregate | All PASS → PASS; any FAIL → FAIL; otherwise UNDERPOWERED |

**Non-inferiority requires mean + half-width ≤ m.** FAIL means this condition was not
established under the implemented rule; inferiority would require L > m. UNDERPOWERED is the
repository's width-based classification, not a standard power analysis.
Margins are RMSE 0.3 m, missed 0.005, ghost 0.01 per step and ID switches 0.2 per run.
The practical threshold is max(floor, fraction × max(D, 0)), with D the baseline's degradation
from the neutral row on the data being judged. Floors are missed 0.01, RMSE/cross-range 1 m,
ghost 0.02 per step and ID switches 0.5 per run; the first two use fraction 0.25.

- **A1:** at least ⌈2W/3⌉ of W frozen winnable maneuver blocks improve in window missed rate;
  no block may worsen in window missed, ghost rate or ID switches. Winnability is selected on
  tuning seeds 1000–1019, not evaluation seeds.
- **CTRL:** A1 against each high-Q control; PASS requires the strongest control, chosen by
  lowest mean window missed over the winnable blocks.
- **A2:** non-inferiority against EKF on every neutral block and margin.
- **B1:** improvement in run RMSE and cross-range at 0.5° and 1° in every block.
- **B2:** equivalence of EKF+bias to EKF at zero bias for every margin.
- **B3:** non-inferiority of the shift in run-mean absolute bias error against 0.05°, plus
  equivalence for every tracking margin on every maneuver row.

No tuning candidate met the neutral margins; frozen settings use a fallback, not an optimality
claim. Evaluation seeds 0–49 overlap earlier baseline experiments; seeds 0–9 generated the drag
hypothesis. Its separate H-drag check sums missed rates over 5 s bins covering 25–45 s in the
separated 5 deg/s turn: a reduction of at least 0.5 with an interval excluding zero in every
block supports it, also reported on seeds 10 onward. Five-seed pilots are not evidence.
Implementation: [improvement_criteria.py](../src/fusion/improvement_criteria.py), `:35-47`,
`:99-143`, `:215-231`, `:365-627`; see [RESULTS §6.1](RESULTS.md#61-limitations-of-the-frozen-parameters-verbatim).

## 7. IMM motion model

### 7.1 Modes and transition probabilities

All modes share the four-dimensional state. CV uses the process in §2. With θ = ω Δt,
a coordinated-turn mode uses the nested block matrix below (positive ω turns counter-clockwise):

```math
F_{\mathrm{CT}}=\begin{pmatrix}I_2&\tfrac1\omega\begin{pmatrix}\sin\theta&-(1-\cos\theta)\\1-\cos\theta&\sin\theta\end{pmatrix}\\0&\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}\end{pmatrix}
```

As ω → 0 this becomes CV. CT retains the CV process-noise matrix, adding random acceleration
after rotation. IMM-A uses two CV modes (σ_a = 0.5 and 2.0); IMM-B uses CV with σ_a = 0.5
and CT modes at ±7.5 deg/s with σ_a = 2.0. These fixed rates do not exactly match truth turns
of 5, 10 or 20 deg/s. A single-mode IMM follows the EKF path without mixing or combination.

The symmetric Markov transition is specified per radar scan by stay probability a = 0.95.
For M modes and n filter steps per scan, its per-step n-th root is:

```math
\begin{aligned}
&\lambda=\frac{Ma-1}{M-1},\qquad \lambda_s=\lambda^{1/n},\\
&\Pi_{ii}=\frac{1+(M-1)\lambda_s}{M},\qquad
\Pi_{ij}=\frac{1-\Pi_{ii}}{M-1}\quad(i\ne j)
\end{aligned}
```

Π_ij is the probability of mode i transitioning to j. Equal switching probability to every
other mode is an assumption; the code requires a > 1/M for the nonnegative root used here.
Implementation: [imm.py](../src/fusion/filters/imm.py), `:48-116`; [improvement_experiment.py](../src/fusion/improvement_experiment.py), `:111-116`, `:194-202`.

### 7.2 Mixing, prediction and sequential updates

For current mode probabilities μ_i, predicted probabilities c_j and mixing weights w_ij:

```math
c_j=\sum_i\Pi_{ij}\mu_i,\qquad w_{ij}=\frac{\Pi_{ij}\mu_i}{c_j}
```

```math
\begin{aligned}
&x_{0j}=\sum_iw_{ij}x_i,\\
&P_{0j}=\sum_iw_{ij}\big[P_i+(x_i-x_{0j})(x_i-x_{0j})^{\top}\big]
\end{aligned}
```

The covariance includes disagreement between mode means. **Approximation:** mixing replaces
a Gaussian mixture by one Gaussian with matching first two moments. Each mode predicts with
F_j, Q_j, then applies the EKF for each measurement. Its innovation likelihood and posterior are:

```math
\begin{aligned}
&\ln L_j=-\tfrac12\big(d_j^2+\ln\det S_j+m\ln(2\pi)\big),\\
&\mu_j^+=\frac{\mu_j^-L_j}{\sum_l\mu_l^-L_l}
\end{aligned}
```

**Sequential-update requirement:** μ⁻ = c for the first measurement after prediction;
for each later measurement in the same step, μ⁻ is the preceding measurement's posterior μ⁺.
Thus a radar–camera step starts the camera update from the radar posterior. There is no second
mixing or prediction between measurements. Computation uses log weights and log-sum-exp;
if every posterior weight is nonfinite, probabilities remain at their pre-update values.

The reported state combines the current mode estimates, again including their spread:

```math
\begin{aligned}
&\hat x=\sum_j\mu_j\hat x_j,\\
&P=\sum_j\mu_j\big[P_j+(\hat x_j-\hat x)(\hat x_j-\hat x)^{\top}\big]
\end{aligned}
```

Gating and camera ambiguity use this combined estimate. Implementation:
[imm.py](../src/fusion/filters/imm.py), `:129-219`, `:402-452`.
**Empirical limits:** primary maneuver A1 passes, but neutral A2 fails for both arms;
held-out A1 fails for IMM-A and passes for IMM-B. Neither beats the stronger descriptive
high-Q controls on maneuver rows. Those controls were not tuned; the frozen σ_a = 1 EKF
comparison also fails for both IMM arms on maneuver rows. See [RESULTS §6.2–6.4](RESULTS.md#62-verdicts-heading-lines-verbatim-from-criteria_8btxt).

## 8. Camera-bias estimation and correction

### 8.1 Identifiability and measurement pairs

**Identity under symmetry assumptions:** camera-only data leave b at its prior if detection
and clutter are bearing-independent, motion dynamics/noise commute with planar rotations, and
the trajectory prior is rotation-invariant. Rotating trajectory and velocity about the camera
by b produces the same bearings as bias b. CV and CT satisfy the dynamics condition; a
radar-initialised prior does not satisfy the prior condition. A finite bearing sector also
breaks the stated symmetry. This is not an unconditional claim about all camera systems.

Unbiased radar bearing breaks the symmetry; range at the rotation centre does not.
The estimator uses simultaneous radar–camera bearing differences, not kinematic information
between scans. A time offset adds motion-induced bearing change; a lever arm adds parallax.

Only tracks confirmed at that step's radar association, updated by both sensors, and whose
radar measurement gates to exactly one confirmed track contribute a pair. The pair is wrapped:

```math
\delta=\mathrm{wrap}(z_c-z_{r,\beta})=b+v_c-v_{r,\beta}\quad\text{locally in angle}
```

```math
\sigma_p^2=\sigma_\beta^2+\sigma_c^2,\qquad
|\delta|\le\sqrt{\gamma_1(\sigma_p^2+\sigma_s^2)}
```

The default pair gate is 5.77°. **Assumptions:** unbiased radar, correct pair assignment,
correct noise variances, independent Gaussian pairs. Gating truncates tails; evidence ignores
that truncation. Wrong matches within the gate can corrupt the estimate.
Implementation: [camera_bias.py](../src/fusion/tracker/camera_bias.py), `:96-186`; [multi_target.py](../src/fusion/tracker/multi_target.py), `:373-393`.

### 8.2 Spike-and-slab estimator

H₀ fixes b = 0; H₁ starts at N(0, σ_s²), σ_s = 1°, with random-walk variance q Δt,
q = (0.1°)²/600 rad²/s. The simulated bias is constant and does not exercise this drift model.
Under H₁ the scalar state is (b₁, v₁); prediction adds q Δt to v₁. For each accepted pair:

```math
\begin{aligned}
&S_1=v_1+\sigma_p^2,\\
&\ell\leftarrow\ell-\tfrac12\left(\frac{(\delta-b_1)^2}{S_1}+\ln S_1\right)
+\tfrac12\left(\frac{\delta^2}{\sigma_p^2}+\ln\sigma_p^2\right),\\
&k=v_1/S_1,\qquad b_1\leftarrow b_1+k(\delta-b_1),\\
&v_1\leftarrow\max\big((1-k)v_1,10^{-18}\big)
\end{aligned}
```

Here ℓ is log-odds of H₁; frozen experiments use prior 0.01 (the configuration default is 0.5).
Its update is used only in spike_slab mode; always mode uses the same H₁ scalar filter with full weight. With pair data D:

```math
\begin{aligned}
&P_1=(1+e^{-\ell})^{-1},\qquad \hat b=P_1b_1=E[b\mid D],\\
&s_b^2=\mathrm{Var}(b\mid D)=P_1v_1+P_1(1-P_1)b_1^2
\end{aligned}
```

**Identity under the model:** s_b² is the variance of a point mass at zero mixed with the
H₁ Gaussian, not the sampling variance of b̂. For a fixed true bias, estimator mean squared
error includes shrinkage bias; s_b² describes that error only under a correctly specified
generative model/prior. Always mode returns (b₁, v₁); oracle mode returns (b_true, 0).
Implementation: [camera_bias.py](../src/fusion/tracker/camera_bias.py), `:34-71`, `:144-186`.

### 8.3 Sensitivity correction and covariance assumptions

Raw track filters carry V = ∂x̂/∂b, with J_b = 1 for camera and 0 for radar:

```math
V\leftarrow FV\quad\text{on prediction},\qquad
V\leftarrow(I-KH)V+KJ_b\quad\text{on update}
```

```math
\hat x_c=\hat x-V\hat b,\qquad P_c=P+s_b^2 VV^{\top}
```

**Approximation:** the derivative freezes K and H, omitting their state dependence, including
(∂K/∂b)ν. IMM sensitivities are mixed and combined with mode weights, but derivatives of those
weights with respect to b are ignored. Let u = x_true − x̂(0) and w = b_true − b̂. Then:

```math
\begin{aligned}
&\mathrm{Cov}(u-Vw)=\mathrm{Cov}(u)+V\,\mathrm{Var}(w)V^{\top}\\
&\quad-\mathrm{Cov}(u,w)V^{\top}-V\,\mathrm{Cov}(w,u)
\end{aligned}
```

The reported P_c keeps only the first two terms, assuming P represents unbiased-track error
and replacing Var(w) by the model-based s_b². **Unverified assumption:** cross-covariances are
zero. Tracks and bias pairs share measurements; independence has not been established, and
the omitted terms' sign and size were not measured. P_c may be too small or too large.
The approximation also discards the mixture's shape. Shared b̂ makes corrections across tracks
correlated; individual P_c values are marginal covariances and omit inter-track correlation.

Correction is output-only: b̂ is never fed back into association or raw filters. This avoids
entering a measurement into a track update twice; it does **not** prove independent errors.
Implementation: [sensitivity.py](../src/fusion/filters/sensitivity.py), `:23-45`, `:81`; [imm.py](../src/fusion/filters/imm.py), `:300-303`, `:383-395`, `:412-415`, `:431-439`;
[multi_target.py](../src/fusion/tracker/multi_target.py), `:364-393`.

**Empirical limits:** B1 fails at 0.5°; correction is partial at 1°. B2 and B3 pass in the
reported tests. Oracle mean-NEES intervals containing four do not establish calibration or
validate the added variance term (oracle variance is zero). Always-on NEES at 1° has lower
endpoint 4.0021, just above nominal four despite rounded values suggesting otherwise
(crossing scene, clutter 0, seeds 0–49; pooled state NEES at the 200 m diagnostic distance).
The fixed-bias experiments do not validate the prior 0.01 or calibration of s_b². See
[RESULTS §6.6](RESULTS.md#66-camera-bias-estimate), [Table C](RESULTS.md#table-c-camera-bias-estimate)
and the provenance qualification below for diagnostics not published in those tables.

## 9. Limitations and failure modes

| Disturbance or limitation | Mechanism and symptom |
|---|---|
| Maneuver or sustained acceleration | Violates the CV white-acceleration assumption; lag, out-of-gate innovations, ID switches and high NEES |
| Bias, lever arm or time offset | Wrong bearing model; cross-range error and NEES may rise without much change in missed rate |
| Noise underestimated | Covariances and gates shrink; rejected detections, ghosts and ID switches increase |
| Radar outage | Unaware lifecycle loses identities; aware coasting grows uncertainty and retains ghosts |
| Close targets | Overlapping gates create association ambiguity and withhold camera information |
| Camera clutter during outage | Accepted clutter resets coast_steps and prolongs ghosts |
| Sensor proximity | Bearing Jacobian becomes ill-conditioned; tracker guard and model singularity checks reject geometry |

The truth simulator can schedule turns, sustained acceleration, camera bias, timing offsets,
lever arms and noise mismatch. Its turn-plus-acceleration step is a split integration, not the
exact joint continuous-time solution. CT modes use the same deterministic turn part but fixed
rates. Most robustness rows change one factor at a time; combined disturbances appear in the
"hard" improvement group. These are simulation studies, with no real sensor data or evidence
of robustness to arbitrary combined disturbances, unresolved detections or non-planar motion.
Implementation: [maneuvers.py](../src/fusion/maneuvers.py), `:1-22`, `:193-221`.
Evidence and scope: [RESULTS §5](RESULTS.md#5-silent-errors-and-robustness),
[§8](RESULTS.md#8-general-limitations).

## 10. Implementation and reproducibility

Equations describe the implementation at `a7dc570`. Regression tests pin behavior in `tests/test_ekf.py`,
`test_gating.py`, `test_assignment.py`, `test_track.py`, `test_outage_tracker.py`,
`test_imm.py`, `test_camera_bias.py` and `test_sensitivity.py`.

Published evidence is in RESULTS.md. Additional pair-level bias and per-seed NEES diagnostics
come from git-ignored `results/improvement_bias.txt` and `results/improvement_results.pkl`.
The evaluation header in `results/criteria_8b.txt` records commit `16185d2`, seeds 0–49;
the bias text has no header and is tied to that run only by its file date (2026-10-07), as is
the pickle. Tracker, estimator and score source was unchanged through `a7dc570`; the only
change to `improvement_experiment.py` was a docstring. Evaluation was not rerun for this reference.

Those unpublished diagnostics were previously recomputed from local artifacts, not generated
by this documentation edit. They are not reproducible from a clean checkout without rerunning
`python scripts/run_improvement_experiment.py --stage eval --workers 14` (reported runtime
about 90 minutes with 14 workers). See [RESULTS §9](RESULTS.md#9-reproduction) for reproduction
and [§0](RESULTS.md#0-how-to-read-this) for result provenance and conventions.
