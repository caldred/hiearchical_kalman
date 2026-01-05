# Understanding Kalman Filters: A Data Scientist's Guide

*A comprehensive introduction to Kalman Filters, using this baseball skill estimation system as a practical example.*

---

## Table of Contents

1. [What Problem Are We Solving?](#what-problem-are-we-solving)
2. [The Intuition Behind Kalman Filters](#the-intuition-behind-kalman-filters)
3. [The Core Kalman Filter Algorithm](#the-core-kalman-filter-algorithm)
4. [From Basic to Hierarchical: This Repository](#from-basic-to-hierarchical-this-repository)
5. [Learning Parameters with EM](#learning-parameters-with-em)
6. [Code Walkthrough](#code-walkthrough)
7. [Practical Tips and Gotchas](#practical-tips-and-gotchas)

---

## What Problem Are We Solving?

### The Noisy Measurement Problem

Imagine you're trying to estimate a baseball player's true batting skill. Every time they step up to the plate, you observe their performance—bat speed, launch angle, hit velocity—but these observations are **noisy**:

- A player might have a bad day due to fatigue
- Weather conditions affect performance
- The quality of pitches varies
- Random chance plays a role

**The question**: Given a sequence of noisy observations over time, how do we estimate the player's *true underlying skill*?

### Why Not Just Average?

You might think: "Just average all the observations!" But this misses something crucial:

1. **Skills change over time** — A player's skill today is more relevant than their skill 6 months ago
2. **Recent data matters more** — But we shouldn't throw away historical data entirely
3. **Some observations are noisier than others** — A single at-bat tells us less than a 50-game average

**The Kalman Filter solves exactly this problem**: It optimally combines prior beliefs about a hidden state with new noisy observations, accounting for how both the state and observations evolve over time.

---

## The Intuition Behind Kalman Filters

### A Thought Experiment: The GPS Analogy

Imagine you're navigating with two information sources:

1. **Your car's speedometer + steering wheel** (prediction): You know roughly where you should be based on how fast you were going and which direction you turned.

2. **Your GPS reading** (observation): It tells you where you actually are, but it's a bit noisy—sometimes off by 10 meters.

How do you combine these?

- If your GPS is very accurate (low noise), trust it more
- If your prediction model is very accurate, trust it more
- The optimal combination depends on the *relative reliability* of each source

**This is exactly what a Kalman Filter does!**

### The Two-Step Dance

Every Kalman Filter iteration has two steps:

```
┌─────────────────────────────────────────────────────────────┐
│                      PREDICT STEP                           │
│  "Where do I think the state is, based on how it evolves?"  │
│                                                              │
│  • Use previous estimate + knowledge of state dynamics       │
│  • Uncertainty GROWS (we're less sure as time passes)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      UPDATE STEP                             │
│  "Now that I see the measurement, let me correct my belief" │
│                                                              │
│  • Compare prediction to actual observation                  │
│  • Blend them based on their relative uncertainties          │
│  • Uncertainty SHRINKS (we learned something!)               │
└─────────────────────────────────────────────────────────────┘
```

### Visualizing the Process

```
True Skill:     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    (hidden - we never see this directly)

Observations:   *   *  *    * *   *  *   *    *  *   *  *
                    (scattered around true value - noisy!)

Kalman Est:     ──────────────────────────────────────────
                    (smoothly tracks the true value)

Uncertainty:    ▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                (starts high, shrinks as we see more data)
```

---

## The Core Kalman Filter Algorithm

Now let's get into the math. Don't worry—we'll build it up piece by piece.

### The System Model

We model our system with two equations:

#### 1. State Evolution (How skills change over time)

```
z_t = z_{t-1} + w_t     where w_t ~ N(0, Q)
```

In plain English:
- `z_t` = the true skill at time t (what we want to estimate)
- `z_{t-1}` = the skill at the previous time step
- `w_t` = random "process noise" — skills drift a little each period
- `Q` = how much we expect skills to drift (process noise variance)

**This is a "random walk" model**: The skill tomorrow is today's skill plus some random change.

> **In this codebase**: Each player's skill evolves as a random walk. The process noise `Q` controls how quickly we expect skills to change. Higher Q means skills are more volatile.

#### 2. Observation Model (How we measure skills)

```
y_t = C @ z_t + n_t     where n_t ~ N(0, R)
```

In plain English:
- `y_t` = what we actually observe (noisy measurement)
- `C` = observation matrix (which skills we can see)
- `z_t` = the true underlying skill
- `n_t` = measurement noise
- `R` = how noisy our measurements are

**Key insight**: We never see `z_t` directly—only `y_t`, which is `z_t` plus noise.

> **In this codebase**: Some skills like `bat_speed` are directly observed. Others like `damage_value` depend on multiple skills. The `C` matrix selects which skills we observe at each time.

### The Predict Step

When we move to a new time period, we first predict where we think the state is:

```python
# From kalman.py - the predict step
z_pred = z_prev           # Predicted state (random walk: stays the same)
P_pred = P_prev + Q       # Predicted covariance (uncertainty grows!)
```

**Why does uncertainty grow?** Because time passing means more random drift could have occurred. It's like predicting where a wandering person will be—the further into the future, the less certain we are.

### The Update Step

When we get a new observation, we update our belief:

```python
# From kalman.py - the update step (simplified)
# Innovation: how much did observation differ from prediction?
v = y - C @ z_pred

# Innovation covariance: total uncertainty in the innovation
S = C @ P_pred @ C.T + R

# Kalman Gain: how much to trust the observation vs prediction
K = P_pred @ C.T @ inv(S)

# Updated state: blend prediction with observation
z_upd = z_pred + K @ v

# Updated covariance: uncertainty shrinks after seeing data
P_upd = (I - K @ C) @ P_pred
```

### Understanding the Kalman Gain

The **Kalman Gain** `K` is the secret sauce. It tells us how much to adjust our estimate when we see new data.

```
K = P_pred @ C.T @ inv(S)
```

**Intuition**:
- If observation noise `R` is small (accurate measurements) → `K` is large → trust observations more
- If prediction uncertainty `P_pred` is small (confident prediction) → `K` is small → trust prediction more
- If `P_pred` is large and `R` is small → `K` approaches 1 → almost fully trust observation

```
          Prediction Uncertainty
K ∝  ─────────────────────────────────────
     Prediction Uncertainty + Observation Noise
```

### A Concrete Example

Let's walk through a simple example with made-up numbers:

```
Time 0: Initial belief
  - Estimated skill: z = 100 (e.g., bat speed in mph)
  - Uncertainty: P = 25 (variance)

Time 1: Predict step
  - Process noise Q = 4
  - z_pred = 100 (skill stays same under random walk)
  - P_pred = 25 + 4 = 29 (uncertainty grew)

Time 1: Observe y = 105 (noisy measurement)
  - Observation noise R = 9

Time 1: Update step
  - Innovation: v = 105 - 100 = 5
  - Innovation covariance: S = 29 + 9 = 38
  - Kalman gain: K = 29/38 = 0.76
  - Updated state: z_upd = 100 + 0.76 × 5 = 103.8
  - Updated covariance: P_upd = (1 - 0.76) × 29 = 6.96

Result:
  - New estimate: 103.8 (moved toward observation)
  - New uncertainty: 6.96 (much lower than before!)
```

The filter "split the difference" between prediction (100) and observation (105), weighted by their relative uncertainties. And our confidence improved dramatically!

---

## From Basic to Hierarchical: This Repository

This repository extends the basic Kalman Filter in two important ways:

### 1. Hierarchical Skills via a DAG

Some skills **depend on other skills**. In baseball:

```
                    bat_speed ──────┐
                                    ├──► hard_hit_velocity ──┐
                    bat_length ─────┘                        │
                                                             ├──► damage_value
                    attack_angle ───┬──► launch_angle_value ─┘
                                    │
     swing_vertical_bat_angle ──────┘
```

**Key insight**: A player's `hard_hit_velocity` isn't a completely independent skill—it's determined by their `bat_speed` and `bat_length`.

This is modeled using a **Directed Acyclic Graph (DAG)** and a weight matrix `W`:

```python
# From dag.py - computing total effects
# W[i,j] = direct effect of skill j on skill i
# B = (I - W)^(-1) = total effects (including indirect paths)

def compute_B_matrix(self, W: np.ndarray) -> np.ndarray:
    """Compute the total effects matrix B = (I - W)^(-1)."""
    I = np.eye(W.shape[0])
    B = np.linalg.inv(I - W)
    return B
```

**The B matrix**: If skill j affects skill k, and skill k affects skill i, then B captures the total (direct + indirect) effect of j on i.

### 2. Intrinsic vs. Observed Skills

The filter distinguishes between:

- **Intrinsic state `z`**: The underlying latent skill factors
- **Observed skill `s`**: What we actually measure, which is `s = B @ z`

```python
# From kalman.py - converting internal state to skills
def get_skill_estimates(self):
    """Convert intrinsic state z to skill estimates s = B @ z."""
    skill_mean = self.B @ self.z
    skill_cov = self.B @ self.P @ self.B.T
    return skill_mean, np.diag(skill_cov)
```

### 3. Handling Missing Observations

Not every skill is observed at every time step. The system handles this elegantly:

```python
# From kalman.py - partial observation update
def update(self, y_partial, obs_mask, obs_vars, obs_counts):
    """
    Update with partial observations.

    obs_mask: which skills are observed (boolean array)
    """
    if not obs_mask.any():
        return  # No observations, just keep prediction

    # Build observation matrix C that selects observed skills
    C = self.B[obs_mask, :]

    # Proceed with standard update using only observed skills
    ...
```

---

## Learning Parameters with EM

So far, we've assumed we know the noise parameters Q and R. But how do we learn them from data?

### The Expectation-Maximization (EM) Algorithm

EM alternates between two steps:

```
┌────────────────────────────────────────────────────────────┐
│  E-STEP: Given current parameters, estimate hidden states  │
│                                                             │
│  • Run Kalman filter forward (filtering)                    │
│  • Run smoother backward (get better estimates)             │
│  • Compute expected sufficient statistics                   │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  M-STEP: Given hidden state estimates, update parameters   │
│                                                             │
│  • Estimate Q from how much states changed between times    │
│  • Estimate R from residuals (observed - predicted)         │
│  • Estimate W from skill relationships                      │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     Repeat until convergence
```

### The Rauch-Tung-Striebel Smoother

The **smoother** is crucial for EM. While filtering only uses past data, smoothing uses the **entire sequence** to get better estimates:

```
Filtering: Uses data up to time t
           z_filt[t] = E[z_t | y_1, y_2, ..., y_t]

Smoothing: Uses ALL data
           z_smooth[t] = E[z_t | y_1, y_2, ..., y_T]
```

**Why does this help?** Consider estimating a player's skill in week 3. Filtering only sees weeks 1-3. But smoothing also sees weeks 4-10, which provides additional information about week 3!

```python
# From kalman.py - RTS smoother (simplified)
def smooth_sequence(self, T, z_filt, P_filt, z_pred_seq, P_pred_seq):
    """Rauch-Tung-Striebel backward smoothing."""
    z_smooth = [None] * T
    P_smooth = [None] * T

    # Initialize: last smoothed = last filtered
    z_smooth[T-1] = z_filt[T-1]
    P_smooth[T-1] = P_filt[T-1]

    # Backward pass
    for t in range(T-2, -1, -1):
        # Smoother gain
        J = P_filt[t] @ np.linalg.solve(P_pred_seq[t+1], np.eye(n)).T

        # Smoothed estimates
        z_smooth[t] = z_filt[t] + J @ (z_smooth[t+1] - z_pred_seq[t+1])
        P_smooth[t] = P_filt[t] + J @ (P_smooth[t+1] - P_pred_seq[t+1]) @ J.T

    return z_smooth, P_smooth
```

### M-Step: Estimating Parameters

#### Process Noise Q

Q represents how much skills drift over time. We estimate it from the expected squared changes:

```python
# From parameters.py - Q estimation
def estimate_process_noise(self, smooth_results):
    """
    Q_i = E[(z_t - z_{t-1})^2]
        = E[z_t^2] - 2*E[z_t*z_{t-1}] + E[z_{t-1}^2]
    """
    # Collect smoothed states and covariances
    # Compute expected squared differences
    # Average across all time steps and players
```

**Intuition**: If the smoothed states change a lot between time steps, Q should be large.

#### Observation Noise R

R represents measurement noise. We estimate it from the prediction residuals:

```python
# From parameters.py - R estimation
def estimate_observation_noise(self, observations, predictions):
    """
    R_i = E[(y_i - predicted_i)^2]
    """
    residuals = observations - predictions
    R = np.mean(residuals ** 2, axis=0)
    return R
```

**Intuition**: If our predictions are far from observations, either R is large or our model is wrong.

#### Edge Weights W

For hierarchical skills, we learn how parent skills affect child skills:

```python
# From parameters.py - W estimation (simplified)
def estimate_edge_weights(self, smoothed_skills, dag):
    """
    For each child skill:
      child = intercept + sum(weight_i * parent_i)

    Solved via weighted least squares.
    """
    for child_idx in range(n_skills):
        parent_indices = dag.get_parents(child_idx)
        if not parent_indices:
            continue  # No parents, no weights to learn

        # Gather parent skill values as features
        X = smoothed_skills[:, parent_indices]
        y = smoothed_skills[:, child_idx]

        # Weighted least squares
        weights = observation_counts
        W[child_idx, parent_indices] = weighted_lstsq(X, y, weights)
```

---

## Code Walkthrough

Let's trace through how the main components work together.

### File Structure

```
hiearchical_kalman/
├── model.py          # Main HierarchicalKalmanFilter class
├── kalman.py         # Core Kalman filter + smoother
├── parameters.py     # EM parameter estimation
├── dag.py            # Directed Acyclic Graph for skill hierarchy
├── preprocessing.py  # Data binning and normalization
└── run_model_on_parquet.py  # Example usage script
```

### The Main Fitting Loop

```python
# From model.py - simplified fit() method
class HierarchicalKalmanFilter:

    def fit(self, df, max_iter=50, tol=1e-4):
        """Train the model using EM algorithm."""

        # 1. Preprocess: bin observations by time
        binned_data = preprocess(df, self.bin_size)

        # 2. Initialize parameters
        self._initialize_parameters(binned_data)

        # 3. EM iterations
        for iteration in range(max_iter):

            # E-Step: Filter and smooth all players
            smooth_results = {}
            for player_id in players:
                # Forward filter
                z_filt, P_filt = self.kalman.filter_sequence(
                    observations[player_id]
                )
                # Backward smooth
                z_smooth, P_smooth = self.kalman.smooth_sequence(
                    z_filt, P_filt
                )
                smooth_results[player_id] = (z_smooth, P_smooth)

            # M-Step: Update parameters
            self.Q = estimate_process_noise(smooth_results)
            self.R = estimate_observation_noise(smooth_results, observations)
            self.W = estimate_edge_weights(smooth_results, self.dag)
            self.B = self.dag.compute_B_matrix(self.W)

            # Check convergence
            log_likelihood = compute_log_likelihood()
            if converged(log_likelihood, prev_ll, tol):
                break
            prev_ll = log_likelihood

        # 4. Store final filtered states for each player
        self._update_player_states(binned_data)
```

### Making Predictions

```python
# From model.py - getting skill estimates
def get_estimates(self, player_ids=None):
    """Get current skill estimates for players."""
    results = {}

    for player_id in player_ids:
        state = self.player_states[player_id]

        # Convert intrinsic state to skills: s = B @ z
        skill_means = self.B @ state.z
        skill_vars = np.diag(self.B @ state.P @ self.B.T)

        results[player_id] = {
            'skills': dict(zip(self.skill_names, skill_means)),
            'uncertainties': dict(zip(self.skill_names, np.sqrt(skill_vars)))
        }

    return results

# Forward prediction (what will the skill be in N time bins?)
def predict_forward(self, player_id, n_bins):
    """Predict skill evolution forward in time."""
    state = self.player_states[player_id]
    z, P = state.z.copy(), state.P.copy()

    predictions = []
    for _ in range(n_bins):
        # Random walk: state stays same, uncertainty grows
        P = P + np.diag(self.Q)  # Add process noise

        skill_means = self.B @ z
        skill_vars = np.diag(self.B @ P @ self.B.T)
        predictions.append((skill_means, skill_vars))

    return predictions
```

### Numerical Stability Tricks

The code includes several important stability measures:

```python
# From kalman.py - Joseph form for numerical stability
def _update_core(self, y, C, R_diag):
    """Numerically stable covariance update."""

    # Instead of: P = (I - KC) @ P
    # Use Joseph form: P = (I-KC) @ P @ (I-KC)' + K @ R @ K'
    # This guarantees P stays positive semi-definite

    I_KC = np.eye(self.n) - K @ C
    P_upd = I_KC @ P_pred @ I_KC.T + K @ np.diag(R_diag) @ K.T

    # Force symmetry (floating point can break it)
    P_upd = 0.5 * (P_upd + P_upd.T)

    return z_upd, P_upd
```

---

## Practical Tips and Gotchas

### 1. Initializing Covariance

Start with reasonable uncertainty. Too small = filter ignores data. Too large = filter overreacts.

```python
# Good: Start uncertain, let data inform
P_init = np.eye(n_skills) * initial_variance  # e.g., 1.0

# Bad: Start overconfident
P_init = np.eye(n_skills) * 0.001  # Will ignore early observations!
```

### 2. Choosing Q and R

- **Q too small**: Filter is overconfident, slow to adapt to true changes
- **Q too large**: Filter is jittery, overreacts to noise
- **R too small**: Filter trusts noisy observations too much
- **R too large**: Filter ignores useful information

**Rule of thumb**: Start with both around 1.0 and let EM learn them, or use domain knowledge.

### 3. Observation Scaling

This codebase scales observation noise by sample size:

```python
# More observations = more precise measurement
effective_R = R / n_observations
```

**Why?** If you observe 100 at-bats vs. 1 at-bat, the average of 100 is much more reliable.

### 4. Damping in EM

Parameter updates are damped to prevent oscillation:

```python
# From model.py - damped parameter update
theta_new = (1 - alpha) * theta_old + alpha * theta_estimated
```

Without damping, EM can oscillate or diverge. The code uses `alpha ≈ 0.98-1.0`.

### 5. Convergence Criteria

Multiple criteria are checked:

```python
# From model.py - convergence check
converged = (
    abs(ll_new - ll_old) / abs(ll_old) < rel_tol  # Relative LL change
    and abs(ll_new - ll_old) < abs_tol             # Absolute LL change
    and parameter_change < param_tol               # Parameters stabilized
)
```

### 6. Handling the DAG

The DAG must be acyclic (no circular dependencies):

```python
# From dag.py - cycle detection
def _validate_dag(self):
    """Ensure graph has no cycles using DFS."""
    # Uses three-color DFS: WHITE (unvisited), GRAY (in progress), BLACK (done)
    # If we revisit a GRAY node, there's a cycle!
```

---

## Summary: The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HIERARCHICAL KALMAN FILTER                       │
│                                                                       │
│  INPUT: Noisy observations of player skills over time                │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. PREPROCESSING                                            │    │
│  │     • Bin observations by time period                        │    │
│  │     • Normalize skills (z-score)                             │    │
│  │     • Handle missing data                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  2. EM ALGORITHM (iterative)                                 │    │
│  │                                                               │    │
│  │     E-Step:                                                   │    │
│  │     • Forward filter (estimate states given past data)        │    │
│  │     • Backward smooth (refine using all data)                 │    │
│  │                                                               │    │
│  │     M-Step:                                                   │    │
│  │     • Update Q (process noise)                                │    │
│  │     • Update R (observation noise)                            │    │
│  │     • Update W (skill dependencies)                           │    │
│  │                                                               │    │
│  │     Repeat until likelihood converges                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  3. OUTPUT                                                   │    │
│  │     • Skill estimates (s = B @ z)                             │    │
│  │     • Uncertainty bounds (from P covariance)                  │    │
│  │     • Can predict forward in time                             │    │
│  │     • Can update incrementally with new data                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  OUTPUT: Denoised skill estimates with uncertainty quantification    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Takeaways

1. **Kalman Filters optimally combine prediction and observation** based on their relative uncertainties

2. **The two-step dance**: Predict (uncertainty grows) → Update (uncertainty shrinks)

3. **The Kalman Gain** determines how much to trust new observations

4. **Smoothing** gives better estimates than filtering by using future data

5. **EM learns the noise parameters** from data automatically

6. **Hierarchical structure** captures that some skills depend on others

7. **Numerical stability matters** — use Joseph form, symmetrize matrices, damp updates

---

## Further Reading

- **Kalman's Original Paper**: "A New Approach to Linear Filtering and Prediction Problems" (1960)
- **Bishop's Pattern Recognition and Machine Learning**: Chapter 13 on Sequential Data
- **Shumway & Stoffer's Time Series Analysis**: Comprehensive treatment of state space models
- **Murphy's Machine Learning: A Probabilistic Perspective**: Chapters on Linear Gaussian Models

---

*This guide was written to accompany the `hiearchical_kalman` repository. For implementation details, refer to the source files, particularly `kalman.py` for the core algorithm and `model.py` for the full system.*
