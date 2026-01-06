# Pillars 2.0: A Kalman Filter Approach to Hierarchical Talent Estimation

## Executive Summary

This document proposes a new implementation of the Pillars talent estimation system using a hierarchical Kalman filter. The new approach addresses three fundamental limitations of the current Pillars implementation:

1. **Temporal granularity**: The current system operates only at the season level, preventing intra-season tracking of player development
2. **Anchoring at observed nodes**: Nodes with significant observed data can become "stuck" to their historical values, slow to update even when lower-level evidence shifts substantially
3. **Underweighting of uncertain features**: The linear models that compute priors systematically underweight skills that are often unobserved, even when those skills are highly informative when available

The Kalman filter approach solves all three issues through a unified probabilistic framework that:
- Tracks skill evolution at configurable time intervals (e.g., weekly)
- Propagates information bidirectionally through the skill hierarchy
- Learns observation uncertainty per skill, automatically calibrating the weight given to each data source

This document explains the methodology, demonstrates the approach on synthetic data, and presents experiments that specifically validate improvements over the current system's limitations.

---

## Table of Contents

1. [Background: The Pillars Framework](#1-background-the-pillars-framework)
2. [Limitations of the Current Implementation](#2-limitations-of-the-current-implementation)
3. [The Kalman Filter Approach](#3-the-kalman-filter-approach)
4. [How This Solves Each Limitation](#4-how-this-solves-each-limitation)
5. [Experiments and Validation](#5-experiments-and-validation)
6. [Implementation Details](#6-implementation-details)
7. [Next Steps](#7-next-steps)

---

## 1. Background: The Pillars Framework

The Pillars framework models player talent as a directed acyclic graph (DAG) where higher-level skills are influenced by lower-level component skills. For example:

```
[Skill A] ──┐
            ├──▶ [Skill C] ──┐
[Skill B] ──┤                ├──▶ [Skill E]
            └──▶ [Skill D] ──┘
```

In this structure:
- **Root nodes** (A, B) represent fundamental skills with no parent dependencies
- **Intermediate nodes** (C, D) combine information from their parents
- **Leaf nodes** (E) represent high-level outcomes that aggregate multiple skill components

The goal is to estimate each player's true underlying talent at each node, accounting for:
- Measurement noise in observations
- The hierarchical relationships between skills
- Changes in talent over time

---

## 2. Limitations of the Current Implementation

### 2.1 Season-Level Only

The current Pillars implementation operates exclusively at the season level. This creates several problems:

- **No intra-season tracking**: We cannot see how a player's skills evolve during a season
- **Delayed signal detection**: A player who improves mid-season won't show that improvement until the following year's estimate
- **Inference discontinuities**: Running inference mid-season is "out of distribution" relative to how the model was trained, causing unpredictable jumps in how prior vs. observed data are weighted

### 2.2 Anchoring at Observed Nodes

When a node has substantial observed data, the current system can become overly anchored to that historical evidence. This is particularly problematic for high-level nodes like overall production metrics:

**Example scenario**: A player's component skills (measured via tracking data) show substantial improvement early in a season. However, their top-level production metric, which has many observations from prior seasons, updates slowly because:
- The point estimate from observed data carries significant weight
- The prior from component skills, while shifted, competes with this historical anchor
- It takes accumulating contradictory observations to overcome the anchor

This creates a lag between when underlying skills change and when top-level estimates reflect that change.

### 2.3 Underweighting Uncertain Features

The linear regression models that compute priors have a subtle but significant flaw: they systematically underweight skills that are frequently unobserved.

**Why this happens**: When training the linear models that predict parent skills from children, most players contribute only their population-average point estimate for rarely-observed skills. This suppresses the learned coefficient, even if the relationship is very strong when the skill *is* actually observed.

**Consequence**: Skills that are informative but intermittently available (e.g., [PLACEHOLDER: example of a skill that's only observed in certain situations]) contribute less to the prior than they should.

---

## 3. The Kalman Filter Approach

### 3.1 Core Concept

The Kalman filter is a recursive algorithm for estimating the state of a dynamic system from noisy observations. It maintains:

- A **state estimate**: Our best guess of the true underlying values
- An **uncertainty estimate**: How confident we are in that guess

At each time step, the filter:
1. **Predicts** how the state might have evolved (with growing uncertainty)
2. **Updates** the prediction using any new observations (reducing uncertainty)

The key insight is that the Kalman filter optimally balances prior knowledge against new evidence, weighted by their respective uncertainties.

### 3.2 Adapting to Hierarchical Skills

We extend the standard Kalman filter to handle our hierarchical skill structure:

**State representation**: We track *intrinsic* skills (the root-level talents) as our hidden state. All other skills are deterministic functions of these intrinsic skills through the DAG structure.

**Observation model**: When we observe any skill in the hierarchy, we're getting noisy information about the underlying intrinsic skills. The hierarchy defines how intrinsic skills combine to produce observable outcomes.

**Key equations**:

Let $z_t$ be the vector of intrinsic skills at time $t$, and $s_t$ be the vector of all skills (both intrinsic and derived):

$$s_t = B \cdot z_t$$

where $B$ is derived from the DAG edge weights and captures both direct and indirect effects.

When we observe skill $i$ with value $y$:
$$y = s_i + \epsilon, \quad \epsilon \sim N(0, R_i)$$

The filter computes the optimal update to our belief about $z_t$ given this observation.

### 3.3 Learning the Parameters

The model learns from data using the Expectation-Maximization (EM) algorithm:

1. **E-step**: Given current parameters, estimate the most likely skill trajectories for all players
2. **M-step**: Given these trajectories, update the parameters (edge weights, noise levels)

Parameters learned:
- **Edge weights** ($W$): How strongly each parent skill influences each child
- **Process noise** ($Q$): How much each intrinsic skill drifts per time period
- **Observation noise** ($R$): How noisy measurements are for each skill
- **Population prior**: The distribution of skills across the player population

---

## 4. How This Solves Each Limitation

### 4.1 Solving the Temporal Granularity Problem

**Current approach**: One estimate per season, with opaque weighting between prior and observed data.

**Kalman approach**:
- Observations are binned into configurable time windows (e.g., weekly)
- The filter updates beliefs after each bin, producing a continuous trajectory
- Uncertainty grows during gaps (off-season, injured list) and shrinks when data arrives
- No discontinuities or special handling needed for partial seasons

**Benefit**: We can now answer questions like "How did this player's skills evolve from April to September?" and get properly calibrated uncertainty throughout.

### 4.2 Solving the Anchoring Problem

**Current approach**: Observed data at a node competes with the prior in a way that can over-anchor to historical observations.

**Kalman approach**:
- All updates flow through the covariance matrix, which tracks uncertainty across the entire skill hierarchy
- When component skills shift, the uncertainty in derived skills *automatically increases* because the prediction and observation now disagree
- The Kalman gain formula naturally handles this: as the prior becomes less certain, observations get more weight

**Mathematically**: The Kalman gain is:
$$K = P_{pred} C^T (C P_{pred} C^T + R)^{-1}$$

When the prior uncertainty $P_{pred}$ is large (because component skills shifted), $K$ increases, giving more weight to new observations and less to the prior.

**Benefit**: Top-level skills respond appropriately to shifts in component skills, without artificial anchoring.

### 4.3 Solving the Underweighting Problem

**Current approach**: Linear regression with point estimates underweights rarely-observed skills.

**Kalman approach**:
- The model learns observation noise $R_i$ separately for each skill
- Rarely-observed skills naturally get higher $R$ (more noise per observation)
- But this doesn't suppress their *relationship* to other skills—the edge weights $W$ are estimated using the full posterior, not just point estimates
- Bidirectional inference means that even unobserved skills get updated when their children (or parents) are observed

**Key insight**: The edge weight estimation uses *smoothed* state estimates, which incorporate uncertainty. A skill that's rarely observed will have high uncertainty in its smoothed estimate, and this uncertainty propagates correctly into the weight estimation via weighted least squares.

**Benefit**: Skills are weighted by their *informativeness when observed*, not by their observation frequency.

---

## 5. Experiments and Validation

### 5.1 Experiment Setup

We validate the approach using synthetic data where ground truth is known. The synthetic data generator:

- Creates a known DAG structure with true edge weights
- Simulates player skill trajectories as random walks
- Generates observations with realistic patterns (varying frequency by skill, seasonal gaps)

This allows us to directly compare estimated skills against true values.

### 5.2 Experiment 1: Intra-Season Skill Tracking

**Goal**: Demonstrate that the Kalman filter accurately tracks skill changes within a season at weekly resolution.

**Method**: We generated synthetic data for 100 players over a 26-week season, with skills evolving as random walks. The model was fit using weekly time bins, and we compared estimated trajectories against the known true values.

**Results**:

| Skill | RMSE | Bias | 95% Coverage |
|-------|------|------|--------------|
| A | 0.303 | 0.007 | 93.1% |
| B | 0.302 | -0.027 | 94.4% |
| C | 0.218 | -0.015 | 93.8% |
| D | 0.202 | 0.011 | 93.0% |
| E | 0.173 | 0.005 | 91.4% |

**Interpretation**: The Kalman filter accurately tracks weekly skill evolution with near-zero bias. The 95% confidence intervals achieve close to nominal coverage (91-94%), indicating well-calibrated uncertainty estimates. Skills with more observations (C, D, E) have lower RMSE, as expected.

![Intra-Season Tracking](experiment1_intraseason_tracking.png)

### 5.3 Experiment 2: Responsiveness to Talent Changes

**Goal**: Demonstrate that when component skills change substantially, the model responds quickly rather than anchoring to historical values.

**Method**: We simulated 50 players over 52 weeks where every player experienced a sudden +1.5 standard deviation improvement in Skill A at week 26. This tests whether the model can detect and respond to mid-season talent changes, and whether the change propagates through the hierarchy (A → C → E).

**Results**:

| Skill | RMSE (Before Change) | RMSE (Weeks 27-31) | RMSE (Settled, 35+) |
|-------|---------------------|-------------------|---------------------|
| A | 0.265 | 0.628 | 0.215 |
| C | 0.168 | 0.316 | 0.135 |
| E | 0.147 | 0.168 | 0.131 |

**Interpretation**:
- The elevated RMSE immediately after the change (weeks 27-31) shows the expected lag as the model detects the shift
- By week 35, RMSE returns to baseline levels—actually *better* than before, because the model has more data
- The change in A propagates appropriately to downstream skills C and E
- The model does **not** anchor to pre-change values; it adapts within approximately 5-8 weeks

![Responsiveness to Talent Changes](experiment2_responsiveness.png)

### 5.4 Experiment 3: Handling Rarely-Observed Skills

**Goal**: Demonstrate that rarely-observed skills maintain appropriate edge weights and contribute meaningfully when they are observed for a specific player.

**Method**: We generated data for 200 players where Skill A was observed only 5% of weeks (compared to 40-80% for other skills). This simulates skills that are only measurable in specific game situations. We then tested whether: (1) the edge weight A→C is still recovered correctly, and (2) when A *is* observed for a specific player, it properly informs their estimate.

**Results - Edge Weight Recovery**:

| Edge | True Weight | Learned Weight | Error |
|------|-------------|----------------|-------|
| A → C | 0.600 | 0.652 | 0.052 |
| B → C | 0.300 | 0.250 | 0.050 |
| B → D | 0.500 | 0.460 | 0.040 |
| C → E | 0.400 | 0.440 | 0.040 |
| D → E | 0.350 | 0.401 | 0.051 |

Despite A being observed only ~5% of the time, its edge weight is recovered with similar accuracy to densely-observed skills.

**Results - Information Flow Test**:

We created two test players with identical Skill E observations, but one also had Skill A observed. Both players had true Skill A = 1.5 (well above average).

| Metric | E-only Player | E+A Player | Change |
|--------|---------------|------------|--------|
| A estimate | -0.045 | 1.683 | +1.73 |
| A uncertainty (std) | 0.959 | 0.113 | **-88%** |

**Interpretation**:
- The A→C edge weight is learned correctly despite sparse observations
- When A *is* observed for a player, their A estimate improves dramatically (error drops from 1.5 to 0.18)
- Uncertainty on A drops by 88%, showing the model properly weights the sparse observations when available
- This contrasts with linear regression approaches where rarely-observed features would have suppressed coefficients regardless of their informativeness

### 5.5 Parameter Recovery

We validate that the EM algorithm correctly recovers the true data-generating parameters across multiple experiments:

**Edge Weights** (from Experiment 2, 50 players, 52 weeks):

| Edge | True | Learned | Error |
|------|------|---------|-------|
| A → C | 0.60 | 0.66 | 0.06 |
| B → C | 0.30 | 0.50 | 0.20 |
| B → D | 0.50 | 0.57 | 0.07 |
| C → E | 0.40 | 0.42 | 0.02 |
| D → E | 0.35 | 0.47 | 0.12 |

**Observation Noise** (sqrt R):

| Skill | True | Learned |
|-------|------|---------|
| A | 0.50 | 0.49 |
| B | 0.60 | 0.59 |
| C | 0.70 | 0.69 |
| D | 0.60 | 0.60 |
| E | 0.80 | 0.80 |

Observation noise is recovered with high accuracy across all skills. Edge weights show more variation, which is expected given the complexity of disentangling hierarchical relationships from noisy data

---

## 6. Implementation Details

### 6.1 Architecture

The implementation consists of several modules:

- **dag.py**: DAG structure, topological sort, B-matrix computation
- **kalman.py**: Core Kalman filter and RTS smoother
- **parameters.py**: EM parameter estimation
- **model.py**: Main `HierarchicalKalmanFilter` class
- **preprocessing.py**: Time binning and data preparation

### 6.2 Computational Considerations

- **Parallelization**: Player-level filtering runs in parallel using joblib
- **Numerical stability**: Uses Joseph form for covariance updates, matrix solves instead of inversions
- **Scalability**: Tested on 200 players over 52 time bins (~10,000 player-weeks) with 25 EM iterations completing in reasonable time

### 6.3 Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_size_days` | 7 | Width of time bins for aggregating observations |
| `offseason_dates` | (configurable) | Date range for off-season (increased process noise) |
| `max_iter` | 30 | Maximum EM iterations |

---

## 7. Next Steps

### 7.1 Validation on Real Data

- Apply to historical data and compare against current Pillars estimates
- Validate calibration: Do 95% intervals contain the true value 95% of the time?
- Compare predictive accuracy on held-out future data

### 7.2 Integration

- Define the production DAG structure for real skill hierarchies
- Establish data pipelines for observation ingestion
- Create reporting and visualization tools

### 7.3 Extensions

- **Off-season modeling**: Currently uses a multiplicative factor for off-season drift; could model more sophisticated dynamics
- **Non-stationarity**: Allow edge weights to vary over time (e.g., aging curves)
- **Heterogeneous players**: Allow process noise to vary by player (some players more volatile than others)

---

## Appendix A: Mathematical Details

### A.1 State Space Model

**State transition** (random walk):
$$z_t = z_{t-1} + w_t, \quad w_t \sim N(0, Q)$$

**Observation model**:
$$y_t = C_t B z_t + v_t, \quad v_t \sim N(0, R_t)$$

where:
- $z_t \in \mathbb{R}^k$ is the vector of intrinsic skills
- $B \in \mathbb{R}^{n \times k}$ maps intrinsic skills to all skills via $(I - W)^{-1}$
- $C_t$ is a selection matrix for which skills are observed at time $t$
- $Q = \text{diag}(q_1, \ldots, q_k)$ is process noise (skill drift)
- $R_t$ is observation noise for the observed skills

### A.2 Kalman Filter Equations

**Predict**:
$$\hat{z}_{t|t-1} = \hat{z}_{t-1|t-1}$$
$$P_{t|t-1} = P_{t-1|t-1} + Q$$

**Update**:
$$\tilde{y}_t = y_t - C_t B \hat{z}_{t|t-1}$$
$$S_t = C_t B P_{t|t-1} B^T C_t^T + R_t$$
$$K_t = P_{t|t-1} B^T C_t^T S_t^{-1}$$
$$\hat{z}_{t|t} = \hat{z}_{t|t-1} + K_t \tilde{y}_t$$
$$P_{t|t} = (I - K_t C_t B) P_{t|t-1}$$

### A.3 Rauch-Tung-Striebel Smoother

For parameter estimation, we use backward smoothing to incorporate future observations:

$$J_t = P_{t|t} P_{t+1|t}^{-1}$$
$$\hat{z}_{t|T} = \hat{z}_{t|t} + J_t (\hat{z}_{t+1|T} - \hat{z}_{t+1|t})$$
$$P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t^T$$

---

## Appendix B: Glossary

- **DAG**: Directed Acyclic Graph—the structure defining skill relationships
- **Intrinsic skill**: A root-level skill with no parents in the DAG
- **Derived skill**: A skill computed as a function of its parent skills
- **Process noise**: How much a skill is expected to drift per time period
- **Observation noise**: How noisy a single measurement is
- **Kalman gain**: The weight given to new observations vs. the prior prediction
- **Smoothing**: Using future observations to refine past estimates (for parameter learning)

