# Inner Product Analysis for Simulated Neural Data

### Inner Product

For two signals $x$ and $y$, we define the inner product as:

**For vectors** (1D signals):
$$\langle x, y \rangle = \sum_{i=1}^{n} x_i y_i = x^T y$$

**For matrices** (2D signals with channels and time):
$$\langle X, Y \rangle_F = \sum_{i=1}^{m} \sum_{j=1}^{n} X_{ij} Y_{ij} = \text{tr}(X^T Y)$$

<!-- where $X, Y \in \mathbb{R}^{m \times n}$ and $\text{tr}(\cdot)$ denotes the trace operator. -->

### Orthogonality

Two signals $x$ and $y$ are considered **orthogonal** if their inner product is zero (within numerical tolerance):

$$x \perp y \iff |\langle x, y \rangle| < \epsilon \text{, where $\epsilon = 10^{-10}$ is the numerical tolerance.}$$

## Data Structure

The simulated data consists of:
- **Raw data**: $R \in \mathbb{R}^{T \times C \times N}$ (trials × channels × timesteps)
- **Ground truth**: $G \in \mathbb{R}^{T \times C \times N}$ (clean neural signals)
- **Artifacts**: $A \in \mathbb{R}^{T \times C \times N}$ (stimulation artifacts)
- **LFP**: $L \in \mathbb{R}^{T \times C \times N}$ (local field potential)

where:
- $T$ = number of trials
- $C$ = number of channels  
- $N$ = number of timesteps

## Analysis Methods

### 1. Trial-wise Analysis

For each trial $t \in \{1, 2, \ldots, T\}$, compute the inner product between ground truth and artifacts across all channels:

$$\langle G_t, A_t \rangle_F = \sum_{c=1}^{C} \sum_{n=1}^{N} G_{tcn} A_{tcn}$$

This measures the overall correlation between clean signals and artifacts for each trial.

### 2. Channel-wise Analysis

For each trial $t$ and channel $c$, compute the inner product:

$$\langle G_{tc}, A_{tc} \rangle = \sum_{n=1}^{N} G_{tcn} A_{tcn}$$

This measures the correlation between ground truth and artifacts for each specific channel within each trial.

### 3. Orthogonality Assessment

For each channel pair $(t, c)$, determine orthogonality:

$$G_{tc} \perp A_{tc} \iff |\langle G_{tc}, A_{tc} \rangle| < \epsilon$$

## Statistical Summary

### Trial-wise Statistics

- **Mean inner product**: $\bar{\mu}_T = \frac{1}{T} \sum_{t=1}^{T} \langle G_t, A_t \rangle_F$
- **Standard deviation**: $\sigma_T = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (\langle G_t, A_t \rangle_F - \bar{\mu}_T)^2}$
- **Range**: $[\min_t \langle G_t, A_t \rangle_F, \max_t \langle G_t, A_t \rangle_F]$

### Channel-wise Statistics

- **Mean inner product**: $\bar{\mu}_C = \frac{1}{TC} \sum_{t=1}^{T} \sum_{c=1}^{C} \langle G_{tc}, A_{tc} \rangle$
- **Standard deviation**: $\sigma_C = \sqrt{\frac{1}{TC-1} \sum_{t=1}^{T} \sum_{c=1}^{C} (\langle G_{tc}, A_{tc} \rangle - \bar{\mu}_C)^2}$

### Orthogonality Percentage

$$\text{Orthogonality \%} = \frac{100}{TC} \sum_{t=1}^{T} \sum_{c=1}^{C} \mathbf{1}_{|\langle G_{tc}, A_{tc} \rangle| < \epsilon}$$

where $\mathbf{1}_{\cdot}$ is the indicator function.
