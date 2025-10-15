# Mathematical Objective Function for LowRankTV

The core objective function:

$$
\min_{L,S} \quad \|L\|_* + \lambda \cdot \text{TV}(S) \quad \text{subject to} \quad X = L + S
$$

Let's break down each component of this equation.

### 1. The Low-Rank Artifact, Nuclear norm: $\|L\|_*$
$$
\|L\|_* = \sum_{i} \sigma_i(L)
$$

### 2. LFP: $\lambda \cdot \text{TV}(S)$

*  **Total Variation Norm** of the matrix $S$. 
    $$
    \text{TV}(S) = \sum_{c=1}^{\text{Channels}} \sum_{t=1}^{\text{Timesteps}-1} |S_{c,t+1} - S_{c,t}|
    $$
   measures the "jaggedness" / total amount of change in the signal over time
