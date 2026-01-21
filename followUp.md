You are modifying an EXISTING Monte Carlo weather simulation implementation
that currently:

- Initializes a 2D grid (Nx, Ny)
- Updates the grid in place at each timestep using stochastic forcing
- Stores ONLY the final timestep state per grid cell

This refactor MUST PRESERVE the current physics update logic and in-place
time stepping.

Your goal is to EXTEND the implementation so that RANDOM NUMBER GENERATOR
QUALITY is validated NATURALLY during the simulation, without storing
full grid history.

====================================================
1. CORE CHANGE (MANDATORY)
====================================================

Do NOT change the time-marching model.

INSTEAD:
- Capture and store PER-TIMESTEP STATISTICAL DATA
- Capture and store PER-TIMESTEP RNG OBSERVATIONS
- Continue storing ONLY the final grid state

At no point should full 2D grid snapshots be stored for all timesteps.

====================================================
2. WHAT TO STORE PER TIMESTEP
====================================================

For each timestep and each RNG distribution used in the simulation,
record ONLY:

A. RNG SAMPLE DATA (ONE OF THE FOLLOWING)
   - A histogram of generated RNG values, OR
   - A statistically representative subsample

B. AGGREGATED STATISTICS
   - Mean
   - Variance
   - Min / Max

C. METADATA
   - Timestep index
   - RNG generator type
   - Seed / subsequence identifiers
   - GPU and stream identifiers

This data must be small and independent of (Nx, Ny).

====================================================
3. RNG OBSERVATION REQUIREMENTS
====================================================

RNG values must be observed at EVERY place they are used:
- Initial condition perturbation
- Stochastic forcing during each timestep
- Discrete event generation

RNG observation must not interfere with physics updates.

====================================================
4. ANALYSIS LOGIC (MANDATORY)
====================================================

Use the per-timestep data to compute:

- Mean stability over time
- Variance growth or stationarity
- Histogram shape consistency over time
- Frequency of extreme values
- Temporal autocorrelation of RNG samples
- Cross-stream independence checks

These analyses must be driven solely by the stored per-timestep data,
NOT by full grid history.

====================================================
5. FAILURE DETECTION
====================================================

Explicitly detect and report:

- Bias (mean drift over timesteps)
- Correlation (temporal or cross-stream)
- Missing distribution tails
- RNG stream overlap
- Non-deterministic results for fixed seeds
- Changes in results when GPU count changes

Each detected issue must be tied to a measurable signal from the stored data.

====================================================
6. PARALLEL SAFETY
====================================================

Ensure that:
- RNG observation is thread-safe
- Histogram and reduction operations use hipCUB
- No race conditions are introduced
- Results are deterministic for fixed seeds

====================================================
7. OUTPUT
====================================================

At program completion:
- Print or write time-series summaries of statistics
- Print histogram summaries per distribution
- Clearly indicate pass/fail status for RNG quality

No visual plotting is required.

====================================================
8. DOCUMENTATION UPDATE
====================================================

Add a documentation section explaining:

- Why storing only the final grid state is sufficient for physics
- Why per-timestep statistical capture is sufficient for RNG validation
- How RNG quality issues manifest in the stored metrics
- Why full grid history is unnecessary for this purpose

====================================================
9. CONSTRAINTS
====================================================

- Do NOT store full grid data for all timesteps
- Do NOT change the numerical scheme
- Do NOT reduce parallelism
- Do NOT add unnecessary memory overhead

====================================================
Apply these changes cleanly and provide the updated implementation.