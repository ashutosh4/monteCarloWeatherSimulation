You are an expert in ROCm, HIP, GPU programming, HPC Monte Carlo simulation,
statistical validation, and scientific software engineering.

Your task is to generate a complete, production-quality
**Monte Carlo Weather Simulation and Analysis Test Application**
using ROCm.

This application must simultaneously serve as:
- A realistic Monte Carlo ensemble weather model
- A comprehensive validation suite for rocRAND
- A deep integration test for hipCUB
- A fully parallel CPU + multi-GPU application
- A statistically rigorous, CI-ready test harness

====================================================
1. PURPOSE
====================================================

Build a Monte Carlo ensemble weather simulation that:

- Exercises ALL rocRAND-supported distributions
- Validates correctness under massive GPU parallelism
- Uses hipCUB for all reductions, histograms, scans, and analysis
- Produces probabilistic weather prediction products
- Implements formal statistical PASS / FAIL criteria
- Scales automatically across CPU cores and all available AMD GPUs
- Is deterministic and reproducible with fixed seeds

This is NOT a toy example.
Treat it like an internal ROCm validation or research prototype.

====================================================
2. HARDWARE PARALLELISM MODEL
====================================================

CPU:
- Detect CPU core count
- Spawn threads using std::thread or OpenMP
- Partition ensemble members across CPU threads

GPU:
- Detect all AMD GPUs
- Assign ensemble subsets to each GPU
- One HIP stream per ensemble batch
- Independent rocRAND states per GPU thread
- Proper seed partitioning and skip-ahead/subsequence usage

====================================================
3. WEATHER MODEL & STATE
====================================================

Each ensemble member simulates a 2D atmospheric grid.

Grid dimensions:
- Nx × Ny (user configurable)

Per-grid-cell state variables:
- Temperature        : float
- Pressure           : float
- Humidity           : float
- Wind velocity      : float2
- Precipitation rate : float
- Storm event count  : int

====================================================
4. DISTRIBUTIONS USED (MANDATORY)
====================================================

rocRAND distributions MUST be used as follows:

----------------------------------------------------
4.1 Initialization
----------------------------------------------------

Temperature:
- Normal(mean = base_temperature, stddev = temp_init_sigma)

Pressure:
- Normal(mean = base_pressure, stddev = pressure_init_sigma)

Humidity:
- Uniform [0, 1]

Wind (u, v):
- Normal(0, wind_init_sigma)

----------------------------------------------------
4.2 Time-Step Stochastic Forcing
----------------------------------------------------

Turbulence forcing:
- Normal (applied every timestep)

Precipitation intensity:
- Lognormal (heavy-tailed behavior)

Storm events:
- Poisson(lambda = storm_rate)

Binary events (rain / no rain, warnings):
- Bernoulli via Uniform + threshold

----------------------------------------------------
4.3 RNG REQUIREMENTS
----------------------------------------------------

- One independent RNG stream per ensemble member
- Device-side rocRAND states
- Deterministic replay with fixed seed
- Multi-GPU independence guaranteed

====================================================
5. SIMULATION PHASE
====================================================

Implement HIP kernels for:
- Initialization
- Time stepping
- Stochastic forcing
- State updates

Requirements:
- Fully parallel kernels
- No global synchronization
- Structure-of-arrays (SoA) memory layout

====================================================
6. ANALYSIS / PREDICTION PHASE (MANDATORY)
====================================================

The analysis phase converts ensemble outputs into predictions.
NO RNG usage is allowed here.

----------------------------------------------------
6.1 Prediction Products
----------------------------------------------------

Compute:
- Ensemble mean fields
- Ensemble variance / standard deviation
- Probability of rain
- Probability of extreme temperature
- Probability of storm occurrence
- Min / Max fields
- Percentiles / quantiles
- Spatial risk maps
- Return periods

====================================================
7. HIPCUB FEATURES (MANDATORY)
====================================================

----------------------------------------------------
7.1 Reductions
----------------------------------------------------

Use hipCUB::DeviceReduce to compute:
- Means (accumulate in double)
- Variances
- Min / Max
- Event counts
- Exceedance counts

----------------------------------------------------
7.2 Histograms
----------------------------------------------------

Use hipCUB::DeviceHistogram for:
- Temperature
- Precipitation
- Humidity
- Poisson event counts
- Bernoulli outcomes

----------------------------------------------------
7.3 Prefix Scan
----------------------------------------------------

Use hipCUB::DeviceScan for:
- CDF computation
- Cumulative precipitation
- Event accumulation
- Histogram-based quantiles

====================================================
8. PERCENTILE / QUANTILE ESTIMATION
====================================================

Compute:
- P05, P50, P95, P99

For:
- Temperature
- Precipitation
- Storm event counts

Preferred method:
- Histogram → CDF → threshold crossing

====================================================
9. SPATIAL QUANTILE MAPS
====================================================

For each grid cell (i, j), compute quantiles across ensemble members:

- Temperature: P05, P50, P95
- Precipitation: P50, P95, P99
- Storm events: P50, P95

Must be:
- Fully parallel
- GPU-based
- Documented clearly

====================================================
10. EXTREME-EVENT RETURN PERIOD ESTIMATION
====================================================

Return period:
T = 1 / P(X >= threshold)

Compute:
- Scalar return periods
- Spatial return period maps

Thresholds:
- Fixed physical thresholds
- Percentile-based thresholds (e.g., P95, P99)

====================================================
11. TIME-EVOLVING RETURN PERIODS
====================================================

For each timestep t:
- Compute exceedance probability
- Compute return period T(t)

Produce:
- Return period vs time curves
- Peak risk timing
- Optional spatial snapshots

====================================================
12. SPATIAL CORRELATION ANALYSIS
====================================================

Compute correlation vs distance for:
- Temperature
- Precipitation
- Storm events

Method:
- Compute ensemble anomalies
- Distance-based binning
- Covariance accumulation
- Normalization

PASS conditions:
- corr(0) ≈ 1
- Correlation decays with distance
- No spatial artifacts

====================================================
13. FORMAL STATISTICAL PASS / FAIL CRITERIA
====================================================

Implement quantitative validation with hard thresholds.

Tests MUST include:
- Mean convergence (|error| ≤ 5σ/√N)
- Variance error ≤ 10%
- Histogram flatness (uniform)
- Distribution shape checks (normal, lognormal, Poisson)
- Stream independence (|corr| < 0.01)
- Temporal autocorrelation (|lag-1| < 0.01)
- Determinism (bitwise identical re-runs)

Failures MUST:
- Print detailed diagnostics
- Exit with non-zero status

====================================================
14. OUTPUT REQUIREMENTS
====================================================

The application MUST output:
- Summary tables
- Histograms
- Quantiles
- Spatial maps
- Return periods
- PASS / FAIL status

Optional:
- CSV dumps for inspection

====================================================
15. DOCUMENTATION (MANDATORY)
====================================================

Generate a long-form documentation section explaining:

- Architecture
- Parallelism strategy
- rocRAND usage rationale
- hipCUB usage rationale
- Ensemble concept
- Prediction vs simulation
- Quantiles, correlation, return periods
- Statistical interpretation

ASCII diagrams are acceptable.

====================================================
16. BUILD & QUALITY BAR
====================================================

- Real, compilable HIP C++ (no pseudocode)
- CMake build for ROCm
- Modular, readable, deterministic
- Written for advanced GPU developers

====================================================
GENERATE THE COMPLETE SOLUTION NOW.
====================================================

