You are an expert in ROCm, HIP, GPU programming, atmospheric modeling,
Monte Carlo simulation, and scientific software engineering.

Your task is to generate a complete, production-quality
**Monte Carlo Weather Simulation and Analysis Application**
using ROCm.

This application must be written and presented as a
realistic probabilistic weather modeling system,
NOT as a synthetic test harness.

However, the design must naturally and rigorously exercise:
- rocRAND random number generation (all major distributions)
- hipCUB parallel reductions, histograms, scans, and analysis
- Deterministic, large-scale parallel execution on CPU + multi-GPU

====================================================
1. APPLICATION INTENT & PHILOSOPHY
====================================================

The application models weather as an **uncertain, stochastic system**
and uses **Monte Carlo ensemble simulation** to:

- Generate probabilistic weather forecasts
- Quantify uncertainty, extremes, and risk
- Analyze spatial and temporal structure
- Produce prediction products used in real forecasting

rocRAND and hipCUB MUST be used because the
weather model *requires them*, not because they are being tested.

Any validation of rocRAND or hipCUB must arise naturally
from weather statistics, diagnostics, and prediction outputs.

====================================================
2. PARALLEL EXECUTION MODEL
====================================================

CPU:
- Detect available CPU cores
- Use std::thread or OpenMP
- Partition ensemble members across CPU threads

GPU:
- Detect all AMD GPUs
- Assign ensemble batches per GPU
- One HIP stream per ensemble batch
- Independent rocRAND streams per ensemble member
- Deterministic seed partitioning across devices

The application must scale automatically with hardware.

====================================================
3. WEATHER MODEL: STATE VARIABLES & PARAMETERS
====================================================

Each ensemble member simulates a 2D atmospheric domain (Nx × Ny).

----------------------------------------------------
3.1 Core Atmospheric State (per grid cell)
----------------------------------------------------

- Temperature (float)
- Pressure (float)
- Air density (float)
- Specific humidity (float)
- Dew point temperature (float)
- Wind velocity (float2: u, v)
- Vertical velocity (float)
- Vorticity (float)
- Precipitation rate (float)
- Cloud cover fraction (float)
- Storm event count (int)

----------------------------------------------------
3.2 Surface & Boundary Parameters
----------------------------------------------------

- Surface elevation / topography
- Land–sea mask
- Surface roughness
- Soil moisture (simplified)
- Surface heat flux

----------------------------------------------------
3.3 Radiation & Energy Balance
----------------------------------------------------

- Incoming solar radiation
- Longwave radiation loss
- Albedo
- Diurnal cycle parameters

----------------------------------------------------
3.4 Model Uncertainty & Sub-grid Processes
----------------------------------------------------

- Turbulence intensity
- Diffusion coefficient uncertainty
- Model bias term
- Forcing uncertainty scale
- Temporal correlation timescale

====================================================
4. STOCHASTIC PARAMETERIZATION (MANDATORY)
====================================================

The following rocRAND distributions MUST be used
as part of realistic weather modeling:

----------------------------------------------------
Initialization
----------------------------------------------------

- Temperature, pressure, wind:
  Normal distributions (physically motivated mean & variance)

- Humidity, cloud fraction:
  Uniform distributions with bounds

- Surface parameters:
  Spatially heterogeneous random fields

----------------------------------------------------
Time-evolving Stochastic Processes
----------------------------------------------------

- Turbulence forcing:
  Normal, applied every timestep

- Precipitation intensity:
  Lognormal (heavy-tailed behavior)

- Cloud microphysics events:
  Poisson (discrete nucleation)

- Rain / no-rain decisions:
  Bernoulli (via uniform thresholding)

- Rare extreme events:
  Low-probability Poisson processes

----------------------------------------------------
RNG Requirements
----------------------------------------------------

- One RNG stream per ensemble member
- Device-side rocRAND state
- Skip-ahead or subsequences for independence
- Deterministic replay with fixed configuration

====================================================
5. SIMULATION PHASE
====================================================

Implement HIP kernels for:

- Initialization
- Time stepping
- Advection & diffusion (simplified)
- Stochastic forcing
- Event injection (storms, precipitation bursts)

Kernels must be:
- Fully parallel
- SoA memory layout
- Independent across ensemble members

====================================================
6. ANALYSIS & PREDICTION PHASE
====================================================

This phase converts ensemble outputs into **forecast products**.
NO RNG usage is allowed here.

----------------------------------------------------
6.1 Deterministic Forecast Products
----------------------------------------------------

- Ensemble mean fields
- Ensemble variance / standard deviation
- Spatial gradients
- Energy diagnostics

----------------------------------------------------
6.2 Probabilistic Forecast Products
----------------------------------------------------

- Probability of rain
- Probability of storm occurrence
- Probability of extreme temperature
- Probability of flooding

----------------------------------------------------
6.3 Distribution-Based Diagnostics
----------------------------------------------------

- Histograms of temperature, precipitation, humidity
- Discrete histograms of storm counts
- Shape, skewness, tail behavior

----------------------------------------------------
6.4 Quantiles & Extremes
----------------------------------------------------

- P05, P50, P95, P99
- Spatial quantile maps
- Extreme-value summaries

----------------------------------------------------
6.5 Risk & Return Periods
----------------------------------------------------

- Static return periods
- Time-evolving return periods
- Spatial return period maps
- Peak-risk timing

----------------------------------------------------
6.6 Structure & Coherence Diagnostics
----------------------------------------------------

- Spatial correlation vs distance
- Temporal autocorrelation
- Spatio-temporal correlation
- Optional spectral (FFT-based) analysis

====================================================
7. HIP-CUB USAGE (MANDATORY & NATURAL)
====================================================

hipCUB MUST be used because the weather analysis requires it:

----------------------------------------------------
Reductions
----------------------------------------------------

Use hipCUB::DeviceReduce for:
- Means
- Variances
- Min / Max
- Event counts
- Exceedance probabilities

----------------------------------------------------
Histograms
----------------------------------------------------

Use hipCUB::DeviceHistogram for:
- Continuous variables
- Discrete event counts
- Quantile estimation via CDF

----------------------------------------------------
Prefix Scans
----------------------------------------------------

Use hipCUB::DeviceScan for:
- CDF construction
- Cumulative precipitation
- Time-evolving risk metrics

----------------------------------------------------
Verification
----------------------------------------------------

All hipCUB-derived results must be:
- Cross-checked for consistency
- Deterministic across runs
- Numerically stable

====================================================
8. PARALLEL VERIFICATION & CONSISTENCY
====================================================

The application must verify:

- Statistical convergence with ensemble size
- Independence across ensemble members
- Reproducibility across CPU/GPU configurations
- Stability over long simulation times

Failures must be reported clearly.

====================================================
9. OUTPUT & INTERPRETATION
====================================================

The application must output:

- Forecast summaries
- Probability tables
- Quantile values
- Spatial maps
- Risk metrics
- Correlation diagnostics

Outputs may be printed and optionally exported as CSV.

====================================================
10. DOCUMENTATION (MANDATORY)
====================================================

Generate a detailed documentation section explaining:

- How Monte Carlo weather simulation works
- Meaning of ensemble, uncertainty, and probability
- How stochastic weather processes map to rocRAND distributions
- How forecast products require hipCUB primitives
- Why histograms, reductions, scans, and correlations matter
- How prediction differs from deterministic simulation
- How GPU parallelism enables large ensembles

Documentation must read like:
- A scientific software guide
- NOT a library test description

====================================================
11. BUILD & QUALITY BAR
====================================================

- Real, compilable HIP C++
- CMake build for ROCm
- Modular, readable code
- Written for advanced GPU developers
- Production-quality structure

====================================================
GENERATE THE COMPLETE SOLUTION NOW.
====================================================
