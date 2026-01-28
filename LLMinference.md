You are an expert GPU systems engineer with deep knowledge of ROCm internals.
Generate a production-quality LLM inference diagnostic application whose
primary goal is to exercise, isolate, and validate ROCm components under
realistic transformer inference workloads.

========================
HIGH-LEVEL GOAL
========================
Build a C++ + HIP / PyTorch hybrid LLM inference application that:
- Runs autoregressive transformer inference (token-by-token decode)
- Treats inference as a diagnostic workload
- Explicitly tests ROCm components (HIP runtime, rocBLAS, rocRAND, hipCUB,
  rocPRIM, ROCr memory subsystem)
- Provides composable testing modes and detailed telemetry
- Surfaces expected failure signatures per ROCm component

========================
ARCHITECTURE REQUIREMENTS
========================
Organize the codebase as:

/src
  /core
    TokenLoopController.{h,cpp}
    KVCacheManager.{h,cpp}
    MemoryTracker.{h,cpp}
    MetricsCollector.{h,cpp}
  /rocm
    RocblasHooks.{h,cpp}
    RocRandHooks.{h,cpp}
    HipCubHooks.{h,cpp}
    RocPrimHooks.{h,cpp}
    HipRuntimeHooks.{h,cpp}
  /modes
    ModeConfig.{h,cpp}
    PrecisionModes.{h,cpp}
    StreamModes.{h,cpp}
    CacheModes.{h,cpp}
    SamplingModes.{h,cpp}
  main.cpp
/CMakeLists.txt
/README.md

========================
INFERENCE PIPELINE
========================
Implement explicit transformer inference stages:

1. Model load
   - Explicit hipMalloc / hipMemcpy for weights
   - Memory usage snapshot before and after

2. Prompt prefill
   - Large GEMMs via rocBLAS
   - Reduction ops via hipCUB / rocPRIM

3. Autoregressive decode loop (core focus)
   For each token:
     - QKV GEMM (rocBLAS)
     - Attention score reductions (hipCUB / rocPRIM)
     - Softmax
     - KV cache append / reuse
     - Sampling (rocRAND)
     - Token selection

4. Decode termination
   - Final memory snapshot
   - Latency summary

DO NOT use a single `model.generate()` abstraction.
All major ops must be visible and hookable.

========================
ROCM COMPONENT TESTING
========================
Explicitly exercise and log:

HIP Runtime
- hipMalloc / hipFree churn
- hipMemcpy H2D / D2H
- hipStreamCreate / hipEvent timing

rocBLAS
- Mixed-size GEMMs (small QKV, large FFN)
- FP16, BF16, FP32 paths
- Log GEMM shapes per token

rocRAND
- RNG state per sequence
- Top-k and top-p sampling
- Deterministic vs nondeterministic modes

hipCUB / rocPRIM
- Reductions (softmax, layernorm)
- Prefix sums for attention offsets
- Selection/sort for sampling

ROCr / Memory subsystem
- HBM usage tracking
- Fragmentation detection
- KV cache growth behavior

========================
COMPOSABLE MODES
========================
Implement modes that can be enabled simultaneously:

--precision=fp16|bf16|fp32
--batch-size=N
--prompt-length=N
--max-tokens=N

--single-stream
--multi-stream
--stream-per-sequence

--kv-cache-reuse
--kv-cache-no-reuse
--kv-cache-evict

--sampling=greedy|topk|topp
--rng-deterministic
--rng-nondeterministic

--disable-rocblas
--disable-rocrand
--disable-hipcub
--disable-kv-cache

Modes must be orthogonal and composable.

========================
METRICS & TELEMETRY
========================
Collect and print:

Per-token latency (min / max / avg)
First-token latency
Tokens/sec
HBM usage over time
KV cache size per token
GEMM time vs reduction time vs sampling time

========================
FAILURE SIGNATURES
========================
For each ROCm component, document and detect:

HIP Runtime
- Memory leak: free memory monotonically decreases
- Stream deadlock: decode stalls

rocBLAS
- NaNs in logits
- Throughput collapse with mixed GEMM sizes

rocRAND
- Identical outputs across different seeds
- Divergent distributions with same seed

hipCUB / rocPRIM
- Softmax sums != 1.0
- LayerNorm variance drift

KV Cache / ROCr
- Sudden latency spikes after N tokens
- OOM despite reported free memory

Print warnings when detected.

========================
IMPLEMENTATION CONSTRAINTS
========================
- Use CMake
- Clean separation between inference logic and ROCm hooks
- No fused kernels that hide library usage
- Code must compile on ROCm 5.x+
- Prefer clarity over micro-optimizations

========================
DOCUMENTATION
========================
Generate README.md explaining:
- How inference maps to ROCm components
- How to enable each test mode
- How to interpret failure signatures
- Example commands for common test scenarios

========================
OUTPUT
========================
Generate all source files, headers, and README content.
Provide clear TODO markers where a real model implementation would be plugged in.