BEGIN PROMPT

You are a senior ML systems engineer designing a GPU validation workload for AMD ROCm.

OBJECTIVE
Write a production-quality Python LLM training application whose primary purpose is to exercise and validate ROCm components using composable stress modes. This must be a real Transformer training workload, not a toy example, and should be suitable for ROCm regression, soak, and CI testing.

CORE DESIGN PRINCIPLE (CRITICAL)
Stress modes must be composable.
Each stress mode must modify training parameters only (not hard-coded logic).
Multiple stress modes can be enabled simultaneously.
A special flag --stress all must enable every stress mode.
Stress modes must be implemented as pure configuration mutators that transform a base training configuration.

TECHNOLOGY STACK (MANDATORY)
- PyTorch (ROCm build)
- Hugging Face Transformers
- Hugging Face Accelerate
- Hugging Face Datasets
- Optional: DeepSpeed (guarded behind a flag)
Do NOT use CUDA-specific APIs.

PROJECT STRUCTURE (MUST FOLLOW EXACTLY)

rocm_conformance_llm/
├── train.py               # main entrypoint
├── model.py               # configurable Transformer model
├── data.py                # dataset + tokenization
├── stress_modes.py        # composable stress mode definitions
├── config.py              # base config + merged stress profiles
├── utils/
│   ├── logging.py
│   └── reproducibility.py
└── README.md

CLI INTERFACE REQUIREMENTS

The application must support:
--stress gemm
--stress gemm,rng
--stress gemm,rng,comm,memory,fusion
--stress all

CLI parsing rules:
- --stress accepts a comma-separated list
- "all" expands to all known stress modes
- Duplicate modes are ignored
- Order must not affect final behavior

BASE TRAINING CONFIGURATION

Define a single base configuration object using a dataclass:

@dataclass
class TrainingConfig:
    hidden_size: int
    num_layers: int
    ffn_multiplier: float
    dropout: float
    batch_size: int
    seq_len: int
    grad_accum_steps: int
    grad_sync_interval: int
    activation_checkpointing: bool
    mixed_precision: str

STRESS MODE ARCHITECTURE (MANDATORY)

Each stress mode must implement the following interface:

class StressMode:
    name: str
    def apply(self, cfg: TrainingConfig) -> TrainingConfig:
        ...

Stress modes must:
- Only modify relevant fields
- Never invalidate the configuration
- Be safely composable with other modes

REQUIRED STRESS MODES

1) gemm
Targets: rocBLAS, hipBLASLt
Effects:
- Increase hidden size
- Increase FFN width
- Prefer FP16 or BF16

2) rng
Targets: rocRAND
Effects:
- Increase dropout globally
- Enable stochastic layers
- Enforce fixed random seeds
- Include determinism and replay checks

3) comm
Targets: RCCL
Effects:
- Reduce batch size
- Increase gradient synchronization frequency
- Enable multi-GPU logging

4) memory
Targets: ROCr memory allocator
Effects:
- Increase sequence length (8k+ configurable)
- Enable activation checkpointing
- Increase gradient accumulation
- Optional artificial memory pressure

5) fusion
Targets: MIOpen
Effects:
- Increase number of LayerNorms
- Use GELU or SwiGLU activations
- Deep residual stacks

STRESS MODE COMPOSITION LOGIC (REQUIRED)

Implement deterministic composition logic:

def compose_stress_modes(base_cfg: TrainingConfig, modes: List[StressMode]) -> TrainingConfig:
    for mode in modes:
        base_cfg = mode.apply(base_cfg)
    return base_cfg

TRAINING LOOP REQUIREMENTS

The training loop must:
- Perform forward and backward passes
- Compute loss and update parameters
- Support FP16/BF16 mixed precision
- Support gradient accumulation
- Support multi-GPU execution via Accelerate
- Log per step:
  - Loss
  - Step time
  - Tokens per second
  - GPU memory usage (best effort)
  - Active stress modes

ROCm-AWARE DESIGN CONSTRAINTS

- Assume execution on AMD GPUs via ROCm
- Use torch.cuda (HIP backend)
- Prefer large GEMMs and fused ops
- Pin host memory where applicable
- Avoid CUDA-specific assumptions

EXPECTED FAILURE SIGNATURES (CRITICAL)

The application must document and, where feasible, detect and log expected failure signatures per ROCm component. These are NOT hard assertions, but diagnostic signals intended for triage.

rocBLAS / hipBLASLt:
- NaNs or Infs appearing only under FP16/BF16
- Loss divergence when GEMM sizes increase
- Non-deterministic results across identical runs
- Sudden performance cliffs after autotuning
- Incorrect results only for specific matrix shapes

rocRAND:
- Loss curves diverging despite fixed seeds
- Inconsistent dropout masks across ranks
- Replay runs not matching original loss values
- Sensitivity to gradient accumulation boundaries

RCCL:
- Training hangs during backward or optimizer step
- All-reduce latency spikes
- Rank desynchronization or timeouts
- Progress stalls only at synchronization points
- Errors that appear only at scale (N > 1 GPUs)

ROCr Memory Allocator:
- Gradual increase in allocated memory per step
- Non-deterministic OOMs at same configuration
- OOM only after long runs (fragmentation)
- Large allocation latency spikes
- Failure to recover after checkpoint reload

MIOpen:
- Kernel count regressions between runs
- Performance regressions after warmup
- Autotuning instability across executions
- Incorrect outputs when fusion is enabled
- Large variance in step time

HIP Runtime / Driver:
- Illegal memory access errors
- Kernel launch failures under pressure
- GPU resets or hangs during long runs
- Failures that only appear after hours of execution

The application should log enough context (active stress modes, step number, config snapshot) to make these failures diagnosable.

OBSERVABILITY AND DEBUGGING

Include:
- Clear logging identifying active stress modes
- Reproducibility utilities (seed control)
- Optional hooks for timing forward/backward passes
- Optional memory usage snapshots
- Deterministic replay support for rng mode

README.md REQUIREMENTS

The README must explain:
- Purpose of the application
- Each stress mode and which ROCm components it targets
- Expected failure signatures per ROCm component
- How composable stress modes work
- Example commands for:
  - Single mode
  - Multiple modes
  - --stress all
- Single-GPU and multi-GPU usage

ACCEPTANCE CRITERIA

The generated code must:
- Run on ROCm-enabled PyTorch
- Support arbitrary combinations of stress modes
- Make ROCm component stress explicit and intentional
- Be suitable for long-running stability and regression testing
- Clearly surface failure signals in logs

EXPLICITLY DO NOT

- Use CUDA APIs
- Create non-composable special cases
- Hardcode GPU counts
- Assume NVIDIA hardware

STYLE AND QUALITY EXPECTATIONS

- Production-grade structure
- Clean separation of concerns
- Clear comments explaining how each stress mode affects ROCm components
- Explicit documentation of failure signatures
- Prioritize correctness, clarity, and debuggability

END PROMPT
