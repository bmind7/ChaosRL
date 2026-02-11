# ChaosRL Technical Overview

## Purpose and Audience
This document is a contributor-facing technical map of the ChaosRL codebase. It focuses on implementation structure, data flow, and supported capabilities so engineers can modify or extend the system without reverse-engineering from scratch.

The scope is intentionally code-centric. It prioritizes architecture and APIs in C# over scene/prefab authoring details.

## Project Snapshot
- Engine: Unity `6000.2.10f1`
- Language: C#
- Primary runtime assembly: `ChaosRL` (`Assets/ChaosRL/ChaosRL.asmdef`)
- Supporting gameplay script assembly: `Assembly-CSharp` (`Assets/Src/ChaosAgent.cs`)
- Core compute-oriented dependencies:
  - `com.unity.burst`
  - `com.unity.collections`
  - `com.unity.mathematics`
- Primary domains:
  - Tensor autodiff engine
  - Neural network primitives (MLP + optimizer + regularizer)
  - PPO training loop for continuous control
  - Runtime agent/environment integration

## Core Subsystems
| Subsystem | Primary Responsibility | Main Types |
|---|---|---|
| Autodiff Core | Tensor storage, shape ops, operator overloading, dynamic graph backprop | `Tensor`, `TensorOps`, `TensorJobs`, `Value`, `ValueExtensions` |
| NN Stack | Feed-forward layers, model composition, optimization, L2 penalty | `Layer`, `MLP`, `AdamOptimizer`, `L2Regularizer` |
| RL Training Loop | PPO rollout buffering, GAE computation, minibatch optimization | `Academy` |
| Probability + Utility | Gaussian policy math, RNG, minibatch extraction | `Dist`, `RandomHub`, `Utils` |
| Runtime Integration | Physics-agent loop and action/observation bridge to trainer | `ChaosAgent`, `ArenaGridSpawner` |
| Validation + Performance Tooling | Correctness tests and benchmark harnesses | `*Tests.cs`, `AutodiffBenchmarks`, `MatMulBenchmarkUI` |

## Class-Level API Reference
Each class entry follows the same mini-spec format:
- Responsibility
- Key public members/methods
- Core contracts (shape or behavior)
- Main collaborators

### `Tensor` (`Assets/ChaosRL/Autodiff/Tensor.cs`)
- Responsibility: N-dimensional tensor with row-major storage and dynamic autograd graph.
- Key public members/methods:
  - Properties: `Data`, `Grad`, `Shape`, `Size`, `Name`, `Children`, `IsScalar`, `RequiresGrad`
  - Indexing/utilities: `this[params int[]]`, `ToFlatIndex(...)`, `ZeroGrad()`, `Backward()`, `Dispose()`
  - Ops: `+`, `-`, `*`, `/`, `Pow`, `Sqrt`, `Exp`, `Log`, `ReLU`, `Tanh`, `Clamp`, static `Max/Min`
  - Tensor ops: `MatMul`, `Sum`, `Mean`, `Max`
  - Shape ops: `Normalize`, `Unsqueeze`, `Squeeze`, `Slice`, `ExpandLast`, `Reshape`
- Core contracts:
  - Stores data and grad in `NativeArray<float>` with persistent allocation.
  - `MatMul` currently requires 2D tensors with matching inner dimension.
  - Broadcasting is constrained by internal `CanBroadcastModulo(...)` rules.
  - View-like ops (`Reshape`, `Squeeze`, `Unsqueeze`) share underlying storage.
  - `ZeroGrad` uses `UnsafeUtility.MemClear`; `Backward` grad seed uses `UnsafeUtility.MemCpyReplicate`.
  - `Sum` forward/backward uses Burst jobs (`SumReductionJob`, `AddScalarParallelJob`).
- Main collaborators:
  - `Layer`, `MLP`, `Academy`, `Dist`, tests, and benchmark harnesses.
  - Uses `TensorOps` for MatMul and transpose scheduling, which delegates to `TensorJobs` Burst kernels.

### `TensorJobs` (`Assets/ChaosRL/Autodiff/TensorJobs.cs`)
- Responsibility: Burst-compiled job struct definitions for tensor compute kernels.
- Key public structs:
  - `TransposeTiledParallelJob : IJobParallelFor` — cache-friendly tiled transpose (TILE=32)
  - `MatMulNaiveParallelJob : IJobParallelFor` — naive matmul with pre-transposed B, supports accumulate mode
  - `PackBPanelScalarParallelJob : IJobParallelFor` — packs B columns into NR=16-wide panels for GEBP
  - `MatMulGebpScalarParallelJob : IJobParallelFor` — GEBP micro-kernel (MR=6, NR=16) parallelized over row groups
  - `SumReductionJob : IJob` — single-threaded sum reduction to scalar
  - `AddScalarParallelJob : IJobParallelFor` — broadcasts scalar add into target array
- Core contracts:
  - All structs decorated with `[BurstCompile(FloatMode.Fast, FloatPrecision.Low, DisableSafetyChecks=true, OptimizeFor=Performance)]`.
  - Operate on `NativeArray<float>` buffers.
  - GEBP jobs use `KOffset` and `Kc` fields for Kc-blocked iteration over the K dimension.
  - `MatMulNaiveParallelJob` supports overwrite or accumulation mode (`Accumulate`) for backward gradients.
- Main collaborators:
  - Scheduled by `TensorOps` static methods; not invoked directly from `Tensor`.

### `TensorOps` (`Assets/ChaosRL/Autodiff/TensorOps.cs`)
- Responsibility: Static scheduling orchestration for Burst job graphs, auto-selecting the optimal MatMul kernel.
- Key public members/methods:
  - Constants: `KC = 256` (K-block size for L1 residency), `GebpThreshold = 16` (min dimension for GEBP path)
  - `GetBatchSize(int totalWorkItems)` — computes `IJobParallelFor` batch sizing based on worker thread count
  - `ScheduleTranspose(...)` — schedules `TransposeTiledParallelJob` over tiles
  - `ScheduleMatMul(...)` — unified entry point: auto-selects GEBP (dims ≥ 16) or naive path
- Core contracts:
  - GEBP path: loops K in KC-thick slices, packs B into panel layout via `PackBPanelScalarParallelJob`, runs `MatMulGebpScalarParallelJob`, auto-disposes packed buffers.
  - Naive path: transposes B first, then runs `MatMulNaiveParallelJob`.
  - Both paths support accumulation mode for backward gradient computation.
- Main collaborators:
  - Called by `Tensor.MatMul` forward and backward paths.
  - Delegates to `TensorJobs` Burst job structs.

### `Value` (`Assets/ChaosRL/Autodiff/Value.cs`) (legacy/auxiliary path)
- Responsibility: Scalar autodiff node for educational and benchmark comparison workflows.
- Key public members/methods:
  - Fields/properties: `Data`, `Grad`, `Children`, `Name`
  - Ops: `+`, `-`, `*`, `/`, unary `-`, `Pow`, `Exp`, `ReLU`, `Tanh`, `Log`, `Clamp`, static `Max/Min`
  - `Backward()`
- Core contracts:
  - Dynamic graph over scalar nodes.
  - `Backward()` builds topological order and seeds output grad to `1`.
- Main collaborators:
  - Used primarily in `ValueTests` and `AutodiffBenchmarks` as comparison baseline.

### `ValueExtensions` (`Assets/ChaosRL/Autodiff/ValueExtensions.cs`)
- Responsibility: Convenience helpers for formatting and converting between `Value` and primitive arrays.
- Key public methods:
  - `ToStringData`, `ToStringFull`, `ToMatrixString`
  - `ToValues(float[])`
  - `ToFloats(Value[])`
- Core contracts:
  - Pure utility methods; no training loop ownership.
- Main collaborators:
  - Tests and developer inspection workflows.

### `Layer` (`Assets/ChaosRL/NN/Layer.cs`)
- Responsibility: Single dense layer (`input @ weights + bias`) with optional `Tanh`.
- Key public members/methods:
  - Fields: `NumInputs`, `NumOutputs`
  - Property: `Parameters` (weights, bias)
  - Methods: `Forward(Tensor)`, `ZeroGrad()`
- Core contracts:
  - Expects 2D input shape `[batch, numInputs]`.
  - Weight shape `[numInputs, numOutputs]`; bias shape `[numOutputs]`.
  - Uses random initialization scaled by input width.
- Main collaborators:
  - Built and orchestrated by `MLP`.
  - Uses tensor math for forward/backward behavior.

### `MLP` (`Assets/ChaosRL/NN/MLP.cs`)
- Responsibility: Multi-layer perceptron composed from `Layer` instances.
- Key public members/methods:
  - Fields: `NumInputs`, `NumOutputs`, `LayerSizes`
  - Property: `Parameters` (iterates over all layer parameters)
  - Methods: `Forward(Tensor)`, `ZeroGrad()`
- Core contracts:
  - Input must be 2D with feature size equal to `NumInputs`.
  - Last-layer nonlinearity is configurable (`lastLayerNonLin`).
- Main collaborators:
  - Instantiated by `Academy` as policy and value networks.
  - Optimized via `AdamOptimizer`; regularized via `L2Regularizer`.

### `AdamOptimizer` (`Assets/ChaosRL/NN/AdamOptimizer.cs`)
- Responsibility: Adam parameter updates over one or more parameter groups.
- Key public members/methods:
  - Property: `Parameters`
  - Methods: `Step(float learningRate)`, `ResetState()`
- Core contracts:
  - Maintains flattened moment buffers `_m` and `_v` with per-parameter offsets.
  - Applies bias-corrected updates each step.
- Main collaborators:
  - Called by `Academy.UpdateNetworks()` after backpropagation.

### `L2Regularizer` (`Assets/ChaosRL/NN/L2Regularizer.cs`)
- Responsibility: Computes L2 penalty tensor across parameter groups.
- Key public methods:
  - `Compute(float coefficient, float scale = 0.5f)`
- Core contracts:
  - Returns zero tensor when coefficient or scale is non-positive.
  - Accumulates sum of squared parameters.
- Main collaborators:
  - Integrated into `Academy` total loss.

### `Academy` (`Assets/ChaosRL/RL/Academy.cs`)
- Responsibility: End-to-end PPO orchestration, including rollout buffering and parameter updates.
- Key public members/methods:
  - Singleton access: `Instance`
  - `NumEnvs` property
  - `RequestDecision(int agentIdx, float[] observation, bool done, float reward)`
  - `ComputeGAE()`
  - `UpdateNetworks()`
- Core contracts:
  - Continuous-action PPO with Gaussian policy:
    - Policy head outputs mean and log-std (packed in one tensor).
    - Value head outputs scalar state value.
  - Stores rollout tensors for observations, actions, log-probs, values, rewards, done flags.
  - Update trigger occurs when rollout buffer is full and env stepping alignment condition is met.
- Main collaborators:
  - Uses `MLP`, `AdamOptimizer`, `L2Regularizer`, `Dist`, `Utils.GetMinibatch`, `RandomHub`, and `Tensor`.
  - Called by `ChaosAgent` per decision step.

### `ArenaGridSpawner` (`Assets/ChaosRL/RL/ArenaGridSpawner.cs`)
- Responsibility: Runtime utility to instantiate multiple arena prefabs based on configured environment count.
- Key public behavior:
  - Reads `Academy.Instance.NumEnvs`
  - Computes cube-like grid dimensions
  - Spawns up to requested environment count
- Core contracts:
  - Requires `_arenaPrefab`.
  - Optional centering and spacing controls.
- Main collaborators:
  - Coordinates scene runtime setup for parallel environment instances.

### `Dist` (`Assets/ChaosRL/Utils/Dist.cs`)
- Responsibility: Gaussian distribution helper operating on tensors, including differentiable sampling.
- Key public members/methods:
  - Properties: `Mean`, `StdDev`
  - Factory: `StandardNormal()`
  - Methods: `Sample()`, `Pdf(Tensor)`, `LogProb(Tensor)`, `Entropy()`
- Core contracts:
  - Mean and std shapes must match exactly.
  - Sampling uses reparameterization (`mean + std * z`) for gradient flow.
- Main collaborators:
  - Used by `Academy` policy action sampling and log-prob/entropy calculations.

### `RandomHub` (`Assets/ChaosRL/Utils/RandomHub.cs`)
- Responsibility: Centralized thread-safe RNG wrapper for deterministic and shared randomness.
- Key public methods:
  - `SetSeed(int)`, `NextDouble()`, `NextFloat(...)`, `NextInt(...)`, `Shuffle<T>(T[])`
- Core contracts:
  - Uses internal locking around `System.Random` operations.
- Main collaborators:
  - Used in training minibatch shuffling, distribution sampling, and initialization paths.

### `Utils` (`Assets/ChaosRL/Utils/Utils.cs`)
- Responsibility: Data movement helper for minibatch extraction.
- Key public method:
  - `GetMinibatch(Tensor source, int[] indices, int start, int batchSize)`
- Core contracts:
  - Supports only 1D or 2D source tensors.
  - Returns new tensor with copied minibatch data.
- Main collaborators:
  - Used inside `Academy.UpdateNetworks()` minibatch loop.

### `ChaosAgent` (`Assets/Src/ChaosAgent.cs`)
- Responsibility: Physics loop integration and API boundary between environment and trainer.
- Key public methods:
  - `ResetState()`
  - `CollectObservations()`
  - `ApplyActions(float[])`
- Core contracts:
  - Collects 7D observation (rotation quaternion + relative ball position).
  - Calls `Academy.Instance.RequestDecision(...)` on configured step interval.
  - Applies smoothed, clamped continuous tilt actions.
- Main collaborators:
  - Directly interfaces with `Academy` and Unity physics components.

## Training and Optimization Data Flows
### 1) Per-step decision flow
```text
ChaosAgent.FixedUpdate
  -> AddProximityReward
  -> (every step interval) CollectObservations()
  -> Academy.RequestDecision(agentIdx, obs, done, reward)
       -> obs Tensor [1, input]
       -> policy = policyMLP.Forward(obs)
       -> value  = valueMLP.Forward(obs)
       -> split policy output:
            mean   = policy[:, 0:actionSize]
            logStd = policy[:, actionSize:2*actionSize]
       -> std = Exp(logStd + offset)
       -> dist = Dist(mean, std)
       -> actionSample = dist.Sample()
       -> logProb = Sum(LogProb(actionSample))
       -> write rollout buffers:
            observations, actions, logProb, value, done, reward
       -> if buffer full and env-step alignment satisfied:
            UpdateNetworks()
       -> return clamped action to ChaosAgent
  -> ChaosAgent.ApplyActions(action)
  -> if done: ChaosAgent.ResetState()
```

### 2) PPO update flow
```text
Academy.UpdateNetworks
  -> ComputeGAE()
       -> reverse-time per-env delta/gae accumulation
       -> produce advantages[T,N], returns[T,N]
  -> trim bootstrap row
  -> reshape rollout buffers to flat batch [T*N, ...]
  -> normalize advantages
  -> for epoch in updateEpochs:
       -> shuffle indices (RandomHub.Shuffle)
       -> for minibatch:
            -> mb tensors = Utils.GetMinibatch(...)
            -> policy/value forward passes
            -> build Dist(mean, std)
            -> newLogProb, entropy, ratio
            -> clipped PPO policy loss
            -> value loss (MSE-style)
            -> l2 penalty
            -> totalLoss = pg - beta*entropy + vCoef*value + l2
            -> policy.ZeroGrad(); value.ZeroGrad()
            -> totalLoss.Backward()
            -> optimizer.Step(learningRate)
  -> log aggregate training stats
```

## Capability Matrix
| Capability Area | Implemented Behavior |
|---|---|
| Tensor elementwise math | `+`, `-`, `*`, `/`, `Pow`, `Sqrt`, `Exp`, `Log`, `ReLU`, `Tanh`, `Clamp`, elementwise `Max/Min` |
| Tensor reduction ops | `Sum`, `Mean`, `Max` over full tensor or along selected dimension |
| Matrix multiplication | 2D `MatMul` with GEBP kernel (MR=6, NR=16, KC=256), auto-fallback to naive for small dims, Burst-compiled forward and backward |
| Shape/transformation ops | `Reshape`, `Slice`, `Unsqueeze`, `Squeeze`, `ExpandLast`, indexer access |
| Autograd graph | Dynamic graph via `Children` and per-op backward closures; topological backward traversal |
| Gradient control | `RequiresGrad` propagation to support training vs inference pathways |
| View semantics | `Reshape`, `Squeeze`, `Unsqueeze` share underlying data/grad storage |
| PPO algorithm pieces | GAE, clipped policy objective, entropy bonus, value loss, L2 penalty, minibatch SGD with multiple epochs |
| Continuous action policy | Gaussian policy with learned mean/log-std and reparameterized sampling |
| Optimizer support | Adam with flattened moment buffers across parameter groups |
| Utility support | Seedable RNG (`RandomHub`), minibatch extraction helper (`Utils.GetMinibatch`), Gaussian distribution math (`Dist`) |
| Legacy comparison path | Scalar autodiff via `Value` retained for tests/benchmarks and educational clarity |

## Benchmark Methodology
Benchmark coverage is implemented in two surfaces:
- Editor test benchmark class: `Assets/ChaosRL/Tests/AutodiffBenchmarks.cs`
- Runtime benchmark UI: `Assets/ChaosRL/MatMulBenchmarkUI.cs`

Methodology characteristics:
- Uses warmup passes before timed loops.
- Uses repeated iterations and reports average operation time.
- Compares scalar graph path (`Value`) and tensor path (`Tensor`) for overlapping workloads.
- Measures operation families:
  - Elementwise addition/multiplication across 1D/2D/3D sizes
  - Complex forward chains (`x * w + b -> tanh`)
  - Forward + backward pass workflows
  - Matrix multiplication and matrix multiplication with backward
  - Allocation-overhead-focused comparison scenario
- Runtime benchmark UI focuses on matrix multiplication and matrix multiplication + backward across multiple matrix sizes.
- Runtime benchmark UI also supports kernel comparison mode (GEBP vs Naive) across matrix sizes.

The benchmark code is intended for relative implementation comparison and regression visibility, not as a cross-hardware absolute performance contract.

## Test and Verification Coverage
Tests are organized under `Assets/ChaosRL/Tests` in an Editor-only test assembly (`ChaosRLTests.asmdef`).

Coverage map by file:
- `TensorTests.cs`
  - Tensor construction, indexing, broadcasting behavior, math ops, gradients, matmul, reductions, shape transforms, `RequiresGrad`, and error handling.
- `ValueTests.cs`
  - Scalar autodiff forward/backward correctness across arithmetic and activation operations.
- `DistTests.cs`
  - Distribution construction, sampling behavior, log-prob/pdf/entropy values, shape checks, and gradient flow.
- `AdamOptimizerTests.cs`
  - Update correctness across scalar/vector/matrix cases, multiple groups, state reset, and hyperparameter behavior.
- `L2RegularizerTests.cs`
  - L2 computation correctness across coefficients, scaling, parameter groups, and constructor validations.
- `MLPTests.cs`
  - End-to-end learning sanity check for a small supervised mapping task.
- `AutodiffBenchmarks.cs`
  - Benchmark harness coverage for relative operation timing across Value and Tensor paths.

## Architectural Boundaries (Implemented Constraints)
These are current capability boundaries explicitly visible in code contracts and inline comments:
- `Tensor.MatMul(...)` currently supports only 2D tensor inputs.
- Broadcasting in tensor elementwise ops is intentionally constrained by `CanBroadcastModulo(...)` logic and documented TODO notes.
- `Tensor.Normalize(...)` supports:
  - whole tensor (`dim = null`)
  - first dimension (`dim = 0`)
  - last dimension (`dim = -1` or `dim = rank - 1`)
- `Tensor.ExpandLast(...)` expands by adding/repeating only a new last dimension.
- `Utils.GetMinibatch(...)` currently supports only 1D and 2D tensors.
- `Dist` requires exact shape match between mean/std/input tensors (no implicit broadcasting inside distribution methods).
- `Tensor` manages persistent native memory and exposes `Dispose()` for deterministic release.

## Source Index
Core architecture files:
- `Assets/ChaosRL/Autodiff/Tensor.cs`
- `Assets/ChaosRL/Autodiff/TensorJobs.cs`
- `Assets/ChaosRL/Autodiff/TensorOps.cs`
- `Assets/ChaosRL/Autodiff/Value.cs`
- `Assets/ChaosRL/Autodiff/ValueExtensions.cs`
- `Assets/ChaosRL/NN/Layer.cs`
- `Assets/ChaosRL/NN/MLP.cs`
- `Assets/ChaosRL/NN/AdamOptimizer.cs`
- `Assets/ChaosRL/NN/L2Regularizer.cs`
- `Assets/ChaosRL/RL/Academy.cs`
- `Assets/ChaosRL/RL/ArenaGridSpawner.cs`
- `Assets/ChaosRL/Utils/Dist.cs`
- `Assets/ChaosRL/Utils/RandomHub.cs`
- `Assets/ChaosRL/Utils/Utils.cs`
- `Assets/Src/ChaosAgent.cs`

Validation and benchmark files:
- `Assets/ChaosRL/Tests/ChaosRLTests.asmdef`
- `Assets/ChaosRL/Tests/TensorTests.cs`
- `Assets/ChaosRL/Tests/ValueTests.cs`
- `Assets/ChaosRL/Tests/DistTests.cs`
- `Assets/ChaosRL/Tests/AdamOptimizerTests.cs`
- `Assets/ChaosRL/Tests/L2RegularizerTests.cs`
- `Assets/ChaosRL/Tests/MLPTests.cs`
- `Assets/ChaosRL/Tests/AutodiffBenchmarks.cs`
- `Assets/ChaosRL/MatMulBenchmarkUI.cs`

Project metadata:
- `ProjectSettings/ProjectVersion.txt`
- `Packages/manifest.json`
- `Assets/ChaosRL/ChaosRL.asmdef`
