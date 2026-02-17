# ChaosRL Technical Overview

## Purpose and Audience
This document is a contributor-focused map of the ChaosRL codebase. It describes current architecture, data flow, and implementation boundaries so feature work can be done without reverse-engineering from scratch.

The focus is runtime C# systems (autodiff, NN, PPO, and compute backends), not scene authoring.

## Project Snapshot
- Engine: Unity `6000.2.10f1`
- Language: C#
- Runtime assembly: `ChaosRL` (`Assets/ChaosRL/ChaosRL.asmdef`)
- Agent bridge assembly: `Assembly-CSharp` (`Assets/Src/ChaosAgent.cs`)
- Core packages: Burst, Collections, Mathematics
- Primary domains:
  - Tensor autodiff engine
  - MLP + optimizer stack
  - PPO training loop
  - Runtime environment/agent integration

## Core Subsystems
| Subsystem | Responsibility | Main Types |
|---|---|---|
| Autodiff core | Tensor graph, storage ownership, backend dispatch, shape ops | `Tensor`, `TensorStorage`, `TensorScope`, `TensorDevice` |
| Backend abstraction | Device-agnostic kernel API and CPU implementation | `ITensorBackend`, `CpuBackend` |
| Burst kernels and scheduling | Low-level jobs and matmul orchestration | `TensorJobs`, `CpuMatMulOps` |
| Neural network stack | Dense layers, model composition, optimization, regularization | `Layer`, `MLP`, `AdamOptimizer`, `L2Regularizer` |
| RL training | PPO rollouts, GAE, minibatch updates | `Academy`, `ArenaGridSpawner` |
| Probability and utilities | Gaussian policy math, RNG, minibatch extraction | `Dist`, `RandomHub`, `Utils` |
| Runtime integration | Physics observations/actions and trainer handoff | `ChaosAgent` |
| Validation and benchmarks | Edit-mode tests and performance harness | `ChaosRL.Tests`, `AutodiffBenchmarks`, `MatMulBenchmarkUI` |

## Backend and Memory Model
### Tensor storage ownership
- `Tensor` stores data/grad in `TensorStorage` objects (`Data`, `Grad`).
- `TensorStorage` wraps `NativeArray<float>` with ref-counting (`AddRef`, `Release`).
- View ops (`Reshape`, `Squeeze`, `Unsqueeze`) share storage and increment ref-counts.
- Final release disposes persistent native memory.

### Device abstraction
- `TensorDevice` declares `CPU` and `GPU`.
- `Tensor.ResolveBackend(...)` currently resolves:
  - `CPU` -> shared `CpuBackend`
  - `GPU` -> throws until a GPU backend registration path exists
- `TensorStorage.Allocate(...)` currently throws for `TensorDevice.GPU`.

### Backend contracts
- `ITensorBackend` defines forward/backward kernels for:
  - element-wise binary and unary ops
  - reductions (`Sum`, `Max` variants)
  - matmul forward/backward
  - data movement (`Copy`, `SliceCopy`, `ExpandLast`, `Gather`)
  - utility operations (`ZeroGrad`, `FillOnes`)
  - optimizer step (`AdamStep`)
- `CpuBackend` implements all methods with Burst jobs and synchronous completion.

### Tensor lifetime helpers
- `Tensor` implements `IDisposable` and finalizer-based fallback.
- `TensorScope` tracks tensors in scope (`[ThreadStatic]` current scope).
- Disposing a scope disposes tracked tensors in reverse order.
- Test suites use this to avoid NativeArray leakage.

## Class Notes
### `Tensor` (`Assets/ChaosRL/Autodiff/Tensor.cs`)
- N-D tensor with dynamic autograd graph.
- Public API includes `Data`, `Grad`, `Shape`, `Size`, `Device`, `Backend`, `Scalar`, indexer, math ops, reductions, shape ops, `Backward`, `ZeroGrad`, `To(...)`.
- Operator flow:
  1. validate shape and device compatibility
  2. allocate result tensor
  3. run forward through backend
  4. set `RequiresGrad`
  5. attach `_backward` closure calling backend backward kernel
- `Backward()` builds topological order and seeds output grad through backend `FillOnes(...)`.

### `TensorStorage` (`Assets/ChaosRL/Autodiff/TensorStorage.cs`)
- Ref-counted buffer wrapper over persistent `NativeArray<float>`.
- Supports managed copy helpers and low-level clear/fill operations.
- CPU-only today; GPU allocation path is declared but not implemented.

### `ITensorBackend` / `CpuBackend`
- `ITensorBackend` is the single operation contract for devices.
- `CpuBackend` schedules `TensorJobs` and completes each operation synchronously.
- Backward kernels for broadcast/alias cases use serial fallbacks to avoid write races.

### `TensorJobs` (`Assets/ChaosRL/Autodiff/TensorJobs.cs`)
- Contains Burst jobs for element-wise forward/backward, reductions, slicing/expand, optimizer step, transpose, and matmul kernels.
- Matmul jobs include naive path plus packed-panel GEBP path.

### `CpuMatMulOps` (`Assets/ChaosRL/Autodiff/CpuMatMulOps.cs`)
- Internal CPU-specific MatMul orchestration for `CpuBackend`.
- Orchestrates transpose and matmul scheduling.
- Auto-selects naive vs GEBP path by dimension threshold.

### `Layer` and `MLP` (`Assets/ChaosRL/NN`)
- `Layer` = dense affine transform + optional `Tanh`.
- `MLP` chains layers and exposes flattened parameter enumeration.
- Both now implement `IDisposable` and guard public methods when disposed.

### `AdamOptimizer` (`Assets/ChaosRL/NN/AdamOptimizer.cs`)
- Maintains flattened moment buffers (`TensorStorage _m`, `_v`) with per-parameter offsets.
- Calls backend `AdamStep(...)` per parameter tensor.
- Implements dispose/finalizer pattern and `ResetState()`.

### `Academy` (`Assets/ChaosRL/RL/Academy.cs`)
- Singleton PPO trainer.
- Handles rollout collection, GAE computation, reshaping/normalization, minibatch SGD, and logging.
- Scalar reads use `Tensor.Scalar` where scalar tensors are expected.

### `Utils` (`Assets/ChaosRL/Utils/Utils.cs`)
- `GetMinibatch(...)` supports 1D/2D tensors.
- Uses backend `Gather(...)` so minibatch extraction is device-agnostic.

## Training Flow Summary
1. `ChaosAgent` sends observation/reward/done to `Academy.RequestDecision(...)`.
2. Policy/value networks run forward passes.
3. Policy output is split to mean/log-std, then sampled through `Dist`.
4. Rollout buffers store obs/action/logprob/value/reward/done.
5. On rollout completion, `Academy.UpdateNetworks()`:
   - computes GAE and returns
   - flattens trajectories
   - normalizes advantages
   - runs PPO minibatch updates for configured epochs
   - steps Adam optimizers

## Test and Benchmark Coverage
- Test assembly: `Assets/ChaosRL/Tests/ChaosRLTests.asmdef`
- Shared base: `TensorScopedTestBase` creates/disposes `TensorScope` per test.
- Core test files:
  - `TensorTests.cs`
  - `ValueTests.cs`
  - `DistTests.cs`
  - `AdamOptimizerTests.cs`
  - `L2RegularizerTests.cs`
  - `MLPTests.cs`
- Benchmark harness:
  - `AutodiffBenchmarks.cs` (explicit/manual run, performance category)
  - `MatMulBenchmarkUI.cs` runtime benchmark scene helper

## Current Constraints
- `MatMul` supports only 2D tensors.
- Broadcasting remains limited to tail-shape compatibility (`CanBroadcastModulo`).
- `Normalize` supports whole tensor, `dim = 0`, or last dimension.
- `ExpandLast` only appends/repeats a new last dimension.
- `Utils.GetMinibatch` supports only 1D and 2D tensors.
- GPU backend is declared but not implemented.

## Source Index
Autodiff and backend files:
- `Assets/ChaosRL/Autodiff/Tensor.cs`
- `Assets/ChaosRL/Autodiff/TensorStorage.cs`
- `Assets/ChaosRL/Autodiff/TensorScope.cs`
- `Assets/ChaosRL/Autodiff/TensorDevice.cs`
- `Assets/ChaosRL/Autodiff/ITensorBackend.cs`
- `Assets/ChaosRL/Autodiff/CpuBackend.cs`
- `Assets/ChaosRL/Autodiff/CpuMatMulOps.cs`
- `Assets/ChaosRL/Autodiff/TensorJobs.cs`
- `Assets/ChaosRL/Autodiff/Value.cs`
- `Assets/ChaosRL/Autodiff/ValueExtensions.cs`

NN, RL, and utility files:
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

Test and metadata files:
- `Assets/ChaosRL/Tests/ChaosRLTests.asmdef`
- `Assets/ChaosRL/Tests/TensorScopedTestBase.cs`
- `Assets/ChaosRL/Tests/TensorTests.cs`
- `Assets/ChaosRL/Tests/ValueTests.cs`
- `Assets/ChaosRL/Tests/DistTests.cs`
- `Assets/ChaosRL/Tests/AdamOptimizerTests.cs`
- `Assets/ChaosRL/Tests/L2RegularizerTests.cs`
- `Assets/ChaosRL/Tests/MLPTests.cs`
- `Assets/ChaosRL/Tests/AutodiffBenchmarks.cs`
- `Assets/ChaosRL/AssemblyInfo.cs`
- `ProjectSettings/ProjectVersion.txt`
