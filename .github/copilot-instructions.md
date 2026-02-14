# ChaosRL — Copilot Instructions

## Project Overview
Unity 6 (C#) project implementing PPO reinforcement learning from scratch with a custom autodiff engine. No external ML frameworks — everything (tensor math, autograd, neural nets, PPO) is hand-rolled. Uses Unity Burst/Jobs for CPU-vectorised compute.

## Architecture
- **Autodiff Core** (`Assets/ChaosRL/Autodiff/`): `Tensor` (row-major N-D with dynamic autograd graph), `TensorStorage` (ref-counted buffer wrapper), `ITensorBackend` (device-agnostic compute interface), `CpuBackend` (Burst job implementation of `ITensorBackend`), `TensorDevice` (CPU/GPU enum), `TensorOps` (MatMul orchestration & batch-size helpers), `TensorJobs` (Burst-compiled kernels). `Value` is the legacy scalar autodiff path kept for tests/benchmarks.
- **NN Stack** (`Assets/ChaosRL/NN/`): `Layer` (dense + optional Tanh), `MLP` (multi-layer perceptron), `AdamOptimizer`, `L2Regularizer`.
- **RL Training** (`Assets/ChaosRL/RL/`): `Academy` (singleton MonoBehaviour, PPO loop with GAE, rollout buffers, minibatch SGD), `ArenaGridSpawner`.
- **Agent** (`Assets/Src/ChaosAgent.cs`): Physics-side agent collecting observations and applying actions. Lives in `Assembly-CSharp`, not the `ChaosRL` assembly.
- **Utils** (`Assets/ChaosRL/Utils/`): `Dist` (Gaussian with reparameterised sampling), `RandomHub` (thread-safe RNG), `Utils` (minibatch extraction).

All production code uses the flat `ChaosRL` namespace; tests use `ChaosRL.Tests`.

## Coding Conventions
- **Allman braces** — opening `{` on its own line.
- **Spaces inside parentheses and brackets**: `Method( arg1, arg2 )`, `array[ i ]`, `new[] { 1, 2 }`.
- **Section dividers**: `//------------------------------------------------------------------` between members/methods.
- **Private fields**: `_camelCase` with underscore prefix. Public fields: `PascalCase`.
- **XML doc comments** (`/// <summary>`) on public API surface. Inline `//` for implementation notes.
- **`// TODO:`** and **`// Note:`** for future work and domain caveats.
- **No sub-namespaces** despite folder structure — all production types are in `namespace ChaosRL`.

## Tensor Operator Pattern
Every operator/activation follows this structure — maintain it when adding new ops:
1. Validate shapes (e.g., `CanBroadcastModulo`) and device match (`ValidateDeviceMatch`).
2. Allocate result tensor, passing parents as children: `new Tensor( shape, children: new[] { a, b }, "opName" )`.
3. Dispatch forward compute through `result.Backend` (e.g., `result.Backend.Add(...)`).
4. Set `result.RequiresGrad` from children; early-return if no grad needed.
5. Assign `result._backward` as an `Action` closure that calls the appropriate `Backend` backward method (e.g., `Backend.AddBackward(...)`).
6. Return result.

## Burst Jobs
- **Job structs** go in `TensorJobs.cs`; **element-wise scheduling** goes in `CpuBackend.cs`; **MatMul orchestration** stays in `TensorOps.cs`.
- Attribute: `[BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low, DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]`.
- Use `IJobParallelFor` with `[ReadOnly]`/`[WriteOnly]`/`[NativeDisableParallelForRestriction]` annotations.
- Batch size via `TensorOps.GetBatchSize()` based on `JobsUtility.JobWorkerCount`.

## Memory Management
- `Tensor` implements `IDisposable`; backing memory lives in `TensorStorage` (ref-counted `NativeArray<float>`, `Allocator.Persistent`).
- View ops (`Reshape`, `Squeeze`, `Unsqueeze`) share `TensorStorage` via `AddRef()`/`Release()` — the buffer is freed when the ref-count reaches zero.
- Temporary job buffers use `Allocator.TempJob` and must be disposed after use or via `NativeArray.Dispose(JobHandle)`.

## Backend Abstraction
- `ITensorBackend` defines device-agnostic forward and backward kernels. Each `Tensor` caches its `Backend` at creation via `Tensor.ResolveBackend()`.
- `CpuBackend` implements every kernel by scheduling Burst jobs on `TensorStorage.Buffer` (`NativeArray<float>`).
- `TensorDevice.GPU` is declared but not yet implemented — `ResolveBackend` throws until a GPU backend is registered.
- When adding a new op, add its method to `ITensorBackend`, implement it in `CpuBackend`, and call it from the `Tensor` operator / `_backward` closure.

## Tests
- NUnit-based in `Assets/ChaosRL/Tests/` (editor test assembly `ChaosRLTests.asmdef`).
- Naming: `MethodUnderTest_Scenario_ExpectedBehavior` (e.g., `MatMul_2x3_Times_3x2_CorrectResult`).
- Float assertions: `Assert.That( value, Is.EqualTo( expected ).Within( 1e-6 ) )`.
- Run via Unity Test Runner (Edit Mode tests).

## Known Constraints
- `MatMul` supports only 2D tensors.
- Broadcasting limited to tail-shape matching via `CanBroadcastModulo`.
- `Normalize` works only on whole tensor, dim=0, or last dimension.
- `ExpandLast` only adds/repeats a new last dimension.
- `GetMinibatch` supports only 1D/2D tensors.
- GPU backend (`TensorDevice.GPU`) is declared but not yet implemented.

## Building & Benchmarking
- Open in Unity `6000.2.10f1`. Scene: `Dojo` (training), `MatMultBenchmark` (perf).
- For benchmarks, build with **IL2CPP** backend to get production-representative numbers.
- Policy typically converges around 200k–300k steps. If entropy drops below 1.0 and rewards stay below 100, restart.
