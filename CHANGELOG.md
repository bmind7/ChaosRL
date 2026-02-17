# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-16

### Added
- Backend abstraction layer for tensor compute: `ITensorBackend` plus CPU implementation `CpuBackend`
- `TensorStorage` as a ref-counted buffer wrapper for tensor data/grad lifetime management
- `TensorDevice` enum (`CPU`, `GPU`) and device-aware tensor construction paths
- `TensorScope` for deterministic scoped tensor disposal
- `TensorScopedTestBase` to enforce per-test tensor scope cleanup
- `Assets/ChaosRL/AssemblyInfo.cs` with `InternalsVisibleTo("ChaosRLTests")`

### Changed
- Renamed `TensorOps` to `CpuMatMulOps` - explicitly scoped as an internal CPU-specific MatMul orchestration helper for `CpuBackend`, removing ambiguity with the backend abstraction layer
- Consolidated `GetBatchSize()` into `CpuBackend` as a single `internal static` helper - both `CpuBackend` and `CpuMatMulOps` now share the same batch-size policy
- Tightened `CpuMatMulOps` visibility to `internal` (class, constants, and scheduling methods) - it is an implementation detail of `CpuBackend`, not part of the public API
- `Tensor` now dispatches operations through a cached backend and validates device compatibility per operation
- View operations (`Reshape`, `Squeeze`, `Unsqueeze`) share `TensorStorage` with reference counting
- `Tensor.Backward()` seeds gradients through backend utilities; scalar readback uses `Tensor.Scalar`
- `Utils.GetMinibatch(...)` now uses backend `Gather(...)` and preserves source tensor device
- `Layer`, `MLP`, and `AdamOptimizer` now implement `IDisposable`
- Performance benchmark fixture is explicit/manual (`AutodiffBenchmarks`) and updated for explicit tensor disposal
- Technical overview document moved to `Doc/TECH_OVERVIEW.md`

### Fixed
- CPU backend backward paths avoid race conditions for aliasing and broadcast accumulation cases

## [0.3.0] - 2026-02-11

### Added
- GEBP (Generalized Efficient Panel-Block) MatMul kernel with MR=6, NR=16 micro-tiles
- Kc blocking (KC=256) for L1 cache residency in GEBP kernel
- `TensorOps` static class — unified Burst job scheduling orchestration
- `PackBPanelScalarParallelJob` — packs B into NR-wide column-panel layout for GEBP
- `MatMulGebpScalarParallelJob` — Burst GEBP micro-kernel parallelized over MR-row groups
- `SumReductionJob` — Burst-compiled sum reduction (replaces managed loop)
- `AddScalarParallelJob` — Burst-compiled scalar broadcast for Sum backward

### Changed
- MatMul 2048² performance: **254 → 546 GFLOP/s** (~2.15× speedup)
- `Tensor.MatMul` forward/backward delegates to `TensorOps.ScheduleMatMul` (auto-selects GEBP or naive based on dimension threshold)
- `Tensor.ZeroGrad()` uses `UnsafeUtility.MemClear` instead of managed loop
- `Tensor.Backward()` seed uses `UnsafeUtility.MemCpyReplicate` instead of managed loop
- `Tensor.Sum()` forward/backward replaced with Burst jobs
- `TransposeParallelJob` replaced by `TransposeTiledParallelJob` with TILE=32 cache blocking
- `MatMulJob` (IJob) replaced by parallelized GEBP and naive kernels

### Removed
- `TransposeParallelJob` (superseded by `TransposeTiledParallelJob`)
- `MatMulJob` (superseded by `MatMulGebpScalarParallelJob`)

## [0.2.0] - 2025-11-20

### Added
- Improved performance for regular ops **40x-100x** for MatMul **200x-400x**
- **Data-oriented Tensor class** with vectorized operations (no SIMD yet)
- **Broadcasting support** for tensor operations (limited to tail shape matching)
- **Tensor-based neural network** modules and optimizer
- **PPO implementation using Tensors** instead of scalar autodiff
- Tensor operations: `MatMul`, `Sum`, `Mean`, `Max`, `Sqrt`, `Normalize`
- Tensor manipulation: `Reshape`, `Slice`, `Squeeze`, `Unsqueeze`, `ExpandLast`
- Multi-dimensional indexer for Tensor class
- `RequiresGrad` flag for gradient control in tensors
- Performance benchmarks for Tensor vs Value operations
- Unit tests for tensor operations and MLP learning capability

### Changed
- Old implementation of PPO based on Value class is removed

### Known Issues
- Broadcasting is limited and only supports shape mismatch where tail shapes match
- `Normalize()` method only works on whole tensor, dim=0, or last dimension

## [0.1.0] - Initial Release

### Added
- Basic PPO implementation with scalar autodiff engine
- Ball balance demo environment
- Single-threaded training in Unity
