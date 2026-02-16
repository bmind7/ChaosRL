# ChaosRL — Copilot Instructions

For full project details — architecture, coding conventions, operator patterns, memory management, backend abstraction, tests, and known constraints — see [AGENTS.md](../AGENTS.md).

## Quick Reference
- **Unity 6 / C#** — hand-rolled PPO with custom autodiff, Burst/Jobs for compute.
- **Namespace**: `ChaosRL` (production), `ChaosRL.Tests` (tests).
- **Style**: Allman braces, spaces inside parens/brackets, `_camelCase` private fields, `PascalCase` public fields.
- **New ops**: follow the Tensor Operator Pattern (validate → allocate → forward → grad check → `_backward` closure).
- **New Burst jobs**: struct in `TensorJobs.cs`, scheduling in `CpuBackend.cs`, MatMul orchestration in `TensorOps.cs`.
- **New backend methods**: add to `ITensorBackend` → implement in `CpuBackend` → call from `Tensor`.

