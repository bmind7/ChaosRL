# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
