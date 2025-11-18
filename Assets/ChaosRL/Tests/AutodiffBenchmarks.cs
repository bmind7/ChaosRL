using System;
using System.Diagnostics;
using NUnit.Framework;
using ChaosRL;
using Debug = UnityEngine.Debug;

namespace ChaosRL.Tests
{
    /// <summary>
    /// Performance benchmarks comparing scalar Value vs vectorized Tensor operations.
    /// Run in Release mode for accurate results.
    /// </summary>
    public class AutodiffBenchmarks
    {
        private const int WarmupIterations = 10;
        private const int BenchmarkIterations = 100;

        //------------------------------------------------------------------
        // 1D Benchmarks
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Addition_1D_256()
        {
            BenchmarkOperation( new[] { 256 }, "Addition" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Addition_1D_512()
        {
            BenchmarkOperation( new[] { 512 }, "Addition" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Addition_1D_1024()
        {
            BenchmarkOperation( new[] { 1024 }, "Addition" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Multiplication_1D_256()
        {
            BenchmarkOperation( new[] { 256 }, "Multiplication" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Multiplication_1D_512()
        {
            BenchmarkOperation( new[] { 512 }, "Multiplication" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Multiplication_1D_1024()
        {
            BenchmarkOperation( new[] { 1024 }, "Multiplication" );
        }
        //------------------------------------------------------------------
        // 2D Benchmarks
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Addition_2D_16x16()
        {
            BenchmarkOperation( new[] { 16, 16 }, "Addition" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Addition_2D_32x32()
        {
            BenchmarkOperation( new[] { 32, 32 }, "Addition" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Multiplication_2D_16x16()
        {
            BenchmarkOperation( new[] { 16, 16 }, "Multiplication" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Multiplication_2D_32x32()
        {
            BenchmarkOperation( new[] { 32, 32 }, "Multiplication" );
        }
        //------------------------------------------------------------------
        // 3D Benchmarks
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Addition_3D_8x8x8()
        {
            BenchmarkOperation( new[] { 8, 8, 8 }, "Addition" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Addition_3D_16x16x4()
        {
            BenchmarkOperation( new[] { 16, 16, 4 }, "Addition" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Multiplication_3D_8x8x8()
        {
            BenchmarkOperation( new[] { 8, 8, 8 }, "Multiplication" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Multiplication_3D_16x16x4()
        {
            BenchmarkOperation( new[] { 16, 16, 4 }, "Multiplication" );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ComplexForward_1D_256()
        {
            BenchmarkComplexForward( new[] { 256 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ComplexForward_1D_512()
        {
            BenchmarkComplexForward( new[] { 512 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ComplexForward_1D_1024()
        {
            BenchmarkComplexForward( new[] { 1024 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ComplexForward_2D_32x32()
        {
            BenchmarkComplexForward( new[] { 32, 32 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ComplexForward_3D_16x16x4()
        {
            BenchmarkComplexForward( new[] { 16, 16, 4 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ForwardBackward_1D_256()
        {
            BenchmarkForwardBackward( new[] { 256 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ForwardBackward_1D_512()
        {
            BenchmarkForwardBackward( new[] { 512 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ForwardBackward_1D_1024()
        {
            BenchmarkForwardBackward( new[] { 1024 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ForwardBackward_2D_32x32()
        {
            BenchmarkForwardBackward( new[] { 32, 32 } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_ForwardBackward_3D_16x16x4()
        {
            BenchmarkForwardBackward( new[] { 16, 16, 4 } );
        }
        //------------------------------------------------------------------
        // Matrix Multiplication Benchmarks
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_MatMul_32x16_16x32()
        {
            BenchmarkMatMul( 32, 16, 32 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_MatMul_Forward_32x16_16x128()
        {
            BenchmarkMatMulForwardBackward( 32, 16, 128 );
        }
        //------------------------------------------------------------------

        // Tensor-Only MatMul Benchmarks (various sizes)
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMul_64x64x64()
        {
            BenchmarkTensorMatMulOnly( 64, 64, 64 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMul_128x128x128()
        {
            BenchmarkTensorMatMulOnly( 128, 128, 128 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMul_256x256x256()
        {
            BenchmarkTensorMatMulOnly( 256, 256, 256 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMul_512x512x512()
        {
            BenchmarkTensorMatMulOnly( 512, 512, 512 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMul_1024x1024x1024()
        {
            BenchmarkTensorMatMulOnly( 1024, 1024, 1024 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMul_2048x2048x2048()
        {
            BenchmarkTensorMatMulOnly( 2048, 2048, 2048 );
        }
        //------------------------------------------------------------------

        // Tensor-Only MatMul Benchmarks with Backward (various sizes)
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMulBackward_64x64x64()
        {
            BenchmarkTensorMatMulWithBackward( 64, 64, 64 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMulBackward_128x128x128()
        {
            BenchmarkTensorMatMulWithBackward( 128, 128, 128 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMulBackward_256x256x256()
        {
            BenchmarkTensorMatMulWithBackward( 256, 256, 256 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMulBackward_512x512x512()
        {
            BenchmarkTensorMatMulWithBackward( 512, 512, 512 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMulBackward_1024x1024x1024()
        {
            BenchmarkTensorMatMulWithBackward( 1024, 1024, 1024 );
        }
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_TensorMatMulBackward_2048x2048x2048()
        {
            BenchmarkTensorMatMulWithBackward( 2048, 2048, 2048 );
        }
        //------------------------------------------------------------------

        // Allocation Overhead Benchmarks
        //------------------------------------------------------------------
        [Test]
        public void Benchmark_Value_Allocation_Overhead()
        {
            const int size = 1024;
            const int iterations = 1000;

            Debug.Log( "\n=== Value Allocation Overhead Analysis ===" );
            Debug.Log( $"Size: {size}, Iterations: {iterations}\n" );

            // Benchmark 1: Just allocation
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int iter = 0; iter < iterations; iter++)
            {
                var values = new Value[ size ];
                for (int i = 0; i < size; i++)
                    values[ i ] = new Value( i * 0.1f );
            }
            sw.Stop();
            var allocTime = sw.Elapsed.TotalMilliseconds;
            Debug.Log( $"Allocation only: {allocTime:F3} ms" );

            // Benchmark 2: Allocation + simple operation
            sw.Restart();
            for (int iter = 0; iter < iterations; iter++)
            {
                var values1 = new Value[ size ];
                var values2 = new Value[ size ];
                for (int i = 0; i < size; i++)
                {
                    values1[ i ] = new Value( i * 0.1f );
                    values2[ i ] = new Value( (size - i) * 0.1f );
                }

                for (int i = 0; i < size; i++)
                    _ = values1[ i ] + values2[ i ];
            }
            sw.Stop();
            var totalTime = sw.Elapsed.TotalMilliseconds;
            Debug.Log( $"Allocation + Addition: {totalTime:F3} ms" );

            // Total three allocations happened values1, values2, and Value during sumation
            var computeTime = totalTime - 3 * allocTime;
            var allocPercent = (3 * allocTime / totalTime) * 100;
            var computePercent = (computeTime / totalTime) * 100;

            Debug.Log( $"Computation only: {computeTime:F3} ms" );
            Debug.Log( $"Allocation overhead: {allocPercent:F1}%" );
            Debug.Log( $"Computation: {computePercent:F1}%" );

            // Compare with Tensor (pre-allocated)
            sw.Restart();
            for (int iter = 0; iter < iterations; iter++)
            {
                var t1 = new Tensor( new int[] { size } );
                var t2 = new Tensor( new int[] { size } );
                for (int i = 0; i < size; i++)
                {
                    t1.Data[ i ] = i * 0.1f;
                    t2.Data[ i ] = (size - i) * 0.1f;
                }
                _ = t1 + t2;
            }
            sw.Stop();
            var tensorTime = sw.Elapsed.TotalMilliseconds;

            Debug.Log( $"Tensor (with allocation): {tensorTime:F3} ms" );
            Debug.Log( $"Speedup over Value: {totalTime / tensorTime:F2}x" );
        }
        //------------------------------------------------------------------
        private void BenchmarkOperation( int[] shape, string operation )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            Debug.Log( $"\n=== {operation} Benchmark (Shape: {string.Join( "x", shape )}, Size: {size}) ===" );

            // Warmup
            for (int i = 0; i < WarmupIterations; i++)
            {
                RunValueOperation( shape, operation );
                RunTensorOperation( shape, operation );
            }

            // Benchmark Value (scalar)
            var swValue = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunValueOperation( shape, operation );
            }
            swValue.Stop();

            // Benchmark Tensor (vectorized)
            var swTensor = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunTensorOperation( shape, operation );
            }
            swTensor.Stop();

            var valueMs = swValue.Elapsed.TotalMilliseconds;
            var tensorMs = swTensor.Elapsed.TotalMilliseconds;
            var speedup = valueMs / tensorMs;

            Debug.Log( $"Value  (scalar):     {valueMs:F3} ms ({valueMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Tensor (vectorized): {tensorMs:F3} ms ({tensorMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Speedup: {speedup:F2}x" );
        }
        //------------------------------------------------------------------
        private void RunValueOperation( int[] shape, string operation )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            var values1 = new Value[ size ];
            var values2 = new Value[ size ];

            for (int i = 0; i < size; i++)
            {
                values1[ i ] = new Value( i * 0.1f );
                values2[ i ] = new Value( (size - i) * 0.1f );
            }

            switch (operation)
            {
                case "Addition":
                    for (int i = 0; i < size; i++)
                        _ = values1[ i ] + values2[ i ];
                    break;
                case "Multiplication":
                    for (int i = 0; i < size; i++)
                        _ = values1[ i ] * values2[ i ];
                    break;
            }
        }
        //------------------------------------------------------------------
        private void RunTensorOperation( int[] shape, string operation )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            var data1 = new float[ size ];
            var data2 = new float[ size ];

            for (int i = 0; i < size; i++)
            {
                data1[ i ] = i * 0.1f;
                data2[ i ] = (size - i) * 0.1f;
            }

            var tensor1 = new Tensor( shape, data1 );
            var tensor2 = new Tensor( shape, data2 );

            switch (operation)
            {
                case "Addition":
                    _ = tensor1 + tensor2;
                    break;
                case "Multiplication":
                    _ = tensor1 * tensor2;
                    break;
            }
        }
        //------------------------------------------------------------------
        private void BenchmarkComplexForward( int[] shape )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            Debug.Log( $"\n=== Complex Forward Pass (Shape: {string.Join( "x", shape )}, Size: {size}) ===" );

            // Warmup
            for (int i = 0; i < WarmupIterations; i++)
            {
                RunValueComplexForward( shape );
                RunTensorComplexForward( shape );
            }

            // Benchmark Value
            var swValue = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunValueComplexForward( shape );
            }
            swValue.Stop();

            // Benchmark Tensor
            var swTensor = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunTensorComplexForward( shape );
            }
            swTensor.Stop();

            var valueMs = swValue.Elapsed.TotalMilliseconds;
            var tensorMs = swTensor.Elapsed.TotalMilliseconds;
            var speedup = valueMs / tensorMs;

            Debug.Log( $"Value  (scalar):     {valueMs:F3} ms ({valueMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Tensor (vectorized): {tensorMs:F3} ms ({tensorMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Speedup: {speedup:F2}x" );
        }
        //------------------------------------------------------------------
        private void RunValueComplexForward( int[] shape )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            // Simulate: y = tanh(x * w + b)
            var x = new Value[ size ];
            var w = new Value[ size ];
            var b = new Value[ size ];

            for (int i = 0; i < size; i++)
            {
                x[ i ] = new Value( i * 0.01f );
                w[ i ] = new Value( 0.5f );
                b[ i ] = new Value( 0.1f );
            }

            for (int i = 0; i < size; i++)
            {
                _ = (x[ i ] * w[ i ] + b[ i ]).Tanh();
            }
        }
        //------------------------------------------------------------------
        private void RunTensorComplexForward( int[] shape )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            // Simulate: y = tanh(x * w + b)
            var xData = new float[ size ];
            var wData = new float[ size ];
            var bData = new float[ size ];

            for (int i = 0; i < size; i++)
            {
                xData[ i ] = i * 0.01f;
                wData[ i ] = 0.5f;
                bData[ i ] = 0.1f;
            }

            var x = new Tensor( shape, xData );
            var w = new Tensor( shape, wData );
            var b = new Tensor( shape, bData );

            _ = (x * w + b).Tanh();
        }
        //------------------------------------------------------------------
        private void BenchmarkForwardBackward( int[] shape )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            Debug.Log( $"\n=== Forward + Backward Pass (Shape: {string.Join( "x", shape )}, Size: {size}) ===" );

            // Warmup
            for (int i = 0; i < WarmupIterations; i++)
            {
                RunValueForwardBackward( shape );
                RunTensorForwardBackward( shape );
            }

            // Benchmark Value
            var swValue = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunValueForwardBackward( shape );
            }
            swValue.Stop();

            // Benchmark Tensor
            var swTensor = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunTensorForwardBackward( shape );
            }
            swTensor.Stop();

            var valueMs = swValue.Elapsed.TotalMilliseconds;
            var tensorMs = swTensor.Elapsed.TotalMilliseconds;
            var speedup = valueMs / tensorMs;

            Debug.Log( $"Value  (scalar):     {valueMs:F3} ms ({valueMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Tensor (vectorized): {tensorMs:F3} ms ({tensorMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Speedup: {speedup:F2}x" );
        }
        //------------------------------------------------------------------
        private void RunValueForwardBackward( int[] shape )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            var x = new Value[ size ];
            var w = new Value[ size ];

            for (int i = 0; i < size; i++)
            {
                x[ i ] = new Value( i * 0.01f );
                w[ i ] = new Value( 0.5f );
            }

            Value sum = 0f;
            for (int i = 0; i < size; i++)
            {
                sum = sum + (x[ i ] * w[ i ]).Tanh();
            }

            sum.Backward();
        }
        //------------------------------------------------------------------
        private void RunTensorForwardBackward( int[] shape )
        {
            int size = 1;
            foreach (var dim in shape)
                size *= dim;

            var xData = new float[ size ];
            var wData = new float[ size ];

            for (int i = 0; i < size; i++)
            {
                xData[ i ] = i * 0.01f;
                wData[ i ] = 0.5f;
            }

            var x = new Tensor( new[] { size }, xData );
            var w = new Tensor( new[] { size }, wData );

            var result = (x * w).Tanh();

            // Sum reduction (manual for now)
            float sum = 0f;
            for (int i = 0; i < result.Size; i++)
                sum += result.Data[ i ];

            var loss = new Tensor( sum );
            loss.Backward();
        }
        //------------------------------------------------------------------
        private void BenchmarkMatMul( int M, int K, int N )
        {
            Debug.Log( $"\n=== MatMul Benchmark ({M}×{K}) @ ({K}×{N}) -> ({M}×{N}) ===" );

            // Warmup
            for (int i = 0; i < WarmupIterations; i++)
            {
                RunValueMatMul( M, K, N );
                RunTensorMatMul( M, K, N );
            }

            // Benchmark Value (scalar)
            var swValue = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunValueMatMul( M, K, N );
            }
            swValue.Stop();

            // Benchmark Tensor (vectorized)
            var swTensor = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunTensorMatMul( M, K, N );
            }
            swTensor.Stop();

            var valueMs = swValue.Elapsed.TotalMilliseconds;
            var tensorMs = swTensor.Elapsed.TotalMilliseconds;
            var speedup = valueMs / tensorMs;

            Debug.Log( $"Value  (scalar):     {valueMs:F3} ms ({valueMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Tensor (vectorized): {tensorMs:F3} ms ({tensorMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Speedup: {speedup:F2}x" );
        }
        //------------------------------------------------------------------
        private void RunValueMatMul( int M, int K, int N )
        {
            // Simulate matrix multiplication using scalar Values
            var a = new Value[ M * K ];
            var b = new Value[ K * N ];

            for (int i = 0; i < M * K; i++)
                a[ i ] = new Value( i * 0.01f );

            for (int i = 0; i < K * N; i++)
                b[ i ] = new Value( i * 0.01f );

            // Manual matrix multiplication
            var result = new Value[ M * N ];
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Value sum = 0f;
                    for (int k = 0; k < K; k++)
                    {
                        sum = sum + (a[ i * K + k ] * b[ k * N + j ]);
                    }
                    result[ i * N + j ] = sum;
                }
            }
        }
        //------------------------------------------------------------------
        private void RunTensorMatMul( int M, int K, int N )
        {
            var aData = new float[ M * K ];
            var bData = new float[ K * N ];

            for (int i = 0; i < M * K; i++)
                aData[ i ] = i * 0.01f;

            for (int i = 0; i < K * N; i++)
                bData[ i ] = i * 0.01f;

            var a = new Tensor( new[] { M, K }, aData );
            var b = new Tensor( new[] { K, N }, bData );

            _ = a.MatMul( b );
        }
        //------------------------------------------------------------------
        private void BenchmarkMatMulForwardBackward( int M, int K, int N )
        {
            Debug.Log( $"\n=== MatMul Forward+Backward ({M}×{K}) @ ({K}×{N}) ===" );

            // Warmup
            for (int i = 0; i < WarmupIterations; i++)
            {
                RunValueMatMulForwardBackward( M, K, N );
                RunTensorMatMulForwardBackward( M, K, N );
            }

            // Benchmark Value
            var swValue = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunValueMatMulForwardBackward( M, K, N );
            }
            swValue.Stop();

            // Benchmark Tensor
            var swTensor = Stopwatch.StartNew();
            for (int i = 0; i < BenchmarkIterations; i++)
            {
                RunTensorMatMulForwardBackward( M, K, N );
            }
            swTensor.Stop();

            var valueMs = swValue.Elapsed.TotalMilliseconds;
            var tensorMs = swTensor.Elapsed.TotalMilliseconds;
            var speedup = valueMs / tensorMs;

            Debug.Log( $"Value  (scalar):     {valueMs:F3} ms ({valueMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Tensor (vectorized): {tensorMs:F3} ms ({tensorMs / BenchmarkIterations:F4} ms/iter)" );
            Debug.Log( $"Speedup: {speedup:F2}x" );
        }
        //------------------------------------------------------------------
        private void RunValueMatMulForwardBackward( int M, int K, int N )
        {
            var a = new Value[ M * K ];
            var b = new Value[ K * N ];

            for (int i = 0; i < M * K; i++)
                a[ i ] = new Value( i * 0.01f );

            for (int i = 0; i < K * N; i++)
                b[ i ] = new Value( i * 0.01f );

            // Matrix multiplication
            var result = new Value[ M * N ];
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Value sum = 0f;
                    for (int k = 0; k < K; k++)
                    {
                        sum = sum + (a[ i * K + k ] * b[ k * N + j ]);
                    }
                    result[ i * N + j ] = sum;
                }
            }

            // Sum reduction and backward
            Value total = 0f;
            for (int i = 0; i < M * N; i++)
                total = total + result[ i ];

            total.Backward();
        }
        //------------------------------------------------------------------
        private void RunTensorMatMulForwardBackward( int M, int K, int N )
        {
            var aData = new float[ M * K ];
            var bData = new float[ K * N ];

            for (int i = 0; i < M * K; i++)
                aData[ i ] = i * 0.01f;

            for (int i = 0; i < K * N; i++)
                bData[ i ] = i * 0.01f;

            var a = new Tensor( new[] { M, K }, aData );
            var b = new Tensor( new[] { K, N }, bData );

            var result = a.MatMul( b );

            // Sum reduction
            float sum = 0f;
            for (int i = 0; i < result.Size; i++)
                sum += result.Data[ i ];

            var loss = new Tensor( sum );
            loss.Backward();
        }
        //------------------------------------------------------------------
        private void BenchmarkTensorMatMulOnly( int M, int K, int N )
        {
            const int warmup = 3;
            const int iterations = 10;

            Debug.Log( $"\n=== Tensor MatMul Benchmark: {M}×{K} @ {K}×{N} ===" );

            // Pre-allocate and populate tensors once
            var a = new Tensor( new[] { M, K } );
            var b = new Tensor( new[] { K, N } );
            for (int j = 0; j < M * K; j++)
                a.Data[ j ] = j * 0.01f;
            for (int j = 0; j < K * N; j++)
                b.Data[ j ] = j * 0.01f;

            // Warmup
            for (int i = 0; i < warmup; i++)
            {
                _ = a.MatMul( b );
            }

            // Benchmark MatMul only
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                _ = a.MatMul( b );
            }
            sw.Stop();

            var avgTime = sw.Elapsed.TotalMilliseconds / iterations;
            var flops = 2.0 * M * K * N; // multiply-add counts as 2 operations
            var gflops = (flops * iterations / sw.Elapsed.TotalSeconds) / 1e9;

            Debug.Log( $"Average time: {avgTime:F3} ms" );
            Debug.Log( $"Performance: {gflops:F2} GFLOPS" );
        }
        //------------------------------------------------------------------
        private void BenchmarkTensorMatMulWithBackward( int M, int K, int N )
        {
            const int warmup = 3;
            const int iterations = 10;

            Debug.Log( $"\n=== Tensor MatMul + Backward Benchmark: {M}×{K} @ {K}×{N} ===" );

            // Pre-allocate and populate tensors once
            var a = new Tensor( new[] { M, K } );
            var b = new Tensor( new[] { K, N } );
            for (int j = 0; j < M * K; j++)
                a.Data[ j ] = j * 0.01f;
            for (int j = 0; j < K * N; j++)
                b.Data[ j ] = j * 0.01f;

            // Warmup
            for (int i = 0; i < warmup; i++)
            {
                var result = a.MatMul( b );
                float sum = 0f;
                for (int k = 0; k < result.Size; k++)
                    sum += result.Data[ k ];
                var loss = new Tensor( sum );
                loss.Backward();
                a.ZeroGrad();
                b.ZeroGrad();
            }

            // Benchmark MatMul + Backward
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var result = a.MatMul( b );
                float sum = 0f;
                for (int k = 0; k < result.Size; k++)
                    sum += result.Data[ k ];
                var loss = new Tensor( sum );
                loss.Backward();
                a.ZeroGrad();
                b.ZeroGrad();
            }
            sw.Stop();

            var avgTime = sw.Elapsed.TotalMilliseconds / iterations;
            // Forward: 2*M*K*N, Backward: ~4*M*K*N (two matmuls for gradients)
            var flops = 6.0 * M * K * N;
            var gflops = (flops * iterations / sw.Elapsed.TotalSeconds) / 1e9;

            Debug.Log( $"Average time: {avgTime:F3} ms" );
            Debug.Log( $"Performance: {gflops:F2} GFLOPS" );
        }
        //------------------------------------------------------------------
    }
}
