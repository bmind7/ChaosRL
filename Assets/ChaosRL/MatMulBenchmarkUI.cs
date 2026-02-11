using System;
using System.Collections;
using System.Diagnostics;
using System.Text;

using Unity.Collections;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

using UnityEngine;

namespace ChaosRL
{
    public class MatMulBenchmarkUI : MonoBehaviour
    {
        //------------------------------------------------------------------
        private string _benchmarkResults = "Press 'Run Benchmark' to start...";
        private Vector2 _scrollPosition;
        private bool _isRunning = false;
        //------------------------------------------------------------------
        private void OnGUI()
        {
            GUILayout.BeginArea( new Rect( 10, 10, Screen.width - 20, Screen.height - 20 ) );

            GUILayout.BeginHorizontal();
            if (GUILayout.Button( "Run Benchmark", GUILayout.Width( 150 ), GUILayout.Height( 40 ) ))
            {
                if (!_isRunning)
                {
                    _isRunning = true;
                    _benchmarkResults = "Running benchmarks... (UI will update progressively)\n";
                    StartCoroutine( RunBenchmarks() );
                }
            }
            if (GUILayout.Button( "Run Kernel Comparison", GUILayout.Width( 200 ), GUILayout.Height( 40 ) ))
            {
                if (!_isRunning)
                {
                    _isRunning = true;
                    _benchmarkResults = "Running kernel comparison... (UI will update progressively)\n";
                    StartCoroutine( RunKernelComparison() );
                }
            }
            GUILayout.EndHorizontal();

            GUILayout.Space( 10 );

            _scrollPosition = GUILayout.BeginScrollView( _scrollPosition );
            GUILayout.TextArea( _benchmarkResults, GUILayout.ExpandHeight( true ) );
            GUILayout.EndScrollView();

            GUILayout.EndArea();
        }
        //------------------------------------------------------------------
        private IEnumerator RunBenchmarks()
        {
            var sb = new StringBuilder();
            sb.AppendLine( $"Benchmark started at {DateTime.Now}" );
            sb.AppendLine();

            // Include square, odd power, and mixed aspect-ratio shapes.
            var shapes = new (int M, int K, int N)[]
            {
                (64, 64, 64),
                (128, 128, 128),
                (256, 256, 256),
                (512, 512, 512),
                (768, 768, 768),
                (1024, 1024, 1024),
                (1536, 1536, 1536),
                (2048, 2048, 2048),
                (1024, 2048, 1024),
                (2048, 1024, 2048),
            };

            sb.AppendLine( "CPU Results (MatMul Only):" );
            sb.AppendLine( new string( '-', 80 ) );
            sb.AppendLine( $"{"Matrix Size",-25} {"Avg Time (ms)",-20} {"Std Dev (ms)",-20} {"GFLOPS",-10}" );
            sb.AppendLine( new string( '-', 80 ) );
            _benchmarkResults = sb.ToString();
            yield return null;

            foreach (var shape in shapes)
            {
                RunTensorMatMulOnly( shape.M, shape.K, shape.N, sb );
                _benchmarkResults = sb.ToString();
                yield return null;
            }
            sb.AppendLine( new string( '-', 80 ) );
            sb.AppendLine();

            sb.AppendLine( "CPU Results (MatMul + Backward):" );
            sb.AppendLine( new string( '-', 80 ) );
            sb.AppendLine( $"{"Matrix Size",-25} {"Avg Time (ms)",-20} {"Std Dev (ms)",-20} {"GFLOPS",-10}" );
            sb.AppendLine( new string( '-', 80 ) );
            _benchmarkResults = sb.ToString();
            yield return null;

            foreach (var shape in shapes)
            {
                RunTensorMatMulWithBackward( shape.M, shape.K, shape.N, sb );
                _benchmarkResults = sb.ToString();
                yield return null;
            }
            sb.AppendLine( new string( '-', 80 ) );

            sb.AppendLine( "Done." );

            _benchmarkResults = sb.ToString();
            _isRunning = false;
        }
        //------------------------------------------------------------------
        private void RunTensorMatMulOnly( int M, int K, int N, StringBuilder sb )
        {
            const int warmup = 3;
            const int iterations = 10;

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
            double[] times = new double[ iterations ];
            for (int i = 0; i < iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                _ = a.MatMul( b );
                sw.Stop();
                times[ i ] = sw.Elapsed.TotalMilliseconds;
            }

            double avgTime = 0;
            foreach (var t in times) avgTime += t;
            avgTime /= iterations;

            double sumSquares = 0;
            foreach (var t in times) sumSquares += (t - avgTime) * (t - avgTime);
            double stdDev = Math.Sqrt( sumSquares / iterations );

            var flops = 2.0 * M * K * N; // multiply-add counts as 2 operations
            var gflops = (flops / (avgTime / 1000.0)) / 1e9;

            string sizeStr = $"{M}x{K} @ {K}x{N}";
            sb.AppendLine( $"{sizeStr,-25} {avgTime,-20:F3} {stdDev,-20:F3} {gflops,-10:F2}" );
        }
        //------------------------------------------------------------------
        private void RunTensorMatMulWithBackward( int M, int K, int N, StringBuilder sb )
        {
            const int warmup = 3;
            const int iterations = 10;

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
                var loss = result.Sum();
                loss.Backward();
                a.ZeroGrad();
                b.ZeroGrad();
            }

            // Benchmark MatMul + Backward
            double[] times = new double[ iterations ];
            for (int i = 0; i < iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                var result = a.MatMul( b );
                var loss = result.Sum();
                loss.Backward();
                a.ZeroGrad();
                b.ZeroGrad();
                sw.Stop();
                times[ i ] = sw.Elapsed.TotalMilliseconds;
            }

            double avgTime = 0;
            foreach (var t in times) avgTime += t;
            avgTime /= iterations;

            double sumSquares = 0;
            foreach (var t in times) sumSquares += (t - avgTime) * (t - avgTime);
            double stdDev = Math.Sqrt( sumSquares / iterations );

            // Forward: 2*M*K*N, Backward: ~4*M*K*N (two matmuls for gradients)
            var flops = 6.0 * M * K * N;
            var gflops = (flops / (avgTime / 1000.0)) / 1e9;

            string sizeStr = $"{M}x{K} @ {K}x{N}";
            sb.AppendLine( $"{sizeStr,-25} {avgTime,-20:F3} {stdDev,-20:F3} {gflops,-10:F2}" );
        }
        //------------------------------------------------------------------
        private static int ComputeBatchSize( int totalWorkItems )
        {
            int workerCount = Math.Max( 1, JobsUtility.JobWorkerCount + 1 );
            int batch = totalWorkItems / (workerCount * 4);
            return Math.Max( 1, batch );
        }
        //------------------------------------------------------------------
        private static double BenchmarkAction( int warmup, int iterations, Action action )
        {
            for (int i = 0; i < warmup; i++)
                action();

            double total = 0;
            for (int i = 0; i < iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                action();
                sw.Stop();
                total += sw.Elapsed.TotalMilliseconds;
            }
            return total / iterations;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Compares two output buffers element-wise.
        /// Returns (maxRelError, avgRelError) where relative error = |ref-test| / max(|ref|, 1e-8).
        /// </summary>
        private static (double maxRel, double avgRel) CompareOutputs(
            NativeArray<float> reference, NativeArray<float> test, int length )
        {
            double maxRel = 0;
            double sumRel = 0;
            for (int i = 0; i < length; i++)
            {
                double refVal = (double)reference[ i ];
                double testVal = (double)test[ i ];
                double absDiff = Math.Abs( refVal - testVal );
                double denom = Math.Max( Math.Abs( refVal ), 1e-8 );
                double rel = absDiff / denom;
                if (rel > maxRel) maxRel = rel;
                sumRel += rel;
            }
            return (maxRel, sumRel / length);
        }
        //------------------------------------------------------------------
        private IEnumerator RunKernelComparison()
        {
            const int warmup = 3;
            const int iterations = 10;
            // Skip painfully slow single-threaded job above this total element count.
            const long singleThreadSkipThreshold = 512L * 512 * 512;

            var sb = new StringBuilder();
            sb.AppendLine( $"Kernel Comparison started at {DateTime.Now}" );
            sb.AppendLine( "Forward-only MatMul \u2014 timing each Burst job in isolation" );
            sb.AppendLine( "Numerical accuracy compared against Single-thread reference" );
            sb.AppendLine();

            var sizes = new int[] { 64, 128, 256, 512, 1024, 2048 };

            foreach (int sz in sizes)
            {
                int M = sz, K = sz, N = sz;
                int outputLen = M * N;
                long totalElements = (long)M * K * N;
                double flops = 2.0 * M * K * N;
                string sizeStr = $"{M}x{K} @ {K}x{N}";

                sb.AppendLine( $"=== {sizeStr} ===" );
                sb.AppendLine( $"  {"Kernel",-24} {"Avg (ms)",10} {"GFLOPS",10} {"MaxRelErr",12} {"AvgRelErr",12}" );
                sb.AppendLine( $"  {new string( '-', 68 )}" );
                _benchmarkResults = sb.ToString();
                yield return null;

                // --- Allocate shared buffers ---
                var aData = new NativeArray<float>( M * K, Allocator.Persistent );
                var bData = new NativeArray<float>( K * N, Allocator.Persistent );
                var btData = new NativeArray<float>( K * N, Allocator.Persistent );
                // Reference output (from single-thread) and per-kernel output
                var cRef = new NativeArray<float>( outputLen, Allocator.Persistent );
                var cData = new NativeArray<float>( outputLen, Allocator.Persistent );

                for (int j = 0; j < M * K; j++) aData[ j ] = j * 0.001f;
                for (int j = 0; j < K * N; j++) bData[ j ] = j * 0.001f;

                // Pre-transpose B into btData (NxK)
                {
                    int tileSize = TransposeTiledParallelJob.TILE;
                    int totalTiles = ((K + tileSize - 1) / tileSize) * ((N + tileSize - 1) / tileSize);
                    new TransposeTiledParallelJob
                    {
                        Input = bData,
                        Output = btData,
                        Rows = K,
                        Cols = N
                    }.Schedule( totalTiles, Math.Max( 1, totalTiles / 8 ) ).Complete();
                }

                // Pre-pack B into column-panel layout for GEBP
                const int NR = MatMulGebpParallelJob.NR;
                int numPanels = (N + NR - 1) / NR;
                int packedSize = numPanels * K * NR;
                var packedB = new NativeArray<float>( packedSize, Allocator.Persistent );
                {
                    new PackBPanelParallelJob
                    {
                        B = bData,
                        PackedB = packedB,
                        K = K,
                        N = N
                    }.Schedule( numPanels, Math.Max( 1, numPanels / 8 ) ).Complete();
                }

                // --- Compute reference output using single-thread job (always, for accuracy comparison) ---
                // For large sizes where single-thread is too slow for full benchmark,
                // still run it once for the reference, just skip the timed iterations.
                new MatMulJob
                {
                    A = aData,
                    B = btData,
                    C = cRef,
                    M = M,
                    K = K,
                    N = N
                }.Run();

                try
                {
                    // 1. Single-threaded MatMulJob
                    if (totalElements <= singleThreadSkipThreshold)
                    {
                        double ms = BenchmarkAction( warmup, iterations, () =>
                        {
                            new MatMulJob
                            {
                                A = aData,
                                B = btData,
                                C = cData,
                                M = M,
                                K = K,
                                N = N
                            }.Run();
                        } );
                        double gf = (flops / (ms / 1000.0)) / 1e9;
                        sb.AppendLine( $"  {"Single-thread",-24} {ms,10:F3} {gf,10:F2} {"(reference)",12} {"—",12}" );
                    }
                    else
                    {
                        sb.AppendLine( $"  {"Single-thread",-24} {"(skipped)",10} {"\u2014",10} {"(reference)",12} {"\u2014",12}" );
                    }
                    _benchmarkResults = sb.ToString();

                    // 2. Naive Parallel (per-element)
                    {
                        int count = M * N;
                        int batch = ComputeBatchSize( count );
                        double ms = BenchmarkAction( warmup, iterations, () =>
                        {
                            new MatMulNaiveParallelJob
                            {
                                A = aData,
                                BT = btData,
                                C = cData,
                                M = M,
                                K = K,
                                N = N,
                                Accumulate = false
                            }.Schedule( count, batch ).Complete();
                        } );
                        double gf = (flops / (ms / 1000.0)) / 1e9;
                        var (maxErr, avgErr) = CompareOutputs( cRef, cData, outputLen );
                        sb.AppendLine( $"  {"Naive Parallel",-24} {ms,10:F3} {gf,10:F2} {maxErr,12:E2} {avgErr,12:E2}" );
                    }
                    _benchmarkResults = sb.ToString();

                    // 3. Row Parallel (per-row)
                    {
                        int batch = ComputeBatchSize( M );
                        double ms = BenchmarkAction( warmup, iterations, () =>
                        {
                            new MatMulRowParallelJob
                            {
                                A = aData,
                                BT = btData,
                                C = cData,
                                M = M,
                                K = K,
                                N = N,
                                Accumulate = false
                            }.Schedule( M, batch ).Complete();
                        } );
                        double gf = (flops / (ms / 1000.0)) / 1e9;
                        var (maxErr, avgErr) = CompareOutputs( cRef, cData, outputLen );
                        sb.AppendLine( $"  {"Row Parallel",-24} {ms,10:F3} {gf,10:F2} {maxErr,12:E2} {avgErr,12:E2}" );
                    }
                    _benchmarkResults = sb.ToString();

                    // 4. Blocked Parallel (tile-parallel)
                    {
                        int tilesM = (M + MatMulBlockedParallelJob.TM - 1) / MatMulBlockedParallelJob.TM;
                        int tilesN = (N + MatMulBlockedParallelJob.TN - 1) / MatMulBlockedParallelJob.TN;
                        int totalTiles = tilesM * tilesN;
                        int batch = ComputeBatchSize( totalTiles );
                        double ms = BenchmarkAction( warmup, iterations, () =>
                        {
                            new MatMulBlockedParallelJob
                            {
                                A = aData,
                                BT = btData,
                                C = cData,
                                M = M,
                                K = K,
                                N = N,
                                Accumulate = false
                            }.Schedule( totalTiles, batch ).Complete();
                        } );
                        double gf = (flops / (ms / 1000.0)) / 1e9;
                        var (maxErr, avgErr) = CompareOutputs( cRef, cData, outputLen );
                        sb.AppendLine( $"  {"Blocked",-24} {ms,10:F3} {gf,10:F2} {maxErr,12:E2} {avgErr,12:E2}" );
                    }
                    _benchmarkResults = sb.ToString();

                    // 5. GEBP Scalar (Burst auto-vectorized pack + matmul, no manual SIMD)
                    {
                        // Pack B using the scalar packer (no intrinsics).
                        const int NR_S = PackBPanelScalarParallelJob.NR;
                        int numPanelsS = (N + NR_S - 1) / NR_S;
                        int packedSizeS = numPanelsS * K * NR_S;
                        var packedBScalar = new NativeArray<float>( packedSizeS, Allocator.TempJob,
                            NativeArrayOptions.ClearMemory );
                        new PackBPanelScalarParallelJob
                        {
                            B = bData,
                            PackedB = packedBScalar,
                            K = K,
                            N = N
                        }.Schedule( numPanelsS, Math.Max( 1, numPanelsS / 8 ) ).Complete();

                        int rowGroups = (M + MatMulGebpScalarParallelJob.MR - 1) / MatMulGebpScalarParallelJob.MR;
                        int batch = ComputeBatchSize( rowGroups );
                        double ms = BenchmarkAction( warmup, iterations, () =>
                        {
                            new MatMulGebpScalarParallelJob
                            {
                                A = aData,
                                PackedB = packedBScalar,
                                C = cData,
                                M = M,
                                K = K,
                                N = N,
                                Accumulate = false
                            }.Schedule( rowGroups, batch ).Complete();
                        } );
                        double gf = (flops / (ms / 1000.0)) / 1e9;
                        var (maxErr, avgErr) = CompareOutputs( cRef, cData, outputLen );
                        sb.AppendLine( $"  {"GEBP (Burst Auto)",-24} {ms,10:F3} {gf,10:F2} {maxErr,12:E2} {avgErr,12:E2}" );
                        packedBScalar.Dispose();
                    }
                    _benchmarkResults = sb.ToString();

                    // 6. GEBP FMA (panel-packed micro-kernel, manual intrinsics)
                    {
                        int rowGroups = (M + MatMulGebpParallelJob.MR - 1) / MatMulGebpParallelJob.MR;
                        int batch = ComputeBatchSize( rowGroups );
                        double ms = BenchmarkAction( warmup, iterations, () =>
                        {
                            new MatMulGebpParallelJob
                            {
                                A = aData,
                                PackedB = packedB,
                                C = cData,
                                M = M,
                                K = K,
                                N = N,
                                Accumulate = false
                            }.Schedule( rowGroups, batch ).Complete();
                        } );
                        double gf = (flops / (ms / 1000.0)) / 1e9;
                        var (maxErr, avgErr) = CompareOutputs( cRef, cData, outputLen );
                        sb.AppendLine( $"  {"GEBP (FMA Manual)",-24} {ms,10:F3} {gf,10:F2} {maxErr,12:E2} {avgErr,12:E2}" );
                    }
                    _benchmarkResults = sb.ToString();
                }
                finally
                {
                    aData.Dispose();
                    bData.Dispose();
                    btData.Dispose();
                    cRef.Dispose();
                    cData.Dispose();
                    packedB.Dispose();
                }

                sb.AppendLine();
                _benchmarkResults = sb.ToString();
                yield return null;
            }

            sb.AppendLine( "Done." );
            _benchmarkResults = sb.ToString();
            _isRunning = false;
        }
        //------------------------------------------------------------------
    }
}
