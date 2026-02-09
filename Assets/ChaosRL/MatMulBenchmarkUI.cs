using System;
using System.Collections;
using System.Diagnostics;
using System.Text;

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
    }
}
