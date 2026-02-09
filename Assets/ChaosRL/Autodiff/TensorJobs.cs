using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace ChaosRL
{
    //------------------------------------------------------------------
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard )]
    public struct TransposeParallelJob : IJobParallelFor
    {
        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> Input;   // (Rows x Cols)
        [WriteOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> Output; // (Cols x Rows)
        public int Rows; // K
        public int Cols; // N

        public void Execute( int index )
        {
            int i = index / Cols;   // row in Input
            int j = index - i * Cols;

            Output[ j * Rows + i ] = Input[ i * Cols + j ];
        }
    }
    //------------------------------------------------------------------
    /// <summary>
    /// Cache-friendly tiled transpose:
    /// Output(Cols x Rows) = Input(Rows x Cols)
    /// Parallelized over tiles rather than individual elements.
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct TransposeTiledParallelJob : IJobParallelFor
    {
        public const int TILE = 32;

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> Input;   // Rows x Cols
        [WriteOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> Output;  // Cols x Rows
        public int Rows;
        public int Cols;

        public void Execute( int tileIndex )
        {
            int tilesPerRow = (Cols + TILE - 1) / TILE;
            int tileRow = tileIndex / tilesPerRow;
            int tileCol = tileIndex - tileRow * tilesPerRow;

            int rowStart = tileRow * TILE;
            int colStart = tileCol * TILE;

            if (rowStart >= Rows || colStart >= Cols)
                return;

            int rowEnd = math.min( rowStart + TILE, Rows );
            int colEnd = math.min( colStart + TILE, Cols );

            for (int r = rowStart; r < rowEnd; r++)
            {
                int srcBase = r * Cols;
                for (int c = colStart; c < colEnd; c++)
                    Output[ c * Rows + r ] = Input[ srcBase + c ];
            }
        }
    }
    //------------------------------------------------------------------
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard )]
    public struct MatMulJob : IJob
    {
        [ReadOnly] public NativeArray<float> A;
        [ReadOnly] public NativeArray<float> B; // Transposed B (NxK)
        [WriteOnly] public NativeArray<float> C;
        public int M, K, N;

        public void Execute()
        {
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float sum = 0f;
                    int rowOffsetA = i * K;
                    int rowOffsetB = j * K;
                    for (int k = 0; k < K; k++)
                    {
                        sum += A[ rowOffsetA + k ] * B[ rowOffsetB + k ];
                    }
                    C[ i * N + j ] = sum;
                }
            }
        }
    }
    //------------------------------------------------------------------
    /// <summary>
    /// Naive GEMM using transposed B for better locality:
    /// C(MxN) = A(MxK) @ B(KxN), where BT is B transposed into (N x K).
    /// Parallelized over output elements (one job index computes one C[i,j]).
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard )]
    public struct MatMulNaiveParallelJob : IJobParallelFor
    {
        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> A; // M x K

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> BT; // N x K (transposed B)

        // Not WriteOnly because backward can accumulate with +=.
        [NativeDisableParallelForRestriction]
        public NativeArray<float> C; // M x N

        public int M, K, N;
        // Used during backward pass to accumulate gradients.
        public bool Accumulate;

        public void Execute( int index )
        {
            int i = index / N;
            int j = index - i * N;

            int aRow = i * K;
            int btRow = j * K;

            float sum = 0f;
            for (int k = 0; k < K; k++)
                sum += A[ aRow + k ] * BT[ btRow + k ];

            if (Accumulate)
                C[ index ] += sum;
            else
                C[ index ] = sum;
        }
    }
    //------------------------------------------------------------------
    /// <summary>
    /// Row-parallel GEMM:
    /// each parallel index computes one output row in C.
    /// Better scheduling overhead than element-parallel for small matrices.
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct MatMulRowParallelJob : IJobParallelFor
    {
        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> A;  // M x K

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> BT; // N x K (transposed B)

        [NativeDisableParallelForRestriction]
        public NativeArray<float> C;  // M x N

        public int M, K, N;
        public bool Accumulate;

        public void Execute( int row )
        {
            if (row >= M)
                return;

            int aRow = row * K;
            int cRow = row * N;

            for (int j = 0; j < N; j++)
            {
                int btRow = j * K;
                float sum = 0f;
                for (int k = 0; k < K; k++)
                    sum += A[ aRow + k ] * BT[ btRow + k ];

                int idx = cRow + j;
                if (Accumulate)
                    C[ idx ] += sum;
                else
                    C[ idx ] = sum;
            }
        }
    }
    //------------------------------------------------------------------
    /// <summary>
    /// Blocked GEMM using transposed B for better cache reuse:
    /// C(MxN) = A(MxK) @ B(KxN), with BT = transpose(B) shaped (N x K).
    /// One parallel index computes one output tile.
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct MatMulBlockedParallelJob : IJobParallelFor
    {
        public const int TM = 8;
        public const int TN = 8;
        public const int TK = 64;

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> A;  // M x K

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> BT; // N x K

        [NativeDisableParallelForRestriction]
        public NativeArray<float> C;  // M x N

        public int M, K, N;
        public bool Accumulate;

        public unsafe void Execute( int tileIndex )
        {
            int tilesN = (N + TN - 1) / TN;
            int tileRow = tileIndex / tilesN;
            int tileCol = tileIndex - tileRow * tilesN;

            int rowStart = tileRow * TM;
            int colStart = tileCol * TN;

            if (rowStart >= M || colStart >= N)
                return;

            int rowCount = math.min( TM, M - rowStart );
            int colCount = math.min( TN, N - colStart );

            float* accum = stackalloc float[ TM * TN ];
            for (int i = 0; i < TM * TN; i++)
                accum[ i ] = 0f;

            for (int kb = 0; kb < K; kb += TK)
            {
                int kEnd = math.min( kb + TK, K );
                for (int i = 0; i < rowCount; i++)
                {
                    int aRow = (rowStart + i) * K;
                    int accRow = i * TN;

                    for (int j = 0; j < colCount; j++)
                    {
                        int btRow = (colStart + j) * K;
                        float sum = accum[ accRow + j ];

                        for (int k = kb; k < kEnd; k++)
                            sum += A[ aRow + k ] * BT[ btRow + k ];

                        accum[ accRow + j ] = sum;
                    }
                }
            }

            for (int i = 0; i < rowCount; i++)
            {
                int cRow = (rowStart + i) * N;
                int accRow = i * TN;

                for (int j = 0; j < colCount; j++)
                {
                    int idx = cRow + colStart + j;
                    float value = accum[ accRow + j ];
                    if (Accumulate)
                        C[ idx ] += value;
                    else
                        C[ idx ] = value;
                }
            }
        }
    }
    //------------------------------------------------------------------
    /// <summary>
    /// High-performance GEBP MatMul using explicit FMA intrinsics:
    /// C(M×N) = A(M×K) @ B(K×N)  -- B is in ORIGINAL row-major layout (NOT transposed).
    ///
    /// Key difference from other kernels: takes B in K×N layout, vectorizes over
    /// output columns (N dimension) using AVX FMA, broadcasting A elements.
    /// This avoids the separate transpose step entirely and achieves near-optimal
    /// FMA throughput with a 6×16 micro-kernel (12 v256 accumulators).
    ///
    /// Parallelized over row-groups of MR=6 rows each.
    /// Falls back to SSE (4-wide) then scalar for CPUs without AVX/FMA.
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct MatMulGebpParallelJob : IJobParallelFor
    {
        /// <summary>Micro-kernel row count (output rows per parallel index).</summary>
        public const int MR = 6;
        /// <summary>Micro-kernel column count (output columns per inner loop step).</summary>
        public const int NR = 16;

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> A;  // M × K  row-major

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> B;  // K × N  row-major (NOT transposed!)

        [NativeDisableParallelForRestriction]
        public NativeArray<float> C;  // M × N  row-major

        public int M, K, N;
        public bool Accumulate;

        //--------------------------------------------------------------
        [SkipLocalsInit]
        public unsafe void Execute( int rowGroup )
        {
            int rowStart = rowGroup * MR;
            if (rowStart >= M)
                return;
            int rowCount = math.min( MR, M - rowStart );

            float* pA = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( A );
            float* pB = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( B );
            float* pC = (float*)NativeArrayUnsafeUtility.GetUnsafePtr( C );

            int j = 0;

            // ---- AVX + FMA path (256-bit, 8 floats/vec) ----
            if (X86.Fma.IsFmaSupported)
            {
                // Full 16-wide micro-tiles
                for (; j + NR <= N; j += NR)
                {
                    if (rowCount == MR)
                        MicroKernelFma6x16( pA, pB, pC, rowStart, j );
                    else
                        MicroKernelFmaGeneric( pA, pB, pC, rowStart, rowCount, j, NR );
                }
                // 8-wide remainder
                if (j + 8 <= N)
                {
                    MicroKernelFmaGeneric( pA, pB, pC, rowStart, rowCount, j, 8 );
                    j += 8;
                }
            }
            // ---- SSE fallback (128-bit, 4 floats/vec) ----
            else if (X86.Sse.IsSseSupported)
            {
                for (; j + 4 <= N; j += 4)
                    MicroKernelSse( pA, pB, pC, rowStart, rowCount, j );
            }

            // ---- Scalar tail for remaining columns ----
            for (; j < N; j++)
            {
                for (int ii = 0; ii < rowCount; ii++)
                {
                    int row = rowStart + ii;
                    float* aRow = pA + row * K;
                    float sum = 0f;
                    for (int k = 0; k < K; k++)
                        sum += aRow[k] * pB[k * N + j];
                    int cIdx = row * N + j;
                    if (Accumulate)
                        pC[cIdx] += sum;
                    else
                        pC[cIdx] = sum;
                }
            }
        }

        //--------------------------------------------------------------
        // Hot path: exactly 6 rows × 16 columns (2 × v256) using FMA.
        // 12 accumulator registers + 2 B loads + 1 A broadcast = 15 YMM regs used.
        //--------------------------------------------------------------
        [SkipLocalsInit]
        private unsafe void MicroKernelFma6x16( float* pA, float* pB, float* pC, int rowStart, int colStart )
        {
            if (!X86.Fma.IsFmaSupported) return;

            v256 c00 = default, c01 = default;
            v256 c10 = default, c11 = default;
            v256 c20 = default, c21 = default;
            v256 c30 = default, c31 = default;
            v256 c40 = default, c41 = default;
            v256 c50 = default, c51 = default;

            float* a0 = pA + rowStart * K;
            float* a1 = a0 + K;
            float* a2 = a1 + K;
            float* a3 = a2 + K;
            float* a4 = a3 + K;
            float* a5 = a4 + K;

            for (int k = 0; k < K; k++)
            {
                float* bk = pB + k * N + colStart;
                v256 b0 = X86.Avx.mm256_loadu_ps( bk );
                v256 b1 = X86.Avx.mm256_loadu_ps( bk + 8 );

                v256 av;

                av = X86.Avx.mm256_set1_ps( a0[k] );
                c00 = X86.Fma.mm256_fmadd_ps( av, b0, c00 );
                c01 = X86.Fma.mm256_fmadd_ps( av, b1, c01 );

                av = X86.Avx.mm256_set1_ps( a1[k] );
                c10 = X86.Fma.mm256_fmadd_ps( av, b0, c10 );
                c11 = X86.Fma.mm256_fmadd_ps( av, b1, c11 );

                av = X86.Avx.mm256_set1_ps( a2[k] );
                c20 = X86.Fma.mm256_fmadd_ps( av, b0, c20 );
                c21 = X86.Fma.mm256_fmadd_ps( av, b1, c21 );

                av = X86.Avx.mm256_set1_ps( a3[k] );
                c30 = X86.Fma.mm256_fmadd_ps( av, b0, c30 );
                c31 = X86.Fma.mm256_fmadd_ps( av, b1, c31 );

                av = X86.Avx.mm256_set1_ps( a4[k] );
                c40 = X86.Fma.mm256_fmadd_ps( av, b0, c40 );
                c41 = X86.Fma.mm256_fmadd_ps( av, b1, c41 );

                av = X86.Avx.mm256_set1_ps( a5[k] );
                c50 = X86.Fma.mm256_fmadd_ps( av, b0, c50 );
                c51 = X86.Fma.mm256_fmadd_ps( av, b1, c51 );
            }

            StoreFma256( pC, (rowStart + 0) * N + colStart, c00, c01 );
            StoreFma256( pC, (rowStart + 1) * N + colStart, c10, c11 );
            StoreFma256( pC, (rowStart + 2) * N + colStart, c20, c21 );
            StoreFma256( pC, (rowStart + 3) * N + colStart, c30, c31 );
            StoreFma256( pC, (rowStart + 4) * N + colStart, c40, c41 );
            StoreFma256( pC, (rowStart + 5) * N + colStart, c50, c51 );
        }

        //--------------------------------------------------------------
        // Generic FMA path for partial rows or 8-wide remainder columns.
        // nrCols must be 8 or 16.
        //--------------------------------------------------------------
        [SkipLocalsInit]
        private unsafe void MicroKernelFmaGeneric( float* pA, float* pB, float* pC,
                                                    int rowStart, int rowCount, int colStart, int nrCols )
        {
            if (!X86.Fma.IsFmaSupported) return;

            // stackalloc accumulator: up to MR × 2 v256 (6 × 2 = 12 v256 = 384 bytes)
            float* accum = stackalloc float[MR * NR];

            for (int i = 0; i < MR * NR; i++)
                accum[i] = 0f;

            for (int k = 0; k < K; k++)
            {
                float* bk = pB + k * N + colStart;
                v256 b0 = X86.Avx.mm256_loadu_ps( bk );
                v256 b1 = nrCols > 8 ? X86.Avx.mm256_loadu_ps( bk + 8 ) : default;

                for (int ii = 0; ii < rowCount; ii++)
                {
                    v256 av = X86.Avx.mm256_set1_ps( pA[(rowStart + ii) * K + k] );
                    float* acc = accum + ii * NR;

                    v256 a0 = X86.Avx.mm256_loadu_ps( acc );
                    a0 = X86.Fma.mm256_fmadd_ps( av, b0, a0 );
                    X86.Avx.mm256_storeu_ps( acc, a0 );

                    if (nrCols > 8)
                    {
                        v256 a1 = X86.Avx.mm256_loadu_ps( acc + 8 );
                        a1 = X86.Fma.mm256_fmadd_ps( av, b1, a1 );
                        X86.Avx.mm256_storeu_ps( acc + 8, a1 );
                    }
                }
            }

            // Write back from accumulator to C
            for (int ii = 0; ii < rowCount; ii++)
            {
                int cBase = (rowStart + ii) * N + colStart;
                float* acc = accum + ii * NR;
                if (nrCols >= 16)
                    StoreFma256FromAccum( pC, cBase, acc );
                else // nrCols == 8
                    StoreFma128FromAccum( pC, cBase, acc );
            }
        }

        //--------------------------------------------------------------
        // SSE fallback: 4-wide accumulator per row
        //--------------------------------------------------------------
        [SkipLocalsInit]
        private unsafe void MicroKernelSse( float* pA, float* pB, float* pC,
                                             int rowStart, int rowCount, int colStart )
        {
            // Up to MR v128 accumulators
            float* accum = stackalloc float[MR * 4];
            for (int i = 0; i < MR * 4; i++)
                accum[i] = 0f;

            for (int k = 0; k < K; k++)
            {
                v128 bv = X86.Sse.loadu_ps( pB + k * N + colStart );
                for (int ii = 0; ii < rowCount; ii++)
                {
                    v128 av = X86.Sse.set1_ps( pA[(rowStart + ii) * K + k] );
                    float* acc = accum + ii * 4;
                    v128 cv = X86.Sse.loadu_ps( acc );
                    cv = X86.Sse.add_ps( cv, X86.Sse.mul_ps( av, bv ) );
                    X86.Sse.storeu_ps( acc, cv );
                }
            }

            for (int ii = 0; ii < rowCount; ii++)
            {
                float* dst = pC + (rowStart + ii) * N + colStart;
                float* acc = accum + ii * 4;
                v128 cv = X86.Sse.loadu_ps( acc );
                if (Accumulate)
                    cv = X86.Sse.add_ps( cv, X86.Sse.loadu_ps( dst ) );
                X86.Sse.storeu_ps( dst, cv );
            }
        }

        //--------------------------------------------------------------
        // Store helpers
        //--------------------------------------------------------------
        private unsafe void StoreFma256( float* pC, int offset, v256 lo, v256 hi )
        {
            if (!X86.Avx.IsAvxSupported) return;

            float* dst = pC + offset;
            if (Accumulate)
            {
                lo = X86.Avx.mm256_add_ps( lo, X86.Avx.mm256_loadu_ps( dst ) );
                hi = X86.Avx.mm256_add_ps( hi, X86.Avx.mm256_loadu_ps( dst + 8 ) );
            }
            X86.Avx.mm256_storeu_ps( dst, lo );
            X86.Avx.mm256_storeu_ps( dst + 8, hi );
        }

        private unsafe void StoreFma256FromAccum( float* pC, int cBase, float* acc )
        {
            if (!X86.Avx.IsAvxSupported) return;

            float* dst = pC + cBase;
            v256 lo = X86.Avx.mm256_loadu_ps( acc );
            v256 hi = X86.Avx.mm256_loadu_ps( acc + 8 );
            if (Accumulate)
            {
                lo = X86.Avx.mm256_add_ps( lo, X86.Avx.mm256_loadu_ps( dst ) );
                hi = X86.Avx.mm256_add_ps( hi, X86.Avx.mm256_loadu_ps( dst + 8 ) );
            }
            X86.Avx.mm256_storeu_ps( dst, lo );
            X86.Avx.mm256_storeu_ps( dst + 8, hi );
        }

        private unsafe void StoreFma128FromAccum( float* pC, int cBase, float* acc )
        {
            if (!X86.Avx.IsAvxSupported) return;

            float* dst = pC + cBase;
            v256 lo = X86.Avx.mm256_loadu_ps( acc );
            if (Accumulate)
                lo = X86.Avx.mm256_add_ps( lo, X86.Avx.mm256_loadu_ps( dst ) );
            X86.Avx.mm256_storeu_ps( dst, lo );
        }
    }
    //------------------------------------------------------------------

}
