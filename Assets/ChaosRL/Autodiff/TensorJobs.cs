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
    /// Packs B(K×N) into column-panel layout for cache-friendly micro-kernel access.
    /// Packed layout: panel[jp] contains K rows of NR consecutive columns.
    /// Storage: PackedB[jp * K * NR + k * NR + jr] = B[k, jp*NR + jr].
    /// Last panel is zero-padded if N is not divisible by NR.
    /// Parallelized over column panels.
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct PackBPanelParallelJob : IJobParallelFor
    {
        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> B;  // K × N row-major

        [WriteOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> PackedB;  // numPanels × K × NR

        public int K, N;

        [SkipLocalsInit]
        public unsafe void Execute( int panelIndex )
        {
            const int NR = MatMulGebpParallelJob.NR;
            int colStart = panelIndex * NR;
            int colCount = math.min( NR, N - colStart );

            float* pB = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( B );
            float* dst = (float*)NativeArrayUnsafeUtility.GetUnsafePtr( PackedB )
                         + panelIndex * K * NR;

            if (colCount == NR)
            {
                if (X86.Avx.IsAvxSupported)
                {
                    for (int k = 0; k < K; k++)
                    {
                        float* src = pB + k * N + colStart;
                        float* d = dst + k * NR;
                        X86.Avx.mm256_storeu_ps( d, X86.Avx.mm256_loadu_ps( src ) );
                        X86.Avx.mm256_storeu_ps( d + 8, X86.Avx.mm256_loadu_ps( src + 8 ) );
                    }
                }
                else
                {
                    for (int k = 0; k < K; k++)
                    {
                        float* src = pB + k * N + colStart;
                        float* d = dst + k * NR;
                        for (int jr = 0; jr < NR; jr++)
                            d[ jr ] = src[ jr ];
                    }
                }
            }
            else
            {
                // Partial panel: copy valid columns, zero-pad rest
                for (int k = 0; k < K; k++)
                {
                    float* src = pB + k * N + colStart;
                    float* d = dst + k * NR;
                    int jr = 0;
                    for (; jr < colCount; jr++)
                        d[ jr ] = src[ jr ];
                    for (; jr < NR; jr++)
                        d[ jr ] = 0f;
                }
            }
        }
    }
    //------------------------------------------------------------------
    /// <summary>
    /// High-performance GEBP MatMul using explicit FMA intrinsics:
    /// C(M×N) = A(M×K) @ B(K×N)  with B pre-packed into column-panel layout.
    ///
    /// Reads PackedB in sequential NR-stride access (64 bytes = 1 cache line),
    /// achieving full L1 throughput regardless of original N.
    /// Uses a 6×16 FMA micro-kernel (12 v256 accumulators).
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
        public NativeArray<float> A;       // M × K  row-major

        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> PackedB; // numPanels × K × NR, panel-packed

        [NativeDisableParallelForRestriction]
        public NativeArray<float> C;       // M × N  row-major

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
            float* pPB = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( PackedB );
            float* pC = (float*)NativeArrayUnsafeUtility.GetUnsafePtr( C );

            int numPanels = (N + NR - 1) / NR;
            int lastPanelCols = N - (numPanels - 1) * NR;
            int fullPanels = lastPanelCols == NR ? numPanels : numPanels - 1;

            // ---- AVX + FMA path (256-bit, 8 floats/vec) ----
            if (X86.Fma.IsFmaSupported)
            {
                // Full NR-wide panels
                for (int jp = 0; jp < fullPanels; jp++)
                {
                    float* panel = pPB + jp * K * NR;
                    int colStart = jp * NR;
                    if (rowCount == MR)
                        MicroKernelFma6x16( pA, panel, pC, rowStart, colStart );
                    else
                        MicroKernelFmaGeneric( pA, panel, pC, rowStart, rowCount, colStart, NR );
                }
                // Last partial panel (if N not divisible by NR)
                if (fullPanels < numPanels)
                {
                    float* panel = pPB + fullPanels * K * NR;
                    int colStart = fullPanels * NR;
                    MicroKernelFmaGeneric( pA, panel, pC, rowStart, rowCount, colStart, lastPanelCols );
                }
            }
            // ---- SSE fallback (128-bit, 4 floats/vec) ----
            else if (X86.Sse.IsSseSupported)
            {
                for (int jp = 0; jp < numPanels; jp++)
                {
                    float* panel = pPB + jp * K * NR;
                    int colStart = jp * NR;
                    int colCount = jp < fullPanels ? NR : lastPanelCols;
                    int jLocal = 0;
                    for (; jLocal + 4 <= colCount; jLocal += 4)
                        MicroKernelSse( pA, panel, pC, rowStart, rowCount, colStart + jLocal, jLocal );
                    for (; jLocal < colCount; jLocal++)
                        ScalarColumn( pA, panel, pC, rowStart, rowCount, colStart, jLocal );
                }
            }
            // ---- Pure scalar fallback ----
            else
            {
                for (int jp = 0; jp < numPanels; jp++)
                {
                    float* panel = pPB + jp * K * NR;
                    int colStart = jp * NR;
                    int colCount = jp < fullPanels ? NR : lastPanelCols;
                    for (int jLocal = 0; jLocal < colCount; jLocal++)
                        ScalarColumn( pA, panel, pC, rowStart, rowCount, colStart, jLocal );
                }
            }
        }

        //--------------------------------------------------------------
        // Scalar helper: computes one output column within a packed panel.
        //--------------------------------------------------------------
        private unsafe void ScalarColumn( float* pA, float* panel, float* pC,
                                           int rowStart, int rowCount, int colStart, int jLocal )
        {
            for (int ii = 0; ii < rowCount; ii++)
            {
                int row = rowStart + ii;
                float* aRow = pA + row * K;
                float sum = 0f;
                for (int k = 0; k < K; k++)
                    sum += aRow[ k ] * panel[ k * NR + jLocal ];
                int cIdx = row * N + colStart + jLocal;
                if (Accumulate)
                    pC[ cIdx ] += sum;
                else
                    pC[ cIdx ] = sum;
            }
        }

        //--------------------------------------------------------------
        // Hot path: exactly 6 rows × 16 columns (2 × v256) using FMA.
        // 12 accumulator registers + 2 B loads + 1 A broadcast = 15 YMM regs.
        // Reads pre-packed panel with stride NR*4 = 64 bytes (1 cache line).
        //--------------------------------------------------------------
        [SkipLocalsInit]
        private unsafe void MicroKernelFma6x16( float* pA, float* panel, float* pC, int rowStart, int colStart )
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
                float* bk = panel + k * NR;
                v256 b0 = X86.Avx.mm256_loadu_ps( bk );
                v256 b1 = X86.Avx.mm256_loadu_ps( bk + 8 );

                v256 av;

                av = X86.Avx.mm256_set1_ps( a0[ k ] );
                c00 = X86.Fma.mm256_fmadd_ps( av, b0, c00 );
                c01 = X86.Fma.mm256_fmadd_ps( av, b1, c01 );

                av = X86.Avx.mm256_set1_ps( a1[ k ] );
                c10 = X86.Fma.mm256_fmadd_ps( av, b0, c10 );
                c11 = X86.Fma.mm256_fmadd_ps( av, b1, c11 );

                av = X86.Avx.mm256_set1_ps( a2[ k ] );
                c20 = X86.Fma.mm256_fmadd_ps( av, b0, c20 );
                c21 = X86.Fma.mm256_fmadd_ps( av, b1, c21 );

                av = X86.Avx.mm256_set1_ps( a3[ k ] );
                c30 = X86.Fma.mm256_fmadd_ps( av, b0, c30 );
                c31 = X86.Fma.mm256_fmadd_ps( av, b1, c31 );

                av = X86.Avx.mm256_set1_ps( a4[ k ] );
                c40 = X86.Fma.mm256_fmadd_ps( av, b0, c40 );
                c41 = X86.Fma.mm256_fmadd_ps( av, b1, c41 );

                av = X86.Avx.mm256_set1_ps( a5[ k ] );
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
        // Generic FMA path for partial rows or partial column panels.
        // Reads from pre-packed panel. Always accumulates NR-wide (zero-padded),
        // but stores only colCount valid columns.
        //--------------------------------------------------------------
        [SkipLocalsInit]
        private unsafe void MicroKernelFmaGeneric( float* pA, float* panel, float* pC,
                                                    int rowStart, int rowCount, int colStart, int colCount )
        {
            if (!X86.Fma.IsFmaSupported) return;

            // Accumulators: always NR-wide (zero-padded B ensures correctness)
            float* accum = stackalloc float[ MR * NR ];
            for (int i = 0; i < MR * NR; i++)
                accum[ i ] = 0f;

            for (int k = 0; k < K; k++)
            {
                float* bk = panel + k * NR;
                v256 b0 = X86.Avx.mm256_loadu_ps( bk );
                v256 b1 = X86.Avx.mm256_loadu_ps( bk + 8 );

                for (int ii = 0; ii < rowCount; ii++)
                {
                    v256 av = X86.Avx.mm256_set1_ps( pA[ (rowStart + ii) * K + k ] );
                    float* acc = accum + ii * NR;

                    v256 a0 = X86.Avx.mm256_loadu_ps( acc );
                    a0 = X86.Fma.mm256_fmadd_ps( av, b0, a0 );
                    X86.Avx.mm256_storeu_ps( acc, a0 );

                    v256 a1 = X86.Avx.mm256_loadu_ps( acc + 8 );
                    a1 = X86.Fma.mm256_fmadd_ps( av, b1, a1 );
                    X86.Avx.mm256_storeu_ps( acc + 8, a1 );
                }
            }

            // Write back only valid columns to C
            for (int ii = 0; ii < rowCount; ii++)
            {
                int cBase = (rowStart + ii) * N + colStart;
                float* acc = accum + ii * NR;
                if (colCount == NR)
                {
                    StoreFma256FromAccum( pC, cBase, acc );
                }
                else if (colCount >= 8)
                {
                    StoreFma128FromAccum( pC, cBase, acc );
                    for (int jr = 8; jr < colCount; jr++)
                    {
                        if (Accumulate) pC[ cBase + jr ] += acc[ jr ];
                        else pC[ cBase + jr ] = acc[ jr ];
                    }
                }
                else
                {
                    for (int jr = 0; jr < colCount; jr++)
                    {
                        if (Accumulate) pC[ cBase + jr ] += acc[ jr ];
                        else pC[ cBase + jr ] = acc[ jr ];
                    }
                }
            }
        }

        //--------------------------------------------------------------
        // SSE fallback: 4-wide accumulator per row.
        // Reads from packed panel at offset jLocal within the NR-wide panel.
        //--------------------------------------------------------------
        [SkipLocalsInit]
        private unsafe void MicroKernelSse( float* pA, float* panel, float* pC,
                                             int rowStart, int rowCount, int cColStart, int jLocal )
        {
            // Up to MR v128 accumulators
            float* accum = stackalloc float[ MR * 4 ];
            for (int i = 0; i < MR * 4; i++)
                accum[ i ] = 0f;

            for (int k = 0; k < K; k++)
            {
                v128 bv = X86.Sse.loadu_ps( panel + k * NR + jLocal );
                for (int ii = 0; ii < rowCount; ii++)
                {
                    v128 av = X86.Sse.set1_ps( pA[ (rowStart + ii) * K + k ] );
                    float* acc = accum + ii * 4;
                    v128 cv = X86.Sse.loadu_ps( acc );
                    cv = X86.Sse.add_ps( cv, X86.Sse.mul_ps( av, bv ) );
                    X86.Sse.storeu_ps( acc, cv );
                }
            }

            for (int ii = 0; ii < rowCount; ii++)
            {
                float* dst = pC + (rowStart + ii) * N + cColStart;
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
