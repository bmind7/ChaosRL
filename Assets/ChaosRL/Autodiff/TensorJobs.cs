using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace ChaosRL
{
    //------------------------------------------------------------------
    /// <summary>
    /// Cache-friendly tiled transpose:
    /// Output(Cols x Rows) = Input(Rows x Cols)
    /// Parallelized over tiles rather than individual elements.
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low,
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
    /// <summary>
    /// Naive GEMM using transposed B for better locality:
    /// C(MxN) = A(MxK) @ B(KxN), where BT is B transposed into (N x K).
    /// Parallelized over output elements (one job index computes one C[i,j]).
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
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
    /// Packs a Kc-thick slice of B(K×N) into column-panel layout for GEBP.
    /// Packed layout: panel[jp] contains Kc rows of NR consecutive columns.
    /// Storage: PackedB[jp * Kc * NR + k * NR + jr] = B[(KOffset+k), jp*NR + jr].
    /// Last panel is zero-padded if N is not divisible by NR.
    /// Burst auto-vectorizes the copy loop (no manual SIMD intrinsics).
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct PackBPanelScalarParallelJob : IJobParallelFor
    {
        public const int NR = 16;

        [NoAlias, ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> B;  // K × N row-major

        [NoAlias, WriteOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> PackedB;  // numPanels × Kc × NR

        public int N;       // B column count (row stride)
        public int KOffset; // first K-row to pack
        public int Kc;      // number of K-rows to pack in this block

        [SkipLocalsInit]
        public unsafe void Execute( int panelIndex )
        {
            int colStart = panelIndex * NR;
            int colCount = math.min( NR, N - colStart );

            float* pB = (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( B );
            float* dst = (float*)NativeArrayUnsafeUtility.GetUnsafePtr( PackedB )
                         + panelIndex * Kc * NR;

            Hint.Assume( Kc > 0 );

            if (Hint.Likely( colCount == NR ))
            {
                // Full panel: constant NR=16 trip count → Burst emits SIMD copy.
                for (int k = 0; k < Kc; k++)
                {
                    float* src = pB + (KOffset + k) * N + colStart;
                    float* d = dst + k * NR;
                    for (int jr = 0; jr < NR; jr++)
                        d[ jr ] = src[ jr ];
                }
            }
            else
            {
                // Partial panel: copy valid columns, zero-pad rest.
                for (int k = 0; k < Kc; k++)
                {
                    float* src = pB + (KOffset + k) * N + colStart;
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
    /// GEBP MatMul: C(M×N) += A(M×K) @ PackedB for a Kc-thick slice of K.
    /// Reads columns KOffset..KOffset+Kc-1 from A, and a Kc-deep packed B panel.
    /// Uses a 6×16 micro-kernel with NO manual SIMD intrinsics — Burst auto-vectorizes.
    /// Parallelized over row-groups of MR=6 rows each.
    ///
    /// Burst auto-vectorization hints applied:
    ///  - [NoAlias] on all NativeArray fields (removes aliasing barriers)
    ///  - Aliasing.ExpectNotAliased on raw pointers
    ///  - Hint.Likely fast path for full MR=6 row groups (common case)
    ///  - Hint.Assume(Kc > 0) to eliminate empty-loop guards
    ///  - Fully unrolled ii dimension with 6 fixed accumulator pointers (helps SROA)
    ///  - Pre-loaded A values and single B load per jr element
    ///  - Constant NR trip count on innermost jr loop (packed B is zero-padded)
    ///  - Accumulate branch hoisted out of jr loop
    ///  - UnsafeUtility.MemClear for accumulator zeroing
    ///  - stackalloc hoisted outside panel loop
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct MatMulGebpScalarParallelJob : IJobParallelFor
    {
        public const int MR = 6;
        public const int NR = 16;

        [NoAlias, ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> A;       // M × K  row-major

        [NoAlias, ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<float> PackedB; // numPanels × Kc × NR, panel-packed

        [NoAlias, NativeDisableParallelForRestriction]
        public NativeArray<float> C;       // M × N  row-major

        public int M, K, N;
        public int KOffset; // first K-column for this Kc block
        public int Kc;      // number of K-columns in this block
        public bool Accumulate;

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

            // Tell Burst the raw pointers don't alias each other.
            Aliasing.ExpectNotAliased( pA, pPB );
            Aliasing.ExpectNotAliased( pA, pC );
            Aliasing.ExpectNotAliased( pPB, pC );

            // Help the compiler elide empty-loop guards.
            Hint.Assume( Kc > 0 );

            int numPanels = (N + NR - 1) / NR;

            // Hoist stackalloc outside the panel loop — one allocation, reused.
            float* accum = stackalloc float[ MR * NR ];

            // Tell Burst accum doesn't alias input/output buffers.
            Aliasing.ExpectNotAliased( accum, pA );
            Aliasing.ExpectNotAliased( accum, pPB );
            Aliasing.ExpectNotAliased( accum, pC );

            for (int jp = 0; jp < numPanels; jp++)
            {
                float* panel = pPB + jp * Kc * NR;
                int colStart = jp * NR;
                int colCount = math.min( NR, N - colStart );

                // Fast zero via memset instead of scalar loop.
                UnsafeUtility.MemClear( accum, MR * NR * sizeof( float ) );

                // ---- Fast path: full MR=6 rows (the common case) ----
                // Fully unrolls the row dimension so LLVM sees 6 independent
                // accumulator streams with fixed pointer offsets, enabling
                // SROA / register promotion of the accumulator tile.
                if (Hint.Likely( rowCount == MR ))
                {
                    // Pre-compute A row pointers (loop-invariant).
                    float* a0 = pA + (rowStart + 0) * K + KOffset;
                    float* a1 = pA + (rowStart + 1) * K + KOffset;
                    float* a2 = pA + (rowStart + 2) * K + KOffset;
                    float* a3 = pA + (rowStart + 3) * K + KOffset;
                    float* a4 = pA + (rowStart + 4) * K + KOffset;
                    float* a5 = pA + (rowStart + 5) * K + KOffset;

                    // Fixed accumulator row pointers — helps LLVM SROA.
                    float* r0 = accum;
                    float* r1 = accum + NR;
                    float* r2 = accum + 2 * NR;
                    float* r3 = accum + 3 * NR;
                    float* r4 = accum + 4 * NR;
                    float* r5 = accum + 5 * NR;

                    for (int k = 0; k < Kc; k++)
                    {
                        float* bk = panel + k * NR;

                        // Load all 6 A values once per k step.
                        float av0 = a0[ k ], av1 = a1[ k ], av2 = a2[ k ];
                        float av3 = a3[ k ], av4 = a4[ k ], av5 = a5[ k ];

                        // Vectorizable inner loop: constant NR=16 trip count,
                        // single B load reused across 6 independent FMA streams.
                        for (int jr = 0; jr < NR; jr++)
                        {
                            float b = bk[ jr ];
                            r0[ jr ] += av0 * b;
                            r1[ jr ] += av1 * b;
                            r2[ jr ] += av2 * b;
                            r3[ jr ] += av3 * b;
                            r4[ jr ] += av4 * b;
                            r5[ jr ] += av5 * b;
                        }
                    }
                }
                // ---- Generic path for the last (partial) row group ----
                else
                {
                    for (int k = 0; k < Kc; k++)
                    {
                        float* bk = panel + k * NR;
                        for (int ii = 0; ii < rowCount; ii++)
                        {
                            float aVal = pA[ (rowStart + ii) * K + KOffset + k ];
                            float* acc = accum + ii * NR;
                            for (int jr = 0; jr < NR; jr++)
                                acc[ jr ] += aVal * bk[ jr ];
                        }
                    }
                }

                // Write back to C — only valid columns, Accumulate branch hoisted.
                for (int ii = 0; ii < rowCount; ii++)
                {
                    int cBase = (rowStart + ii) * N + colStart;
                    float* acc = accum + ii * NR;

                    if (Accumulate)
                    {
                        for (int jr = 0; jr < colCount; jr++)
                            pC[ cBase + jr ] += acc[ jr ];
                    }
                    else
                    {
                        for (int jr = 0; jr < colCount; jr++)
                            pC[ cBase + jr ] = acc[ jr ];
                    }
                }
            }
        }
    }
    //------------------------------------------------------------------
    /// <summary>
    /// Burst-compiled sum reduction: Output[0] = sum of all Input elements.
    /// Single-threaded IJob — Burst auto-vectorizes the accumulation with SIMD.
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct SumReductionJob : IJob
    {
        [ReadOnly] public NativeArray<float> Input;
        [WriteOnly] public NativeArray<float> Output; // length 1

        public void Execute()
        {
            float sum = 0f;
            for (int i = 0; i < Input.Length; i++)
                sum += Input[ i ];
            Output[ 0 ] = sum;
        }
    }
    //------------------------------------------------------------------
    /// <summary>
    /// Burst-compiled scalar broadcast: Target[index] += Value.
    /// Used by Sum backward to broadcast the gradient to all input elements.
    /// </summary>
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low,
                   DisableSafetyChecks = true, OptimizeFor = OptimizeFor.Performance )]
    public struct AddScalarParallelJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<float> Target;
        public float Value;

        public void Execute( int index )
        {
            Target[ index ] += Value;
        }
    }
    //------------------------------------------------------------------
}
