using Unity.Burst;
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
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard )]
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
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard )]
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
    [BurstCompile( FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard )]
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

}
