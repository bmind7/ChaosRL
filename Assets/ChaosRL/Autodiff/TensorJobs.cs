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

}
