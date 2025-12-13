using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace ChaosRL
{
    [BurstCompile]
    public struct TransposeJob : IJob
    {
        [ReadOnly] public NativeArray<float> Input;
        [WriteOnly] public NativeArray<float> Output;
        public int Rows;
        public int Cols;

        public void Execute()
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Output[ j * Rows + i ] = Input[ i * Cols + j ];
                }
            }
        }
    }

    [BurstCompile]
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
}
