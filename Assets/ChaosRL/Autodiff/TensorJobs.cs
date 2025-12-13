using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace ChaosRL
{
    [BurstCompile]
    public struct MatMulJob : IJob
    {
        [ReadOnly] public NativeArray<float> A;
        [ReadOnly] public NativeArray<float> B;
        public NativeArray<float> C;
        public int M, K, N;

        public void Execute()
        {
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < K; k++)
                    {
                        sum += A[ i * K + k ] * B[ k * N + j ];
                    }
                    C[ i * N + j ] = sum;
                }
            }
        }
    }
}
