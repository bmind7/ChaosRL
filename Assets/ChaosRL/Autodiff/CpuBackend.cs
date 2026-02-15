using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

namespace ChaosRL
{
    /// <summary>
    /// CPU backend implementing <see cref="ITensorBackend"/> via Burst-compiled jobs.
    /// All methods are synchronous — they schedule jobs and call .Complete() before returning.
    /// Extracts <see cref="NativeArray{T}"/> from <see cref="TensorStorage"/> for Burst job dispatch.
    /// </summary>
    public class CpuBackend : ITensorBackend
    {
        //==================================================================
        //  Element-wise binary — forward
        //==================================================================

        //------------------------------------------------------------------
        public void Add( TensorStorage a, TensorStorage b, TensorStorage result,
                         int sizeA, int sizeB, int resultSize )
        {
            new ElementAddJob { A = a.Buffer, B = b.Buffer, Result = result.Buffer, SizeA = sizeA, SizeB = sizeB }
                .Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void Mul( TensorStorage a, TensorStorage b, TensorStorage result,
                         int sizeA, int sizeB, int resultSize )
        {
            new ElementMulJob { A = a.Buffer, B = b.Buffer, Result = result.Buffer, SizeA = sizeA, SizeB = sizeB }
                .Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void Div( TensorStorage a, TensorStorage b, TensorStorage result,
                         int sizeA, int sizeB, int resultSize )
        {
            new ElementDivJob { A = a.Buffer, B = b.Buffer, Result = result.Buffer, SizeA = sizeA, SizeB = sizeB }
                .Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ElementMax( TensorStorage a, TensorStorage b, TensorStorage result,
                                int sizeA, int sizeB, int resultSize )
        {
            new ElementMaxJob { A = a.Buffer, B = b.Buffer, Result = result.Buffer, SizeA = sizeA, SizeB = sizeB }
                .Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ElementMin( TensorStorage a, TensorStorage b, TensorStorage result,
                                int sizeA, int sizeB, int resultSize )
        {
            new ElementMinJob { A = a.Buffer, B = b.Buffer, Result = result.Buffer, SizeA = sizeA, SizeB = sizeB }
                .Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }

        //==================================================================
        //  Element-wise binary — backward
        //==================================================================

        //------------------------------------------------------------------
        public void AddBackward( TensorStorage aGrad, TensorStorage bGrad,
                                 TensorStorage resultGrad,
                                 int sizeA, int sizeB, int resultSize,
                                 float aGradScale, float bGradScale )
        {
            var aG = aGrad.Buffer;
            var bG = bGrad.Buffer;
            var rG = resultGrad.Buffer;
            if (aG == bG)
            {
                // Aliased: same tensor on both sides (e.g. a + a). Fall back to sequential.
                for (int i = 0; i < resultSize; i++)
                {
                    float g = rG[ i ];
                    aG[ i % sizeA ] += g * aGradScale;
                    bG[ i % sizeB ] += g * bGradScale;
                }
                return;
            }
            new AddBackwardJob
            {
                AGrad = aG,
                BGrad = bG,
                ResultGrad = rG,
                SizeA = sizeA,
                SizeB = sizeB,
                AGradScale = aGradScale,
                BGradScale = bGradScale
            }.Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void MulBackward( TensorStorage aData, TensorStorage bData,
                                 TensorStorage aGrad, TensorStorage bGrad,
                                 TensorStorage resultGrad,
                                 int sizeA, int sizeB, int resultSize,
                                 float aGradScale, float bGradScale )
        {
            var aD = aData.Buffer;
            var bD = bData.Buffer;
            var aG = aGrad.Buffer;
            var bG = bGrad.Buffer;
            var rG = resultGrad.Buffer;
            if (aG == bG)
            {
                for (int i = 0; i < resultSize; i++)
                {
                    float g = rG[ i ];
                    aG[ i % sizeA ] += bD[ i % sizeB ] * g * aGradScale;
                    bG[ i % sizeB ] += aD[ i % sizeA ] * g * bGradScale;
                }
                return;
            }
            new MulBackwardJob
            {
                AData = aD,
                BData = bD,
                AGrad = aG,
                BGrad = bG,
                ResultGrad = rG,
                SizeA = sizeA,
                SizeB = sizeB,
                AGradScale = aGradScale,
                BGradScale = bGradScale
            }.Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void DivBackward( TensorStorage aData, TensorStorage bData,
                                 TensorStorage aGrad, TensorStorage bGrad,
                                 TensorStorage resultGrad,
                                 int sizeA, int sizeB, int resultSize,
                                 float aGradScale, float bGradScale )
        {
            var aD = aData.Buffer;
            var bD = bData.Buffer;
            var aG = aGrad.Buffer;
            var bG = bGrad.Buffer;
            var rG = resultGrad.Buffer;
            if (aG == bG)
            {
                for (int i = 0; i < resultSize; i++)
                {
                    int idxA = i % sizeA;
                    int idxB = i % sizeB;
                    float g = rG[ i ];
                    float bVal = bD[ idxB ];
                    aG[ idxA ] += (1f / bVal) * g * aGradScale;
                    bG[ idxB ] += (-aD[ idxA ] / (bVal * bVal)) * g * bGradScale;
                }
                return;
            }
            new DivBackwardJob
            {
                AData = aD,
                BData = bD,
                AGrad = aG,
                BGrad = bG,
                ResultGrad = rG,
                SizeA = sizeA,
                SizeB = sizeB,
                AGradScale = aGradScale,
                BGradScale = bGradScale
            }.Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ElementMaxBackward( TensorStorage aData, TensorStorage bData,
                                        TensorStorage aGrad, TensorStorage bGrad,
                                        TensorStorage resultGrad,
                                        int sizeA, int sizeB, int resultSize,
                                        float aGradScale, float bGradScale )
        {
            var aD = aData.Buffer;
            var bD = bData.Buffer;
            var aG = aGrad.Buffer;
            var bG = bGrad.Buffer;
            var rG = resultGrad.Buffer;
            if (aG == bG)
            {
                for (int i = 0; i < resultSize; i++)
                {
                    int idxA = i % sizeA;
                    int idxB = i % sizeB;
                    float g = rG[ i ];
                    float isAMax = aD[ idxA ] >= bD[ idxB ] ? 1f : 0f;
                    aG[ idxA ] += g * isAMax * aGradScale;
                    bG[ idxB ] += g * (1f - isAMax) * bGradScale;
                }
                return;
            }
            new ElementMaxBackwardJob
            {
                AData = aD,
                BData = bD,
                AGrad = aG,
                BGrad = bG,
                ResultGrad = rG,
                SizeA = sizeA,
                SizeB = sizeB,
                AGradScale = aGradScale,
                BGradScale = bGradScale
            }.Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ElementMinBackward( TensorStorage aData, TensorStorage bData,
                                        TensorStorage aGrad, TensorStorage bGrad,
                                        TensorStorage resultGrad,
                                        int sizeA, int sizeB, int resultSize,
                                        float aGradScale, float bGradScale )
        {
            var aD = aData.Buffer;
            var bD = bData.Buffer;
            var aG = aGrad.Buffer;
            var bG = bGrad.Buffer;
            var rG = resultGrad.Buffer;
            if (aG == bG)
            {
                for (int i = 0; i < resultSize; i++)
                {
                    int idxA = i % sizeA;
                    int idxB = i % sizeB;
                    float g = rG[ i ];
                    float isAMin = aD[ idxA ] <= bD[ idxB ] ? 1f : 0f;
                    aG[ idxA ] += g * isAMin * aGradScale;
                    bG[ idxB ] += g * (1f - isAMin) * bGradScale;
                }
                return;
            }
            new ElementMinBackwardJob
            {
                AData = aD,
                BData = bD,
                AGrad = aG,
                BGrad = bG,
                ResultGrad = rG,
                SizeA = sizeA,
                SizeB = sizeB,
                AGradScale = aGradScale,
                BGradScale = bGradScale
            }.Schedule( resultSize, TensorOps.GetBatchSize( resultSize ) ).Complete();
        }

        //==================================================================
        //  Element-wise unary — forward
        //==================================================================

        //------------------------------------------------------------------
        public void Pow( TensorStorage input, TensorStorage result, int size, float exponent )
        {
            new ElementPowJob { Input = input.Buffer, Result = result.Buffer, Exponent = exponent }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void Exp( TensorStorage input, TensorStorage result, int size )
        {
            new ElementExpJob { Input = input.Buffer, Result = result.Buffer }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void Log( TensorStorage input, TensorStorage result, int size )
        {
            new ElementLogJob { Input = input.Buffer, Result = result.Buffer }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ReLU( TensorStorage input, TensorStorage result, int size )
        {
            new ElementReLUJob { Input = input.Buffer, Result = result.Buffer }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void Tanh( TensorStorage input, TensorStorage result, int size )
        {
            new ElementTanhJob { Input = input.Buffer, Result = result.Buffer }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void Clamp( TensorStorage input, TensorStorage result, int size, float min, float max )
        {
            new ElementClampJob { Input = input.Buffer, Result = result.Buffer, Min = min, Max = max }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }

        //==================================================================
        //  Element-wise unary — backward
        //==================================================================

        //------------------------------------------------------------------
        public void PowBackward( TensorStorage inputData, TensorStorage inputGrad,
                                 TensorStorage resultGrad, int size, float exponent )
        {
            new PowBackwardJob { InputData = inputData.Buffer, InputGrad = inputGrad.Buffer, ResultGrad = resultGrad.Buffer, Exponent = exponent }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ExpBackward( TensorStorage resultData, TensorStorage inputGrad,
                                 TensorStorage resultGrad, int size )
        {
            new ExpBackwardJob { ResultData = resultData.Buffer, InputGrad = inputGrad.Buffer, ResultGrad = resultGrad.Buffer }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void LogBackward( TensorStorage inputData, TensorStorage inputGrad,
                                 TensorStorage resultGrad, int size )
        {
            new LogBackwardJob { InputData = inputData.Buffer, InputGrad = inputGrad.Buffer, ResultGrad = resultGrad.Buffer }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ReLUBackward( TensorStorage inputData, TensorStorage inputGrad,
                                  TensorStorage resultGrad, int size )
        {
            new ReLUBackwardJob { InputData = inputData.Buffer, InputGrad = inputGrad.Buffer, ResultGrad = resultGrad.Buffer }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void TanhBackward( TensorStorage resultData, TensorStorage inputGrad,
                                  TensorStorage resultGrad, int size )
        {
            new TanhBackwardJob { ResultData = resultData.Buffer, InputGrad = inputGrad.Buffer, ResultGrad = resultGrad.Buffer }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ClampBackward( TensorStorage inputData, TensorStorage inputGrad,
                                   TensorStorage resultGrad, int size, float min, float max )
        {
            new ClampBackwardJob { InputData = inputData.Buffer, InputGrad = inputGrad.Buffer, ResultGrad = resultGrad.Buffer, Min = min, Max = max }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }

        //==================================================================
        //  Reductions
        //==================================================================

        //------------------------------------------------------------------
        public void Sum( TensorStorage input, TensorStorage output, int size )
        {
            new SumReductionJob { Input = input.Buffer, Output = output.Buffer }.Run();
        }
        //------------------------------------------------------------------
        public void SumBackward( TensorStorage inputGrad, TensorStorage outputGrad, int size )
        {
            float gradVal = outputGrad[ 0 ];
            new AddScalarParallelJob { Target = inputGrad.Buffer, Value = gradVal }
                .Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
        public void SumDim( TensorStorage input, TensorStorage output,
                            int outerSize, int dimSize, int innerSize )
        {
            int outputSize = outerSize * innerSize;
            new SumDimJob { Input = input.Buffer, Output = output.Buffer, DimSize = dimSize, InnerSize = innerSize }
                .Schedule( outputSize, TensorOps.GetBatchSize( outputSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void SumDimBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                                    int outerSize, int dimSize, int innerSize )
        {
            int outputSize = outerSize * innerSize;
            new SumDimBackwardJob { InputGrad = inputGrad.Buffer, OutputGrad = outputGrad.Buffer, DimSize = dimSize, InnerSize = innerSize }
                .Schedule( outputSize, TensorOps.GetBatchSize( outputSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void MaxReduce( TensorStorage input, TensorStorage output, int size, out int maxIdx )
        {
            // Note: single-threaded scan — not worth parallelizing for a single scalar result.
            var inputBuf = input.Buffer;
            float maxVal = float.MinValue;
            int bestIdx = 0;
            for (int i = 0; i < size; i++)
            {
                if (inputBuf[ i ] > maxVal)
                {
                    maxVal = inputBuf[ i ];
                    bestIdx = i;
                }
            }
            output[ 0 ] = maxVal;
            maxIdx = bestIdx;
        }
        //------------------------------------------------------------------
        public void MaxReduceDim( TensorStorage input, TensorStorage output, int[] maxIndices,
                                  int outerSize, int dimSize, int innerSize )
        {
            var inputBuf = input.Buffer;
            int blockSize = dimSize * innerSize;

            for (int outer = 0; outer < outerSize; outer++)
            {
                int baseBlock = outer * blockSize;

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int baseIdx = baseBlock + inner;
                    float maxVal = float.MinValue;
                    int maxLocalIdx = 0;

                    for (int d = 0; d < dimSize; d++)
                    {
                        int idx = baseIdx + d * innerSize;
                        if (inputBuf[ idx ] > maxVal)
                        {
                            maxVal = inputBuf[ idx ];
                            maxLocalIdx = d;
                        }
                    }

                    int outIdx = outer * innerSize + inner;
                    output[ outIdx ] = maxVal;
                    maxIndices[ outIdx ] = baseIdx + maxLocalIdx * innerSize;
                }
            }
        }
        //------------------------------------------------------------------
        public void MaxReduceBackward( TensorStorage inputGrad, TensorStorage outputGrad, int maxIdx )
        {
            inputGrad[ maxIdx ] += outputGrad[ 0 ];
        }
        //------------------------------------------------------------------
        public void MaxReduceDimBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                                          int[] maxIndices, int resultSize )
        {
            for (int i = 0; i < resultSize; i++)
            {
                inputGrad[ maxIndices[ i ] ] += outputGrad[ i ];
            }
        }

        //==================================================================
        //  MatMul
        //==================================================================

        //------------------------------------------------------------------
        public void MatMul( TensorStorage a, TensorStorage b, TensorStorage c,
                            int M, int K, int N, bool accumulate )
        {
            TensorOps.ScheduleMatMul( a.Buffer, b.Buffer, c.Buffer, M, K, N, accumulate ).Complete();
        }
        //------------------------------------------------------------------
        public void MatMulBackward( TensorStorage aData, TensorStorage bData,
                                    TensorStorage aGrad, TensorStorage bGrad,
                                    TensorStorage resultGrad,
                                    int M, int K, int N,
                                    bool aRequiresGrad, bool bRequiresGrad )
        {
            JobHandle dAHandle = default, dBHandle = default;
            NativeArray<float> tempBT = default, tempAT = default;

            int aSize = M * K;
            int bSize = K * N;

            if (aRequiresGrad)
            {
                tempBT = new NativeArray<float>( bSize, Allocator.TempJob );
                var tBH = TensorOps.ScheduleTranspose( bData.Buffer, tempBT, K, N );
                dAHandle = TensorOps.ScheduleMatMul(
                    resultGrad.Buffer, tempBT, aGrad.Buffer,
                    M, N, K, accumulate: true, dependsOn: tBH );
            }

            if (bRequiresGrad)
            {
                tempAT = new NativeArray<float>( aSize, Allocator.TempJob );
                var tAH = TensorOps.ScheduleTranspose( aData.Buffer, tempAT, M, K );
                dBHandle = TensorOps.ScheduleMatMul(
                    tempAT, resultGrad.Buffer, bGrad.Buffer,
                    K, M, N, accumulate: true, dependsOn: tAH );
            }

            JobHandle.CombineDependencies( dAHandle, dBHandle ).Complete();

            if (tempBT.IsCreated) tempBT.Dispose();
            if (tempAT.IsCreated) tempAT.Dispose();
        }

        //==================================================================
        //  Data movement
        //==================================================================

        //------------------------------------------------------------------
        public void Copy( TensorStorage src, int srcOffset,
                          TensorStorage dst, int dstOffset, int count )
        {
            NativeArray<float>.Copy( src.Buffer, srcOffset, dst.Buffer, dstOffset, count );
        }
        //------------------------------------------------------------------
        public void SliceCopy( TensorStorage src, TensorStorage dst,
                               int outerSize, int srcBlockSize, int dstBlockSize,
                               int startOffset, int innerSize )
        {
            new SliceCopyJob
            {
                Src = src.Buffer,
                Dst = dst.Buffer,
                SrcBlockSize = srcBlockSize,
                DstBlockSize = dstBlockSize,
                StartOffset = startOffset,
                InnerSize = innerSize
            }.Schedule( outerSize, TensorOps.GetBatchSize( outerSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void SliceCopyBackward( TensorStorage srcGrad, TensorStorage dstGrad,
                                       int outerSize, int srcBlockSize, int dstBlockSize,
                                       int startOffset, int innerSize )
        {
            new SliceCopyBackwardJob
            {
                SrcGrad = srcGrad.Buffer,
                DstGrad = dstGrad.Buffer,
                SrcBlockSize = srcBlockSize,
                DstBlockSize = dstBlockSize,
                StartOffset = startOffset,
                InnerSize = innerSize
            }.Schedule( outerSize, TensorOps.GetBatchSize( outerSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ExpandLast( TensorStorage input, TensorStorage output,
                                int inputSize, int num )
        {
            new ExpandLastJob { Input = input.Buffer, Output = output.Buffer, Num = num }
                .Schedule( inputSize, TensorOps.GetBatchSize( inputSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void ExpandLastBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                                        int inputSize, int num )
        {
            new ExpandLastBackwardJob { InputGrad = inputGrad.Buffer, OutputGrad = outputGrad.Buffer, Num = num }
                .Schedule( inputSize, TensorOps.GetBatchSize( inputSize ) ).Complete();
        }
        //------------------------------------------------------------------
        public void Gather( TensorStorage source, TensorStorage dest,
                            int[] indices, int startIdx, int count, int featureSize )
        {
            var srcBuf = source.Buffer;
            var dstBuf = dest.Buffer;

            for (int i = 0; i < count; i++)
            {
                int srcRow = indices[ startIdx + i ];
                int srcStart = srcRow * featureSize;
                int dstStart = i * featureSize;
                NativeArray<float>.Copy( srcBuf, srcStart, dstBuf, dstStart, featureSize );
            }
        }

        //==================================================================
        //  Utility
        //==================================================================

        //------------------------------------------------------------------
        public unsafe void ZeroGrad( TensorStorage grad, int size )
        {
            UnsafeUtility.MemClear(
                NativeArrayUnsafeUtility.GetUnsafePtr( grad.Buffer ),
                size * sizeof( float ) );
        }
        //------------------------------------------------------------------
        public unsafe void FillOnes( TensorStorage grad, int size )
        {
            float one = 1f;
            UnsafeUtility.MemCpyReplicate(
                NativeArrayUnsafeUtility.GetUnsafePtr( grad.Buffer ),
                &one, sizeof( float ), size );
        }

        //==================================================================
        //  Optimizer
        //==================================================================

        //------------------------------------------------------------------
        public void AdamStep( TensorStorage data, TensorStorage grad,
                              TensorStorage m, TensorStorage v,
                              int size, int momentOffset,
                              float lr, float beta1, float beta2, float epsilon,
                              float invBias1, float invBias2 )
        {
            new AdamStepJob
            {
                Data = data.Buffer,
                Grad = grad.Buffer,
                M = m.Buffer,
                V = v.Buffer,
                MomentOffset = momentOffset,
                LR = lr,
                Beta1 = beta1,
                Beta2 = beta2,
                Epsilon = epsilon,
                InvBias1 = invBias1,
                InvBias2 = invBias2
            }.Schedule( size, TensorOps.GetBatchSize( size ) ).Complete();
        }
        //------------------------------------------------------------------
    }
}
