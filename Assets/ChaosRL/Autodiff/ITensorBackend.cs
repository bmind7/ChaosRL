namespace ChaosRL
{
    /// <summary>
    /// Abstraction over compute backends (CPU Burst, future GPU compute-shader).
    /// Each method performs a synchronous forward or backward kernel for one tensor operation.
    /// All buffers are passed as device-agnostic <see cref="TensorStorage"/>; each backend
    /// extracts the device-native handle (e.g. NativeArray for CPU, ComputeBuffer for GPU).
    /// Implementations are resolved via <see cref="Tensor.ResolveBackend"/>.
    /// </summary>
    public interface ITensorBackend
    {
        //------------------------------------------------------------------
        //  Element-wise binary — forward
        //------------------------------------------------------------------

        /// <summary>result[i] = a[i % sizeA] + b[i % sizeB]</summary>
        void Add( TensorStorage a, TensorStorage b, TensorStorage result,
                  int sizeA, int sizeB, int resultSize );

        /// <summary>result[i] = a[i % sizeA] * b[i % sizeB]</summary>
        void Mul( TensorStorage a, TensorStorage b, TensorStorage result,
                  int sizeA, int sizeB, int resultSize );

        /// <summary>result[i] = a[i % sizeA] / b[i % sizeB]</summary>
        void Div( TensorStorage a, TensorStorage b, TensorStorage result,
                  int sizeA, int sizeB, int resultSize );

        /// <summary>result[i] = max( a[i % sizeA], b[i % sizeB] )</summary>
        void ElementMax( TensorStorage a, TensorStorage b, TensorStorage result,
                         int sizeA, int sizeB, int resultSize );

        /// <summary>result[i] = min( a[i % sizeA], b[i % sizeB] )</summary>
        void ElementMin( TensorStorage a, TensorStorage b, TensorStorage result,
                         int sizeA, int sizeB, int resultSize );

        //------------------------------------------------------------------
        //  Element-wise binary — backward
        //------------------------------------------------------------------

        /// <summary>Backward for Add: aGrad += resultGrad, bGrad += resultGrad (with broadcast).</summary>
        void AddBackward( TensorStorage aGrad, TensorStorage bGrad,
                          TensorStorage resultGrad,
                          int sizeA, int sizeB, int resultSize,
                          float aGradScale, float bGradScale );

        /// <summary>Backward for Mul: aGrad += bData * resultGrad, bGrad += aData * resultGrad (with broadcast).</summary>
        void MulBackward( TensorStorage aData, TensorStorage bData,
                          TensorStorage aGrad, TensorStorage bGrad,
                          TensorStorage resultGrad,
                          int sizeA, int sizeB, int resultSize,
                          float aGradScale, float bGradScale );

        /// <summary>Backward for Div: da = (1/b)*dout, db = (-a/b^2)*dout (with broadcast).</summary>
        void DivBackward( TensorStorage aData, TensorStorage bData,
                          TensorStorage aGrad, TensorStorage bGrad,
                          TensorStorage resultGrad,
                          int sizeA, int sizeB, int resultSize,
                          float aGradScale, float bGradScale );

        /// <summary>Backward for ElementMax: gradient flows to whichever operand was larger.</summary>
        void ElementMaxBackward( TensorStorage aData, TensorStorage bData,
                                 TensorStorage aGrad, TensorStorage bGrad,
                                 TensorStorage resultGrad,
                                 int sizeA, int sizeB, int resultSize,
                                 float aGradScale, float bGradScale );

        /// <summary>Backward for ElementMin: gradient flows to whichever operand was smaller.</summary>
        void ElementMinBackward( TensorStorage aData, TensorStorage bData,
                                 TensorStorage aGrad, TensorStorage bGrad,
                                 TensorStorage resultGrad,
                                 int sizeA, int sizeB, int resultSize,
                                 float aGradScale, float bGradScale );

        //------------------------------------------------------------------
        //  Element-wise unary — forward
        //------------------------------------------------------------------

        /// <summary>result[i] = pow( input[i], exponent )</summary>
        void Pow( TensorStorage input, TensorStorage result, int size, float exponent );

        /// <summary>result[i] = exp( input[i] )</summary>
        void Exp( TensorStorage input, TensorStorage result, int size );

        /// <summary>result[i] = log( input[i] )</summary>
        void Log( TensorStorage input, TensorStorage result, int size );

        /// <summary>result[i] = max( 0, input[i] )</summary>
        void ReLU( TensorStorage input, TensorStorage result, int size );

        /// <summary>result[i] = tanh( input[i] )</summary>
        void Tanh( TensorStorage input, TensorStorage result, int size );

        /// <summary>result[i] = clamp( input[i], min, max )</summary>
        void Clamp( TensorStorage input, TensorStorage result, int size, float min, float max );

        //------------------------------------------------------------------
        //  Element-wise unary — backward
        //------------------------------------------------------------------

        /// <summary>Backward for Pow: grad += exponent * pow(data, exponent-1) * resultGrad.</summary>
        void PowBackward( TensorStorage inputData, TensorStorage inputGrad,
                          TensorStorage resultGrad, int size, float exponent );

        /// <summary>Backward for Exp: grad += resultData * resultGrad.</summary>
        void ExpBackward( TensorStorage resultData, TensorStorage inputGrad,
                          TensorStorage resultGrad, int size );

        /// <summary>Backward for Log: grad += (1/inputData) * resultGrad.</summary>
        void LogBackward( TensorStorage inputData, TensorStorage inputGrad,
                          TensorStorage resultGrad, int size );

        /// <summary>Backward for ReLU: grad += (inputData > 0 ? 1 : 0) * resultGrad.</summary>
        void ReLUBackward( TensorStorage inputData, TensorStorage inputGrad,
                           TensorStorage resultGrad, int size );

        /// <summary>Backward for Tanh: grad += (1 - resultData^2) * resultGrad.</summary>
        void TanhBackward( TensorStorage resultData, TensorStorage inputGrad,
                           TensorStorage resultGrad, int size );

        /// <summary>Backward for Clamp: grad += resultGrad where inputData is within [min,max].</summary>
        void ClampBackward( TensorStorage inputData, TensorStorage inputGrad,
                            TensorStorage resultGrad, int size, float min, float max );

        //------------------------------------------------------------------
        //  Reductions
        //------------------------------------------------------------------

        /// <summary>Sum all elements to output[0].</summary>
        void Sum( TensorStorage input, TensorStorage output, int size );

        /// <summary>Backward for Sum: inputGrad[i] += outputGrad[0].</summary>
        void SumBackward( TensorStorage inputGrad, TensorStorage outputGrad, int size );

        /// <summary>Sum along a specific dimension.</summary>
        void SumDim( TensorStorage input, TensorStorage output,
                     int outerSize, int dimSize, int innerSize );

        /// <summary>Backward for SumDim: broadcast gradient back over reduced dimension.</summary>
        void SumDimBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                             int outerSize, int dimSize, int innerSize );

        /// <summary>Max reduction to scalar: output[0] = max(input), maxIdx is the flat index of the max.</summary>
        void MaxReduce( TensorStorage input, TensorStorage output, int size, out int maxIdx );

        /// <summary>Max along a specific dimension.</summary>
        void MaxReduceDim( TensorStorage input, TensorStorage output, int[] maxIndices,
                           int outerSize, int dimSize, int innerSize );

        /// <summary>Backward for MaxReduce: gradient flows only to the max element.</summary>
        void MaxReduceBackward( TensorStorage inputGrad, TensorStorage outputGrad, int maxIdx );

        /// <summary>Backward for MaxReduceDim: gradient flows to stored max-element positions.</summary>
        void MaxReduceDimBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                                   int[] maxIndices, int resultSize );

        //------------------------------------------------------------------
        //  MatMul
        //------------------------------------------------------------------

        /// <summary>C(M×N) = A(M×K) @ B(K×N). If accumulate, C += A@B.</summary>
        void MatMul( TensorStorage a, TensorStorage b, TensorStorage c,
                     int M, int K, int N, bool accumulate );

        /// <summary>Backward for MatMul: dA = dC @ B^T, dB = A^T @ dC.</summary>
        void MatMulBackward( TensorStorage aData, TensorStorage bData,
                             TensorStorage aGrad, TensorStorage bGrad,
                             TensorStorage resultGrad,
                             int M, int K, int N,
                             bool aRequiresGrad, bool bRequiresGrad );

        //------------------------------------------------------------------
        //  Data movement
        //------------------------------------------------------------------

        /// <summary>Copy a contiguous block of floats between storages.</summary>
        void Copy( TensorStorage src, int srcOffset,
                   TensorStorage dst, int dstOffset, int count );

        /// <summary>Slice forward: copies strided blocks from src to dst.</summary>
        void SliceCopy( TensorStorage src, TensorStorage dst,
                        int outerSize, int srcBlockSize, int dstBlockSize,
                        int startOffset, int innerSize );

        /// <summary>Slice backward: accumulates gradient back to source positions.</summary>
        void SliceCopyBackward( TensorStorage srcGrad, TensorStorage dstGrad,
                                int outerSize, int srcBlockSize, int dstBlockSize,
                                int startOffset, int innerSize );

        /// <summary>ExpandLast forward: replicate each element num times.</summary>
        void ExpandLast( TensorStorage input, TensorStorage output,
                         int inputSize, int num );

        /// <summary>ExpandLast backward: accumulate gradients from replicated elements.</summary>
        void ExpandLastBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                                 int inputSize, int num );

        /// <summary>
        /// Gather rows from source into dest using an index array.
        /// Each index selects a contiguous block of <paramref name="featureSize"/> elements.
        /// Caller must ensure all indices are in valid range (0 to source.Length / featureSize - 1).
        /// </summary>
        void Gather( TensorStorage source, TensorStorage dest,
                     int[] indices, int startIdx, int count, int featureSize );

        //------------------------------------------------------------------
        //  Utility
        //------------------------------------------------------------------

        /// <summary>Zero all elements in the storage.</summary>
        void ZeroGrad( TensorStorage grad, int size );

        /// <summary>Fill all elements with 1.0f (backward seed).</summary>
        void FillOnes( TensorStorage grad, int size );

        //------------------------------------------------------------------
        //  Optimizer
        //------------------------------------------------------------------

        /// <summary>
        /// Adam optimizer step for a single parameter tensor.
        /// Updates data in-place, reads grad, reads/writes moment buffers.
        /// </summary>
        void AdamStep( TensorStorage data, TensorStorage grad,
                       TensorStorage m, TensorStorage v,
                       int size, int momentOffset,
                       float lr, float beta1, float beta2, float epsilon,
                       float invBias1, float invBias2 );
        //------------------------------------------------------------------
    }
}
