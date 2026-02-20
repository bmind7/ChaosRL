using System;

using UnityEngine;

namespace ChaosRL
{
    /// <summary>
    /// GPU backend implementing <see cref="ITensorBackend"/> via Unity compute shaders.
    /// All methods are synchronous — they dispatch compute kernels and return.
    /// GPU command buffer is implicitly flushed by Unity at frame boundaries
    /// or when a CPU readback (<see cref="GraphicsBuffer.GetData"/>) is issued.
    /// </summary>
    public class GpuBackend : ITensorBackend
    {
        //==================================================================
        //  Constants
        //==================================================================

        private const int THREADS = 256;
        private const int TILE = 16;

        //==================================================================
        //  Shader + kernel indices (cached once at construction)
        //==================================================================

        private readonly ComputeShader _shader;

        // Element-wise binary — forward
        private readonly int _elementAddKernel;
        private readonly int _elementMulKernel;
        private readonly int _elementDivKernel;
        private readonly int _elementMaxKernel;
        private readonly int _elementMinKernel;

        // Element-wise binary — backward
        private readonly int _addBackwardKernel;
        private readonly int _mulBackwardKernel;
        private readonly int _divBackwardKernel;
        private readonly int _elementMaxBackwardKernel;
        private readonly int _elementMinBackwardKernel;

        // Element-wise unary — forward
        private readonly int _powKernel;
        private readonly int _expKernel;
        private readonly int _logKernel;
        private readonly int _reluKernel;
        private readonly int _tanhKernel;
        private readonly int _clampKernel;

        // Element-wise unary — backward
        private readonly int _powBackwardKernel;
        private readonly int _expBackwardKernel;
        private readonly int _logBackwardKernel;
        private readonly int _reluBackwardKernel;
        private readonly int _tanhBackwardKernel;
        private readonly int _clampBackwardKernel;

        // Reductions
        private readonly int _sumReduceKernel;
        private readonly int _sumBackwardKernel;
        private readonly int _sumDimKernel;
        private readonly int _sumDimBackwardKernel;
        private readonly int _maxReduceKernel;
        private readonly int _maxReduceBackwardKernel;
        private readonly int _maxReduceDimKernel;
        private readonly int _maxReduceDimBackwardKernel;

        // MatMul
        private readonly int _matMulForwardKernel;
        private readonly int _transposeKernel;

        // Data movement
        private readonly int _copyKernel;
        private readonly int _sliceCopyKernel;
        private readonly int _sliceCopyBackwardKernel;
        private readonly int _expandLastKernel;
        private readonly int _expandLastBackwardKernel;
        private readonly int _gatherKernel;

        // Utility
        private readonly int _zeroFillKernel;
        private readonly int _oneFillKernel;
        private readonly int _valueFillKernel;

        // Optimizer
        private readonly int _adamStepKernel;

        //==================================================================
        //  Construction
        //==================================================================

        //------------------------------------------------------------------
        public GpuBackend()
        {
            _shader = Resources.Load<ComputeShader>( "TensorOps" );
            if (_shader == null)
                throw new InvalidOperationException(
                    "Could not load TensorOps compute shader from Resources/. " +
                    "Ensure Assets/Resources/TensorOps.compute exists." );

            // Element-wise binary — forward
            _elementAddKernel = _shader.FindKernel( "ElementAdd" );
            _elementMulKernel = _shader.FindKernel( "ElementMul" );
            _elementDivKernel = _shader.FindKernel( "ElementDiv" );
            _elementMaxKernel = _shader.FindKernel( "ElementMax" );
            _elementMinKernel = _shader.FindKernel( "ElementMin" );

            // Element-wise binary — backward
            _addBackwardKernel = _shader.FindKernel( "AddBackward" );
            _mulBackwardKernel = _shader.FindKernel( "MulBackward" );
            _divBackwardKernel = _shader.FindKernel( "DivBackward" );
            _elementMaxBackwardKernel = _shader.FindKernel( "ElementMaxBackward" );
            _elementMinBackwardKernel = _shader.FindKernel( "ElementMinBackward" );

            // Element-wise unary — forward
            _powKernel = _shader.FindKernel( "Pow" );
            _expKernel = _shader.FindKernel( "Exp" );
            _logKernel = _shader.FindKernel( "Log" );
            _reluKernel = _shader.FindKernel( "ReLU" );
            _tanhKernel = _shader.FindKernel( "Tanh" );
            _clampKernel = _shader.FindKernel( "Clamp" );

            // Element-wise unary — backward
            _powBackwardKernel = _shader.FindKernel( "PowBackward" );
            _expBackwardKernel = _shader.FindKernel( "ExpBackward" );
            _logBackwardKernel = _shader.FindKernel( "LogBackward" );
            _reluBackwardKernel = _shader.FindKernel( "ReLUBackward" );
            _tanhBackwardKernel = _shader.FindKernel( "TanhBackward" );
            _clampBackwardKernel = _shader.FindKernel( "ClampBackward" );

            // Reductions
            _sumReduceKernel = _shader.FindKernel( "SumReduce" );
            _sumBackwardKernel = _shader.FindKernel( "SumBackward" );
            _sumDimKernel = _shader.FindKernel( "SumDim" );
            _sumDimBackwardKernel = _shader.FindKernel( "SumDimBackward" );
            _maxReduceKernel = _shader.FindKernel( "MaxReduce" );
            _maxReduceBackwardKernel = _shader.FindKernel( "MaxReduceBackward" );
            _maxReduceDimKernel = _shader.FindKernel( "MaxReduceDim" );
            _maxReduceDimBackwardKernel = _shader.FindKernel( "MaxReduceDimBackward" );

            // MatMul
            _matMulForwardKernel = _shader.FindKernel( "MatMulForward" );
            _transposeKernel = _shader.FindKernel( "Transpose" );

            // Data movement
            _copyKernel = _shader.FindKernel( "Copy" );
            _sliceCopyKernel = _shader.FindKernel( "SliceCopy" );
            _sliceCopyBackwardKernel = _shader.FindKernel( "SliceCopyBackward" );
            _expandLastKernel = _shader.FindKernel( "ExpandLast" );
            _expandLastBackwardKernel = _shader.FindKernel( "ExpandLastBackward" );
            _gatherKernel = _shader.FindKernel( "Gather" );

            // Utility
            _zeroFillKernel = _shader.FindKernel( "ZeroFill" );
            _oneFillKernel = _shader.FindKernel( "OneFill" );
            _valueFillKernel = _shader.FindKernel( "ValueFill" );

            // Optimizer
            _adamStepKernel = _shader.FindKernel( "AdamStep" );
        }

        //==================================================================
        //  Auto-registration
        //==================================================================

        //------------------------------------------------------------------
        /// <summary>
        /// Automatically registers the GPU backend at application startup
        /// if the platform supports compute shaders.
        //==================================================================
        //  Element-wise binary — forward
        //==================================================================

        //------------------------------------------------------------------
        public void Add( TensorStorage a, TensorStorage b, TensorStorage result,
                         int sizeA, int sizeB, int resultSize )
        {
            SetBinaryForwardParams( _elementAddKernel, a, b, result, sizeA, sizeB, resultSize );
            _shader.Dispatch( _elementAddKernel, Groups( resultSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void Mul( TensorStorage a, TensorStorage b, TensorStorage result,
                         int sizeA, int sizeB, int resultSize )
        {
            SetBinaryForwardParams( _elementMulKernel, a, b, result, sizeA, sizeB, resultSize );
            _shader.Dispatch( _elementMulKernel, Groups( resultSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void Div( TensorStorage a, TensorStorage b, TensorStorage result,
                         int sizeA, int sizeB, int resultSize )
        {
            SetBinaryForwardParams( _elementDivKernel, a, b, result, sizeA, sizeB, resultSize );
            _shader.Dispatch( _elementDivKernel, Groups( resultSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ElementMax( TensorStorage a, TensorStorage b, TensorStorage result,
                                int sizeA, int sizeB, int resultSize )
        {
            SetBinaryForwardParams( _elementMaxKernel, a, b, result, sizeA, sizeB, resultSize );
            _shader.Dispatch( _elementMaxKernel, Groups( resultSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ElementMin( TensorStorage a, TensorStorage b, TensorStorage result,
                                int sizeA, int sizeB, int resultSize )
        {
            SetBinaryForwardParams( _elementMinKernel, a, b, result, sizeA, sizeB, resultSize );
            _shader.Dispatch( _elementMinKernel, Groups( resultSize ), 1, 1 );
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
            SetBinaryBackwardParams( _addBackwardKernel, null, null, aGrad, bGrad,
                                     resultGrad, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            _shader.Dispatch( _addBackwardKernel, Groups( resultSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void MulBackward( TensorStorage aData, TensorStorage bData,
                                 TensorStorage aGrad, TensorStorage bGrad,
                                 TensorStorage resultGrad,
                                 int sizeA, int sizeB, int resultSize,
                                 float aGradScale, float bGradScale )
        {
            SetBinaryBackwardParams( _mulBackwardKernel, aData, bData, aGrad, bGrad,
                                     resultGrad, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            _shader.Dispatch( _mulBackwardKernel, Groups( resultSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void DivBackward( TensorStorage aData, TensorStorage bData,
                                 TensorStorage aGrad, TensorStorage bGrad,
                                 TensorStorage resultGrad,
                                 int sizeA, int sizeB, int resultSize,
                                 float aGradScale, float bGradScale )
        {
            SetBinaryBackwardParams( _divBackwardKernel, aData, bData, aGrad, bGrad,
                                     resultGrad, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            _shader.Dispatch( _divBackwardKernel, Groups( resultSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ElementMaxBackward( TensorStorage aData, TensorStorage bData,
                                        TensorStorage aGrad, TensorStorage bGrad,
                                        TensorStorage resultGrad,
                                        int sizeA, int sizeB, int resultSize,
                                        float aGradScale, float bGradScale )
        {
            SetBinaryBackwardParams( _elementMaxBackwardKernel, aData, bData, aGrad, bGrad,
                                     resultGrad, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            _shader.Dispatch( _elementMaxBackwardKernel, Groups( resultSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ElementMinBackward( TensorStorage aData, TensorStorage bData,
                                        TensorStorage aGrad, TensorStorage bGrad,
                                        TensorStorage resultGrad,
                                        int sizeA, int sizeB, int resultSize,
                                        float aGradScale, float bGradScale )
        {
            SetBinaryBackwardParams( _elementMinBackwardKernel, aData, bData, aGrad, bGrad,
                                     resultGrad, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            _shader.Dispatch( _elementMinBackwardKernel, Groups( resultSize ), 1, 1 );
        }

        //==================================================================
        //  Element-wise unary — forward
        //==================================================================

        //------------------------------------------------------------------
        public void Pow( TensorStorage input, TensorStorage result, int size, float exponent )
        {
            SetUnaryForwardParams( _powKernel, input, result, size );
            _shader.SetFloat( "_Exponent", exponent );
            _shader.Dispatch( _powKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void Exp( TensorStorage input, TensorStorage result, int size )
        {
            SetUnaryForwardParams( _expKernel, input, result, size );
            _shader.Dispatch( _expKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void Log( TensorStorage input, TensorStorage result, int size )
        {
            SetUnaryForwardParams( _logKernel, input, result, size );
            _shader.Dispatch( _logKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ReLU( TensorStorage input, TensorStorage result, int size )
        {
            SetUnaryForwardParams( _reluKernel, input, result, size );
            _shader.Dispatch( _reluKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void Tanh( TensorStorage input, TensorStorage result, int size )
        {
            SetUnaryForwardParams( _tanhKernel, input, result, size );
            _shader.Dispatch( _tanhKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void Clamp( TensorStorage input, TensorStorage result, int size, float min, float max )
        {
            SetUnaryForwardParams( _clampKernel, input, result, size );
            _shader.SetFloat( "_MinVal", min );
            _shader.SetFloat( "_MaxVal", max );
            _shader.Dispatch( _clampKernel, Groups( size ), 1, 1 );
        }

        //==================================================================
        //  Element-wise unary — backward
        //==================================================================

        //------------------------------------------------------------------
        public void PowBackward( TensorStorage inputData, TensorStorage inputGrad,
                                 TensorStorage resultGrad, int size, float exponent )
        {
            SetUnaryBackwardParams( _powBackwardKernel, inputData, null, inputGrad, resultGrad, size );
            _shader.SetFloat( "_Exponent", exponent );
            _shader.Dispatch( _powBackwardKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ExpBackward( TensorStorage resultData, TensorStorage inputGrad,
                                 TensorStorage resultGrad, int size )
        {
            SetUnaryBackwardParams( _expBackwardKernel, null, resultData, inputGrad, resultGrad, size );
            _shader.Dispatch( _expBackwardKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void LogBackward( TensorStorage inputData, TensorStorage inputGrad,
                                 TensorStorage resultGrad, int size )
        {
            SetUnaryBackwardParams( _logBackwardKernel, inputData, null, inputGrad, resultGrad, size );
            _shader.Dispatch( _logBackwardKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ReLUBackward( TensorStorage inputData, TensorStorage inputGrad,
                                  TensorStorage resultGrad, int size )
        {
            SetUnaryBackwardParams( _reluBackwardKernel, inputData, null, inputGrad, resultGrad, size );
            _shader.Dispatch( _reluBackwardKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void TanhBackward( TensorStorage resultData, TensorStorage inputGrad,
                                  TensorStorage resultGrad, int size )
        {
            SetUnaryBackwardParams( _tanhBackwardKernel, null, resultData, inputGrad, resultGrad, size );
            _shader.Dispatch( _tanhBackwardKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ClampBackward( TensorStorage inputData, TensorStorage inputGrad,
                                   TensorStorage resultGrad, int size, float min, float max )
        {
            SetUnaryBackwardParams( _clampBackwardKernel, inputData, null, inputGrad, resultGrad, size );
            _shader.SetFloat( "_MinVal", min );
            _shader.SetFloat( "_MaxVal", max );
            _shader.Dispatch( _clampBackwardKernel, Groups( size ), 1, 1 );
        }

        //==================================================================
        //  Reductions
        //==================================================================

        //------------------------------------------------------------------
        public void Sum( TensorStorage input, TensorStorage output, int size )
        {
            int remaining = size;
            int numGroups = Groups( remaining );

            if (numGroups <= 1)
            {
                // Single group — write directly to output
                _shader.SetInt( "_Size", remaining );
                _shader.SetBuffer( _sumReduceKernel, "_Input", input.GpuBuffer );
                _shader.SetBuffer( _sumReduceKernel, "_Output", output.GpuBuffer );
                _shader.Dispatch( _sumReduceKernel, 1, 1, 1 );
                return;
            }

            // Multi-pass: keep reducing until a single group remains.
            // Ping-pong between two temp buffers to avoid read-write hazards.
            var tempA = AllocTemp( numGroups );
            GraphicsBuffer tempB = null;

            try
            {
                // First pass: input -> tempA
                _shader.SetInt( "_Size", remaining );
                _shader.SetBuffer( _sumReduceKernel, "_Input", input.GpuBuffer );
                _shader.SetBuffer( _sumReduceKernel, "_Output", tempA );
                _shader.Dispatch( _sumReduceKernel, numGroups, 1, 1 );

                remaining = numGroups;
                numGroups = Groups( remaining );
                bool readFromA = true;

                // Subsequent passes until one group suffices
                while (numGroups > 1)
                {
                    if (readFromA)
                    {
                        if (tempB == null) tempB = AllocTemp( numGroups );
                        _shader.SetInt( "_Size", remaining );
                        _shader.SetBuffer( _sumReduceKernel, "_Input", tempA );
                        _shader.SetBuffer( _sumReduceKernel, "_Output", tempB );
                        _shader.Dispatch( _sumReduceKernel, numGroups, 1, 1 );
                    }
                    else
                    {
                        _shader.SetInt( "_Size", remaining );
                        _shader.SetBuffer( _sumReduceKernel, "_Input", tempB );
                        _shader.SetBuffer( _sumReduceKernel, "_Output", tempA );
                        _shader.Dispatch( _sumReduceKernel, numGroups, 1, 1 );
                    }

                    remaining = numGroups;
                    numGroups = Groups( remaining );
                    readFromA = !readFromA;
                }

                // Final pass: reduce to output[0]
                var src = readFromA ? tempA : tempB;
                _shader.SetInt( "_Size", remaining );
                _shader.SetBuffer( _sumReduceKernel, "_Input", src );
                _shader.SetBuffer( _sumReduceKernel, "_Output", output.GpuBuffer );
                _shader.Dispatch( _sumReduceKernel, 1, 1, 1 );
            }
            finally
            {
                tempA.Release();
                tempB?.Release();
            }
        }
        //------------------------------------------------------------------
        public void SumBackward( TensorStorage inputGrad, TensorStorage outputGrad, int size )
        {
            _shader.SetInt( "_Size", size );
            _shader.SetBuffer( _sumBackwardKernel, "_InputGrad", inputGrad.GpuBuffer );
            _shader.SetBuffer( _sumBackwardKernel, "_Output", outputGrad.GpuBuffer );
            _shader.Dispatch( _sumBackwardKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void SumDim( TensorStorage input, TensorStorage output,
                            int outerSize, int dimSize, int innerSize )
        {
            int workItems = outerSize * innerSize;
            _shader.SetInt( "_OuterSize", outerSize );
            _shader.SetInt( "_DimSize", dimSize );
            _shader.SetInt( "_InnerSize", innerSize );
            _shader.SetBuffer( _sumDimKernel, "_Input", input.GpuBuffer );
            _shader.SetBuffer( _sumDimKernel, "_Output", output.GpuBuffer );
            _shader.Dispatch( _sumDimKernel, Groups( workItems ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void SumDimBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                                    int outerSize, int dimSize, int innerSize )
        {
            int workItems = outerSize * innerSize;
            _shader.SetInt( "_OuterSize", outerSize );
            _shader.SetInt( "_DimSize", dimSize );
            _shader.SetInt( "_InnerSize", innerSize );
            _shader.SetBuffer( _sumDimBackwardKernel, "_InputGrad", inputGrad.GpuBuffer );
            _shader.SetBuffer( _sumDimBackwardKernel, "_ResultGrad", outputGrad.GpuBuffer );
            _shader.Dispatch( _sumDimBackwardKernel, Groups( workItems ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void MaxReduce( TensorStorage input, TensorStorage output, int size, out int maxIdx )
        {
            int numGroups = Groups( size );

            if (numGroups <= 1)
            {
                using (var idxBuf = AllocTempInt( 1 ))
                {
                    _shader.SetInt( "_Size", size );
                    _shader.SetBuffer( _maxReduceKernel, "_Input", input.GpuBuffer );
                    _shader.SetBuffer( _maxReduceKernel, "_Output", output.GpuBuffer );
                    _shader.SetBuffer( _maxReduceKernel, "_MaxIdxBuf", idxBuf );
                    _shader.Dispatch( _maxReduceKernel, 1, 1, 1 );

                    var idx = new int[ 1 ];
                    idxBuf.GetData( idx );
                    maxIdx = idx[ 0 ];
                }
            }
            else
            {
                // Two-pass
                using (var tempVal = AllocTemp( numGroups ))
                using (var tempIdx = AllocTempInt( numGroups ))
                {
                    // Pass 1: per-group max
                    _shader.SetInt( "_Size", size );
                    _shader.SetBuffer( _maxReduceKernel, "_Input", input.GpuBuffer );
                    _shader.SetBuffer( _maxReduceKernel, "_Output", tempVal );
                    _shader.SetBuffer( _maxReduceKernel, "_MaxIdxBuf", tempIdx );
                    _shader.Dispatch( _maxReduceKernel, numGroups, 1, 1 );

                    // Read back partial results to CPU for final reduction
                    // (numGroups is small, typically < 1000)
                    var partialVals = new float[ numGroups ];
                    var partialIdxs = new int[ numGroups ];
                    tempVal.GetData( partialVals );
                    tempIdx.GetData( partialIdxs );

                    float bestVal = float.MinValue;
                    int bestIdx = 0;
                    for (int i = 0; i < numGroups; i++)
                    {
                        if (partialVals[ i ] > bestVal)
                        {
                            bestVal = partialVals[ i ];
                            bestIdx = partialIdxs[ i ];
                        }
                    }

                    output[ 0 ] = bestVal;
                    maxIdx = bestIdx;
                }
            }
        }
        //------------------------------------------------------------------
        public void MaxReduceDim( TensorStorage input, TensorStorage output, int[] maxIndices,
                                  int outerSize, int dimSize, int innerSize )
        {
            int workItems = outerSize * innerSize;
            using (var idxBuf = AllocTempInt( workItems ))
            {
                _shader.SetInt( "_OuterSize", outerSize );
                _shader.SetInt( "_DimSize", dimSize );
                _shader.SetInt( "_InnerSize", innerSize );
                _shader.SetBuffer( _maxReduceDimKernel, "_Input", input.GpuBuffer );
                _shader.SetBuffer( _maxReduceDimKernel, "_Output", output.GpuBuffer );
                _shader.SetBuffer( _maxReduceDimKernel, "_MaxIdxBuf", idxBuf );
                _shader.Dispatch( _maxReduceDimKernel, Groups( workItems ), 1, 1 );

                // Read back indices to managed array
                idxBuf.GetData( maxIndices );
            }
        }
        //------------------------------------------------------------------
        public void MaxReduceBackward( TensorStorage inputGrad, TensorStorage outputGrad, int maxIdx )
        {
            // Upload the single max index
            using (var idxBuf = AllocTempInt( 1 ))
            {
                idxBuf.SetData( new[] { maxIdx } );
                _shader.SetBuffer( _maxReduceBackwardKernel, "_InputGrad", inputGrad.GpuBuffer );
                _shader.SetBuffer( _maxReduceBackwardKernel, "_Output", outputGrad.GpuBuffer );
                _shader.SetBuffer( _maxReduceBackwardKernel, "_MaxIdxBuf", idxBuf );
                _shader.Dispatch( _maxReduceBackwardKernel, 1, 1, 1 );
            }
        }
        //------------------------------------------------------------------
        public void MaxReduceDimBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                                          int[] maxIndices, int resultSize )
        {
            using (var idxBuf = AllocTempInt( resultSize ))
            {
                idxBuf.SetData( maxIndices );
                _shader.SetInt( "_Size", resultSize );
                _shader.SetBuffer( _maxReduceDimBackwardKernel, "_InputGrad", inputGrad.GpuBuffer );
                _shader.SetBuffer( _maxReduceDimBackwardKernel, "_ResultGrad", outputGrad.GpuBuffer );
                _shader.SetBuffer( _maxReduceDimBackwardKernel, "_MaxIdxBuf", idxBuf );
                _shader.Dispatch( _maxReduceDimBackwardKernel, Groups( resultSize ), 1, 1 );
            }
        }

        //==================================================================
        //  MatMul
        //==================================================================

        //------------------------------------------------------------------
        public void MatMul( TensorStorage a, TensorStorage b, TensorStorage c,
                            int M, int K, int N, bool accumulate )
        {
            _shader.SetInt( "_M", M );
            _shader.SetInt( "_K", K );
            _shader.SetInt( "_N", N );
            _shader.SetInt( "_Accumulate", accumulate ? 1 : 0 );
            _shader.SetBuffer( _matMulForwardKernel, "_A", a.GpuBuffer );
            _shader.SetBuffer( _matMulForwardKernel, "_B", b.GpuBuffer );
            _shader.SetBuffer( _matMulForwardKernel, "_Result", c.GpuBuffer );

            // The kernel computes a (TILE*2)x(TILE*2) block per threadgroup
            int tileOut = TILE * 2;
            int gx = (N + tileOut - 1) / tileOut;
            int gy = (M + tileOut - 1) / tileOut;
            _shader.Dispatch( _matMulForwardKernel, gx, gy, 1 );
        }
        //------------------------------------------------------------------
        public void MatMulBackward( TensorStorage aData, TensorStorage bData,
                                    TensorStorage aGrad, TensorStorage bGrad,
                                    TensorStorage resultGrad,
                                    int M, int K, int N,
                                    bool aRequiresGrad, bool bRequiresGrad )
        {
            // dA = dC(M,N) @ B^T(N,K)   — need B transposed (K,N) -> (N,K)
            // dB = A^T(K,M) @ dC(M,N)   — need A transposed (M,K) -> (K,M)

            GraphicsBuffer tempBT = null;
            GraphicsBuffer tempAT = null;

            try
            {
                if (aRequiresGrad)
                {
                    int bSize = K * N;
                    tempBT = AllocTempRaw( bSize );
                    DispatchTranspose( bData.GpuBuffer, tempBT, K, N, bSize );
                    // dA(M,K) += dC(M,N) @ BT(N,K)
                    MatMul_Internal( resultGrad.GpuBuffer, tempBT, aGrad.GpuBuffer, M, N, K, accumulate: true );
                }

                if (bRequiresGrad)
                {
                    int aSize = M * K;
                    tempAT = AllocTempRaw( aSize );
                    DispatchTranspose( aData.GpuBuffer, tempAT, M, K, aSize );
                    // dB(K,N) += AT(K,M) @ dC(M,N)
                    MatMul_Internal( tempAT, resultGrad.GpuBuffer, bGrad.GpuBuffer, K, M, N, accumulate: true );
                }
            }
            finally
            {
                tempBT?.Release();
                tempAT?.Release();
            }
        }

        //==================================================================
        //  Data movement
        //==================================================================

        //------------------------------------------------------------------
        public void Copy( TensorStorage src, int srcOffset,
                          TensorStorage dst, int dstOffset, int count )
        {
            _shader.SetInt( "_SrcOffset", srcOffset );
            _shader.SetInt( "_DstOffset", dstOffset );
            _shader.SetInt( "_Count", count );
            _shader.SetBuffer( _copyKernel, "_Src", src.GpuBuffer );
            _shader.SetBuffer( _copyKernel, "_Dst", dst.GpuBuffer );
            _shader.Dispatch( _copyKernel, Groups( count ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void SliceCopy( TensorStorage src, TensorStorage dst,
                               int outerSize, int srcBlockSize, int dstBlockSize,
                               int startOffset, int innerSize )
        {
            _shader.SetInt( "_OuterSize", outerSize );
            _shader.SetInt( "_SrcBlockSize", srcBlockSize );
            _shader.SetInt( "_DstBlockSize", dstBlockSize );
            _shader.SetInt( "_StartOffset", startOffset );
            _shader.SetInt( "_InnerSize", innerSize );
            _shader.SetBuffer( _sliceCopyKernel, "_Src", src.GpuBuffer );
            _shader.SetBuffer( _sliceCopyKernel, "_Dst", dst.GpuBuffer );
            _shader.Dispatch( _sliceCopyKernel, Groups( outerSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void SliceCopyBackward( TensorStorage srcGrad, TensorStorage dstGrad,
                                       int outerSize, int srcBlockSize, int dstBlockSize,
                                       int startOffset, int innerSize )
        {
            // Note: in the compute shader, _Src reads dstGrad, _Dst writes to srcGrad
            _shader.SetInt( "_OuterSize", outerSize );
            _shader.SetInt( "_SrcBlockSize", srcBlockSize );
            _shader.SetInt( "_DstBlockSize", dstBlockSize );
            _shader.SetInt( "_StartOffset", startOffset );
            _shader.SetInt( "_InnerSize", innerSize );
            _shader.SetBuffer( _sliceCopyBackwardKernel, "_Src", dstGrad.GpuBuffer );
            _shader.SetBuffer( _sliceCopyBackwardKernel, "_Dst", srcGrad.GpuBuffer );
            _shader.Dispatch( _sliceCopyBackwardKernel, Groups( outerSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ExpandLast( TensorStorage input, TensorStorage output,
                                int inputSize, int num )
        {
            _shader.SetInt( "_Size", inputSize );
            _shader.SetInt( "_Num", num );
            _shader.SetBuffer( _expandLastKernel, "_Input", input.GpuBuffer );
            _shader.SetBuffer( _expandLastKernel, "_Result", output.GpuBuffer );
            _shader.Dispatch( _expandLastKernel, Groups( inputSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void ExpandLastBackward( TensorStorage inputGrad, TensorStorage outputGrad,
                                        int inputSize, int num )
        {
            _shader.SetInt( "_Size", inputSize );
            _shader.SetInt( "_Num", num );
            _shader.SetBuffer( _expandLastBackwardKernel, "_InputGrad", inputGrad.GpuBuffer );
            _shader.SetBuffer( _expandLastBackwardKernel, "_ResultGrad", outputGrad.GpuBuffer );
            _shader.Dispatch( _expandLastBackwardKernel, Groups( inputSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void Gather( TensorStorage source, TensorStorage dest,
                            int[] indices, int startIdx, int count, int featureSize )
        {
            // Upload the index slice to a temporary GPU buffer
            var idxSlice = new int[ count ];
            Array.Copy( indices, startIdx, idxSlice, 0, count );

            using (var idxBuf = AllocTempInt( count ))
            {
                idxBuf.SetData( idxSlice );

                _shader.SetInt( "_Count", count );
                _shader.SetInt( "_FeatureSize", featureSize );
                _shader.SetBuffer( _gatherKernel, "_Src", source.GpuBuffer );
                _shader.SetBuffer( _gatherKernel, "_Dst", dest.GpuBuffer );
                _shader.SetBuffer( _gatherKernel, "_Indices", idxBuf );
                _shader.Dispatch( _gatherKernel, Groups( count ), 1, 1 );
            }
        }

        //==================================================================
        //  Utility
        //==================================================================

        //------------------------------------------------------------------
        public void ZeroGrad( TensorStorage grad, int size )
        {
            _shader.SetInt( "_Size", size );
            _shader.SetBuffer( _zeroFillKernel, "_Result", grad.GpuBuffer );
            _shader.Dispatch( _zeroFillKernel, Groups( size ), 1, 1 );
        }
        //------------------------------------------------------------------
        public void FillOnes( TensorStorage grad, int size )
        {
            _shader.SetInt( "_Size", size );
            _shader.SetBuffer( _oneFillKernel, "_Result", grad.GpuBuffer );
            _shader.Dispatch( _oneFillKernel, Groups( size ), 1, 1 );
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
            _shader.SetInt( "_Size", size );
            _shader.SetInt( "_MomentOffset", momentOffset );
            _shader.SetFloat( "_LR", lr );
            _shader.SetFloat( "_Beta1", beta1 );
            _shader.SetFloat( "_Beta2", beta2 );
            _shader.SetFloat( "_Epsilon", epsilon );
            _shader.SetFloat( "_InvBias1", invBias1 );
            _shader.SetFloat( "_InvBias2", invBias2 );
            _shader.SetBuffer( _adamStepKernel, "_Data", data.GpuBuffer );
            _shader.SetBuffer( _adamStepKernel, "_Grad", grad.GpuBuffer );
            _shader.SetBuffer( _adamStepKernel, "_MBuf", m.GpuBuffer );
            _shader.SetBuffer( _adamStepKernel, "_VBuf", v.GpuBuffer );
            _shader.Dispatch( _adamStepKernel, Groups( size ), 1, 1 );
        }

        //==================================================================
        //  Internal helpers
        //==================================================================

        //------------------------------------------------------------------
        private static int Groups( int workItems )
        {
            return Math.Max( 1, (workItems + THREADS - 1) / THREADS );
        }
        //------------------------------------------------------------------
        private static GraphicsBuffer AllocTemp( int count )
        {
            return new GraphicsBuffer( GraphicsBuffer.Target.Structured, Math.Max( 1, count ), sizeof( float ) );
        }
        //------------------------------------------------------------------
        private static GraphicsBuffer AllocTempInt( int count )
        {
            return new GraphicsBuffer( GraphicsBuffer.Target.Structured, Math.Max( 1, count ), sizeof( int ) );
        }
        //------------------------------------------------------------------
        private static GraphicsBuffer AllocTempRaw( int count )
        {
            return new GraphicsBuffer( GraphicsBuffer.Target.Structured, Math.Max( 1, count ), sizeof( float ) );
        }
        //------------------------------------------------------------------
        private void SetBinaryForwardParams( int kernel,
                                             TensorStorage a, TensorStorage b, TensorStorage result,
                                             int sizeA, int sizeB, int resultSize )
        {
            _shader.SetInt( "_SizeA", sizeA );
            _shader.SetInt( "_SizeB", sizeB );
            _shader.SetInt( "_ResultSize", resultSize );
            _shader.SetBuffer( kernel, "_A", a.GpuBuffer );
            _shader.SetBuffer( kernel, "_B", b.GpuBuffer );
            _shader.SetBuffer( kernel, "_Result", result.GpuBuffer );
        }
        //------------------------------------------------------------------
        private void SetBinaryBackwardParams( int kernel,
                                              TensorStorage aData, TensorStorage bData,
                                              TensorStorage aGrad, TensorStorage bGrad,
                                              TensorStorage resultGrad,
                                              int sizeA, int sizeB, int resultSize,
                                              float aGradScale, float bGradScale )
        {
            _shader.SetInt( "_SizeA", sizeA );
            _shader.SetInt( "_SizeB", sizeB );
            _shader.SetInt( "_ResultSize", resultSize );
            _shader.SetFloat( "_AGradScale", aGradScale );
            _shader.SetFloat( "_BGradScale", bGradScale );

            if (aData != null) _shader.SetBuffer( kernel, "_AData", aData.GpuBuffer );
            if (bData != null) _shader.SetBuffer( kernel, "_BData", bData.GpuBuffer );
            _shader.SetBuffer( kernel, "_AGrad", aGrad.GpuBuffer );
            _shader.SetBuffer( kernel, "_BGrad", bGrad.GpuBuffer );
            _shader.SetBuffer( kernel, "_ResultGrad", resultGrad.GpuBuffer );
        }
        //------------------------------------------------------------------
        private void SetUnaryForwardParams( int kernel,
                                            TensorStorage input, TensorStorage result, int size )
        {
            _shader.SetInt( "_Size", size );
            _shader.SetBuffer( kernel, "_Input", input.GpuBuffer );
            _shader.SetBuffer( kernel, "_Result", result.GpuBuffer );
        }
        //------------------------------------------------------------------
        private void SetUnaryBackwardParams( int kernel,
                                             TensorStorage inputData, TensorStorage resultData,
                                             TensorStorage inputGrad, TensorStorage resultGrad,
                                             int size )
        {
            _shader.SetInt( "_Size", size );

            if (inputData != null) _shader.SetBuffer( kernel, "_InputData", inputData.GpuBuffer );
            if (resultData != null) _shader.SetBuffer( kernel, "_ResultData", resultData.GpuBuffer );
            _shader.SetBuffer( kernel, "_InputGrad", inputGrad.GpuBuffer );
            _shader.SetBuffer( kernel, "_ResultGrad", resultGrad.GpuBuffer );
        }
        //------------------------------------------------------------------
        /// <summary>Dispatches a transpose of an MxN input into an NxM result.</summary>
        private void DispatchTranspose( GraphicsBuffer input, GraphicsBuffer result,
                                        int rows, int cols, int totalSize )
        {
            _shader.SetInt( "_M", rows );
            _shader.SetInt( "_N", cols );
            _shader.SetInt( "_Size", totalSize );
            _shader.SetBuffer( _transposeKernel, "_Input", input );
            _shader.SetBuffer( _transposeKernel, "_Result", result );
            _shader.Dispatch( _transposeKernel, Groups( totalSize ), 1, 1 );
        }
        //------------------------------------------------------------------
        /// <summary>Internal MatMul dispatch using raw GraphicsBuffer handles.</summary>
        private void MatMul_Internal( GraphicsBuffer a, GraphicsBuffer b, GraphicsBuffer c,
                                      int M, int K, int N, bool accumulate )
        {
            _shader.SetInt( "_M", M );
            _shader.SetInt( "_K", K );
            _shader.SetInt( "_N", N );
            _shader.SetInt( "_Accumulate", accumulate ? 1 : 0 );
            _shader.SetBuffer( _matMulForwardKernel, "_A", a );
            _shader.SetBuffer( _matMulForwardKernel, "_B", b );
            _shader.SetBuffer( _matMulForwardKernel, "_Result", c );

            // The kernel computes a (TILE*2)x(TILE*2) block per threadgroup
            int tileOut = TILE * 2;
            int gx = (N + tileOut - 1) / tileOut;
            int gy = (M + tileOut - 1) / tileOut;
            _shader.Dispatch( _matMulForwardKernel, gx, gy, 1 );
        }
        //------------------------------------------------------------------
    }
}
