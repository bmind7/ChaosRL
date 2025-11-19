using System;
using System.Collections.Generic;

namespace ChaosRL
{
    /// <summary>
    /// Neural network layer using Tensor for efficient batch processing.
    /// Performs matrix multiplication: output = input @ weights + bias
    /// where input is (batch_size, num_inputs) and weights is (num_inputs, num_outputs).
    /// </summary>
    public class LayerTensor
    {
        //------------------------------------------------------------------
        public readonly int NumInputs;
        public readonly int NumOutputs;

        public IEnumerable<Tensor> Parameters
        {
            get
            {
                yield return _weights;
                yield return _bias;
            }
        }

        private readonly Tensor _weights; // Shape: (num_inputs, num_outputs)
        private readonly Tensor _bias;    // Shape: (num_outputs,)
        private readonly bool _nonLin;
        //------------------------------------------------------------------
        /// <summary>
        /// Creates a new layer with He initialization.
        /// </summary>
        /// <param name="numInputs">Number of input features</param>
        /// <param name="numOutputs">Number of output features</param>
        /// <param name="nonLin">Whether to apply non-linearity (Tanh) activation</param>
        public LayerTensor( int numInputs, int numOutputs, bool nonLin = true )
        {
            if (numInputs <= 0) throw new ArgumentOutOfRangeException( nameof( numInputs ), "numInputs must be > 0" );
            if (numOutputs <= 0) throw new ArgumentOutOfRangeException( nameof( numOutputs ), "numOutputs must be > 0" );

            this.NumInputs = numInputs;
            this.NumOutputs = numOutputs;
            this._nonLin = nonLin;

            // Initialize weights with He initialization for better gradient flow
            var limit = MathF.Sqrt( 6f / numInputs );
            var weightData = new float[ numInputs * numOutputs ];
            for (int i = 0; i < weightData.Length; i++)
                weightData[ i ] = RandomHub.NextFloat( -limit, limit );

            _weights = new Tensor( new[] { numInputs, numOutputs }, weightData, "weights" );

            // Initialize bias to zero
            var biasData = new float[ numOutputs ];
            _bias = new Tensor( new[] { numOutputs }, biasData, "bias" );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Forward pass for a batch of inputs.
        /// </summary>
        /// <param name="input">Input tensor of shape (batch_size, num_inputs)</param>
        /// <returns>Output tensor of shape (batch_size, num_outputs)</returns>
        public Tensor Forward( Tensor input )
        {
            if (input.Shape.Length != 2)
                throw new ArgumentException( $"Expected 2D input tensor, got shape [{string.Join( ", ", input.Shape )}]" );

            int inputFeatures = input.Shape[ 1 ];

            if (inputFeatures != this.NumInputs)
                throw new ArgumentException( $"Expected {this.NumInputs} input features, got {inputFeatures}" );

            // Linear transformation: output = input @ weights
            // input: (batch_size, num_inputs)
            // weights: (num_inputs, num_outputs)
            // output: (batch_size, num_outputs)
            var output = input.MatMul( _weights );

            output = output + _bias;

            // Apply non-linearity if enabled
            if (_nonLin)
                output = output.Tanh();

            return output;
        }
        //------------------------------------------------------------------
        public void ZeroGrad()
        {
            _weights.ZeroGrad();
            _bias.ZeroGrad();
        }
        //------------------------------------------------------------------
        public override string ToString()
        {
            return $"LayerTensor(NumInputs: {this.NumInputs}, NumOutputs: {this.NumOutputs}, NonLin: {_nonLin})";
        }
        //------------------------------------------------------------------
    }
}
