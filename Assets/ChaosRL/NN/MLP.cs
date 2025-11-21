using System;
using System.Collections.Generic;

namespace ChaosRL
{
    /// <summary>
    /// Multi-layer perceptron using Tensor for efficient batch processing.
    /// Chains multiple Layer instances to create a deep neural network.
    /// </summary>
    public class MLP
    {
        //------------------------------------------------------------------
        public readonly int NumInputs;
        public readonly int NumOutputs;
        public readonly int[] LayerSizes;

        public IEnumerable<Tensor> Parameters
        {
            get
            {
                foreach (var layer in _layers)
                    foreach (var parameter in layer.Parameters)
                        yield return parameter;
            }
        }

        private readonly Layer[] _layers;
        //------------------------------------------------------------------
        /// <summary>
        /// Creates a multi-layer perceptron with specified architecture.
        /// </summary>
        /// <param name="numInputs">Number of input features</param>
        /// <param name="layerSizes">Array of layer output sizes. Example: [8, 8, 4] creates 3 layers</param>
        /// <param name="lastLayerNonLin">Whether to apply non-linearity to the last layer (default: false for value/logit outputs)</param>
        public MLP( int numInputs, int[] layerSizes, bool lastLayerNonLin = false )
        {
            if (numInputs <= 0) throw new ArgumentOutOfRangeException( nameof( numInputs ), "numInputs must be > 0" );
            if (layerSizes == null) throw new ArgumentNullException( nameof( layerSizes ) );
            if (layerSizes.Length == 0) throw new ArgumentException( "layerSizes must contain at least one layer", nameof( layerSizes ) );

            foreach (var s in layerSizes)
                if (s <= 0) throw new ArgumentOutOfRangeException( nameof( layerSizes ), "All layer sizes must be > 0" );

            this.NumInputs = numInputs;
            this.LayerSizes = (int[])layerSizes.Clone();
            this.NumOutputs = layerSizes[ ^1 ];

            _layers = new Layer[ layerSizes.Length ];
            int prev = numInputs;
            for (int i = 0; i < layerSizes.Length; i++)
            {
                int curr = layerSizes[ i ];
                bool isLast = (i == layerSizes.Length - 1);
                bool useNonLin = isLast ? lastLayerNonLin : true;
                _layers[ i ] = new Layer( prev, curr, nonLin: useNonLin );
                prev = curr;
            }
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

            if (input.Shape[ 1 ] != this.NumInputs)
                throw new ArgumentException( $"Expected {this.NumInputs} input features, got {input.Shape[ 1 ]}" );

            var x = input;
            for (int i = 0; i < _layers.Length; i++)
                x = _layers[ i ].Forward( x );

            return x;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Resets gradients for all parameters to zero.
        /// Call this before each backward pass.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var layer in _layers)
                layer.ZeroGrad();
        }
        //------------------------------------------------------------------
        public override string ToString()
        {
            return $"MLPTensor(NumInputs: {this.NumInputs}, NumOutputs: {this.NumOutputs}, Layers: {_layers.Length})";
        }
        //------------------------------------------------------------------
    }
}
