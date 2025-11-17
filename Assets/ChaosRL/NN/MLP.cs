using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace ChaosRL
{
    public class MLP
    {
        //------------------------------------------------------------------
        public readonly int NumInputs;
        public readonly int NumOutputs;
        public readonly int[] LayerSizes;

        public IEnumerable<Value> Parameters
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
        // MLP configured by input size and an array of layer sizes
        // Example: new MLP(3, new[]{ 8, 8, 4 }) creates 3 layers: 3->8, 8->8, 8->4
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
                // Allow last layer to drop nonlinearity when producing logits or values
                _layers[ i ] = new Layer( prev, curr, nonLin: useNonLin );
                prev = curr;
            }
        }
        //------------------------------------------------------------------
        // Single-sample forward using a span for the input vector
        public Value[] Forward( ReadOnlySpan<Value> inputs )
        {
            if (inputs.Length != this.NumInputs) throw new ArgumentException( $"Expected {this.NumInputs} inputs, got {inputs.Length}", nameof( inputs ) );

            var x = _layers[ 0 ].Forward( inputs );
            for (int i = 1; i < _layers.Length; i++)
                x = _layers[ i ].Forward( x );

            return x;
        }
        //------------------------------------------------------------------
        // Batch forward pass for a rectangular 2D array of inputs (rows x NumInputs)
        public Value[,] Forward( Value[,] inputs2D )
        {
            if (inputs2D == null) throw new ArgumentNullException( nameof( inputs2D ) );

            int rows = inputs2D.GetLength( 0 );
            int cols = inputs2D.GetLength( 1 );
            if (cols != this.NumInputs)
                throw new ArgumentException( $"Second dimension length {cols} != NumInputs {this.NumInputs}", nameof( inputs2D ) );

            var outputs = new Value[ rows, this.NumOutputs ];
            for (int r = 0; r < rows; r++)
            {
                // Create a span over the current row without copying
                ref var rowStart = ref inputs2D[ r, 0 ];
                var rowSpan = MemoryMarshal.CreateSpan( ref rowStart, cols ); // reinterpret row for the single-sample path

                // Call the single-sample forward for this row using the span
                var outRow = this.Forward( rowSpan );
                for (int oc = 0; oc < outRow.Length; oc++)
                    outputs[ r, oc ] = outRow[ oc ];
            }

            return outputs;
        }
        //------------------------------------------------------------------
        public void Backward()
        {
            foreach (var layer in _layers)
                layer.Backward();
        }
        //------------------------------------------------------------------
        public void ZeroGrad()
        {
            foreach (var layer in _layers)
                layer.ZeroGrad();
        }
        //------------------------------------------------------------------
        public override string ToString()
        {
            return $"MLP(NumInputs: {this.NumInputs}, NumOutputs: {this.NumOutputs}, Layers: {_layers.Length})";
        }
        //------------------------------------------------------------------
    }
}
