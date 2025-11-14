using System;
using System.Collections.Generic;

namespace ChaosRL
{
    public class Layer
    {
        //------------------------------------------------------------------
        public readonly int NumInputs;
        public readonly int NumOutputs;

        public IEnumerable<Value> Parameters
        {
            get
            {
                foreach (var neuron in _neurons)
                    foreach (var parameter in neuron.Parameters)
                        yield return parameter;
            }
        }

        private readonly Neuron[] _neurons;
        //------------------------------------------------------------------
        public Layer( int numInputs, int numOutputs, bool nonLin = true )
        {
            if (numInputs <= 0) throw new ArgumentOutOfRangeException( nameof( numInputs ), "numInputs must be > 0" );
            if (numOutputs <= 0) throw new ArgumentOutOfRangeException( nameof( numOutputs ), "numOutputs must be > 0" );

            this.NumInputs = numInputs;
            this.NumOutputs = numOutputs;

            _neurons = new Neuron[ numOutputs ];
            for (int i = 0; i < numOutputs; i++)
                _neurons[ i ] = new Neuron( numInputs, nonLin );
        }
        //------------------------------------------------------------------
        // Single-layer forward taking a span as input vector
        public Value[] Forward( ReadOnlySpan<Value> inputs )
        {
            if (inputs.Length != this.NumInputs) throw new ArgumentException( $"Expected {this.NumInputs} inputs, got {inputs.Length}", nameof( inputs ) );

            var outputs = new Value[ this.NumOutputs ];
            for (int i = 0; i < this.NumOutputs; i++)
                outputs[ i ] = _neurons[ i ].Forward( inputs );

            return outputs;
        }
        //------------------------------------------------------------------
        public void Backward()
        {
            foreach (var n in _neurons)
                n.Backward();
        }
        //------------------------------------------------------------------
        public void ZeroGrad()
        {
            foreach (var n in _neurons)
                n.ZeroGrad();
        }
        //------------------------------------------------------------------
        public override string ToString()
        {
            return $"Layer(NumInputs: {this.NumInputs}, NumOutputs: {this.NumOutputs})";
        }
        //------------------------------------------------------------------
    }
}
