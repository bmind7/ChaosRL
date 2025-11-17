using System;
using System.Collections.Generic;

namespace ChaosRL
{
    public class Neuron
    {
        //------------------------------------------------------------------
        public IEnumerable<Value> Parameters
        {
            get
            {
                for (int i = 0; i < _weights.Length; i++)
                    yield return _weights[ i ];

                yield return _bias;
            }
        }

        private readonly Value[] _weights;
        private readonly Value _bias;
        private readonly bool _nonLin;
        //------------------------------------------------------------------
        public Neuron( int numInputs, bool nonLin = true )
        {
            _weights = new Value[ numInputs ];
            // He-style init keeps forward variance steady with tanh
            var limit = MathF.Sqrt( 6f / numInputs );
            for (int i = 0; i < numInputs; i++)
                _weights[ i ] = new Value( RandomHub.NextFloat( -limit, limit ) );

            _bias = 0f;
            _nonLin = nonLin;
        }
        //------------------------------------------------------------------
        // Single-sample forward taking a span to avoid copies
        public Value Forward( ReadOnlySpan<Value> inputs )
        {
            if (inputs.Length != _weights.Length) throw new ArgumentException( "Mismatched lengths" );

            var output = new Value( 0f );
            for (int i = 0; i < inputs.Length; i++)
                output += inputs[ i ] * _weights[ i ];

            output += _bias;

            if (_nonLin)
                // Tanh keeps activations bounded which helps training stability here
                // output = output.ReLU();
                output = output.Tanh();

            return output;
        }
        //------------------------------------------------------------------
        public void Backward()
        {
            foreach (var weight in _weights)
                weight.Backward();

            _bias.Backward();
        }
        //------------------------------------------------------------------
        public void ZeroGrad()
        {
            foreach (var weight in _weights)
            {
                weight.Grad = 0f;
            }

            _bias.Grad = 0f;
        }
        //------------------------------------------------------------------
        public override string ToString()
        {
            return $"Neuron(weights count: [{_weights.Length}], bias: {_bias}, nonLin: {_nonLin})";
        }
        //------------------------------------------------------------------
    }
}