using System;
using System.Collections.Generic;

namespace ChaosRL
{
    /// <summary>
    /// L2 (weight decay) regularizer for Tensor-based neural networks.
    /// Computes the sum of squared parameters scaled by a coefficient.
    /// </summary>
    public class L2RegularizerTensor
    {
        //------------------------------------------------------------------
        private readonly Tensor[] _parameters;
        //------------------------------------------------------------------
        public L2RegularizerTensor( IEnumerable<IEnumerable<Tensor>> parameterGroups )
        {
            if (parameterGroups == null) throw new ArgumentNullException( nameof( parameterGroups ) );

            var collected = new List<Tensor>();
            foreach (var group in parameterGroups)
                foreach (var parameter in group)
                    if (parameter != null)
                        collected.Add( parameter );

            if (collected.Count == 0)
                throw new ArgumentException( "Parameter collection must not be empty", nameof( parameterGroups ) );

            _parameters = collected.ToArray();
        }
        //------------------------------------------------------------------
        public Tensor Compute( float coefficient, float scale = 0.5f )
        {
            if (coefficient <= 0f)
                return new Tensor( 0f );

            Tensor total = new Tensor( 0f );
            for (int i = 0; i < _parameters.Length; i++)
            {
                var squared = _parameters[ i ] * _parameters[ i ];
                total += squared.Sum();
            }

            return total * (scale * coefficient);
        }
        //------------------------------------------------------------------
    }
}
