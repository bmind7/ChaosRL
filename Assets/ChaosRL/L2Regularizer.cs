using System;
using System.Collections.Generic;

namespace ChaosRL
{
    public class L2Regularizer
    {
        //------------------------------------------------------------------
        private readonly Value[] _parameters;
        //------------------------------------------------------------------
        public L2Regularizer( IEnumerable<IEnumerable<Value>> parameterGroups )
        {
            if (parameterGroups == null) throw new ArgumentNullException( nameof( parameterGroups ) );

            var collected = new List<Value>();
            foreach (var group in parameterGroups)
                foreach (var parameter in group)
                    if (parameter != null)
                        collected.Add( parameter );

            if (collected.Count == 0)
                throw new ArgumentException( "Parameter collection must not be empty", nameof( parameterGroups ) );

            _parameters = collected.ToArray();
        }
        //------------------------------------------------------------------
        public Value Compute( float coefficient, float scale = 0.5f )
        {
            if (coefficient <= 0f)
                return 0f;

            Value total = 0f;
            for (int i = 0; i < _parameters.Length; i++)
                total += _parameters[ i ] * _parameters[ i ];

            return total * (scale * coefficient);
        }
        //------------------------------------------------------------------
    }
}
