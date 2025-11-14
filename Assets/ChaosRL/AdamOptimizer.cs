using System;
using System.Collections.Generic;

namespace ChaosRL
{
    public class AdamOptimizer
    {
        //------------------------------------------------------------------
        public IReadOnlyList<Value> Parameters => _parameters;

        private readonly Value[] _parameters;
        private readonly float[] _m;
        private readonly float[] _v;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private float _beta1Pow = 1f;
        private float _beta2Pow = 1f;
        private long _step = 0;
        //------------------------------------------------------------------
        public AdamOptimizer( IEnumerable<IEnumerable<Value>> parameterGroups, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f )
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
            _m = new float[ _parameters.Length ];
            _v = new float[ _parameters.Length ];
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _beta1Pow = 1f;
            _beta2Pow = 1f;
            _step = 0;
        }
        //------------------------------------------------------------------
        public void Step( float learningRate )
        {
            _step++;
            _beta1Pow *= _beta1;
            _beta2Pow *= _beta2;

            float invBias1 = 1f / (1f - _beta1Pow);
            float invBias2 = 1f / (1f - _beta2Pow);
            for (int i = 0; i < _parameters.Length; i++)
            {
                var parameter = _parameters[ i ];
                float grad = parameter.Grad;

                _m[ i ] = _beta1 * _m[ i ] + (1 - _beta1) * grad;
                _v[ i ] = _beta2 * _v[ i ] + (1 - _beta2) * grad * grad;

                float mHat = _m[ i ] * invBias1;
                float vHat = _v[ i ] * invBias2;

                parameter.Data -= learningRate * mHat / (MathF.Sqrt( vHat ) + _epsilon);
            }
        }
        //------------------------------------------------------------------
        public void ResetState()
        {
            Array.Clear( _m, 0, _m.Length );
            Array.Clear( _v, 0, _v.Length );
            _beta1Pow = 1f;
            _beta2Pow = 1f;
            _step = 0;
        }
        //------------------------------------------------------------------
    }
}
