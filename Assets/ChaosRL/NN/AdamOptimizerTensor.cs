using System;
using System.Collections.Generic;

namespace ChaosRL
{
    public class AdamOptimizerTensor
    {
        //------------------------------------------------------------------
        public IReadOnlyList<Tensor> Parameters => _parameters;

        private readonly Tensor[] _parameters;
        private readonly int[] _parameterOffsets;
        private readonly float[] _m;
        private readonly float[] _v;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private float _beta1Pow = 1f;
        private float _beta2Pow = 1f;
        private long _step = 0;
        //------------------------------------------------------------------
        public AdamOptimizerTensor( IEnumerable<IEnumerable<Tensor>> parameterGroups, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f )
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
            _parameterOffsets = new int[ _parameters.Length ];

            // Calculate total size and offsets
            int totalSize = 0;
            for (int i = 0; i < _parameters.Length; i++)
            {
                _parameterOffsets[ i ] = totalSize;
                totalSize += _parameters[ i ].Size;
            }

            _m = new float[ totalSize ];
            _v = new float[ totalSize ];

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
                int offset = _parameterOffsets[ i ];

                // Update moments element-wise
                for (int j = 0; j < parameter.Size; j++)
                {
                    int idx = offset + j;
                    float grad = parameter.Grad[ j ];

                    _m[ idx ] = _beta1 * _m[ idx ] + (1 - _beta1) * grad;
                    _v[ idx ] = _beta2 * _v[ idx ] + (1 - _beta2) * grad * grad;

                    float mHat = _m[ idx ] * invBias1;
                    float vHat = _v[ idx ] * invBias2;

                    parameter.Data[ j ] -= learningRate * mHat / (MathF.Sqrt( vHat ) + _epsilon);
                }
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
