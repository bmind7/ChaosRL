using System;
using System.Collections.Generic;

namespace ChaosRL
{
    public class AdamOptimizer : IDisposable
    {
        //------------------------------------------------------------------
        public IReadOnlyList<Tensor> Parameters => _parameters;

        private readonly Tensor[] _parameters;
        private readonly int[] _parameterOffsets;
        private TensorStorage _m;
        private TensorStorage _v;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private float _beta1Pow = 1f;
        private float _beta2Pow = 1f;
        private long _step = 0;
        private bool _disposed;
        //------------------------------------------------------------------
        public AdamOptimizer( IEnumerable<IEnumerable<Tensor>> parameterGroups, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f )
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

            // Validate all parameters live on the same device
            var device = _parameters[ 0 ].Device;
            for (int i = 1; i < _parameters.Length; i++)
            {
                if (_parameters[ i ].Device != device)
                    throw new ArgumentException(
                        $"All parameters must be on the same device. Parameter 0 is on {device}, " +
                        $"but parameter {i} is on {_parameters[ i ].Device}." );
            }

            // Calculate total size and offsets
            int totalSize = 0;
            for (int i = 0; i < _parameters.Length; i++)
            {
                _parameterOffsets[ i ] = totalSize;
                totalSize += _parameters[ i ].Size;
            }

            _m = TensorStorage.Allocate( totalSize, _parameters[ 0 ].Device );
            _v = TensorStorage.Allocate( totalSize, _parameters[ 0 ].Device );

            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _beta1Pow = 1f;
            _beta2Pow = 1f;
            _step = 0;
        }
        //------------------------------------------------------------------
        ~AdamOptimizer()
        {
            Dispose( false );
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

                var backend = parameter.Backend;
                backend.AdamStep(
                    parameter.Data, parameter.Grad, _m, _v,
                    parameter.Size, offset,
                    learningRate, _beta1, _beta2, _epsilon,
                    invBias1, invBias2 );
            }
        }
        //------------------------------------------------------------------
        public void ResetState()
        {
            _m?.Clear();
            _v?.Clear();

            _beta1Pow = 1f;
            _beta2Pow = 1f;
            _step = 0;
        }
        //------------------------------------------------------------------
        public void Dispose()
        {
            Dispose( true );
            GC.SuppressFinalize( this );
        }
        //------------------------------------------------------------------
        private void Dispose( bool disposing )
        {
            if (_disposed)
                return;

            _disposed = true;

            _m?.Release();
            _v?.Release();
            _m = null;
            _v = null;
        }
        //------------------------------------------------------------------
    }
}
