using System;
using System.Collections.Generic;

namespace ChaosRL
{
    /// <summary>
    /// Tracks tensors created within the current execution context and disposes them
    /// deterministically when the scope ends.
    /// </summary>
    public sealed class TensorScope : IDisposable
    {
        [ThreadStatic]
        private static TensorScope _current;

        private readonly TensorScope _parent;
        private readonly List<Tensor> _trackedTensors = new List<Tensor>();
        private bool _disposed;

        //------------------------------------------------------------------
        public TensorScope()
        {
            _parent = _current;
            _current = this;
        }
        //------------------------------------------------------------------
        internal static void Track( Tensor tensor )
        {
            _current?._trackedTensors.Add( tensor );
        }
        //------------------------------------------------------------------
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            if (ReferenceEquals( _current, this ))
                _current = _parent;

            for (int i = _trackedTensors.Count - 1; i >= 0; i--)
            {
                _trackedTensors[ i ]?.Dispose();
            }

            _trackedTensors.Clear();
        }
        //------------------------------------------------------------------
    }
}
