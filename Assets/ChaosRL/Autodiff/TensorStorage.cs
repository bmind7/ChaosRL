using System;
using System.Threading;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace ChaosRL
{
    /// <summary>
    /// Ref-counted wrapper around a contiguous float buffer.
    /// Owns the backing <see cref="NativeArray{T}"/> (CPU) and manages its lifetime
    /// via reference counting so that view tensors can safely share storage.
    /// </summary>
    public class TensorStorage : IDisposable
    {
        //------------------------------------------------------------------
        /// <summary>
        /// Raw backing buffer. Internal so only the ChaosRL assembly (CpuBackend, CpuMatMulOps,
        /// TensorJobs) can access the <see cref="NativeArray{T}"/> directly. External code
        /// should use the element indexer <c>this[int]</c> or <see cref="CopyFrom"/>/<see cref="CopyTo"/>.
        /// </summary>
        internal ref NativeArray<float> Buffer => ref _buffer;

        /// <summary>Which device this storage resides on.</summary>
        public TensorDevice Device => _device;

        /// <summary>Number of elements in this storage.</summary>
        public int Length => _buffer.Length;

        /// <summary>Current reference count. Starts at 1 when allocated.</summary>
        public int RefCount => _refCount;

        /// <summary>Whether the underlying buffer has been disposed.</summary>
        public bool IsCreated => _buffer.IsCreated;

        private NativeArray<float> _buffer;
        private readonly TensorDevice _device;
        private int _refCount;
        private bool _disposed;

        //------------------------------------------------------------------
        /// <summary>
        /// Allocates a new storage buffer of the given size on the specified device.
        /// The buffer is zero-initialized.
        /// </summary>
        public static TensorStorage Allocate( int size, TensorDevice device = TensorDevice.CPU )
        {
            if (size <= 0)
                throw new ArgumentException( $"Storage size must be positive, got {size}", nameof( size ) );

            if (device == TensorDevice.GPU)
                throw new NotSupportedException( "GPU storage is not yet implemented" );

            var buffer = new NativeArray<float>( size, Allocator.Persistent, NativeArrayOptions.ClearMemory );
            return new TensorStorage( buffer, device );
        }
        //------------------------------------------------------------------
        private TensorStorage( NativeArray<float> buffer, TensorDevice device )
        {
            _buffer = buffer;
            _device = device;
            _refCount = 1;
            _disposed = false;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Element accessor for convenience (scalar readback, test assertions).
        /// Not intended for hot-path computation — use <see cref="AsNativeArray"/> for jobs.
        /// </summary>
        public float this[ int i ]
        {
            get => _buffer[ i ];
            set => _buffer[ i ] = value;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Returns the raw <see cref="NativeArray{T}"/> for passing to Burst jobs.
        /// Throws if this storage is GPU-resident (use the future AsComputeBuffer path instead).
        /// </summary>
        public NativeArray<float> AsNativeArray()
        {
            if (_device != TensorDevice.CPU)
                throw new InvalidOperationException(
                    "Cannot get NativeArray from GPU storage. Transfer to CPU first via Tensor.ToCpu()." );

            return _buffer;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Increments the reference count. Called when a view tensor shares this storage.
        /// </summary>
        public void AddRef()
        {
            if (_disposed)
                throw new ObjectDisposedException( nameof( TensorStorage ) );

            Interlocked.Increment( ref _refCount );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Decrements the reference count and disposes the buffer when it reaches zero.
        /// </summary>
        public void Release()
        {
            if (_disposed)
                return;

            int newCount = Interlocked.Decrement( ref _refCount );
            if (newCount <= 0)
            {
                _disposed = true;
                if (_buffer.IsCreated)
                    _buffer.Dispose();
                _buffer = default;
            }
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Copies data from a managed float array into the buffer.
        /// </summary>
        public void CopyFrom( float[] source )
        {
            if (source == null)
                throw new ArgumentNullException( nameof( source ) );
            if (source.Length != _buffer.Length)
                throw new ArgumentException(
                    $"Source length {source.Length} doesn't match buffer length {_buffer.Length}" );

            _buffer.CopyFrom( source );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Copies data from the buffer into a managed float array.
        /// </summary>
        public void CopyTo( float[] destination )
        {
            if (destination == null)
                throw new ArgumentNullException( nameof( destination ) );
            if (destination.Length != _buffer.Length)
                throw new ArgumentException(
                    $"Destination length {destination.Length} doesn't match buffer length {_buffer.Length}" );

            _buffer.CopyTo( destination );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Copies data from another TensorStorage into this one.
        /// Both storages must have the same length.
        /// </summary>
        public void CopyFrom( TensorStorage source )
        {
            if (source == null)
                throw new ArgumentNullException( nameof( source ) );
            if (source.Length != _buffer.Length)
                throw new ArgumentException(
                    $"Source length {source.Length} doesn't match buffer length {_buffer.Length}" );

            NativeArray<float>.Copy( source._buffer, _buffer );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Zeroes the entire buffer. Used by ZeroGrad.
        /// </summary>
        public unsafe void Clear()
        {
            UnsafeUtility.MemClear(
                NativeArrayUnsafeUtility.GetUnsafePtr( _buffer ),
                _buffer.Length * sizeof( float ) );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Fills the entire buffer with a single value. Used to seed backward grad with 1.0.
        /// </summary>
        public unsafe void Fill( float value )
        {
            UnsafeUtility.MemCpyReplicate(
                NativeArrayUnsafeUtility.GetUnsafePtr( _buffer ),
                &value, sizeof( float ), _buffer.Length );
        }
        //------------------------------------------------------------------
        public void Dispose()
        {
            Release();
        }
        //------------------------------------------------------------------
    }
}
