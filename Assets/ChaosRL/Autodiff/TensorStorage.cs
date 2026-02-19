using System;
using System.Threading;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

using UnityEngine;

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
        //  GPU utility delegates — registered by GpuBackend at startup so
        //  that Clear / Fill / Allocate can dispatch compute kernels instead
        //  of allocating managed arrays for CPU -> GPU upload.
        //------------------------------------------------------------------

        internal static Action<GraphicsBuffer, int> GpuZeroFill;
        internal static Action<GraphicsBuffer, int, float> GpuValueFill;

        //------------------------------------------------------------------
        /// <summary>
        /// Raw CPU backing buffer. Internal so only the ChaosRL assembly (CpuBackend, CpuMatMulOps,
        /// TensorJobs) can access the <see cref="NativeArray{T}"/> directly. External code
        /// should use the element indexer <c>this[int]</c> or <see cref="CopyFrom"/>/<see cref="CopyTo"/>.
        /// Only valid when <see cref="Device"/> is <see cref="TensorDevice.CPU"/>.
        /// </summary>
        internal ref NativeArray<float> Buffer => ref _buffer;

        /// <summary>
        /// Raw GPU backing buffer. Internal so only <see cref="GpuBackend"/> can access it directly.
        /// Only valid when <see cref="Device"/> is <see cref="TensorDevice.GPU"/>.
        /// </summary>
        internal GraphicsBuffer GpuBuffer => _gpuBuffer;

        /// <summary>Which device this storage resides on.</summary>
        public TensorDevice Device => _device;

        /// <summary>Number of elements in this storage.</summary>
        public int Length => _size;

        /// <summary>Current reference count. Starts at 1 when allocated.</summary>
        public int RefCount => _refCount;

        /// <summary>Whether the underlying buffer has been disposed.</summary>
        public bool IsCreated => _device == TensorDevice.GPU ? _gpuBuffer != null : _buffer.IsCreated;

        private NativeArray<float> _buffer;
        private GraphicsBuffer _gpuBuffer;
        private readonly TensorDevice _device;
        private readonly int _size;
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
            {
                var gpuBuf = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, size, sizeof( float ) );
                // Zero-initialize via compute dispatch if available, else managed upload
                if (GpuZeroFill != null)
                    GpuZeroFill( gpuBuf, size );
                else
                    gpuBuf.SetData( new float[ size ] );
                return new TensorStorage( gpuBuf, size );
            }

            var buffer = new NativeArray<float>( size, Allocator.Persistent, NativeArrayOptions.ClearMemory );
            return new TensorStorage( buffer, device );
        }
        //------------------------------------------------------------------
        private TensorStorage( NativeArray<float> buffer, TensorDevice device )
        {
            _buffer = buffer;
            _device = device;
            _size = buffer.Length;
            _refCount = 1;
            _disposed = false;
        }
        //------------------------------------------------------------------
        private TensorStorage( GraphicsBuffer gpuBuffer, int size )
        {
            _gpuBuffer = gpuBuffer;
            _device = TensorDevice.GPU;
            _size = size;
            _refCount = 1;
            _disposed = false;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Element accessor for convenience (scalar readback, test assertions).
        /// Not intended for hot-path computation — use <see cref="AsNativeArray"/> for jobs.
        /// </summary>
        /// <remarks>
        /// GPU path performs a synchronous readback/upload per access — debug/test use only.
        /// </remarks>
        public float this[ int i ]
        {
            get
            {
                if (_device == TensorDevice.GPU)
                {
                    var tmp = new float[ 1 ];
                    _gpuBuffer.GetData( tmp, 0, i, 1 );
                    return tmp[ 0 ];
                }
                return _buffer[ i ];
            }
            set
            {
                if (_device == TensorDevice.GPU)
                {
                    _gpuBuffer.SetData( new[] { value }, 0, i, 1 );
                    return;
                }
                _buffer[ i ] = value;
            }
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
        /// Returns the <see cref="GraphicsBuffer"/> for passing to compute shader dispatches.
        /// Throws if this storage is CPU-resident.
        /// </summary>
        public GraphicsBuffer AsGraphicsBuffer()
        {
            if (_device != TensorDevice.GPU)
                throw new InvalidOperationException(
                    "Cannot get GraphicsBuffer from CPU storage. Transfer to GPU first via Tensor.ToGpu()." );

            return _gpuBuffer;
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
                if (_device == TensorDevice.GPU)
                {
                    _gpuBuffer?.Release();
                    _gpuBuffer = null;
                }
                else
                {
                    if (_buffer.IsCreated)
                        _buffer.Dispose();
                    _buffer = default;
                }
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
            if (source.Length != _size)
                throw new ArgumentException(
                    $"Source length {source.Length} doesn't match buffer length {_size}" );

            if (_device == TensorDevice.GPU)
                _gpuBuffer.SetData( source );
            else
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
            if (destination.Length != _size)
                throw new ArgumentException(
                    $"Destination length {destination.Length} doesn't match buffer length {_size}" );

            if (_device == TensorDevice.GPU)
                _gpuBuffer.GetData( destination );
            else
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
            if (source.Length != _size)
                throw new ArgumentException(
                    $"Source length {source.Length} doesn't match buffer length {_size}" );

            if (_device != source._device)
                throw new InvalidOperationException(
                    $"Cannot copy between different devices ({source._device} -> {_device}). Use Tensor.To() for cross-device transfer." );

            if (_device == TensorDevice.GPU)
                Graphics.CopyBuffer( source._gpuBuffer, _gpuBuffer );
            else
                NativeArray<float>.Copy( source._buffer, _buffer );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Zeroes the entire buffer. Used by ZeroGrad.
        /// </summary>
        public unsafe void Clear()
        {
            if (_device == TensorDevice.GPU)
            {
                if (GpuZeroFill != null)
                    GpuZeroFill( _gpuBuffer, _size );
                else
                    _gpuBuffer.SetData( new float[ _size ] );
                return;
            }
            UnsafeUtility.MemClear(
                NativeArrayUnsafeUtility.GetUnsafePtr( _buffer ),
                _size * sizeof( float ) );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Fills the entire buffer with a single value. Used to seed backward grad with 1.0.
        /// </summary>
        public unsafe void Fill( float value )
        {
            if (_device == TensorDevice.GPU)
            {
                if (GpuValueFill != null)
                    GpuValueFill( _gpuBuffer, _size, value );
                else
                {
                    var tmp = new float[ _size ];
                    for (int i = 0; i < _size; i++)
                        tmp[ i ] = value;
                    _gpuBuffer.SetData( tmp );
                }
                return;
            }
            UnsafeUtility.MemCpyReplicate(
                NativeArrayUnsafeUtility.GetUnsafePtr( _buffer ),
                &value, sizeof( float ), _size );
        }
        //------------------------------------------------------------------
        public void Dispose()
        {
            Release();
        }
        //------------------------------------------------------------------
    }
}
