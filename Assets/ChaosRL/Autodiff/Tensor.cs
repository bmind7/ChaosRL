using System;
using System.Collections.Generic;
using System.Text;

using Unity.Collections;

namespace ChaosRL
{
    /// <summary>
    /// Multi-dimensional tensor with automatic differentiation support.
    /// Data is stored in row-major (C-style) contiguous layout for cache efficiency.
    /// Storage is managed by <see cref="TensorStorage"/> with ref-counting for view tensors.
    /// Compute is dispatched through the cached <see cref="Backend"/> resolved at creation.
    /// </summary>
    public class Tensor : IDisposable
    {
        //------------------------------------------------------------------
        //  Backend registration
        //------------------------------------------------------------------
        private static readonly CpuBackend _cpuBackend = new CpuBackend();
        private static ITensorBackend _gpuBackend;

        private static ITensorBackend ResolveBackend( TensorDevice device )
        {
            switch (device)
            {
                case TensorDevice.CPU:
                    return _cpuBackend;

                case TensorDevice.GPU:
                    if (_gpuBackend == null)
                        throw new InvalidOperationException(
                            "No GPU backend registered. Set Tensor.GpuBackend before using GPU tensors." );
                    return _gpuBackend;

                default:
                    throw new ArgumentOutOfRangeException( nameof( device ), device, "Unknown tensor device" );
            }
        }

        //------------------------------------------------------------------
        /// <summary>Underlying ref-counted data storage.</summary>
        public TensorStorage DataStorage { get; private set; }

        /// <summary>Underlying ref-counted gradient storage.</summary>
        public TensorStorage GradStorage { get; private set; }

        /// <summary>Convenience accessor — returns the NativeArray for Burst jobs and element access.</summary>
        public ref NativeArray<float> Data => ref DataStorage.Buffer;

        /// <summary>Convenience accessor — returns the NativeArray for Burst jobs and element access.</summary>
        public ref NativeArray<float> Grad => ref GradStorage.Buffer;

        public int[] Shape { get; private set; }
        public int Size { get; private set; }
        public string Name { get; set; }
        public HashSet<Tensor> Children { get; private set; }
        public bool IsScalar => Size == 1;
        public bool RequiresGrad { get; set; }

        /// <summary>Which compute device this tensor resides on.</summary>
        public TensorDevice Device => DataStorage.Device;

        /// <summary>Cached compute backend for this tensor's device.</summary>
        public ITensorBackend Backend { get; private set; }

        private Action _backward;
        private bool _disposed;

        //------------------------------------------------------------------
        public float this[ params int[] indices ]
        {
            get => DataStorage[ ToFlatIndex( indices ) ];
            set => DataStorage[ ToFlatIndex( indices ) ] = value;
        }
        //------------------------------------------------------------------
        private void ValidateAndCalculateSize( int[] shape )
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException( "Shape must have at least one dimension", nameof( shape ) );

            foreach (var dim in shape)
                if (dim <= 0)
                    throw new ArgumentException( "All dimensions must be positive", nameof( shape ) );

            Shape = (int[])shape.Clone();
            Size = 1;
            foreach (var dim in Shape)
                Size *= dim;
        }
        //------------------------------------------------------------------
        private static void ValidateDeviceMatch( Tensor a, Tensor b )
        {
            if (a.Device != b.Device)
                throw new InvalidOperationException(
                    $"Tensor device mismatch: '{a.Name}' on {a.Device}, '{b.Name}' on {b.Device}. Use Tensor.To() to transfer." );
        }
        //------------------------------------------------------------------
        // TODO: switch to params int[] for shape to improve usability
        public Tensor( int[] shape, float[] data = null, string name = "", bool requiresGrad = true,
                       TensorDevice device = TensorDevice.CPU )
        {
            ValidateAndCalculateSize( shape );

            if (data != null && data.Length != Size)
                throw new ArgumentException( $"Data length {data.Length} doesn't match shape size {Size}" );

            DataStorage = TensorStorage.Allocate( Size, device );
            GradStorage = TensorStorage.Allocate( Size, device );

            if (data != null)
                DataStorage.CopyFrom( data );

            Name = name;
            Children = new HashSet<Tensor>();
            RequiresGrad = requiresGrad;
            Backend = ResolveBackend( device );
            _backward = null;

            TensorScope.Track( this );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// View constructor — shares the same <see cref="TensorStorage"/> as the owner tensor.
        /// Increments the ref-count on both data and grad storage.
        /// </summary>
        private Tensor( int[] shape, TensorStorage dataStorage, TensorStorage gradStorage,
                        Tensor[] children, string name, bool requiresGrad )
        {
            ValidateAndCalculateSize( shape );

            if (dataStorage == null || gradStorage == null)
                throw new ArgumentNullException( "Storage must not be null for view tensors" );
            if (!dataStorage.IsCreated || !gradStorage.IsCreated)
                throw new ArgumentException( "Storage must be created for view tensors" );
            if (dataStorage.Length != Size || gradStorage.Length != Size)
                throw new ArgumentException( $"View tensor storage length must match shape size {Size}" );

            dataStorage.AddRef();
            gradStorage.AddRef();

            DataStorage = dataStorage;
            GradStorage = gradStorage;
            Backend = ResolveBackend( dataStorage.Device );
            Name = name;
            Children = new HashSet<Tensor>();
            if (children != null)
                foreach (var child in children)
                    Children.Add( child );
            RequiresGrad = requiresGrad;
            _backward = null;

            TensorScope.Track( this );
        }
        //------------------------------------------------------------------
        public Tensor( int[] shape, Tensor[] children, string name = "",
                       TensorDevice device = TensorDevice.CPU ) : this( shape, (float[])null, name, device: device )
        {
            if (children != null)
                foreach (var child in children)
                    Children.Add( child );
        }
        //------------------------------------------------------------------
        // Scalar tensor constructor
        public Tensor( float scalar, string name = "", bool requiresGrad = true ) : this( new[] { 1 }, new[] { scalar }, name, requiresGrad )
        {
        }
        //------------------------------------------------------------------
        ~Tensor()
        {
            Dispose( false );
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

            // Ref-counted: Release decrements and disposes the backing buffer when count reaches zero.
            // View tensors called AddRef at creation, so this is always safe.
            GradStorage?.Release();
            DataStorage?.Release();

            GradStorage = null;
            DataStorage = null;
        }
        //------------------------------------------------------------------
        public static implicit operator Tensor( float f )
        {
            return new Tensor( f );
        }
        //------------------------------------------------------------------
        // Element-wise addition
        public static Tensor operator +( Tensor a, Tensor b )
        {
            ValidateDeviceMatch( a, b );

            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for addition" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "+", device: a.Device );
            a.Backend.Add( a.DataStorage, b.DataStorage, result.DataStorage, sizeA, sizeB, resultSize );

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                a.Backend.AddBackward( a.GradStorage, b.GradStorage, result.GradStorage, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            };
            return result;
        }
        //------------------------------------------------------------------
        // Element-wise multiplication
        public static Tensor operator *( Tensor a, Tensor b )
        {
            ValidateDeviceMatch( a, b );

            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for multiplication" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "*", device: a.Device );
            a.Backend.Mul( a.DataStorage, b.DataStorage, result.DataStorage, sizeA, sizeB, resultSize );

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                a.Backend.MulBackward( a.DataStorage, b.DataStorage, a.GradStorage, b.GradStorage, result.GradStorage, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Tensor operator -( Tensor a )
        {
            return a * -1f;
        }
        //------------------------------------------------------------------
        public static Tensor operator -( Tensor a, Tensor b )
        {
            return a + (-b);
        }
        //------------------------------------------------------------------
        public static Tensor operator /( Tensor a, Tensor b )
        {
            ValidateDeviceMatch( a, b );

            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for division" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "/", device: a.Device );
            a.Backend.Div( a.DataStorage, b.DataStorage, result.DataStorage, sizeA, sizeB, resultSize );

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                a.Backend.DivBackward( a.DataStorage, b.DataStorage, a.GradStorage, b.GradStorage, result.GradStorage, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Pow( float exponent )
        {
            var result = new Tensor( Shape, new[] { this }, $"^{exponent}", device: Device );
            Backend.Pow( DataStorage, result.DataStorage, Size, exponent );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                Backend.PowBackward( DataStorage, GradStorage, result.GradStorage, Size, exponent );
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Sqrt()
        {
            return Pow( 0.5f );
        }
        //------------------------------------------------------------------
        public Tensor Exp()
        {
            var result = new Tensor( Shape, new[] { this }, "exp", device: Device );
            Backend.Exp( DataStorage, result.DataStorage, Size );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                Backend.ExpBackward( result.DataStorage, GradStorage, result.GradStorage, Size );
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Log()
        {
            var result = new Tensor( Shape, new[] { this }, "log", device: Device );
            Backend.Log( DataStorage, result.DataStorage, Size );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                Backend.LogBackward( DataStorage, GradStorage, result.GradStorage, Size );
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor ReLU()
        {
            var result = new Tensor( Shape, new[] { this }, "ReLU", device: Device );
            Backend.ReLU( DataStorage, result.DataStorage, Size );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                Backend.ReLUBackward( DataStorage, GradStorage, result.GradStorage, Size );
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Tanh()
        {
            var result = new Tensor( Shape, new[] { this }, "tanh", device: Device );
            Backend.Tanh( DataStorage, result.DataStorage, Size );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                Backend.TanhBackward( result.DataStorage, GradStorage, result.GradStorage, Size );
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Clamp( float min, float max )
        {
            var result = new Tensor( Shape, new[] { this }, "clamp", device: Device );
            Backend.Clamp( DataStorage, result.DataStorage, Size, min, max );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                Backend.ClampBackward( DataStorage, GradStorage, result.GradStorage, Size, min, max );
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Tensor Max( Tensor a, Tensor b )
        {
            ValidateDeviceMatch( a, b );

            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for max" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "max", device: a.Device );
            a.Backend.ElementMax( a.DataStorage, b.DataStorage, result.DataStorage, sizeA, sizeB, resultSize );

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                a.Backend.ElementMaxBackward( a.DataStorage, b.DataStorage, a.GradStorage, b.GradStorage, result.GradStorage, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Tensor Min( Tensor a, Tensor b )
        {
            ValidateDeviceMatch( a, b );

            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for min" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "min", device: a.Device );
            a.Backend.ElementMin( a.DataStorage, b.DataStorage, result.DataStorage, sizeA, sizeB, resultSize );

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                a.Backend.ElementMinBackward( a.DataStorage, b.DataStorage, a.GradStorage, b.GradStorage, result.GradStorage, sizeA, sizeB, resultSize, aGradScale, bGradScale );
            };
            return result;
        }

        //------------------------------------------------------------------
        /// <summary>
        /// Matrix multiplication: (M×K) @ (K×N) -> (M×N)
        /// Supports 2D tensors for now.
        /// </summary>
        public Tensor MatMul( Tensor other )
        {
            ValidateDeviceMatch( this, other );

            if (Shape.Length != 2 || other.Shape.Length != 2)
                throw new ArgumentException( "MatMul requires 2D tensors" );

            int M = Shape[ 0 ];
            int K = Shape[ 1 ];
            int N = other.Shape[ 1 ];

            if (other.Shape[ 0 ] != K)
                throw new ArgumentException( $"Inner dimensions must match: ({M}x{K}) @ ({other.Shape[ 0 ]}x{N})" );

            var result = new Tensor( new[] { M, N }, new[] { this, other }, "matmul", device: Device );

            // Forward: C = A @ B
            Backend.MatMul( DataStorage, other.DataStorage, result.DataStorage, M, K, N, accumulate: false );

            result.RequiresGrad = this.RequiresGrad || other.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward: dA = dC @ B^T, dB = A^T @ dC
            result._backward = () =>
            {
                Backend.MatMulBackward(
                    DataStorage, other.DataStorage,
                    GradStorage, other.GradStorage,
                    result.GradStorage,
                    M, K, N,
                    this.RequiresGrad, other.RequiresGrad );
            };

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Sum all elements to a scalar tensor.
        /// </summary>
        public Tensor Sum()
        {
            var result = new Tensor( new[] { 1 }, new[] { this }, "sum", device: Device );
            Backend.Sum( DataStorage, result.DataStorage, Size );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                Backend.SumBackward( GradStorage, result.GradStorage, Size );
            };

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Sum along a specific dimension.
        /// For example, if shape is [2, 3, 4] and dim=1, result shape is [2, 4].
        /// </summary>
        public Tensor Sum( int dim )
        {
            int rank = Shape.Length;

            // Support negative dims
            if (dim < 0) dim += rank;
            if (dim < 0 || dim >= rank)
                throw new ArgumentException(
                    $"Dimension {dim} out of range for shape with {rank} dimensions" );

            // Build output shape by dropping the reduced dim
            var outShape = new int[ rank - 1 ];
            for (int i = 0, j = 0; i < rank; i++)
                if (i != dim)
                    outShape[ j++ ] = Shape[ i ];

            // Edge case: full reduction -> scalar
            if (outShape.Length == 0)
                return Sum(); // your existing full sum

            var result = new Tensor( outShape, new[] { this }, $"sum(dim={dim})", device: Device );

            int dimSize = Shape[ dim ];

            // product of sizes after "dim"
            int innerSize = 1;
            for (int i = dim + 1; i < rank; i++)
                innerSize *= Shape[ i ];

            // number of elements in one [dim, inner] block
            int blockSize = dimSize * innerSize;

            // how many such blocks we have (product of sizes before "dim")
            int outerSize = Size / blockSize;

            // Forward: sum along dim via backend
            Backend.SumDim( DataStorage, result.DataStorage, outerSize, dimSize, innerSize );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward: broadcast grad back over reduced dim
            result._backward = () =>
            {
                Backend.SumDimBackward( GradStorage, result.GradStorage, outerSize, dimSize, innerSize );
            };

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Mean of all elements to a scalar tensor.
        /// </summary>
        public Tensor Mean()
        {
            var sum = Sum();
            return sum / (float)Size;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Mean along a specific dimension.
        /// For example, if shape is [2, 3, 4] and dim=1, result shape is [2, 4].
        /// </summary>
        public Tensor Mean( int dim )
        {
            int rank = Shape.Length;

            // Support negative dims
            if (dim < 0) dim += rank;
            if (dim < 0 || dim >= rank)
                throw new ArgumentException(
                    $"Dimension {dim} out of range for shape with {rank} dimensions" );

            var sum = Sum( dim );
            var dimSize = Shape[ dim ];
            return sum / (float)dimSize;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Normalizes the tensor: (x - mean) / sqrt(variance + epsilon)
        /// Supports normalization over entire tensor (dim=null), first dimension (dim=0), 
        /// or last dimension (dim=-1 or dim=rank-1).
        /// </summary>
        /// <param name="dim">Optional dimension to normalize along. 
        /// null: normalize entire tensor
        /// 0: normalize along first dimension (batch normalization style)
        /// -1 or last: normalize along last dimension (layer normalization style)</param>
        /// <param name="epsilon">Small value to prevent division by zero (default 1e-5)</param>
        /// <returns>Normalized tensor</returns>
        public Tensor Normalize( int? dim = null, float epsilon = 1e-5f )
        {
            if (!dim.HasValue)
            {
                // Normalize entire tensor
                var mean = Mean();
                var centered = this - mean;
                var variance = (centered * centered).Mean();
                var std = (variance + epsilon).Sqrt();
                return centered / std;
            }

            int rank = Shape.Length;
            int actualDim = dim.Value;
            if (actualDim < 0) actualDim += rank;

            // Only support dim=0 or dim=last
            if (actualDim != 0 && actualDim != rank - 1)
                throw new ArgumentException(
                    $"Normalize only supports dim=null (all), dim=0 (first), or dim={rank - 1} (last). Got dim={dim.Value}" );

            if (actualDim == 0)
            {
                // Normalize along first dimension (batch norm style)
                var mean = Mean( 0 );
                var centered = this - mean;
                var variance = (centered * centered).Mean( 0 );
                var std = (variance + epsilon).Sqrt();
                return centered / std;
            }
            else
            {
                // Normalize along last dimension (layer norm style)
                // Use ExpandLast to match dimensions for broadcasting
                var mean = Mean( rank - 1 );
                var meanExpanded = mean.ExpandLast( Shape[ rank - 1 ] );
                var centered = this - meanExpanded;
                var variance = (centered * centered).Mean( rank - 1 );
                var varianceExpanded = variance.ExpandLast( Shape[ rank - 1 ] );
                var std = (varianceExpanded + epsilon).Sqrt();
                return centered / std;
            }
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Maximum value across all elements to a scalar tensor.
        /// </summary>
        public Tensor Max()
        {
            var result = new Tensor( new[] { 1 }, new[] { this }, "max", device: Device );
            Backend.MaxReduce( DataStorage, result.DataStorage, Size, out int maxIdx );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                // Gradient only flows to the max element
                Grad[ maxIdx ] += result.Grad[ 0 ];
            };

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Maximum values along a specific dimension.
        /// For example, if shape is [2, 3, 4] and dim=1, result shape is [2, 4].
        /// </summary>
        public Tensor Max( int dim )
        {
            int rank = Shape.Length;

            // Support negative dims
            if (dim < 0) dim += rank;
            if (dim < 0 || dim >= rank)
                throw new ArgumentException(
                    $"Dimension {dim} out of range for shape with {rank} dimensions" );

            // Build output shape by dropping the reduced dim
            var outShape = new int[ rank - 1 ];
            for (int i = 0, j = 0; i < rank; i++)
                if (i != dim)
                    outShape[ j++ ] = Shape[ i ];

            // Edge case: full reduction -> scalar
            if (outShape.Length == 0)
                return Max();

            var result = new Tensor( outShape, new[] { this }, $"max(dim={dim})", device: Device );

            // Calculate strides
            int dimSize = Shape[ dim ];
            int innerSize = 1;
            for (int i = dim + 1; i < rank; i++)
                innerSize *= Shape[ i ];

            int blockSize = dimSize * innerSize;
            int outerSize = Size / blockSize;

            // Store indices of max elements for backward pass
            var maxIndices = new int[ result.Size ];

            // Forward: find max along dimension
            for (int outer = 0; outer < outerSize; outer++)
            {
                int baseBlock = outer * blockSize;

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int baseIdx = baseBlock + inner;
                    float maxVal = float.MinValue;
                    int maxLocalIdx = 0;

                    for (int d = 0; d < dimSize; d++)
                    {
                        int idx = baseIdx + d * innerSize;
                        if (Data[ idx ] > maxVal)
                        {
                            maxVal = Data[ idx ];
                            maxLocalIdx = d;
                        }
                    }

                    int outIdx = outer * innerSize + inner;
                    result.Data[ outIdx ] = maxVal;
                    maxIndices[ outIdx ] = baseIdx + maxLocalIdx * innerSize;
                }
            }

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward: gradient only flows to max elements
            result._backward = () =>
            {
                for (int i = 0; i < result.Size; i++)
                {
                    Grad[ maxIndices[ i ] ] += result.Grad[ i ];
                }
            };

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Adds a dimension of size 1 at the specified position (PyTorch: unsqueeze).
        /// This is a view operation - shares the same Data and Grad arrays.
        /// </summary>
        /// <param name="dim">Dimension index where to insert the new axis. Supports negative indexing.</param>
        /// <returns>New tensor view with expanded shape</returns>
        public Tensor Unsqueeze( int dim )
        {
            int rank = Shape.Length;

            // Support negative indexing: -1 means after last dim
            if (dim < 0) dim += rank + 1;
            if (dim < 0 || dim > rank)
                throw new ArgumentException(
                    $"Dimension {dim} out of range for tensor with {rank} dimensions (valid range: -{rank - 1} to {rank})" );

            // Build new shape with inserted dimension
            var newShape = new int[ rank + 1 ];
            for (int i = 0; i < dim; i++)
                newShape[ i ] = Shape[ i ];
            newShape[ dim ] = 1;
            for (int i = dim; i < rank; i++)
                newShape[ i + 1 ] = Shape[ i ];

            // Create view tensor sharing data/grad (no allocation)
            return new Tensor( newShape, DataStorage, GradStorage, new[] { this }, $"unsqueeze({dim})", RequiresGrad );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Removes dimensions of size 1 (PyTorch: squeeze).
        /// This is a view operation - shares the same Data and Grad arrays.
        /// </summary>
        /// <param name="dim">Optional dimension to squeeze. If null, removes all dimensions of size 1.</param>
        /// <returns>New tensor view with reduced shape</returns>
        public Tensor Squeeze( int? dim = null )
        {
            int rank = Shape.Length;

            if (dim.HasValue)
            {
                // Squeeze specific dimension
                int d = dim.Value;

                // Support negative indexing
                if (d < 0) d += rank;
                if (d < 0 || d >= rank)
                    throw new ArgumentException(
                        $"Dimension {dim.Value} out of range for tensor with {rank} dimensions" );

                if (Shape[ d ] != 1)
                    throw new ArgumentException(
                        $"Cannot squeeze dimension {d} with size {Shape[ d ]} (must be 1)" );

                // Build new shape without the squeezed dimension
                var newShape = new int[ rank - 1 ];
                for (int i = 0, j = 0; i < rank; i++)
                    if (i != d)
                        newShape[ j++ ] = Shape[ i ];

                // Handle edge case: squeezing last dimension results in empty shape
                if (newShape.Length == 0)
                    newShape = new[] { 1 }; // Keep as scalar [1]

                // Create view tensor sharing data/grad (no allocation)
                return new Tensor( newShape, DataStorage, GradStorage, new[] { this }, $"squeeze({d})", RequiresGrad );
            }
            else
            {
                // Squeeze all dimensions of size 1
                var newShapeList = new List<int>();
                for (int i = 0; i < rank; i++)
                    if (Shape[ i ] != 1)
                        newShapeList.Add( Shape[ i ] );

                // If all dimensions were 1, keep as scalar
                if (newShapeList.Count == 0)
                    newShapeList.Add( 1 );

                var newShape = newShapeList.ToArray();

                // Create view tensor sharing data/grad (no allocation)
                return new Tensor( newShape, DataStorage, GradStorage, new[] { this }, "squeeze()", RequiresGrad );
            }
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Extracts a slice along a specific dimension starting from 'start' and taking 'length' elements.
        /// All other dimensions remain unchanged. Gradients flow back to the original tensor.
        /// </summary>
        /// <param name="dim">The dimension to slice along</param>
        /// <param name="start">Starting index in that dimension</param>
        /// <param name="length">Number of elements to take from that dimension</param>
        /// <returns>New tensor with reduced size along the specified dimension</returns>
        public Tensor Slice( int dim, int start, int length )
        {
            int rank = Shape.Length;

            // Support negative indexing
            if (dim < 0) dim += rank;
            if (dim < 0 || dim >= rank)
                throw new ArgumentException(
                    $"Dimension {dim} out of range for tensor with {rank} dimensions" );

            if (start < 0 || start >= Shape[ dim ])
                throw new ArgumentException(
                    $"Start index {start} out of range for dimension {dim} with size {Shape[ dim ]}" );

            if (length <= 0 || start + length > Shape[ dim ])
                throw new ArgumentException(
                    $"Length {length} invalid for dimension {dim}: start={start}, size={Shape[ dim ]}" );

            // Build result shape: same as input but with reduced size at 'dim'
            var resultShape = (int[])Shape.Clone();
            resultShape[ dim ] = length;

            var result = new Tensor( resultShape, new[] { this }, $"slice(dim={dim},start={start})", device: Device );

            // Calculate strides
            int innerSize = 1;
            for (int i = dim + 1; i < rank; i++)
                innerSize *= Shape[ i ];

            int blockSize = Shape[ dim ] * innerSize;
            int outerSize = Size / blockSize;
            int sliceBlockSize = length * innerSize;

            // Forward: copy sliced elements via backend
            Backend.SliceCopy( DataStorage, result.DataStorage, outerSize, blockSize, sliceBlockSize, start * innerSize, sliceBlockSize );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward: accumulate gradients back to source positions
            result._backward = () =>
            {
                Backend.SliceCopyBackward( GradStorage, result.GradStorage, outerSize, blockSize, sliceBlockSize, start * innerSize, sliceBlockSize );
            };

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Expands the tensor by repeating the last dimension.
        /// For example, shape [3, 2] with num=5 becomes [3, 2, 5] where each element is replicated 5 times.
        /// </summary>
        /// <param name="num">Number of times to repeat the last dimension</param>
        /// <returns>New tensor with expanded shape</returns>
        public Tensor ExpandLast( int num )
        {
            if (num <= 0)
                throw new ArgumentException( $"Number of repetitions must be positive, got {num}" );

            // Build new shape: original shape + new dimension of size num
            var newShape = new int[ Shape.Length + 1 ];
            for (int i = 0; i < Shape.Length; i++)
                newShape[ i ] = Shape[ i ];
            newShape[ Shape.Length ] = num;

            var result = new Tensor( newShape, new[] { this }, $"expandLast({num})", device: Device );

            // Forward pass: replicate each element num times via backend
            Backend.ExpandLast( DataStorage, result.DataStorage, Size, num );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward pass: accumulate gradients from all replications
            result._backward = () =>
            {
                Backend.ExpandLastBackward( GradStorage, result.GradStorage, Size, num );
            };

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Reshapes the tensor to a new shape while keeping the same data.
        /// This is a view operation - shares the same Data and Grad arrays.
        /// </summary>
        /// <param name="newShape">New shape dimensions. Total size must match current size.</param>
        /// <returns>New tensor view with reshaped data</returns>
        public Tensor Reshape( params int[] newShape )
        {
            if (newShape == null || newShape.Length == 0)
                throw new ArgumentException( "New shape cannot be null or empty" );

            // Calculate total size of new shape
            int newSize = 1;
            foreach (int dim in newShape)
            {
                if (dim <= 0)
                    throw new ArgumentException( $"All dimensions must be positive, got {dim}" );
                newSize *= dim;
            }

            // Verify total size matches
            if (newSize != Size)
                throw new ArgumentException(
                    $"Total size must remain the same. Current size: {Size}, new size: {newSize}" );

            // Create view tensor sharing data/grad (no allocation)
            return new Tensor( newShape, DataStorage, GradStorage, new[] { this }, $"reshape({string.Join( ",", newShape )})", RequiresGrad );
        }
        //------------------------------------------------------------------
        private static bool CanBroadcastModulo( Tensor a, Tensor b )
        {
            int sizeA = a.Size;
            int sizeB = b.Size;

            // TODO: fix issue when (2,3) + (3,2) incorrectly allowed
            // 1) Both scalars or equal sizes always OK
            if (sizeA == sizeB)
                return true;

            // 2) One is scalar
            if (sizeA == 1 || sizeB == 1)
                return true;

            // 3) Larger must be clean multiple of smaller
            int bigger = Math.Max( sizeA, sizeB );
            int smaller = Math.Min( sizeA, sizeB );

            if (bigger % smaller != 0)
                return false;

            // 4) Enforce that one is 1D or classic bias pattern
            //    This stops invalid multi-dimensional flattening patterns.
            if (a.Shape.Length > 1 && b.Shape.Length > 1)
            {
                // Only allow classic bias: [M,N] + [N]
                // Check if one shape is suffix of the other
                int[] shorter = a.Shape.Length < b.Shape.Length ? a.Shape : b.Shape;
                int[] longer = a.Shape.Length < b.Shape.Length ? b.Shape : a.Shape;

                // Check if shorter matches end of longer
                int offset = longer.Length - shorter.Length;
                for (int i = 0; i < shorter.Length; i++)
                {
                    if (shorter[ i ] != longer[ offset + i ])
                        return false;
                }
            }

            return true;
        }
        //------------------------------------------------------------------
        public int ToFlatIndex( int[] indices )
        {
            if (indices == null || indices.Length != Shape.Length)
                throw new ArgumentException(
                    $"Expected {Shape.Length} indices but got {indices?.Length ?? 0}" );

            int flatIndex = 0;
            int stride = 1;

            // Convert multi-dimensional indices to flat index (row-major order)
            for (int i = Shape.Length - 1; i >= 0; i--)
            {
                if (indices[ i ] < 0 || indices[ i ] >= Shape[ i ])
                    throw new IndexOutOfRangeException(
                        $"Index {indices[ i ]} is out of range for dimension {i} with size {Shape[ i ]}" );

                flatIndex += indices[ i ] * stride;
                stride *= Shape[ i ];
            }

            return flatIndex;
        }
        //------------------------------------------------------------------
        public void Backward()
        {
            var topo = new List<Tensor>();
            var visited = new HashSet<Tensor>();

            void BuildTopo( Tensor t )
            {
                if (visited.Contains( t ))
                    return;

                visited.Add( t );
                foreach (var child in t.Children)
                    BuildTopo( child );
                topo.Add( t );
            }

            BuildTopo( this );

            Backend.FillOnes( GradStorage, GradStorage.Length );

            topo.Reverse();
            foreach (var t in topo)
                t._backward?.Invoke();
        }
        //------------------------------------------------------------------
        public void ZeroGrad()
        {
            GradStorage.Clear();
        }
        //------------------------------------------------------------------
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append( "Tensor(" );
            sb.Append( string.Join( "x", Shape ) );
            sb.Append( ") [" );

            int preview = Math.Min( 5, Size );
            for (int i = 0; i < preview; i++)
            {
                sb.Append( Data[ i ].ToString( "G6" ) );
                if (i < preview - 1) sb.Append( ", " );
            }
            if (Size > preview) sb.Append( "..." );

            sb.Append( "]" );
            return sb.ToString();
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Creates a copy of this tensor on the specified device.
        /// If already on the target device, returns the same tensor (no copy).
        /// </summary>
        public Tensor To( TensorDevice targetDevice )
        {
            if (Device == targetDevice)
                return this;

            var result = new Tensor( (int[])Shape.Clone(), device: targetDevice, name: Name, requiresGrad: RequiresGrad );

            // Copy data from source storage to destination via CPU-accessible readback
            var tempData = new float[ Size ];
            DataStorage.CopyTo( tempData );
            result.DataStorage.CopyFrom( tempData );

            var tempGrad = new float[ Size ];
            GradStorage.CopyTo( tempGrad );
            result.GradStorage.CopyFrom( tempGrad );

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>Convenience for <see cref="To"/>(<see cref="TensorDevice.CPU"/>).</summary>
        public Tensor ToCpu() => To( TensorDevice.CPU );

        /// <summary>Convenience for <see cref="To"/>(<see cref="TensorDevice.GPU"/>).</summary>
        public Tensor ToGpu() => To( TensorDevice.GPU );
        //------------------------------------------------------------------
    }
}
