using System;
using System.Collections.Generic;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Profiling;

namespace ChaosRL
{
    /// <summary>
    /// Multi-dimensional tensor with automatic differentiation support.
    /// Data is stored in row-major (C-style) contiguous layout for cache efficiency.
    /// </summary>
    public class Tensor : IDisposable
    {
        //------------------------------------------------------------------
        public ref NativeArray<float> Data => ref _data;
        public ref NativeArray<float> Grad => ref _grad;
        public int[] Shape { get; private set; }
        public int Size { get; private set; }
        public string Name { get; set; }
        public HashSet<Tensor> Children { get; private set; }
        public bool IsScalar => Size == 1;
        public bool RequiresGrad { get; set; }

        private NativeArray<float> _data;
        private NativeArray<float> _grad;
        // For view tensors, keep a strong reference to the tensor that owns the storage.
        // This prevents the owning tensor from being GC'd/finalized while views are still alive.
        private Tensor _storageOwner;
        private Action _backward;
        private bool _disposed;

        private const int GebpMatMulThreshold = 16;
        private const int MinScheduleBatch = 1;

        private static readonly ProfilerMarker sMatMulTransposeTimeMarker = new ProfilerMarker( "ChaosRL.MatMul.transpose_time" );
        private static readonly ProfilerMarker sMatMulGemmTimeMarker = new ProfilerMarker( "ChaosRL.MatMul.gemm_time" );
        private static readonly ProfilerMarker sMatMulBackwardDATimeMarker = new ProfilerMarker( "ChaosRL.MatMul.backward_dA_time" );
        private static readonly ProfilerMarker sMatMulBackwardDBTimeMarker = new ProfilerMarker( "ChaosRL.MatMul.backward_dB_time" );
        //------------------------------------------------------------------
        public float this[ params int[] indices ]
        {
            get => Data[ ToFlatIndex( indices ) ];
            set => Data[ ToFlatIndex( indices ) ] = value;
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
        private static Tensor ResolveStorageOwner( Tensor t )
        {
            if (t == null)
                throw new ArgumentNullException( nameof( t ) );

            var current = t;
            while (current._storageOwner != null)
                current = current._storageOwner;

            return current;
        }

        //------------------------------------------------------------------
        // TODO: switch to params int[] for shape to improve usability
        public Tensor( int[] shape, float[] data = null, string name = "", bool requiresGrad = true )
        {
            ValidateAndCalculateSize( shape );

            if (data != null && data.Length != Size)
                throw new ArgumentException( $"Data length {data.Length} doesn't match shape size {Size}" );

            _data = new NativeArray<float>( Size, Allocator.Persistent, NativeArrayOptions.ClearMemory );
            _grad = new NativeArray<float>( Size, Allocator.Persistent, NativeArrayOptions.ClearMemory );
            _storageOwner = null;

            if (data != null)
                _data.CopyFrom( data );

            Name = name;
            Children = new HashSet<Tensor>();
            RequiresGrad = requiresGrad;
            _backward = null;
        }
        //------------------------------------------------------------------
        private Tensor( int[] shape, Tensor storageOwner, Tensor[] children, string name, bool requiresGrad )
        {
            ValidateAndCalculateSize( shape );

            var owner = ResolveStorageOwner( storageOwner );
            if (owner._disposed)
                throw new ObjectDisposedException( nameof( Tensor ), "Storage owner tensor is disposed" );

            if (!owner._data.IsCreated || !owner._grad.IsCreated)
                throw new ArgumentException( "Data/Grad must be created for view tensors" );
            if (owner._data.Length != Size || owner._grad.Length != Size)
                throw new ArgumentException( $"View tensor storage length must match shape size {Size}" );

            _data = owner._data;
            _grad = owner._grad;
            _storageOwner = owner;
            Name = name;
            Children = new HashSet<Tensor>();
            if (children != null)
                foreach (var child in children)
                    Children.Add( child );
            RequiresGrad = requiresGrad;
            _backward = null;
        }
        //------------------------------------------------------------------
        public Tensor( int[] shape, Tensor[] children, string name = "" ) : this( shape, (float[])null, name )
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
        // TODO: move to ArenaPool and TensorStorage pattern
        // Relying on finalizer to catch undisposed tensors is bad practice
        // Finalizers are non-deterministic and can lead to memory leaks
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

            // Only the owning tensor is responsible for disposing the shared storage.
            // Views keep a strong reference to the owner via _storageOwner.
            if (_storageOwner == null)
            {
                if (_grad.IsCreated)
                    _grad.Dispose();
                if (_data.IsCreated)
                    _data.Dispose();
            }

            _grad = default;
            _data = default;
            _storageOwner = null;
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
            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for addition" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "+" );
            for (int i = 0; i < resultSize; i++)
                result.Data[ i ] = a.Data[ i % sizeA ] + b.Data[ i % sizeB ];

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                for (int i = 0; i < resultSize; i++)
                {
                    a.Grad[ i % sizeA ] += result.Grad[ i ] * aGradScale;
                    b.Grad[ i % sizeB ] += result.Grad[ i ] * bGradScale;
                }
            };
            return result;
        }
        //------------------------------------------------------------------
        // Element-wise multiplication
        public static Tensor operator *( Tensor a, Tensor b )
        {
            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for multiplication" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "*" );
            for (int i = 0; i < resultSize; i++)
                result.Data[ i ] = a.Data[ i % sizeA ] * b.Data[ i % sizeB ];

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                for (int i = 0; i < resultSize; i++)
                {
                    a.Grad[ i % sizeA ] += b.Data[ i % sizeB ] * result.Grad[ i ] * aGradScale;
                    b.Grad[ i % sizeB ] += a.Data[ i % sizeA ] * result.Grad[ i ] * bGradScale;
                }
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
            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for division" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "/" );
            for (int i = 0; i < resultSize; i++)
                result.Data[ i ] = a.Data[ i % sizeA ] / b.Data[ i % sizeB ];

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                for (int i = 0; i < resultSize; i++)
                {
                    int idxA = i % sizeA;
                    int idxB = i % sizeB;
                    a.Grad[ idxA ] += (1f / b.Data[ idxB ]) * result.Grad[ i ] * aGradScale;
                    b.Grad[ idxB ] += (-a.Data[ idxA ] / (b.Data[ idxB ] * b.Data[ idxB ])) * result.Grad[ i ] * bGradScale;
                }
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Pow( float exponent )
        {
            var result = new Tensor( Shape, new[] { this }, $"^{exponent}" );
            for (int i = 0; i < Size; i++)
                result.Data[ i ] = MathF.Pow( Data[ i ], exponent );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                for (int i = 0; i < Size; i++)
                    Grad[ i ] += exponent * MathF.Pow( Data[ i ], exponent - 1 ) * result.Grad[ i ];
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
            var result = new Tensor( Shape, new[] { this }, "exp" );
            for (int i = 0; i < Size; i++)
                result.Data[ i ] = MathF.Exp( Data[ i ] );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                for (int i = 0; i < Size; i++)
                    Grad[ i ] += result.Data[ i ] * result.Grad[ i ];
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Log()
        {
            var result = new Tensor( Shape, new[] { this }, "log" );
            for (int i = 0; i < Size; i++)
                result.Data[ i ] = MathF.Log( Data[ i ] );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                for (int i = 0; i < Size; i++)
                    Grad[ i ] += (1f / Data[ i ]) * result.Grad[ i ];
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor ReLU()
        {
            var result = new Tensor( Shape, new[] { this }, "ReLU" );
            for (int i = 0; i < Size; i++)
                result.Data[ i ] = Data[ i ] > 0 ? Data[ i ] : 0;

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                for (int i = 0; i < Size; i++)
                    Grad[ i ] += (Data[ i ] > 0 ? 1f : 0f) * result.Grad[ i ];
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Tanh()
        {
            var result = new Tensor( Shape, new[] { this }, "tanh" );
            for (int i = 0; i < Size; i++)
                result.Data[ i ] = MathF.Tanh( Data[ i ] );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                for (int i = 0; i < Size; i++)
                    Grad[ i ] += (1f - result.Data[ i ] * result.Data[ i ]) * result.Grad[ i ];
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Clamp( float min, float max )
        {
            var result = new Tensor( Shape, new[] { this }, "clamp" );
            for (int i = 0; i < Size; i++)
                result.Data[ i ] = Math.Max( min, Math.Min( max, Data[ i ] ) );

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                for (int i = 0; i < Size; i++)
                    if (Data[ i ] >= min && Data[ i ] <= max)
                        Grad[ i ] += result.Grad[ i ];
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Tensor Max( Tensor a, Tensor b )
        {
            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for max" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "max" );
            for (int i = 0; i < resultSize; i++)
                result.Data[ i ] = Math.Max( a.Data[ i % sizeA ], b.Data[ i % sizeB ] );

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                for (int i = 0; i < resultSize; i++)
                {
                    int idxA = i % sizeA;
                    int idxB = i % sizeB;
                    float isAMax = a.Data[ idxA ] >= b.Data[ idxB ] ? 1f : 0f;
                    a.Grad[ idxA ] += result.Grad[ i ] * isAMax * aGradScale;
                    b.Grad[ idxB ] += result.Grad[ i ] * (1f - isAMax) * bGradScale;
                }
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Tensor Min( Tensor a, Tensor b )
        {
            if (CanBroadcastModulo( a, b ) == false)
                throw new ArgumentException(
                    $"Cannot broadcast shapes [{string.Join( ",", a.Shape )}] and [{string.Join( ",", b.Shape )}] for min" );

            int sizeA = a.Size;
            int sizeB = b.Size;
            int resultSize = Math.Max( sizeA, sizeB );
            var resultShape = resultSize == sizeA ? a.Shape : b.Shape;

            var result = new Tensor( resultShape, new[] { a, b }, "min" );
            for (int i = 0; i < resultSize; i++)
                result.Data[ i ] = Math.Min( a.Data[ i % sizeA ], b.Data[ i % sizeB ] );

            result.RequiresGrad = a.RequiresGrad || b.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            float aGradScale = a.RequiresGrad ? 1f : 0f;
            float bGradScale = b.RequiresGrad ? 1f : 0f;

            result._backward = () =>
            {
                for (int i = 0; i < resultSize; i++)
                {
                    int idxA = i % sizeA;
                    int idxB = i % sizeB;
                    float isAMin = a.Data[ idxA ] <= b.Data[ idxB ] ? 1f : 0f;
                    a.Grad[ idxA ] += result.Grad[ i ] * isAMin * aGradScale;
                    b.Grad[ idxB ] += result.Grad[ i ] * (1f - isAMin) * bGradScale;
                }
            };
            return result;
        }
        //------------------------------------------------------------------
        // Job scheduling helpers for MatMul and transpose kernels.
        private static int GetBatchSize( int totalWorkItems )
        {
            int workerCount = Math.Max( 1, JobsUtility.JobWorkerCount + 1 );
            int batch = totalWorkItems / (workerCount * 4);
            return Math.Max( MinScheduleBatch, batch );
        }
        //------------------------------------------------------------------
        private static JobHandle ScheduleTranspose(
            NativeArray<float> input,
            NativeArray<float> output,
            int rows,
            int cols,
            JobHandle dependsOn = default )
        {
            int tileSize = TransposeTiledParallelJob.TILE;
            int tileRows = (rows + tileSize - 1) / tileSize;
            int tileCols = (cols + tileSize - 1) / tileSize;
            int totalTiles = tileRows * tileCols;

            var transposeJob = new TransposeTiledParallelJob
            {
                Input = input,
                Output = output,
                Rows = rows,
                Cols = cols
            };

            int batch = GetBatchSize( totalTiles );
            return transposeJob.Schedule( totalTiles, batch, dependsOn );
        }
        //------------------------------------------------------------------
        private static JobHandle ScheduleMatMul(
            NativeArray<float> a,
            NativeArray<float> bt,
            NativeArray<float> c,
            int m,
            int k,
            int n,
            bool accumulate,
            JobHandle dependsOn = default )
        {
            int totalElements = m * n;
            var naiveJob = new MatMulNaiveParallelJob
            {
                A = a,
                BT = bt,
                C = c,
                M = m,
                K = k,
                N = n,
                Accumulate = accumulate
            };

            int batch = GetBatchSize( totalElements );
            return naiveJob.Schedule( totalElements, batch, dependsOn );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Schedules a GEBP MatMul: C(m×n) = A(m×k) @ B(k×n).
        /// B must be in ORIGINAL row-major layout (NOT transposed).
        /// Packs B into column-panel layout once, then runs GEBP micro-kernels
        /// that read the packed buffer sequentially for full L1 throughput.
        /// The packed buffer is auto-disposed when the returned handle completes.
        /// </summary>
        private static JobHandle ScheduleGebpMatMul(
            NativeArray<float> a,
            NativeArray<float> b,
            NativeArray<float> c,
            int m,
            int k,
            int n,
            bool accumulate,
            JobHandle dependsOn = default )
        {
            const int NR = PackBPanelScalarParallelJob.NR;
            int numPanels = (n + NR - 1) / NR;
            int packedSize = numPanels * k * NR;

            var packedB = new NativeArray<float>( packedSize, Allocator.TempJob );

            // Phase 1: Pack B into column-panel layout (parallel over panels)
            var packJob = new PackBPanelScalarParallelJob
            {
                B = b,
                PackedB = packedB,
                K = k,
                N = n
            };
            int packBatch = GetBatchSize( numPanels );
            var packHandle = packJob.Schedule( numPanels, packBatch, dependsOn );

            // Phase 2: GEBP micro-kernels read from packed B (parallel over row-groups)
            int rowGroups = (m + MatMulGebpScalarParallelJob.MR - 1) / MatMulGebpScalarParallelJob.MR;
            var gebpJob = new MatMulGebpScalarParallelJob
            {
                A = a,
                PackedB = packedB,
                C = c,
                M = m,
                K = k,
                N = n,
                Accumulate = accumulate
            };
            int batch = GetBatchSize( rowGroups );
            var gebpHandle = gebpJob.Schedule( rowGroups, batch, packHandle );

            // Auto-dispose packed buffer after GEBP completes
            return packedB.Dispose( gebpHandle );
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Matrix multiplication: (M×K) @ (K×N) -> (M×N)
        /// Supports 2D tensors for now.
        /// </summary>
        public Tensor MatMul( Tensor other )
        {
            // Validate 2D tensors
            if (Shape.Length != 2 || other.Shape.Length != 2)
                throw new ArgumentException( "MatMul requires 2D tensors" );

            int M = Shape[ 0 ]; // rows of this
            int K = Shape[ 1 ]; // cols of this / rows of other
            int N = other.Shape[ 1 ]; // cols of other

            if (other.Shape[ 0 ] != K)
                throw new ArgumentException( $"Inner dimensions must match: ({M}x{K}) @ ({other.Shape[ 0 ]}x{N})" );

            var result = new Tensor( new[] { M, N }, new[] { this, other }, "matmul" );

            // Forward pass:
            // Use GEBP FMA kernel when matrix is large enough — reads B in original
            // K×N layout, eliminating the separate transpose step entirely.
            bool useGebp = M >= GebpMatMulThreshold &&
                           K >= GebpMatMulThreshold &&
                           N >= GebpMatMulThreshold;

            if (useGebp)
            {
                using (sMatMulGemmTimeMarker.Auto())
                {
                    var mmHandle = ScheduleGebpMatMul(
                        Data,
                        other.Data,
                        result.Data,
                        M,
                        K,
                        N,
                        accumulate: false );

                    mmHandle.Complete();
                }
            }
            else
            {
                // Small matrix path: transpose B then use row/blocked kernel
                using (var nativeBT = new NativeArray<float>( other.Size, Allocator.TempJob ))
                {
                    JobHandle transposeHandle;
                    using (sMatMulTransposeTimeMarker.Auto())
                    {
                        transposeHandle = ScheduleTranspose( other.Data, nativeBT, K, N );
                    }

                    using (sMatMulGemmTimeMarker.Auto())
                    {
                        var mmHandle = ScheduleMatMul(
                            Data,
                            nativeBT,
                            result.Data,
                            M,
                            K,
                            N,
                            accumulate: false,
                            dependsOn: transposeHandle );

                        mmHandle.Complete();
                    }
                }
            }

            result.RequiresGrad = this.RequiresGrad || other.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward pass
            result._backward = () =>
            {
                if (useGebp)
                {
                    // GEBP backward: schedule dA and dB transposes + matmuls concurrently
                    JobHandle dAHandle = default, dBHandle = default;
                    NativeArray<float> tempBT = default, tempAT = default;
                    bool hasDa = false, hasDb = false;

                    // dL/dA = dC(MxN) @ B^T(NxK) => (MxK)
                    if (this.RequiresGrad)
                    {
                        using (sMatMulBackwardDATimeMarker.Auto())
                        {
                            tempBT = new NativeArray<float>( other.Size, Allocator.TempJob );
                            var tBH = ScheduleTranspose( other.Data, tempBT, K, N );
                            dAHandle = ScheduleGebpMatMul(
                                result.Grad, tempBT, Grad,
                                M, N, K,
                                accumulate: true,
                                dependsOn: tBH );
                            hasDa = true;
                        }
                    }

                    // dL/dB = A^T(KxM) @ dC(MxN) => (KxN)
                    // dC is already MxN row-major — no transpose needed!
                    if (other.RequiresGrad)
                    {
                        using (sMatMulBackwardDBTimeMarker.Auto())
                        {
                            tempAT = new NativeArray<float>( Size, Allocator.TempJob );
                            var tAH = ScheduleTranspose( Data, tempAT, M, K );
                            dBHandle = ScheduleGebpMatMul(
                                tempAT, result.Grad, other.Grad,
                                K, M, N,
                                accumulate: true,
                                dependsOn: tAH );
                            hasDb = true;
                        }
                    }

                    // Wait for both to complete concurrently
                    if (hasDa && hasDb)
                        JobHandle.CombineDependencies( dAHandle, dBHandle ).Complete();
                    else if (hasDa)
                        dAHandle.Complete();
                    else if (hasDb)
                        dBHandle.Complete();

                    // Dispose temp buffers
                    if (tempBT.IsCreated) tempBT.Dispose();
                    if (tempAT.IsCreated) tempAT.Dispose();
                }
                else
                {
                    // Small matrix backward path (original BT-based approach)
                    JobHandle dAHandle = default;
                    bool dAScheduled = false;

                    if (this.RequiresGrad)
                    {
                        using (sMatMulBackwardDATimeMarker.Auto())
                        {
                            dAHandle = ScheduleMatMul(
                                result.Grad, other.Data, Grad,
                                M, N, K,
                                accumulate: true );
                            dAScheduled = true;
                        }
                    }

                    if (other.RequiresGrad)
                    {
                        using (var nativeAT = new NativeArray<float>( Size, Allocator.TempJob ))
                        using (var nativeDCT = new NativeArray<float>( result.Size, Allocator.TempJob ))
                        {
                            JobHandle tAHandle, tDCHandle;
                            using (sMatMulTransposeTimeMarker.Auto())
                            {
                                tAHandle = ScheduleTranspose( Data, nativeAT, M, K );
                                tDCHandle = ScheduleTranspose( result.Grad, nativeDCT, M, N );
                            }

                            JobHandle dBHandle;
                            using (sMatMulBackwardDBTimeMarker.Auto())
                            {
                                var dBDeps = JobHandle.CombineDependencies( tAHandle, tDCHandle );
                                dBHandle = ScheduleMatMul(
                                    nativeAT, nativeDCT, other.Grad,
                                    K, M, N,
                                    accumulate: true,
                                    dependsOn: dBDeps );
                            }

                            if (dAScheduled)
                                JobHandle.CombineDependencies( dAHandle, dBHandle ).Complete();
                            else
                                dBHandle.Complete();
                            return;
                        }
                    }

                    if (dAScheduled)
                        dAHandle.Complete();
                }
            };

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Sum all elements to a scalar tensor.
        /// </summary>
        public Tensor Sum()
        {
            var result = new Tensor( new[] { 1 }, new[] { this }, "sum" );

            float sum = 0f;
            for (int i = 0; i < Size; i++)
                sum += Data[ i ];
            result.Data[ 0 ] = sum;

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            result._backward = () =>
            {
                // Gradient broadcasts to all input elements
                for (int i = 0; i < Size; i++)
                    Grad[ i ] += result.Grad[ 0 ];
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

            var result = new Tensor( outShape, new[] { this }, $"sum(dim={dim})" );

            int dimSize = Shape[ dim ];

            // product of sizes after "dim"
            int innerSize = 1;
            for (int i = dim + 1; i < rank; i++)
                innerSize *= Shape[ i ];

            // number of elements in one [dim, inner] block
            int blockSize = dimSize * innerSize;

            // how many such blocks we have (product of sizes before "dim")
            int outerSize = Size / blockSize;

            // Forward: for each (outer, inner) sum over dim
            for (int outer = 0; outer < outerSize; outer++)
            {
                int baseBlock = outer * blockSize;

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int baseIdx = baseBlock + inner;
                    float sum = 0f;

                    for (int d = 0; d < dimSize; d++)
                        sum += Data[ baseIdx + d * innerSize ];

                    result.Data[ outer * innerSize + inner ] = sum;
                }
            }

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward: broadcast grad back over reduced dim
            result._backward = () =>
            {
                for (int outer = 0; outer < outerSize; outer++)
                {
                    int baseBlock = outer * blockSize;

                    for (int inner = 0; inner < innerSize; inner++)
                    {
                        int baseIdx = baseBlock + inner;
                        float g = result.Grad[ outer * innerSize + inner ];

                        for (int d = 0; d < dimSize; d++)
                            Grad[ baseIdx + d * innerSize ] += g;
                    }
                }
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
            var result = new Tensor( new[] { 1 }, new[] { this }, "max" );

            float maxVal = float.MinValue;
            int maxIdx = 0;
            for (int i = 0; i < Size; i++)
            {
                if (Data[ i ] > maxVal)
                {
                    maxVal = Data[ i ];
                    maxIdx = i;
                }
            }
            result.Data[ 0 ] = maxVal;

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

            var result = new Tensor( outShape, new[] { this }, $"max(dim={dim})" );

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
            return new Tensor( newShape, this, new[] { this }, $"unsqueeze({dim})", RequiresGrad );
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
                return new Tensor( newShape, this, new[] { this }, $"squeeze({d})", RequiresGrad );
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
                return new Tensor( newShape, this, new[] { this }, "squeeze()", RequiresGrad );
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

            var result = new Tensor( resultShape, new[] { this }, $"slice(dim={dim},start={start})" );

            // Calculate strides
            int innerSize = 1;
            for (int i = dim + 1; i < rank; i++)
                innerSize *= Shape[ i ];

            int blockSize = Shape[ dim ] * innerSize;
            int outerSize = Size / blockSize;
            int sliceBlockSize = length * innerSize;

            // Forward: copy sliced elements
            for (int outer = 0; outer < outerSize; outer++)
            {
                int srcBase = outer * blockSize + start * innerSize;
                int dstBase = outer * sliceBlockSize;

                for (int i = 0; i < sliceBlockSize; i++)
                    result.Data[ dstBase + i ] = Data[ srcBase + i ];
            }

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward: accumulate gradients back to source positions
            result._backward = () =>
            {
                for (int outer = 0; outer < outerSize; outer++)
                {
                    int srcBase = outer * blockSize + start * innerSize;
                    int dstBase = outer * sliceBlockSize;

                    for (int i = 0; i < sliceBlockSize; i++)
                        Grad[ srcBase + i ] += result.Grad[ dstBase + i ];
                }
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

            var result = new Tensor( newShape, new[] { this }, $"expandLast({num})" );

            // Forward pass: replicate each element num times
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < num; j++)
                {
                    result.Data[ i * num + j ] = Data[ i ];
                }
            }

            result.RequiresGrad = this.RequiresGrad;
            if (result.RequiresGrad == false)
                return result;

            // Backward pass: accumulate gradients from all replications
            result._backward = () =>
            {
                for (int i = 0; i < Size; i++)
                {
                    for (int j = 0; j < num; j++)
                    {
                        Grad[ i ] += result.Grad[ i * num + j ];
                    }
                }
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
            return new Tensor( newShape, this, new[] { this }, $"reshape({string.Join( ",", newShape )})", RequiresGrad );
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

            for (int i = 0; i < Grad.Length; i++)
                Grad[ i ] = 1f;

            topo.Reverse();
            foreach (var t in topo)
                t._backward?.Invoke();
        }
        //------------------------------------------------------------------
        public void ZeroGrad()
        {
            for (int i = 0; i < Grad.Length; i++)
                Grad[ i ] = 0f;
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
    }
}
