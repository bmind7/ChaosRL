using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ChaosRL
{
    /// <summary>
    /// Multi-dimensional tensor with automatic differentiation support.
    /// Data is stored in row-major (C-style) contiguous layout for cache efficiency.
    /// </summary>
    public class Tensor
    {
        //------------------------------------------------------------------
        public float[] Data { get; private set; }
        public float[] Grad { get; private set; }
        public int[] Shape { get; private set; }
        public int Size { get; private set; }
        public string Name { get; set; }
        public HashSet<Tensor> Children { get; private set; }
        public bool IsScalar => Size == 1;

        private Action _backward;
        //------------------------------------------------------------------
        public Tensor( int[] shape, float[] data = null, string name = "" )
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

            Data = data ?? new float[ Size ];
            Grad = new float[ Size ];
            Name = name;
            Children = new HashSet<Tensor>();
            _backward = null;

            if (data != null && data.Length != Size)
                throw new ArgumentException( $"Data length {data.Length} doesn't match shape size {Size}" );
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
        public Tensor( float scalar, string name = "" ) : this( new[] { 1 }, new[] { scalar }, name )
        {
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
            // Handle scalar broadcasting
            if (a.IsScalar && !b.IsScalar)
                a = a.Broadcast( b.Shape );
            else if (b.IsScalar && !a.IsScalar)
                b = b.Broadcast( a.Shape );

            if (!ShapesMatch( a.Shape, b.Shape ))
                throw new ArgumentException( "Shapes must match for addition" );

            var result = new Tensor( a.Shape, new[] { a, b }, "+" );
            for (int i = 0; i < a.Size; i++)
                result.Data[ i ] = a.Data[ i ] + b.Data[ i ];

            result._backward = () =>
            {
                for (int i = 0; i < a.Size; i++)
                {
                    a.Grad[ i ] += result.Grad[ i ];
                    b.Grad[ i ] += result.Grad[ i ];
                }
            };
            return result;
        }
        //------------------------------------------------------------------
        // Element-wise multiplication
        public static Tensor operator *( Tensor a, Tensor b )
        {
            // Handle scalar broadcasting
            if (a.IsScalar && !b.IsScalar)
                a = a.Broadcast( b.Shape );
            else if (b.IsScalar && !a.IsScalar)
                b = b.Broadcast( a.Shape );

            if (!ShapesMatch( a.Shape, b.Shape ))
                throw new ArgumentException( "Shapes must match for multiplication" );

            var result = new Tensor( a.Shape, new[] { a, b }, "*" );
            for (int i = 0; i < a.Size; i++)
                result.Data[ i ] = a.Data[ i ] * b.Data[ i ];

            result._backward = () =>
            {
                for (int i = 0; i < a.Size; i++)
                {
                    a.Grad[ i ] += b.Data[ i ] * result.Grad[ i ];
                    b.Grad[ i ] += a.Data[ i ] * result.Grad[ i ];
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
            // Handle scalar broadcasting
            if (a.IsScalar && !b.IsScalar)
                a = a.Broadcast( b.Shape );
            else if (b.IsScalar && !a.IsScalar)
                b = b.Broadcast( a.Shape );

            if (!ShapesMatch( a.Shape, b.Shape ))
                throw new ArgumentException( "Shapes must match for division" );

            var result = new Tensor( a.Shape, new[] { a, b }, "/" );
            for (int i = 0; i < a.Size; i++)
                result.Data[ i ] = a.Data[ i ] / b.Data[ i ];

            result._backward = () =>
            {
                for (int i = 0; i < a.Size; i++)
                {
                    a.Grad[ i ] += (1f / b.Data[ i ]) * result.Grad[ i ];
                    b.Grad[ i ] += (-a.Data[ i ] / (b.Data[ i ] * b.Data[ i ])) * result.Grad[ i ];
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

            result._backward = () =>
            {
                for (int i = 0; i < Size; i++)
                    Grad[ i ] += exponent * MathF.Pow( Data[ i ], exponent - 1 ) * result.Grad[ i ];
            };
            return result;
        }
        //------------------------------------------------------------------
        public Tensor Exp()
        {
            var result = new Tensor( Shape, new[] { this }, "exp" );
            for (int i = 0; i < Size; i++)
                result.Data[ i ] = MathF.Exp( Data[ i ] );

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
            if (!ShapesMatch( a.Shape, b.Shape ))
                throw new ArgumentException( "Shapes must match for max" );

            var result = new Tensor( a.Shape, new[] { a, b }, "max" );
            for (int i = 0; i < a.Size; i++)
                result.Data[ i ] = Math.Max( a.Data[ i ], b.Data[ i ] );

            result._backward = () =>
            {
                for (int i = 0; i < a.Size; i++)
                {
                    if (a.Data[ i ] >= b.Data[ i ])
                        a.Grad[ i ] += result.Grad[ i ];
                    else
                        b.Grad[ i ] += result.Grad[ i ];
                }
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Tensor Min( Tensor a, Tensor b )
        {
            if (!ShapesMatch( a.Shape, b.Shape ))
                throw new ArgumentException( "Shapes must match for min" );

            var result = new Tensor( a.Shape, new[] { a, b }, "min" );
            for (int i = 0; i < a.Size; i++)
                result.Data[ i ] = Math.Min( a.Data[ i ], b.Data[ i ] );

            result._backward = () =>
            {
                for (int i = 0; i < a.Size; i++)
                {
                    if (a.Data[ i ] <= b.Data[ i ])
                        a.Grad[ i ] += result.Grad[ i ];
                    else
                        b.Grad[ i ] += result.Grad[ i ];
                }
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
            // Validate 2D tensors
            if (Shape.Length != 2 || other.Shape.Length != 2)
                throw new ArgumentException( "MatMul requires 2D tensors" );

            int M = Shape[ 0 ]; // rows of this
            int K = Shape[ 1 ]; // cols of this / rows of other
            int N = other.Shape[ 1 ]; // cols of other

            if (other.Shape[ 0 ] != K)
                throw new ArgumentException( $"Inner dimensions must match: ({M}×{K}) @ ({other.Shape[ 0 ]}×{N})" );

            var result = new Tensor( new[] { M, N }, new[] { this, other }, "matmul" );

            // Forward pass: C[i,j] = sum_k A[i,k] * B[k,j]
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < K; k++)
                    {
                        sum += Data[ i * K + k ] * other.Data[ k * N + j ];
                    }
                    result.Data[ i * N + j ] = sum;
                }
            }

            // Backward pass
            result._backward = () =>
            {
                // dL/dA[i,k] = sum_j dL/dC[i,j] * B[k,j]
                for (int i = 0; i < M; i++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        float grad_sum = 0f;
                        for (int j = 0; j < N; j++)
                        {
                            grad_sum += result.Grad[ i * N + j ] * other.Data[ k * N + j ];
                        }
                        Grad[ i * K + k ] += grad_sum;
                    }
                }

                // dL/dB[k,j] = sum_i dL/dC[i,j] * A[i,k]
                for (int k = 0; k < K; k++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        float grad_sum = 0f;
                        for (int i = 0; i < M; i++)
                        {
                            grad_sum += result.Grad[ i * N + j ] * Data[ i * K + k ];
                        }
                        other.Grad[ k * N + j ] += grad_sum;
                    }
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

            // innerSize = product of sizes after "dim"
            int innerSize = 1;
            for (int i = dim + 1; i < rank; i++)
                innerSize *= Shape[ i ];

            // blockSize = dimSize * innerSize = number of elements in one [dim, inner] block
            int blockSize = dimSize * innerSize;

            // outerSize = how many such blocks we have (product of sizes before "dim")
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

            Array.Fill( Grad, 1f );

            topo.Reverse();
            foreach (var t in topo)
                t._backward?.Invoke();
        }
        //------------------------------------------------------------------
        public void ZeroGrad()
        {
            Array.Clear( Grad, 0, Grad.Length );
        }
        //------------------------------------------------------------------
        private static bool ShapesMatch( int[] shape1, int[] shape2 )
        {
            if (shape1.Length != shape2.Length)
                return false;

            for (int i = 0; i < shape1.Length; i++)
                if (shape1[ i ] != shape2[ i ])
                    return false;

            return true;
        }
        //------------------------------------------------------------------
        private Tensor Broadcast( int[] targetShape )
        {
            if (!IsScalar)
                return this;

            var broadcasted = new Tensor( targetShape, Children.ToArray() );
            var scalarValue = Data[ 0 ];
            for (int i = 0; i < broadcasted.Size; i++)
                broadcasted.Data[ i ] = scalarValue;

            return broadcasted;
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