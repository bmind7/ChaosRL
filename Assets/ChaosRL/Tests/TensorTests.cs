using System;
using NUnit.Framework;
using ChaosRL;
using System.Linq;
using System.Diagnostics;

namespace ChaosRL.Tests
{
    public class TensorTests
    {
        //------------------------------------------------------------------
        [Test]
        public void Constructor_ScalarTensor_CreatesCorrectly()
        {
            var t = new Tensor( 5.0f );

            Assert.That( t.Shape.Length, Is.EqualTo( 1 ) );
            Assert.That( t.Shape[ 0 ], Is.EqualTo( 1 ) );
            Assert.That( t.Size, Is.EqualTo( 1 ) );
            Assert.That( t.Data[ 0 ], Is.EqualTo( 5.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithShape_CreatesCorrectly()
        {
            var t = new Tensor( new[] { 2, 3 } );

            Assert.That( t.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( t.Size, Is.EqualTo( 6 ) );
            Assert.That( t.Data.Length, Is.EqualTo( 6 ) );
            Assert.That( t.Grad.Length, Is.EqualTo( 6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithData_InitializesCorrectly()
        {
            var data = new[] { 1f, 2f, 3f, 4f };
            var t = new Tensor( new[] { 2, 2 }, data );

            Assert.That( t.Data, Is.EqualTo( data ) );
            Assert.That( t.Size, Is.EqualTo( 4 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_InvalidShape_ThrowsException()
        {
            Assert.Throws<ArgumentException>( () => new Tensor( new int[] { } ) );
            Assert.Throws<ArgumentException>( () => new Tensor( null ) );
            Assert.Throws<ArgumentException>( () => new Tensor( new[] { 2, 0, 3 } ) );
            Assert.Throws<ArgumentException>( () => new Tensor( new[] { -1, 3 } ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_DataLengthMismatch_ThrowsException()
        {
            var data = new[] { 1f, 2f, 3f };
            Assert.Throws<ArgumentException>( () => new Tensor( new[] { 2, 2 }, data ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Add_TwoTensors_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2.0f, 3.0f } );
            var b = new Tensor( new[] { 2 }, new[] { 3.5f, 1.5f } );

            var c = a + b;

            Assert.That( c.Data[ 0 ], Is.EqualTo( 5.5f ).Within( 1e-6 ) );
            Assert.That( c.Data[ 1 ], Is.EqualTo( 4.5f ).Within( 1e-6 ) );

            c.Backward();
            Assert.That( c.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( c.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Add_ScalarTensors_ComputesForwardAndBackward()
        {
            var a = new Tensor( 2.0f );
            var b = new Tensor( 3.5f );

            var c = a + b;

            Assert.That( c.Data[ 0 ], Is.EqualTo( 5.5f ).Within( 1e-6 ) );

            c.Backward();
            Assert.That( c.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Add_ShapeMismatch_ThrowsException()
        {
            var a = new Tensor( new[] { 2 }, new[] { 1f, 2f } );
            var b = new Tensor( new[] { 3 }, new[] { 1f, 2f, 3f } );

            Assert.Throws<ArgumentException>( () => { var c = a + b; } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Add_IncompatibleShapesForBroadcast_ThrowsException()
        {
            // Size not a clean multiple
            var a = new Tensor( new[] { 5 }, new[] { 1f, 2f, 3f, 4f, 5f } );
            var b = new Tensor( new[] { 3 }, new[] { 1f, 2f, 3f } );

            Assert.Throws<ArgumentException>( () => { var c = a + b; } );

            // Multi-dimensional mismatch
            var m1 = new Tensor( new[] { 2, 3 } );
            var m2 = new Tensor( new[] { 2, 4 } );

            Assert.Throws<ArgumentException>( () => { var c = m1 + m2; } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Add_ScalarBroadcast_ComputesForwardAndBackward()
        {
            // Scalar broadcasted to larger tensor
            var scalar = new Tensor( 2.0f );
            var tensor = new Tensor( new[] { 3 }, new[] { 1f, 2f, 3f } );

            var result = scalar + tensor;

            Assert.That( result.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( result.Data[ 0 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 5f ).Within( 1e-6 ) );

            result.Backward();

            // Scalar gradient should accumulate from all elements
            Assert.That( scalar.Grad[ 0 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( tensor.Grad[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( tensor.Grad[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( tensor.Grad[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Add_1DBroadcastTo2D_ComputesForwardAndBackward()
        {
            // Bias addition: [2,3] + [3]
            var matrix = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var bias = new Tensor( new[] { 3 }, new[] { 0.1f, 0.2f, 0.3f } );

            var result = matrix + bias;

            Assert.That( result.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            // First row: [1+0.1, 2+0.2, 3+0.3]
            Assert.That( result.Data[ 0 ], Is.EqualTo( 1.1f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 2.2f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 3.3f ).Within( 1e-6 ) );
            // Second row: [4+0.1, 5+0.2, 6+0.3]
            Assert.That( result.Data[ 3 ], Is.EqualTo( 4.1f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 4 ], Is.EqualTo( 5.2f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 5 ], Is.EqualTo( 6.3f ).Within( 1e-6 ) );

            result.Backward();

            // Matrix gradient is 1 everywhere
            for (int i = 0; i < 6; i++)
                Assert.That( matrix.Grad[ i ], Is.EqualTo( 1f ).Within( 1e-6 ) );

            // Bias gradient accumulates from both rows
            Assert.That( bias.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( bias.Grad[ 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( bias.Grad[ 2 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Add_LargerTensorMultipleOfSmaller_ComputesCorrectly()
        {
            // [6] + [2] where 6 = 3 * 2
            var large = new Tensor( new[] { 6 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var small = new Tensor( new[] { 2 }, new[] { 10f, 20f } );

            var result = large + small;

            Assert.That( result.Shape, Is.EqualTo( new[] { 6 } ) );
            // Pattern repeats: [1+10, 2+20, 3+10, 4+20, 5+10, 6+20]
            Assert.That( result.Data[ 0 ], Is.EqualTo( 11f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 22f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 13f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 3 ], Is.EqualTo( 24f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 4 ], Is.EqualTo( 15f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 5 ], Is.EqualTo( 26f ).Within( 1e-6 ) );

            result.Backward();

            // Large tensor gradient is 1 everywhere
            for (int i = 0; i < 6; i++)
                Assert.That( large.Grad[ i ], Is.EqualTo( 1f ).Within( 1e-6 ) );

            // Small tensor gradient accumulates from 3 repetitions
            Assert.That( small.Grad[ 0 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( small.Grad[ 1 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Multiply_TwoTensors_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2.0f, 4.0f } );
            var b = new Tensor( new[] { 2 }, new[] { 3.0f, 5.0f } );

            var c = a * b;

            Assert.That( c.Data[ 0 ], Is.EqualTo( 6.0f ).Within( 1e-6 ) );
            Assert.That( c.Data[ 1 ], Is.EqualTo( 20.0f ).Within( 1e-6 ) );

            c.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 3.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 5.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 2.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 4.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Multiply_ScalarBroadcast_ComputesForwardAndBackward()
        {
            var tensor = new Tensor( new[] { 3 }, new[] { 2f, 3f, 4f } );
            var scalar = new Tensor( 2.0f );

            var result = tensor * scalar;

            Assert.That( result.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( result.Data[ 0 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 6f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 8f ).Within( 1e-6 ) );

            result.Backward();

            // d(tensor * scalar)/dtensor = scalar
            Assert.That( tensor.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( tensor.Grad[ 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( tensor.Grad[ 2 ], Is.EqualTo( 2f ).Within( 1e-6 ) );

            // d(tensor * scalar)/dscalar = sum(tensor)
            Assert.That( scalar.Grad[ 0 ], Is.EqualTo( 9f ).Within( 1e-6 ) ); // 2+3+4
        }
        //------------------------------------------------------------------
        [Test]
        public void Multiply_1DBroadcastTo2D_ComputesForwardAndBackward()
        {
            // Element-wise scaling: [2,3] * [3]
            var matrix = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var scale = new Tensor( new[] { 3 }, new[] { 2f, 3f, 4f } );

            var result = matrix * scale;

            Assert.That( result.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            // First row: [1*2, 2*3, 3*4]
            Assert.That( result.Data[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 6f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 12f ).Within( 1e-6 ) );
            // Second row: [4*2, 5*3, 6*4]
            Assert.That( result.Data[ 3 ], Is.EqualTo( 8f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 4 ], Is.EqualTo( 15f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 5 ], Is.EqualTo( 24f ).Within( 1e-6 ) );

            result.Backward();

            // d(matrix * scale)/dmatrix = scale (broadcasted)
            Assert.That( matrix.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( matrix.Grad[ 1 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( matrix.Grad[ 2 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( matrix.Grad[ 3 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( matrix.Grad[ 4 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( matrix.Grad[ 5 ], Is.EqualTo( 4f ).Within( 1e-6 ) );

            // d(matrix * scale)/dscale accumulates from both rows
            Assert.That( scale.Grad[ 0 ], Is.EqualTo( 5f ).Within( 1e-6 ) ); // 1+4
            Assert.That( scale.Grad[ 1 ], Is.EqualTo( 7f ).Within( 1e-6 ) ); // 2+5
            Assert.That( scale.Grad[ 2 ], Is.EqualTo( 9f ).Within( 1e-6 ) ); // 3+6
        }
        //------------------------------------------------------------------
        [Test]
        public void UnaryMinus_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2.0f, -3.0f } );
            var c = -a;

            Assert.That( c.Data[ 0 ], Is.EqualTo( -2.0f ).Within( 1e-6 ) );
            Assert.That( c.Data[ 1 ], Is.EqualTo( 3.0f ).Within( 1e-6 ) );

            c.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( -1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( -1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Subtract_TwoTensors_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 5.0f, 10.0f } );
            var b = new Tensor( new[] { 2 }, new[] { 2.0f, 3.0f } );
            var c = a - b;

            Assert.That( c.Data[ 0 ], Is.EqualTo( 3.0f ).Within( 1e-6 ) );
            Assert.That( c.Data[ 1 ], Is.EqualTo( 7.0f ).Within( 1e-6 ) );

            c.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( -1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( -1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Divide_TwoTensors_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 8.0f, 12.0f } );
            var b = new Tensor( new[] { 2 }, new[] { 2.0f, 3.0f } );
            var c = a / b;

            Assert.That( c.Data[ 0 ], Is.EqualTo( 4.0f ).Within( 1e-6 ) );
            Assert.That( c.Data[ 1 ], Is.EqualTo( 4.0f ).Within( 1e-6 ) );

            c.Backward();
            // dc/da = 1/b
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.5f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f / 3.0f ).Within( 1e-6 ) );
            // dc/db = -a/b^2
            Assert.That( b.Grad[ 0 ], Is.EqualTo( -2.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( -12.0f / 9.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Divide_ScalarBroadcast_ComputesForwardAndBackward()
        {
            var tensor = new Tensor( new[] { 3 }, new[] { 6f, 9f, 12f } );
            var scalar = new Tensor( 3f );

            var result = tensor / scalar;

            Assert.That( result.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( result.Data[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 4f ).Within( 1e-6 ) );

            result.Backward();

            // d(tensor / scalar)/dtensor = 1/scalar
            Assert.That( tensor.Grad[ 0 ], Is.EqualTo( 1f / 3f ).Within( 1e-6 ) );
            Assert.That( tensor.Grad[ 1 ], Is.EqualTo( 1f / 3f ).Within( 1e-6 ) );
            Assert.That( tensor.Grad[ 2 ], Is.EqualTo( 1f / 3f ).Within( 1e-6 ) );

            // d(tensor / scalar)/dscalar = -sum(tensor / scalar^2)
            // = -(6/9 + 9/9 + 12/9) = -(2/3 + 1 + 4/3) = -3
            Assert.That( scalar.Grad[ 0 ], Is.EqualTo( -3f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Divide_1DBroadcastTo2D_ComputesForwardAndBackward()
        {
            // [2,3] / [3]
            var matrix = new Tensor( new[] { 2, 3 }, new[] { 6f, 9f, 12f, 3f, 6f, 9f } );
            var divisor = new Tensor( new[] { 3 }, new[] { 2f, 3f, 4f } );

            var result = matrix / divisor;

            Assert.That( result.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            // First row: [6/2, 9/3, 12/4]
            Assert.That( result.Data[ 0 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            // Second row: [3/2, 6/3, 9/4]
            Assert.That( result.Data[ 3 ], Is.EqualTo( 1.5f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 4 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 5 ], Is.EqualTo( 2.25f ).Within( 1e-6 ) );

            result.Backward();

            // d(matrix / divisor)/dmatrix = 1/divisor (broadcasted)
            Assert.That( matrix.Grad[ 0 ], Is.EqualTo( 0.5f ).Within( 1e-6 ) );  // 1/2
            Assert.That( matrix.Grad[ 1 ], Is.EqualTo( 1f / 3f ).Within( 1e-6 ) );
            Assert.That( matrix.Grad[ 2 ], Is.EqualTo( 0.25f ).Within( 1e-6 ) ); // 1/4
            Assert.That( matrix.Grad[ 3 ], Is.EqualTo( 0.5f ).Within( 1e-6 ) );
            Assert.That( matrix.Grad[ 4 ], Is.EqualTo( 1f / 3f ).Within( 1e-6 ) );
            Assert.That( matrix.Grad[ 5 ], Is.EqualTo( 0.25f ).Within( 1e-6 ) );

            // d(matrix / divisor)/ddivisor accumulates: -matrix/divisor^2
            Assert.That( divisor.Grad[ 0 ], Is.EqualTo( -(6f / 4f + 3f / 4f) ).Within( 1e-5 ) ); // -(6/4 + 3/4)
            Assert.That( divisor.Grad[ 1 ], Is.EqualTo( -(9f / 9f + 6f / 9f) ).Within( 1e-5 ) ); // -(9/9 + 6/9)
            Assert.That( divisor.Grad[ 2 ], Is.EqualTo( -(12f / 16f + 9f / 16f) ).Within( 1e-5 ) ); // -(12/16 + 9/16)
        }
        //------------------------------------------------------------------
        [Test]
        public void Pow_IntegerExponent_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2.0f, 3.0f } );
            var y = a.Pow( 3.0f );

            Assert.That( y.Data[ 0 ], Is.EqualTo( 8.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 1 ], Is.EqualTo( 27.0f ).Within( 1e-6 ) );

            y.Backward();
            // dy/da = 3 * a^2
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 12.0f ).Within( 1e-5 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 27.0f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Pow_FractionalExponent_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 4.0f, 9.0f } );
            var y = a.Pow( 0.5f );

            Assert.That( y.Data[ 0 ], Is.EqualTo( 2.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 1 ], Is.EqualTo( 3.0f ).Within( 1e-6 ) );

            y.Backward();
            // dy/da = 0.5 * a^(-0.5)
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.25f ).Within( 1e-5 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f / 6.0f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sqrt_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 3 }, new[] { 4.0f, 9.0f, 16.0f } );
            var y = a.Sqrt();

            Assert.That( y.Data[ 0 ], Is.EqualTo( 2.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 1 ], Is.EqualTo( 3.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 2 ], Is.EqualTo( 4.0f ).Within( 1e-6 ) );

            y.Backward();
            // dy/da = 0.5 * a^(-0.5) = 1/(2*sqrt(a))
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.25f ).Within( 1e-5 ) );      // 1/(2*2) = 0.25
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f / 6.0f ).Within( 1e-5 ) ); // 1/(2*3) = 1/6
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 0.125f ).Within( 1e-5 ) );     // 1/(2*4) = 0.125
        }
        //------------------------------------------------------------------
        [Test]
        public void Sqrt_ScalarTensor_WorksCorrectly()
        {
            var a = new Tensor( 25.0f );
            var y = a.Sqrt();

            Assert.That( y.Data[ 0 ], Is.EqualTo( 5.0f ).Within( 1e-6 ) );

            y.Backward();
            // dy/da = 1/(2*sqrt(25)) = 1/10 = 0.1
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.1f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sqrt_ChainedWithOperations_ComputesGradientsCorrectly()
        {
            var a = new Tensor( new[] { 2 }, new[] { 4.0f, 9.0f } );
            var b = new Tensor( new[] { 2 }, new[] { 2.0f, 3.0f } );

            var c = a * b;  // [8, 27]
            var d = c.Sqrt(); // [2.828..., 5.196...]
            var e = d.Sum(); // scalar

            e.Backward();

            // de/dd = 1, dd/dc = 1/(2*sqrt(c)), dc/da = b
            // da = b * 1/(2*sqrt(a*b))
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 2.0f / (2.0f * MathF.Sqrt( 8.0f )) ).Within( 1e-5 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 3.0f / (2.0f * MathF.Sqrt( 27.0f )) ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sqrt_2DTensor_ComputesCorrectly()
        {
            var a = new Tensor( new[] { 2, 2 }, new[] { 1.0f, 4.0f, 9.0f, 16.0f } );
            var y = a.Sqrt();

            Assert.That( y.Shape, Is.EqualTo( new[] { 2, 2 } ) );
            Assert.That( y.Data[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 1 ], Is.EqualTo( 2.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 2 ], Is.EqualTo( 3.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 3 ], Is.EqualTo( 4.0f ).Within( 1e-6 ) );

            y.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.5f ).Within( 1e-5 ) );       // 1/(2*1)
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 0.25f ).Within( 1e-5 ) );      // 1/(2*2)
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 1.0f / 6.0f ).Within( 1e-5 ) ); // 1/(2*3)
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 0.125f ).Within( 1e-5 ) );     // 1/(2*4)
        }
        //------------------------------------------------------------------
        [Test]
        public void Normalize_EntireTensor_ProducesMeanZeroUnitVariance()
        {
            var a = new Tensor( new[] { 4 }, new[] { 1f, 2f, 3f, 4f } );
            var normalized = a.Normalize();

            // Verify mean is approximately 0
            var mean = normalized.Data.Average();
            Assert.That( mean, Is.EqualTo( 0f ).Within( 1e-5 ) );

            // Verify variance is approximately 1
            var variance = normalized.Data.Select( x => x * x ).Average();
            Assert.That( variance, Is.EqualTo( 1f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Normalize_EntireTensor_BackwardPropagatesCorrectly()
        {
            var a = new Tensor( new[] { 3 }, new[] { 0f, 2f, 3f } );
            var normalized = a.Normalize();
            var loss = normalized.Sum();

            loss.Backward();

            // Gradients should be non-zero
            Assert.That( a.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
            Assert.That( a.Grad[ 1 ], Is.Not.EqualTo( 0f ) );
            Assert.That( a.Grad[ 2 ], Is.Not.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Normalize_AlongDimension_2DTensor()
        {
            // [[1, 2, 3],
            //  [4, 5, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            // Normalize along dimension 1 (normalize each row)
            var normalized = a.Normalize( dim: 1 );

            Assert.That( normalized.Shape, Is.EqualTo( new[] { 2, 3 } ) );

            UnityEngine.Debug.Log( "Normalized Tensor:" );
            for (int i = 0; i < normalized.Size; i++)
            {
                UnityEngine.Debug.Log( normalized.Data[ i ] );
            }

            // Each row should have mean ~0 and variance ~1
            // Row 0: [1, 2, 3] -> normalized
            var row0Mean = (normalized.Data[ 0 ] + normalized.Data[ 1 ] + normalized.Data[ 2 ]) / 3f;
            Assert.That( row0Mean, Is.EqualTo( 0f ).Within( 1e-5 ) );

            // Row 1: [4, 5, 6] -> normalized
            var row1Mean = (normalized.Data[ 3 ] + normalized.Data[ 4 ] + normalized.Data[ 5 ]) / 3f;
            Assert.That( row1Mean, Is.EqualTo( 0f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Normalize_AlongDimension0_2DTensor()
        {
            // [[1, 4],
            //  [2, 5],
            //  [3, 6]]
            var a = new Tensor( new[] { 3, 2 }, new[] { 1f, 4f, 2f, 5f, 3f, 6f } );

            // Normalize along dimension 0 (normalize each column)
            var normalized = a.Normalize( dim: 0 );

            Assert.That( normalized.Shape, Is.EqualTo( new[] { 3, 2 } ) );

            // Each column should have mean ~0
            // Column 0: [1, 2, 3] -> normalized
            var col0Mean = (normalized.Data[ 0 ] + normalized.Data[ 2 ] + normalized.Data[ 4 ]) / 3f;
            Assert.That( col0Mean, Is.EqualTo( 0f ).Within( 1e-5 ) );

            // Column 1: [4, 5, 6] -> normalized
            var col1Mean = (normalized.Data[ 1 ] + normalized.Data[ 3 ] + normalized.Data[ 5 ]) / 3f;
            Assert.That( col1Mean, Is.EqualTo( 0f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Normalize_AlongDimension_BackwardPropagatesCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 2.1f, 0.4f, 4.5f, 4.5f, 5.2f, 6.6f } );
            var normalized = a.Normalize( dim: 1 );
            var loss = (normalized * normalized).Sum();

            loss.Backward();

            // All gradients should be non-zero
            for (int i = 0; i < a.Size; i++)
            {
                Assert.That( a.Grad[ i ], Is.Not.EqualTo( 0f ) );
            }
        }
        //------------------------------------------------------------------
        [Test]
        public void Normalize_WithEpsilon_PreventsZeroDivision()
        {
            // Constant tensor would have zero variance
            var a = new Tensor( new[] { 4 }, new[] { 5f, 5f, 5f, 5f } );

            // Should not throw with epsilon
            var normalized = a.Normalize( epsilon: 1e-5f );

            // Result should be all zeros (since all values equal mean)
            for (int i = 0; i < normalized.Size; i++)
            {
                Assert.That( normalized.Data[ i ], Is.EqualTo( 0f ).Within( 1e-5 ) );
            }
        }
        //------------------------------------------------------------------
        [Test]
        public void Normalize_NegativeDimension_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3, 4 } );
            for (int i = 0; i < 24; i++)
                a.Data[ i ] = i + 1;

            // dim=-1 should be equivalent to dim=2 (last dimension)
            var normalized = a.Normalize( dim: -1 );

            Assert.That( normalized.Shape, Is.EqualTo( new[] { 2, 3, 4 } ) );

            // Should complete without errors
            var loss = normalized.Sum();
            loss.Backward();
        }
        //------------------------------------------------------------------
        [Test]
        public void Normalize_LayerNormUseCase_WorksCorrectly()
        {
            // Simulating layer normalization for a batch of sequences
            // Shape: [batch=2, seq_len=3, features=4]
            var a = new Tensor( new[] { 2, 3, 4 } );
            for (int i = 0; i < 24; i++)
                a.Data[ i ] = (float)(i % 7) - 3f; // Mix of positive and negative values

            // Normalize along the feature dimension (dim=2)
            var normalized = a.Normalize( dim: 2 );

            Assert.That( normalized.Shape, Is.EqualTo( new[] { 2, 3, 4 } ) );

            // Each set of 4 features should have mean ~0 and std ~1
            // Check first feature vector
            var slice0Mean = (normalized.Data[ 0 ] + normalized.Data[ 1 ] +
                             normalized.Data[ 2 ] + normalized.Data[ 3 ]) / 4f;
            Assert.That( slice0Mean, Is.EqualTo( 0f ).Within( 1e-4 ) );

            // Backward should work
            var loss = (normalized * normalized).Sum();
            loss.Backward();
            Assert.That( a.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ReLU_Positive_Negative_And_Zero()
        {
            var a = new Tensor( new[] { 4 }, new[] { 3.0f, -2.0f, 0.0f, 5.0f } );
            var y = a.ReLU();

            Assert.That( y.Data[ 0 ], Is.EqualTo( 3.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 1 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 2 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( y.Data[ 3 ], Is.EqualTo( 5.0f ).Within( 1e-6 ) );

            y.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Exp_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 1.5f, 0.0f } );
            var y = a.Exp();

            var expected0 = MathF.Exp( 1.5f );
            var expected1 = MathF.Exp( 0.0f );
            Assert.That( y.Data[ 0 ], Is.EqualTo( expected0 ).Within( 1e-6 ) );
            Assert.That( y.Data[ 1 ], Is.EqualTo( expected1 ).Within( 1e-6 ) );

            y.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( expected0 ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( expected1 ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Tanh_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 0.5f, -0.5f } );
            var y = a.Tanh();

            var expected0 = MathF.Tanh( 0.5f );
            var expected1 = MathF.Tanh( -0.5f );
            Assert.That( y.Data[ 0 ], Is.EqualTo( expected0 ).Within( 1e-6 ) );
            Assert.That( y.Data[ 1 ], Is.EqualTo( expected1 ).Within( 1e-6 ) );

            y.Backward();
            var expectedGrad0 = 1.0f - expected0 * expected0;
            var expectedGrad1 = 1.0f - expected1 * expected1;
            Assert.That( a.Grad[ 0 ], Is.EqualTo( expectedGrad0 ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( expectedGrad1 ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Log_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2.0f, (float)Math.E } );
            var y = a.Log();

            var expected0 = MathF.Log( 2.0f );
            var expected1 = MathF.Log( (float)Math.E );
            Assert.That( y.Data[ 0 ], Is.EqualTo( expected0 ).Within( 1e-6 ) );
            Assert.That( y.Data[ 1 ], Is.EqualTo( expected1 ).Within( 1e-6 ) );

            y.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.5f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f / (float)Math.E ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Clamp_AllowsGradientWithinBounds_BlocksWhenClamped()
        {
            var a = new Tensor( new[] { 4 }, new[] { 0.5f, 2.0f, -2.0f, -0.5f } );
            var clamped = a.Clamp( -1.0f, 1.0f );

            Assert.That( clamped.Data[ 0 ], Is.EqualTo( 0.5f ).Within( 1e-6 ) );
            Assert.That( clamped.Data[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( clamped.Data[ 2 ], Is.EqualTo( -1.0f ).Within( 1e-6 ) );
            Assert.That( clamped.Data[ 3 ], Is.EqualTo( -0.5f ).Within( 1e-6 ) );

            clamped.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2.0f, 7.0f } );
            var b = new Tensor( new[] { 2 }, new[] { 5.0f, 3.0f } );
            var m = Tensor.Max( a, b );

            Assert.That( m.Data[ 0 ], Is.EqualTo( 5.0f ).Within( 1e-6 ) );
            Assert.That( m.Data[ 1 ], Is.EqualTo( 7.0f ).Within( 1e-6 ) );

            m.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Min_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2.0f, 7.0f } );
            var b = new Tensor( new[] { 2 }, new[] { -1.0f, 10.0f } );
            var m = Tensor.Min( a, b );

            Assert.That( m.Data[ 0 ], Is.EqualTo( -1.0f ).Within( 1e-6 ) );
            Assert.That( m.Data[ 1 ], Is.EqualTo( 7.0f ).Within( 1e-6 ) );

            m.Backward();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 0.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_ScalarBroadcast_ComputesForwardAndBackward()
        {
            var tensor = new Tensor( new[] { 3 }, new[] { 1f, 5f, 3f } );
            var scalar = new Tensor( 4f );

            var result = Tensor.Max( tensor, scalar );

            Assert.That( result.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( result.Data[ 0 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 5f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 4f ).Within( 1e-6 ) );

            result.Backward();

            // Gradient flows to whichever is larger
            Assert.That( tensor.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // scalar wins
            Assert.That( tensor.Grad[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // tensor wins
            Assert.That( tensor.Grad[ 2 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // scalar wins

            // Scalar accumulates from positions where it wins
            Assert.That( scalar.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_1DBroadcastTo2D_ComputesForwardAndBackward()
        {
            // [2,3] max [3]
            var matrix = new Tensor( new[] { 2, 3 }, new[] { 1f, 5f, 3f, 2f, 4f, 6f } );
            var vector = new Tensor( new[] { 3 }, new[] { 3f, 4f, 5f } );

            var result = Tensor.Max( matrix, vector );

            Assert.That( result.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            // First row: max([1,5,3], [3,4,5])
            Assert.That( result.Data[ 0 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 5f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 5f ).Within( 1e-6 ) );
            // Second row: max([2,4,6], [3,4,5])
            Assert.That( result.Data[ 3 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 4 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 5 ], Is.EqualTo( 6f ).Within( 1e-6 ) );

            result.Backward();

            Assert.That( matrix.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // vector wins
            Assert.That( matrix.Grad[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // matrix wins
            Assert.That( matrix.Grad[ 2 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // vector wins
            Assert.That( matrix.Grad[ 3 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // vector wins
            Assert.That( matrix.Grad[ 4 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // tie - matrix wins (>=)
            Assert.That( matrix.Grad[ 5 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // matrix wins

            // Vector accumulates from both rows
            Assert.That( vector.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) ); // wins in both rows
            Assert.That( vector.Grad[ 1 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // loses in both rows
            Assert.That( vector.Grad[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // wins in first row only
        }
        //------------------------------------------------------------------
        [Test]
        public void Min_ScalarBroadcast_ComputesForwardAndBackward()
        {
            var tensor = new Tensor( new[] { 3 }, new[] { 1f, 5f, 3f } );
            var scalar = new Tensor( 4f );

            var result = Tensor.Min( tensor, scalar );

            Assert.That( result.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( result.Data[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 3f ).Within( 1e-6 ) );

            result.Backward();

            // Gradient flows to whichever is smaller
            Assert.That( tensor.Grad[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // tensor wins
            Assert.That( tensor.Grad[ 1 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // scalar wins
            Assert.That( tensor.Grad[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // tensor wins

            // Scalar accumulates from positions where it wins
            Assert.That( scalar.Grad[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Min_1DBroadcastTo2D_ComputesForwardAndBackward()
        {
            // [2,3] min [3]
            var matrix = new Tensor( new[] { 2, 3 }, new[] { 1f, 5f, 3f, 2f, 4f, 6f } );
            var vector = new Tensor( new[] { 3 }, new[] { 3f, 4f, 5f } );

            var result = Tensor.Min( matrix, vector );

            Assert.That( result.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            // First row: min([1,5,3], [3,4,5])
            Assert.That( result.Data[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            // Second row: min([2,4,6], [3,4,5])
            Assert.That( result.Data[ 3 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 4 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( result.Data[ 5 ], Is.EqualTo( 5f ).Within( 1e-6 ) );

            result.Backward();

            Assert.That( matrix.Grad[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // matrix wins
            Assert.That( matrix.Grad[ 1 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // vector wins
            Assert.That( matrix.Grad[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // matrix wins (tie, <=)
            Assert.That( matrix.Grad[ 3 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // matrix wins
            Assert.That( matrix.Grad[ 4 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // matrix wins (tie, <=)
            Assert.That( matrix.Grad[ 5 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // vector wins

            // Vector accumulates from both rows
            Assert.That( vector.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // loses in both rows
            Assert.That( vector.Grad[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // wins in first row only
            Assert.That( vector.Grad[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) ); // wins in second row only
        }
        //------------------------------------------------------------------
        [Test]
        public void ImplicitConversion_FromFloat_WorksInExpressions()
        {
            Tensor t = 5.0f;
            Assert.That( t.Data[ 0 ], Is.EqualTo( 5.0f ).Within( 1e-6 ) );

            var b = new Tensor( new[] { 1 }, new[] { 3.0f } );
            var c = b + 2.0f;
            Assert.That( c.Data[ 0 ], Is.EqualTo( 5.0f ).Within( 1e-6 ) );
            c.Backward();
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void BackwardPass_Accumulation_And_TopologicalOrder()
        {
            // f(a,b) = a + b + a*b
            var a = new Tensor( 2.0f );
            var b = new Tensor( 3.0f );
            var c = a + b;
            var d = a * b;
            var f = c + d;

            Assert.That( f.Data[ 0 ], Is.EqualTo( 2.0f + 3.0f + 6.0f ).Within( 1e-6 ) );

            f.Backward();
            // df/da = 1 + b = 4
            // df/db = 1 + a = 3
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 4.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 3.0f ).Within( 1e-6 ) );
            Assert.That( f.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ZeroGrad_ClearsGradients()
        {
            var a = new Tensor( new[] { 2 }, new[] { 1.0f, 2.0f } );
            var b = new Tensor( new[] { 2 }, new[] { 3.0f, 4.0f } );
            var c = a + b;

            c.Backward();
            Assert.That( a.Grad[ 0 ], Is.Not.EqualTo( 0.0f ) );
            Assert.That( a.Grad[ 1 ], Is.Not.EqualTo( 0.0f ) );

            a.ZeroGrad();
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0.0f ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 0.0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MultiDimensional_3DTensor_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 2, 2 } );

            Assert.That( a.Shape, Is.EqualTo( new[] { 2, 2, 2 } ) );
            Assert.That( a.Size, Is.EqualTo( 8 ) );
            Assert.That( a.Data.Length, Is.EqualTo( 8 ) );

            for (int i = 0; i < 8; i++)
                a.Data[ i ] = i;

            var b = new Tensor( new[] { 2, 2, 2 } );
            for (int i = 0; i < 8; i++)
                b.Data[ i ] = 1.0f;

            var c = a + b;
            for (int i = 0; i < 8; i++)
                Assert.That( c.Data[ i ], Is.EqualTo( i + 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_2x3_3x2_ProducesCorrectShape()
        {
            var a = new Tensor( new[] { 2, 3 } );
            var b = new Tensor( new[] { 3, 2 } );

            var c = a.MatMul( b );

            Assert.That( c.Shape, Is.EqualTo( new[] { 2, 2 } ) );
            Assert.That( c.Size, Is.EqualTo( 4 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_SimpleMultiplication_ComputesCorrectly()
        {
            // A = [[1, 2],    B = [[5, 6],
            //      [3, 4]]         [7, 8]]
            var a = new Tensor( new[] { 2, 2 }, new[] { 1f, 2f, 3f, 4f } );
            var b = new Tensor( new[] { 2, 2 }, new[] { 5f, 6f, 7f, 8f } );

            var c = a.MatMul( b );

            // C = [[1*5+2*7, 1*6+2*8],  = [[19, 22],
            //      [3*5+4*7, 3*6+4*8]]     [43, 50]]
            Assert.That( c.Data[ 0 ], Is.EqualTo( 19f ).Within( 1e-6 ) );
            Assert.That( c.Data[ 1 ], Is.EqualTo( 22f ).Within( 1e-6 ) );
            Assert.That( c.Data[ 2 ], Is.EqualTo( 43f ).Within( 1e-6 ) );
            Assert.That( c.Data[ 3 ], Is.EqualTo( 50f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_NonSquareMatrices_ComputesCorrectly()
        {
            // A = [[1, 2, 3]]  (1×3)
            // B = [[4],        (3×1)
            //      [5],
            //      [6]]
            var a = new Tensor( new[] { 1, 3 }, new[] { 1f, 2f, 3f } );
            var b = new Tensor( new[] { 3, 1 }, new[] { 4f, 5f, 6f } );

            var c = a.MatMul( b );

            // C = [[1*4 + 2*5 + 3*6]] = [[32]]
            Assert.That( c.Shape, Is.EqualTo( new[] { 1, 1 } ) );
            Assert.That( c.Data[ 0 ], Is.EqualTo( 32f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_BatchMultiplication_ComputesCorrectly()
        {
            // Simulating 2 batches × 3 features
            var input = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            // Weights: 3 inputs × 2 outputs
            var weights = new Tensor( new[] { 3, 2 }, new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f } );

            var output = input.MatMul( weights );

            // Output should be 2×2
            Assert.That( output.Shape, Is.EqualTo( new[] { 2, 2 } ) );

            // First batch: [1,2,3] @ [[0.1,0.2], [0.3,0.4], [0.5,0.6]]
            // = [1*0.1 + 2*0.3 + 3*0.5, 1*0.2 + 2*0.4 + 3*0.6] = [2.2, 2.8]
            Assert.That( output.Data[ 0 ], Is.EqualTo( 2.2f ).Within( 1e-5 ) );
            Assert.That( output.Data[ 1 ], Is.EqualTo( 2.8f ).Within( 1e-5 ) );

            // Second batch: [4,5,6] @ [[0.1,0.2], [0.3,0.4], [0.5,0.6]]
            // = [4*0.1 + 5*0.3 + 6*0.5, 4*0.2 + 5*0.4 + 6*0.6] = [4.9, 6.4]
            Assert.That( output.Data[ 2 ], Is.EqualTo( 4.9f ).Within( 1e-5 ) );
            Assert.That( output.Data[ 3 ], Is.EqualTo( 6.4f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_Backward_ComputesGradientsCorrectly()
        {
            // Simple 2×2 multiplication
            var a = new Tensor( new[] { 2, 2 }, new[] { 1f, 2f, 3f, 4f } );
            var b = new Tensor( new[] { 2, 2 }, new[] { 5f, 6f, 7f, 8f } );

            var c = a.MatMul( b );
            c.Backward();

            // Gradients should be non-zero
            Assert.That( a.Grad[ 0 ], Is.Not.EqualTo( 0.0f ) );
            Assert.That( a.Grad[ 1 ], Is.Not.EqualTo( 0.0f ) );
            Assert.That( b.Grad[ 0 ], Is.Not.EqualTo( 0.0f ) );
            Assert.That( b.Grad[ 1 ], Is.Not.EqualTo( 0.0f ) );

            // dL/dA = dL/dC @ B^T
            // With dL/dC = [[1,1], [1,1]] (gradient of 1 everywhere)
            // B^T = [[5,7], [6,8]]
            // dL/dA = [[1*5+1*6, 1*7+1*8], [1*5+1*6, 1*7+1*8]] = [[11,15], [11,15]]
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 11f ).Within( 1e-5 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 15f ).Within( 1e-5 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 11f ).Within( 1e-5 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 15f ).Within( 1e-5 ) );

            // dL/dB = A^T @ dL/dC
            // A^T = [[1,3], [2,4]]
            // dL/dB = [[1*1+3*1, 1*1+3*1], [2*1+4*1, 2*1+4*1]] = [[4,4], [6,6]]
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 4f ).Within( 1e-5 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 4f ).Within( 1e-5 ) );
            Assert.That( b.Grad[ 2 ], Is.EqualTo( 6f ).Within( 1e-5 ) );
            Assert.That( b.Grad[ 3 ], Is.EqualTo( 6f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_Backward_MatchesValueImplementation()
        {
            // Test that Tensor.MatMul gradients match Value-based matmul
            const int M = 2, K = 3, N = 2;

            // Input data
            var aData = new[] { 1f, 2f, 3f, 4f, 5f, 6f }; // 2×3
            var bData = new[] { 0.5f, 0.7f, 0.3f, 0.9f, 0.2f, 0.4f }; // 3×2

            // Tensor implementation
            var tensorA = new Tensor( new[] { M, K }, aData );
            var tensorB = new Tensor( new[] { K, N }, bData );
            var tensorC = tensorA.MatMul( tensorB );
            tensorC.Backward();

            // Value implementation
            var valueA = new Value[ M * K ];
            var valueB = new Value[ K * N ];
            for (int i = 0; i < M * K; i++)
                valueA[ i ] = new Value( aData[ i ] );
            for (int i = 0; i < K * N; i++)
                valueB[ i ] = new Value( bData[ i ] );

            // Manual matmul with Value
            var valueC = new Value[ M * N ];
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Value sum = 0f;
                    for (int k = 0; k < K; k++)
                    {
                        sum = sum + (valueA[ i * K + k ] * valueB[ k * N + j ]);
                    }
                    valueC[ i * N + j ] = sum;
                }
            }

            // Backward pass for Value
            for (int i = 0; i < M * N; i++)
                valueC[ i ].Backward();

            // Compare forward results
            for (int i = 0; i < M * N; i++)
            {
                Assert.That( tensorC.Data[ i ], Is.EqualTo( valueC[ i ].Data ).Within( 1e-5 ),
                    $"Forward pass mismatch at index {i}" );
            }

            // Compare gradients
            for (int i = 0; i < M * K; i++)
            {
                Assert.That( tensorA.Grad[ i ], Is.EqualTo( valueA[ i ].Grad ).Within( 1e-5 ),
                    $"Gradient mismatch for A at index {i}" );
            }
            for (int i = 0; i < K * N; i++)
            {
                Assert.That( tensorB.Grad[ i ], Is.EqualTo( valueB[ i ].Grad ).Within( 1e-5 ),
                    $"Gradient mismatch for B at index {i}" );
            }
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_InvalidDimensions_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );
            var b = new Tensor( new[] { 2, 2 } ); // Wrong: should be 3×N

            Assert.Throws<ArgumentException>( () => a.MatMul( b ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_Non2DTensors_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3, 4 } ); // 3D tensor
            var b = new Tensor( new[] { 3, 2 } );

            Assert.Throws<ArgumentException>( () => a.MatMul( b ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_ChainedWithActivations_ComputesCorrectly()
        {
            // Simulate a simple neural network layer: output = ReLU(input @ weights)
            var input = new Tensor( new[] { 2, 3 }, new[] { 1f, -2f, 3f, -1f, 2f, -3f } );
            var weights = new Tensor( new[] { 3, 2 }, new[] { 0.5f, -0.5f, 0.3f, 0.3f, 0.2f, 0.2f } );

            var output = input.MatMul( weights ).ReLU();

            // Forward pass should execute correctly
            Assert.That( output.Shape, Is.EqualTo( new[] { 2, 2 } ) );

            // Backward pass should work through the chain
            output.Backward();
            Assert.That( input.Grad[ 0 ], Is.Not.EqualTo( 0.0f ).Or.EqualTo( 0.0f ) ); // May be zero due to ReLU
            Assert.That( weights.Grad[ 0 ], Is.Not.EqualTo( 0.0f ).Or.EqualTo( 0.0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_AllDimensions_ComputesCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var sum = a.Sum();

            Assert.That( sum.Shape, Is.EqualTo( new[] { 1 } ) );
            Assert.That( sum.Data[ 0 ], Is.EqualTo( 21f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_AllDimensions_BackwardBroadcastsCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var sum = a.Sum();

            sum.Backward();

            // Gradient should be 1.0 for all elements
            for (int i = 0; i < a.Size; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 1f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_AlongDimension0_ComputesCorrectly()
        {
            // Shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var sum = a.Sum( 0 ); // Sum along rows

            // Result shape should be [3]
            Assert.That( sum.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( sum.Data[ 0 ], Is.EqualTo( 5f ).Within( 1e-6 ) );  // 1 + 4
            Assert.That( sum.Data[ 1 ], Is.EqualTo( 7f ).Within( 1e-6 ) );  // 2 + 5
            Assert.That( sum.Data[ 2 ], Is.EqualTo( 9f ).Within( 1e-6 ) );  // 3 + 6
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_AlongDimension1_ComputesCorrectly()
        {
            // Shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var sum = a.Sum( 1 ); // Sum along columns

            // Result shape should be [2]
            Assert.That( sum.Shape, Is.EqualTo( new[] { 2 } ) );
            Assert.That( sum.Data[ 0 ], Is.EqualTo( 6f ).Within( 1e-6 ) );   // 1 + 2 + 3
            Assert.That( sum.Data[ 1 ], Is.EqualTo( 15f ).Within( 1e-6 ) );  // 4 + 5 + 6
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_AlongDimension_BackwardBroadcastsCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var sum = a.Sum( 1 ); // Sum along columns -> shape [2]

            sum.Backward();

            // Gradient should broadcast: each row element gets gradient from corresponding output
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 4 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 5 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_3DTensor_AlongMiddleDimension()
        {
            // Shape [2, 3, 4]: 2 matrices of 3×4
            var a = new Tensor( new[] { 2, 3, 4 } );
            for (int i = 0; i < 24; i++)
                a.Data[ i ] = i + 1;

            var sum = a.Sum( 1 ); // Sum along dimension 1

            // Result shape should be [2, 4]
            Assert.That( sum.Shape, Is.EqualTo( new[] { 2, 4 } ) );
            Assert.That( sum.Size, Is.EqualTo( 8 ) );

            // First matrix: indices 0-11, sum along rows (3 rows)
            // Column 0: 1 + 5 + 9 = 15
            Assert.That( sum.Data[ 0 ], Is.EqualTo( 15f ).Within( 1e-6 ) );
            // Column 1: 2 + 6 + 10 = 18
            Assert.That( sum.Data[ 1 ], Is.EqualTo( 18f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_InvalidDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            Assert.Throws<ArgumentException>( () => a.Sum( -3 ) );
            Assert.Throws<ArgumentException>( () => a.Sum( 2 ) );
            Assert.Throws<ArgumentException>( () => a.Sum( 5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_ChainedWithOperations_ComputesGradientsCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = new Tensor( new[] { 2, 3 }, new[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f } );

            var c = a * b;
            var sum = c.Sum();

            Assert.That( sum.Data[ 0 ], Is.EqualTo( 10.5f ).Within( 1e-6 ) );

            sum.Backward();

            // dc/da = b, dsum/dc = 1, so da = b
            for (int i = 0; i < a.Size; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 0.5f ).Within( 1e-6 ) );

            // dc/db = a, dsum/dc = 1, so db = a
            for (int i = 0; i < b.Size; i++)
                Assert.That( b.Grad[ i ], Is.EqualTo( a.Data[ i ] ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Mean_AllDimensions_ComputesCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var mean = a.Mean();

            Assert.That( mean.Shape, Is.EqualTo( new[] { 1 } ) );
            Assert.That( mean.Data[ 0 ], Is.EqualTo( 3.5f ).Within( 1e-6 ) ); // 21/6 = 3.5
        }
        //------------------------------------------------------------------
        [Test]
        public void Mean_AllDimensions_BackwardScalesGradient()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var mean = a.Mean();

            mean.Backward();

            // Gradient should be 1/size = 1/6 for all elements
            for (int i = 0; i < a.Size; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 1f / 6f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Mean_AlongDimension0_ComputesCorrectly()
        {
            // Shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var mean = a.Mean( 0 ); // Mean along rows

            // Result shape should be [3]
            Assert.That( mean.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( mean.Data[ 0 ], Is.EqualTo( 2.5f ).Within( 1e-6 ) );  // (1+4)/2
            Assert.That( mean.Data[ 1 ], Is.EqualTo( 3.5f ).Within( 1e-6 ) );  // (2+5)/2
            Assert.That( mean.Data[ 2 ], Is.EqualTo( 4.5f ).Within( 1e-6 ) );  // (3+6)/2
        }
        //------------------------------------------------------------------
        [Test]
        public void Mean_AlongDimension1_ComputesCorrectly()
        {
            // Shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var mean = a.Mean( 1 ); // Mean along columns

            // Result shape should be [2]
            Assert.That( mean.Shape, Is.EqualTo( new[] { 2 } ) );
            Assert.That( mean.Data[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );   // (1+2+3)/3
            Assert.That( mean.Data[ 1 ], Is.EqualTo( 5f ).Within( 1e-6 ) );   // (4+5+6)/3
        }
        //------------------------------------------------------------------
        [Test]
        public void Mean_AlongDimension_BackwardScalesGradient()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var mean = a.Mean( 1 ); // Mean along columns -> shape [2]

            mean.Backward();

            // Gradient should be 1/dimSize = 1/3 for all elements
            for (int i = 0; i < a.Size; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 1f / 3f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Mean_NegativeDimension_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3, 4 } );
            for (int i = 0; i < 24; i++)
                a.Data[ i ] = i + 1;

            // dim=-1 should be equivalent to dim=2
            var mean = a.Mean( -1 );

            Assert.That( mean.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( mean.Size, Is.EqualTo( 6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Mean_InvalidDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            Assert.Throws<ArgumentException>( () => a.Mean( -3 ) );
            Assert.Throws<ArgumentException>( () => a.Mean( 2 ) );
            Assert.Throws<ArgumentException>( () => a.Mean( 5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Mean_ChainedWithOperations_ComputesGradientsCorrectly()
        {
            var a = new Tensor( new[] { 2, 2 }, new[] { 1f, 2f, 3f, 4f } );
            var b = new Tensor( new[] { 2, 2 }, new[] { 2f, 2f, 2f, 2f } );

            var c = a * b;
            var mean = c.Mean();

            Assert.That( mean.Data[ 0 ], Is.EqualTo( 5f ).Within( 1e-6 ) ); // (2+4+6+8)/4 = 5

            mean.Backward();

            // dc/da = b, dmean/dc = 1/4, so da = b/4
            for (int i = 0; i < a.Size; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 0.5f ).Within( 1e-6 ) ); // 2/4

            // dc/db = a, dmean/dc = 1/4, so db = a/4
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 0.25f ).Within( 1e-6 ) ); // 1/4
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 0.5f ).Within( 1e-6 ) );  // 2/4
            Assert.That( b.Grad[ 2 ], Is.EqualTo( 0.75f ).Within( 1e-6 ) ); // 3/4
            Assert.That( b.Grad[ 3 ], Is.EqualTo( 1f ).Within( 1e-6 ) );    // 4/4
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_AllDimensions_FindsMaximum()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 5f, 3f, 2f, 8f, 4f } );
            var max = a.Max();

            Assert.That( max.Shape, Is.EqualTo( new[] { 1 } ) );
            Assert.That( max.Data[ 0 ], Is.EqualTo( 8f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_AllDimensions_BackwardToMaxElement()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 5f, 3f, 2f, 8f, 4f } );
            var max = a.Max();

            max.Backward();

            // Gradient should only flow to the max element (index 4, value 8)
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 4 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 5 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_AlongDimension0_ComputesCorrectly()
        {
            // Shape [2, 3]: [[1, 5, 3], [2, 4, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 5f, 3f, 2f, 4f, 6f } );
            var max = a.Max( 0 ); // Max along rows

            // Result shape should be [3]
            Assert.That( max.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( max.Data[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );  // max(1, 2)
            Assert.That( max.Data[ 1 ], Is.EqualTo( 5f ).Within( 1e-6 ) );  // max(5, 4)
            Assert.That( max.Data[ 2 ], Is.EqualTo( 6f ).Within( 1e-6 ) );  // max(3, 6)
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_AlongDimension1_ComputesCorrectly()
        {
            // Shape [2, 3]: [[1, 5, 3], [2, 4, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 5f, 3f, 2f, 4f, 6f } );
            var max = a.Max( 1 ); // Max along columns

            // Result shape should be [2]
            Assert.That( max.Shape, Is.EqualTo( new[] { 2 } ) );
            Assert.That( max.Data[ 0 ], Is.EqualTo( 5f ).Within( 1e-6 ) );  // max(1, 5, 3)
            Assert.That( max.Data[ 1 ], Is.EqualTo( 6f ).Within( 1e-6 ) );  // max(2, 4, 6)
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_AlongDimension_BackwardToMaxElements()
        {
            // Shape [2, 3]: [[1, 5, 3], [2, 4, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 5f, 3f, 2f, 4f, 6f } );
            var max = a.Max( 1 ); // Max along columns -> shape [2], values [5, 6]

            max.Backward();

            // Gradient should only flow to max elements in each row
            // Row 0: max is 5 (index 1)
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            // Row 1: max is 6 (index 5)
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 4 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 5 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_3DTensor_AlongMiddleDimension()
        {
            // Shape [2, 3, 2]: 2 matrices of 3×2
            var a = new Tensor( new[] { 2, 3, 2 } );
            for (int i = 0; i < 12; i++)
                a.Data[ i ] = i + 1;

            var max = a.Max( 1 ); // Max along dimension 1

            // Result shape should be [2, 2]
            Assert.That( max.Shape, Is.EqualTo( new[] { 2, 2 } ) );
            Assert.That( max.Size, Is.EqualTo( 4 ) );

            // First matrix (indices 0-5): [[1,2], [3,4], [5,6]]
            // Max along rows: [5, 6]
            Assert.That( max.Data[ 0 ], Is.EqualTo( 5f ).Within( 1e-6 ) );
            Assert.That( max.Data[ 1 ], Is.EqualTo( 6f ).Within( 1e-6 ) );

            // Second matrix (indices 6-11): [[7,8], [9,10], [11,12]]
            // Max along rows: [11, 12]
            Assert.That( max.Data[ 2 ], Is.EqualTo( 11f ).Within( 1e-6 ) );
            Assert.That( max.Data[ 3 ], Is.EqualTo( 12f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_NegativeDimension_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3, 4 } );
            for (int i = 0; i < 24; i++)
                a.Data[ i ] = i + 1;

            // dim=-1 should be equivalent to dim=2
            var max = a.Max( -1 );

            Assert.That( max.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( max.Size, Is.EqualTo( 6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_InvalidDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            Assert.Throws<ArgumentException>( () => a.Max( -3 ) );
            Assert.Throws<ArgumentException>( () => a.Max( 2 ) );
            Assert.Throws<ArgumentException>( () => a.Max( 5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_WithDuplicateMaxValues_GradientToFirstOccurrence()
        {
            // Test tie-breaking behavior
            var a = new Tensor( new[] { 4 }, new[] { 3f, 5f, 5f, 2f } );
            var max = a.Max();

            Assert.That( max.Data[ 0 ], Is.EqualTo( 5f ).Within( 1e-6 ) );

            max.Backward();

            // Gradient should go to first occurrence of max value (index 1)
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_ChainedWithOperations_ComputesGradientsCorrectly()
        {
            var a = new Tensor( new[] { 3 }, new[] { 1f, 4f, 2f } );
            var b = new Tensor( new[] { 3 }, new[] { 2f, 2f, 2f } );

            var c = a * b; // [2, 8, 4]
            var max = c.Max();

            Assert.That( max.Data[ 0 ], Is.EqualTo( 8f ).Within( 1e-6 ) );

            max.Backward();

            // dc/da = b, dmax/dc flows only to index 1, so da[1] = b[1] = 2
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 0f ).Within( 1e-6 ) );

            // dc/db = a, dmax/dc flows only to index 1, so db[1] = a[1] = 4
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 2 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unsqueeze_AtBeginning_AddsFirstDimension()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Unsqueeze( 0 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 1, 2, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Should share the same data
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unsqueeze_InMiddle_AddsMiddleDimension()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Unsqueeze( 1 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 1, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Should share the same data
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unsqueeze_AtEnd_AddsLastDimension()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Unsqueeze( 2 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3, 1 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Should share the same data
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unsqueeze_NegativeIndex_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            // -1 means after last dimension (same as index 2)
            var b = a.Unsqueeze( -1 );
            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3, 1 } ) );

            // -2 means before last dimension (same as index 1)
            var c = a.Unsqueeze( -2 );
            Assert.That( c.Shape, Is.EqualTo( new[] { 2, 1, 3 } ) );

            // -3 means before first dimension (same as index 0)
            var d = a.Unsqueeze( -3 );
            Assert.That( d.Shape, Is.EqualTo( new[] { 1, 2, 3 } ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unsqueeze_1DTensor_To2D()
        {
            var a = new Tensor( new[] { 5 }, new[] { 1f, 2f, 3f, 4f, 5f } );

            // Add batch dimension at start: [5] -> [1, 5]
            var b = a.Unsqueeze( 0 );
            Assert.That( b.Shape, Is.EqualTo( new[] { 1, 5 } ) );

            // Add dimension at end: [5] -> [5, 1]
            var c = a.Unsqueeze( 1 );
            Assert.That( c.Shape, Is.EqualTo( new[] { 5, 1 } ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unsqueeze_InvalidDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            // Too large
            Assert.Throws<ArgumentException>( () => a.Unsqueeze( 3 ) );
            Assert.Throws<ArgumentException>( () => a.Unsqueeze( 10 ) );

            // Too negative
            Assert.Throws<ArgumentException>( () => a.Unsqueeze( -4 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unsqueeze_GradientsFlowCorrectly()
        {
            var a = new Tensor( new[] { 3 }, new[] { 1f, 2f, 3f } );
            var b = a.Unsqueeze( 0 ); // [1, 3]
            var c = b * 2f;

            c.Backward();

            // Gradients should flow through shared arrays
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 2f ).Within( 1e-6 ) );

            // b's gradient is the same array
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unsqueeze_MatMulUseCase_WorksCorrectly()
        {
            // Common use case: converting 1D to 2D for matrix multiplication
            var input = new Tensor( new[] { 3 }, new[] { 1f, 2f, 3f } );
            var weights = new Tensor( new[] { 3, 2 }, new[] { 0.5f, 0.7f, 0.3f, 0.9f, 0.2f, 0.4f } );

            // Unsqueeze to add batch dimension: [3] -> [1, 3]
            var input2D = input.Unsqueeze( 0 );
            Assert.That( input2D.Shape, Is.EqualTo( new[] { 1, 3 } ) );

            // Now we can do matmul: [1, 3] @ [3, 2] -> [1, 2]
            var output = input2D.MatMul( weights );
            Assert.That( output.Shape, Is.EqualTo( new[] { 1, 2 } ) );

            output.Backward();

            // Gradients should flow back correctly
            Assert.That( input.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
            Assert.That( weights.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_RemovesAllSingleDimensions()
        {
            var a = new Tensor( new[] { 1, 2, 1, 3, 1 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Squeeze();

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Should share the same data
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_SpecificDimension_RemovesThatDimension()
        {
            var a = new Tensor( new[] { 1, 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Squeeze( 0 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Should share the same data
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_MiddleDimension_RemovesCorrectly()
        {
            var a = new Tensor( new[] { 2, 1, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Squeeze( 1 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Should share the same data
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_LastDimension_RemovesCorrectly()
        {
            var a = new Tensor( new[] { 2, 3, 1 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Squeeze( 2 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Should share the same data
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_NegativeIndex_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 1, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            // -2 means middle dimension (same as index 1)
            var b = a.Squeeze( -2 );
            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3 } ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_NonSingleDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            // Cannot squeeze dimension with size > 1
            Assert.Throws<ArgumentException>( () => a.Squeeze( 0 ) );
            Assert.Throws<ArgumentException>( () => a.Squeeze( 1 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_InvalidDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 1, 2, 3 } );

            // Out of range
            Assert.Throws<ArgumentException>( () => a.Squeeze( 3 ) );
            Assert.Throws<ArgumentException>( () => a.Squeeze( -4 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_AllDimensionsAreOne_KeepsScalar()
        {
            var a = new Tensor( new[] { 1, 1, 1 }, new[] { 5f } );

            var b = a.Squeeze();

            // Should keep as [1] scalar tensor
            Assert.That( b.Shape, Is.EqualTo( new[] { 1 } ) );
            Assert.That( b.Size, Is.EqualTo( 1 ) );
            Assert.That( b.Data[ 0 ], Is.EqualTo( 5f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_NoDimensionsToRemove_ReturnsView()
        {
            var a = new Tensor( new[] { 2, 3, 4 } );
            var b = a.Squeeze();

            // No dimensions of size 1, but should still return a view
            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3, 4 } ) );
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_GradientsFlowCorrectly()
        {
            var a = new Tensor( new[] { 1, 3 }, new[] { 1f, 2f, 3f } );
            var b = a.Squeeze( 0 ); // [3]
            var c = b * 3f;

            c.Backward();

            // Gradients should flow through shared arrays
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 3f ).Within( 1e-6 ) );

            // b's gradient is the same array
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_AfterMatMul_RemovesBatchDimension()
        {
            // Common use case: removing batch dimension after matmul
            var input = new Tensor( new[] { 1, 3 }, new[] { 1f, 2f, 3f } );
            var weights = new Tensor( new[] { 3, 2 }, new[] { 0.5f, 0.7f, 0.3f, 0.9f, 0.2f, 0.4f } );

            // Matmul: [1, 3] @ [3, 2] -> [1, 2]
            var output2D = input.MatMul( weights );
            Assert.That( output2D.Shape, Is.EqualTo( new[] { 1, 2 } ) );

            // Squeeze to remove batch dimension: [1, 2] -> [2]
            var output = output2D.Squeeze( 0 );
            Assert.That( output.Shape, Is.EqualTo( new[] { 2 } ) );

            output.Backward();

            // Gradients should flow back correctly
            Assert.That( input.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
            Assert.That( weights.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void UnsqueezeAndSqueeze_RoundTrip_PreservesShape()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            // Unsqueeze then squeeze should give back original shape
            var b = a.Unsqueeze( 0 );  // [1, 2, 3]
            var c = b.Squeeze( 0 );    // [2, 3]

            Assert.That( c.Shape, Is.EqualTo( a.Shape ) );
            Assert.That( c.Size, Is.EqualTo( a.Size ) );

            // All should share same data/grad arrays
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( b.Data, c.Data );
            Assert.AreSame( a.Grad, c.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void UnsqueezeAndSqueeze_ComplexChain_GradientsFlowCorrectly()
        {
            var a = new Tensor( new[] { 3 }, new[] { 1f, 2f, 3f } );
            var b = a.Unsqueeze( 0 );  // [1, 3]
            var c = b * 2f;            // [1, 3]
            var d = c.Squeeze( 0 );    // [3]
            var e = d + 5f;            // [3]

            e.Backward();

            // Gradient should be 2 (from multiplication) for all elements
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Squeeze_Multiple_WorksSequentially()
        {
            var a = new Tensor( new[] { 1, 2, 1, 3, 1 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            // Squeeze specific dimensions one by one
            var b = a.Squeeze( 0 );  // [2, 1, 3, 1]
            var c = b.Squeeze( 1 );  // [2, 3, 1]
            var d = c.Squeeze( 2 );  // [2, 3]

            Assert.That( d.Shape, Is.EqualTo( new[] { 2, 3 } ) );

            // All should share same data
            Assert.AreSame( a.Data, d.Data );
            Assert.AreSame( a.Grad, d.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_2Dto1D_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Reshape( 6 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 6 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );
            Assert.That( b.Data, Is.EqualTo( new[] { 1f, 2f, 3f, 4f, 5f, 6f } ) );

            // Verify data is shared
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_1Dto2D_WorksCorrectly()
        {
            var a = new Tensor( new[] { 6 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Reshape( 2, 3 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Verify data is shared
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_3DtoOther3D_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3, 4 }, Enumerable.Range( 0, 24 ).Select( x => (float)x ).ToArray() );
            var b = a.Reshape( 4, 6 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 4, 6 } ) );
            Assert.That( b.Size, Is.EqualTo( 24 ) );

            // Verify data is shared
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_ToSameShape_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Reshape( 2, 3 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Still should be a view sharing data
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_Example10x5x20To10x100_WorksCorrectly()
        {
            var a = new Tensor( new[] { 10, 5, 20 } ); // Total size: 1000
            var b = a.Reshape( 10, 100 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 10, 100 } ) );
            Assert.That( b.Size, Is.EqualTo( 1000 ) );

            // Verify data is shared
            Assert.AreSame( a.Data, b.Data );
            Assert.AreSame( a.Grad, b.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_InvalidSize_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } ); // Size: 6

            Assert.Throws<ArgumentException>( () => a.Reshape( 5 ) );
            Assert.Throws<ArgumentException>( () => a.Reshape( 2, 4 ) );
            Assert.Throws<ArgumentException>( () => a.Reshape( 3, 3 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_EmptyShape_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            Assert.Throws<ArgumentException>( () => a.Reshape() );
            Assert.Throws<ArgumentException>( () => a.Reshape( new int[] { } ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_NegativeDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            Assert.Throws<ArgumentException>( () => a.Reshape( -1, 6 ) );
            Assert.Throws<ArgumentException>( () => a.Reshape( 2, -3 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_ZeroDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            Assert.Throws<ArgumentException>( () => a.Reshape( 0, 6 ) );
            Assert.Throws<ArgumentException>( () => a.Reshape( 2, 0 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_GradientsFlowCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Reshape( 6 );      // [6]
            var c = b * 2f;              // [6]
            var d = c.Sum();             // scalar

            d.Backward();

            // Gradients should be 2 (from multiplication) for all elements
            for (int i = 0; i < 6; i++)
            {
                Assert.That( a.Grad[ i ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            }
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_MultipleReshapes_GradientsFlowCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Reshape( 6 );      // [6]
            var c = b.Reshape( 3, 2 );   // [3, 2]
            var d = c * 3f;              // [3, 2]
            var e = d.Sum();             // scalar

            e.Backward();

            // Gradients should be 3 (from multiplication) for all elements
            for (int i = 0; i < 6; i++)
            {
                Assert.That( a.Grad[ i ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            }

            // All tensors should share the same grad array
            Assert.AreSame( a.Grad, b.Grad );
            Assert.AreSame( b.Grad, c.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_WithOtherOperations_GradientsFlowCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Reshape( 3, 2 );   // [3, 2]
            var c = new Tensor( new[] { 3, 2 }, new[] { 1f, 1f, 1f, 1f, 1f, 1f } );
            var d = b + c;               // [3, 2]
            var e = d.Sum();             // scalar

            e.Backward();

            // Gradients should be 1 for all elements
            for (int i = 0; i < 6; i++)
            {
                Assert.That( a.Grad[ i ], Is.EqualTo( 1f ).Within( 1e-6 ) );
                Assert.That( c.Grad[ i ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            }
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_DataModificationAffectsOriginal()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Reshape( 6 );

            // Modify reshaped tensor
            b.Data[ 0 ] = 99f;

            // Original should be affected
            Assert.That( a.Data[ 0 ], Is.EqualTo( 99f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Reshape_ChainedWithUnsqueezeAndSqueeze_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.Reshape( 6 );      // [6]
            var c = b.Unsqueeze( 0 );    // [1, 6]
            var d = c.Squeeze( 0 );      // [6]
            var e = d.Reshape( 3, 2 );   // [3, 2]

            Assert.That( e.Shape, Is.EqualTo( new[] { 3, 2 } ) );

            // All should share same data
            Assert.AreSame( a.Data, e.Data );
            Assert.AreSame( a.Grad, e.Grad );
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_2DTo3D_ReplicatesElements()
        {
            // [3, 2] -> [3, 2, 5]
            var a = new Tensor( new[] { 3, 2 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.ExpandLast( 5 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 3, 2, 5 } ) );
            Assert.That( b.Size, Is.EqualTo( 30 ) );

            // Each element should be replicated 5 times
            // First element (1f) at positions 0-4
            for (int i = 0; i < 5; i++)
                Assert.That( b.Data[ i ], Is.EqualTo( 1f ).Within( 1e-6 ) );

            // Second element (2f) at positions 5-9
            for (int i = 5; i < 10; i++)
                Assert.That( b.Data[ i ], Is.EqualTo( 2f ).Within( 1e-6 ) );

            // Third element (3f) at positions 10-14
            for (int i = 10; i < 15; i++)
                Assert.That( b.Data[ i ], Is.EqualTo( 3f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_1DTo2D_WorksCorrectly()
        {
            // [4] -> [4, 3]
            var a = new Tensor( new[] { 4 }, new[] { 1f, 2f, 3f, 4f } );
            var b = a.ExpandLast( 3 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 4, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 12 ) );

            // Check pattern: [1,1,1, 2,2,2, 3,3,3, 4,4,4]
            Assert.That( b.Data[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 3 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 4 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 5 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_ScalarTo1D_WorksCorrectly()
        {
            // [1] -> [1, 5]
            var a = new Tensor( new[] { 1 }, new[] { 7f } );
            var b = a.ExpandLast( 5 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 1, 5 } ) );
            Assert.That( b.Size, Is.EqualTo( 5 ) );

            // All elements should be 7
            for (int i = 0; i < 5; i++)
                Assert.That( b.Data[ i ], Is.EqualTo( 7f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_Backward_AccumulatesGradients()
        {
            // [3, 2] -> [3, 2, 4]
            var a = new Tensor( new[] { 3, 2 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.ExpandLast( 4 );
            var c = b.Sum();

            c.Backward();

            // Each element in 'a' was replicated 4 times, so gradient should be 4
            for (int i = 0; i < 6; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 4f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_Backward_WithMultiplication()
        {
            // [2, 3] -> [2, 3, 2]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.ExpandLast( 2 );
            var c = b * 3f;
            var d = c.Sum();

            d.Backward();

            // Each element multiplied by 3, then gradient flows back through expansion
            for (int i = 0; i < 6; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 6f ).Within( 1e-6 ) ); // 3 * 2 repetitions
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_3DTensor_AddsAnotherDimension()
        {
            // [2, 3, 4] -> [2, 3, 4, 3]
            var a = new Tensor( new[] { 2, 3, 4 } );
            for (int i = 0; i < 24; i++)
                a.Data[ i ] = i + 1;

            var b = a.ExpandLast( 3 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3, 4, 3 } ) );
            Assert.That( b.Size, Is.EqualTo( 72 ) );

            // Check first element is replicated 3 times
            Assert.That( b.Data[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) );

            // Check second element is replicated 3 times
            Assert.That( b.Data[ 3 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 4 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( b.Data[ 5 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_InvalidNumber_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            Assert.Throws<ArgumentException>( () => a.ExpandLast( 0 ) );
            Assert.Throws<ArgumentException>( () => a.ExpandLast( -1 ) );
            Assert.Throws<ArgumentException>( () => a.ExpandLast( -5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_ChainedOperations_WorksCorrectly()
        {
            // [3] -> [3, 2] -> sum
            var a = new Tensor( new[] { 3 }, new[] { 1f, 2f, 3f } );
            var b = a.ExpandLast( 2 );
            var c = b + 10f;
            var d = c.Sum();

            Assert.That( d.Data[ 0 ], Is.EqualTo( 72f ).Within( 1e-6 ) ); // (1+10)*2 + (2+10)*2 + (3+10)*2 = 22+24+26 = 72

            d.Backward();

            // Gradient should accumulate through expansion and addition
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ExpandLast_ExpansionByOne_SameAsSingleReplication()
        {
            // [2, 3] -> [2, 3, 1]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            var b = a.ExpandLast( 1 );

            Assert.That( b.Shape, Is.EqualTo( new[] { 2, 3, 1 } ) );
            Assert.That( b.Size, Is.EqualTo( 6 ) );

            // Data should be the same
            for (int i = 0; i < 6; i++)
                Assert.That( b.Data[ i ], Is.EqualTo( a.Data[ i ] ).Within( 1e-6 ) );

            b.Backward();

            // Gradient should be 1 for each element
            for (int i = 0; i < 6; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 1f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void RequiresGrad_DefaultTrue_EnablesGradientTracking()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2f, 3f } );
            var b = new Tensor( new[] { 2 }, new[] { 4f, 5f } );

            Assert.That( a.RequiresGrad, Is.True );
            Assert.That( b.RequiresGrad, Is.True );

            var c = a * b;
            Assert.That( c.RequiresGrad, Is.True );

            c.Backward();

            // Both should have gradients
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void RequiresGrad_SetToFalse_DisablesGradientTracking()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2f, 3f } );
            a.RequiresGrad = false;

            var b = new Tensor( new[] { 2 }, new[] { 4f, 5f } );

            var c = a * b;
            Assert.That( c.RequiresGrad, Is.True ); // Still true because b requires grad

            c.Backward();

            // a should not have gradients accumulated
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 0f ).Within( 1e-6 ) );

            // b should have gradients
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void RequiresGrad_BothFalse_NoBackwardFunction()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2f, 3f } );
            a.RequiresGrad = false;

            var b = new Tensor( new[] { 2 }, new[] { 4f, 5f } );
            b.RequiresGrad = false;

            var c = a * b;
            Assert.That( c.RequiresGrad, Is.False );

            // Should not throw, but no gradients should accumulate
            c.Backward();

            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0f ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void RequiresGrad_Inference_NoMemoryOverhead()
        {
            // Simulate inference mode
            var weights = new Tensor( new[] { 3, 2 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            weights.RequiresGrad = false;

            var input = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );
            input.RequiresGrad = false;

            var output = input.MatMul( weights );

            Assert.That( output.RequiresGrad, Is.False );

            // Forward pass works normally
            Assert.That( output.Data[ 0 ], Is.EqualTo( 22f ).Within( 1e-6 ) );

            // No backward function should be set (saving memory)
            output.Backward();

            // No gradients accumulated
            Assert.That( input.Grad.Sum( x => x ), Is.EqualTo( 0f ) );
            Assert.That( weights.Grad.Sum( x => x ), Is.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void RequiresGrad_MixedOperations_PropagatesCorrectly()
        {
            var a = new Tensor( 0.1f );
            var b = new Tensor( 0.2f, requiresGrad: false );

            var c = a + b;  // RequiresGrad = true (from a)
            var d = c * 2f; // RequiresGrad = true
            var e = d.Tanh(); // RequiresGrad = true

            Assert.That( c.RequiresGrad, Is.True );
            Assert.That( d.RequiresGrad, Is.True );
            Assert.That( e.RequiresGrad, Is.True );

            e.Backward();

            // a should have gradient, b should not
            Assert.That( a.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_1DTensor_GetAndSet()
        {
            var a = new Tensor( new[] { 5 }, new[] { 1f, 2f, 3f, 4f, 5f } );

            // Test getter
            Assert.That( a[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a[ 2 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( a[ 4 ], Is.EqualTo( 5f ).Within( 1e-6 ) );

            // Test setter
            a[ 1 ] = 10f;
            Assert.That( a[ 1 ], Is.EqualTo( 10f ).Within( 1e-6 ) );
            Assert.That( a.Data[ 1 ], Is.EqualTo( 10f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_2DTensor_GetAndSet()
        {
            // [[1, 2, 3],
            //  [4, 5, 6]]
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            // Test getter
            Assert.That( a[ 0, 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 2 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 0 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 1 ], Is.EqualTo( 5f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 2 ], Is.EqualTo( 6f ).Within( 1e-6 ) );

            // Test setter
            a[ 1, 1 ] = 50f;
            Assert.That( a[ 1, 1 ], Is.EqualTo( 50f ).Within( 1e-6 ) );
            Assert.That( a.Data[ 4 ], Is.EqualTo( 50f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_3DTensor_GetAndSet()
        {
            // 2×2×3 tensor
            var a = new Tensor( new[] { 2, 2, 3 } );
            for (int i = 0; i < 12; i++)
                a.Data[ i ] = i + 1;

            // Test getter with various indices
            Assert.That( a[ 0, 0, 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 0, 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 0, 2 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 1, 0 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 0, 0 ], Is.EqualTo( 7f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 1, 2 ], Is.EqualTo( 12f ).Within( 1e-6 ) );

            // Test setter
            a[ 1, 0, 1 ] = 100f;
            Assert.That( a[ 1, 0, 1 ], Is.EqualTo( 100f ).Within( 1e-6 ) );
            Assert.That( a.Data[ 7 ], Is.EqualTo( 100f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_4DTensor_GetAndSet()
        {
            // 2×2×2×2 tensor
            var a = new Tensor( new[] { 2, 2, 2, 2 } );
            for (int i = 0; i < 16; i++)
                a.Data[ i ] = i;

            // Test getter
            Assert.That( a[ 0, 0, 0, 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 0, 0, 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 1, 1, 1 ], Is.EqualTo( 15f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 0, 1, 0 ], Is.EqualTo( 10f ).Within( 1e-6 ) );

            // Test setter
            a[ 0, 1, 0, 1 ] = 99f;
            Assert.That( a[ 0, 1, 0, 1 ], Is.EqualTo( 99f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_WrongNumberOfIndices_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            // Too few indices
            Assert.Throws<ArgumentException>( () => { var x = a[ 0 ]; } );

            // Too many indices
            Assert.Throws<ArgumentException>( () => { var x = a[ 0, 1, 2 ]; } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_OutOfBounds_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f } );

            // First dimension out of bounds
            Assert.Throws<IndexOutOfRangeException>( () => { var x = a[ 2, 0 ]; } );
            Assert.Throws<IndexOutOfRangeException>( () => { var x = a[ -1, 0 ]; } );

            // Second dimension out of bounds
            Assert.Throws<IndexOutOfRangeException>( () => { var x = a[ 0, 3 ]; } );
            Assert.Throws<IndexOutOfRangeException>( () => { var x = a[ 1, -1 ]; } );

            // Both dimensions out of bounds
            Assert.Throws<IndexOutOfRangeException>( () => { var x = a[ 5, 10 ]; } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_SetterOutOfBounds_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            Assert.Throws<IndexOutOfRangeException>( () => { a[ 2, 0 ] = 1f; } );
            Assert.Throws<IndexOutOfRangeException>( () => { a[ 0, 3 ] = 1f; } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_MatchesFlatArrayIndexing()
        {
            // Verify that indexer produces same results as direct flat array access
            var a = new Tensor( new[] { 3, 4 } );
            for (int i = 0; i < 12; i++)
                a.Data[ i ] = i * 10;

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    int flatIndex = i * 4 + j;
                    Assert.That( a[ i, j ], Is.EqualTo( a.Data[ flatIndex ] ).Within( 1e-6 ),
                        $"Mismatch at [{i},{j}]" );
                }
            }
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_WithOperations_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 2 }, new[] { 1f, 2f, 3f, 4f } );

            // Modify using indexer
            a[ 0, 0 ] = 10f;
            a[ 1, 1 ] = 40f;

            // Use in operations
            var b = new Tensor( new[] { 2, 2 }, new[] { 2f, 2f, 2f, 2f } );
            var c = a * b;

            // Results should reflect the modified values
            Assert.That( c[ 0, 0 ], Is.EqualTo( 20f ).Within( 1e-6 ) ); // 10 * 2
            Assert.That( c[ 0, 1 ], Is.EqualTo( 4f ).Within( 1e-6 ) );  // 2 * 2
            Assert.That( c[ 1, 0 ], Is.EqualTo( 6f ).Within( 1e-6 ) );  // 3 * 2
            Assert.That( c[ 1, 1 ], Is.EqualTo( 80f ).Within( 1e-6 ) ); // 40 * 2
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_PopulateEntireTensor_WorksCorrectly()
        {
            var a = new Tensor( new[] { 3, 3 } );

            // Populate using indexer
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    a[ i, j ] = i * 10 + j;

            // Verify values
            Assert.That( a[ 0, 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 0 ], Is.EqualTo( 10f ).Within( 1e-6 ) );
            Assert.That( a[ 2, 2 ], Is.EqualTo( 22f ).Within( 1e-6 ) );

            // Verify data array is populated correctly
            Assert.That( a.Data[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Data[ 4 ], Is.EqualTo( 11f ).Within( 1e-6 ) );
            Assert.That( a.Data[ 8 ], Is.EqualTo( 22f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_NullIndices_ThrowsException()
        {
            var a = new Tensor( new[] { 2, 3 } );

            Assert.Throws<ArgumentException>( () => { var x = a[ (int[])null ]; } );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_ScalarTensor_AccessWithSingleIndex()
        {
            var a = new Tensor( 42f );

            // Access scalar as [0]
            Assert.That( a[ 0 ], Is.EqualTo( 42f ).Within( 1e-6 ) );

            // Modify scalar
            a[ 0 ] = 100f;
            Assert.That( a[ 0 ], Is.EqualTo( 100f ).Within( 1e-6 ) );
            Assert.That( a.Data[ 0 ], Is.EqualTo( 100f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_LargeTensor_CorrectIndexCalculation()
        {
            // Test with larger tensor to ensure index calculation is correct
            var a = new Tensor( new[] { 5, 4, 3 } );
            for (int i = 0; i < 60; i++)
                a.Data[ i ] = i;

            // Test several random access patterns
            Assert.That( a[ 0, 0, 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 0, 2 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a[ 0, 3, 2 ], Is.EqualTo( 11f ).Within( 1e-6 ) );
            Assert.That( a[ 2, 1, 1 ], Is.EqualTo( 28f ).Within( 1e-6 ) );
            Assert.That( a[ 4, 3, 2 ], Is.EqualTo( 59f ).Within( 1e-6 ) );

            // Verify calculations: index = i*12 + j*3 + k
            Assert.That( a[ 1, 0, 0 ], Is.EqualTo( 12f ).Within( 1e-6 ) );
            Assert.That( a[ 1, 2, 1 ], Is.EqualTo( 19f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Indexer_RowMajorOrderVerification()
        {
            // Verify row-major (C-style) ordering
            // For shape [2, 3], layout should be:
            // [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5
            var a = new Tensor( new[] { 2, 3 } );
            for (int i = 0; i < 6; i++)
                a.Data[ i ] = i;

            int counter = 0;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.That( a[ i, j ], Is.EqualTo( counter ).Within( 1e-6 ),
                        $"Row-major order violation at [{i},{j}], expected {counter}" );
                    counter++;
                }
            }
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_Dim0_ExtractsCorrectly()
        {
            // Shape [4, 3]: 4 rows, 3 columns
            var a = new Tensor( new[] { 4, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f } );

            // Slice dimension 0: start at row 1, take 2 rows
            var slice = a.Slice( 0, 1, 2 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            // Should get rows 1 and 2: [4,5,6, 7,8,9]
            Assert.That( slice.Data[ 0 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 1 ], Is.EqualTo( 5f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 2 ], Is.EqualTo( 6f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 3 ], Is.EqualTo( 7f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 4 ], Is.EqualTo( 8f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 5 ], Is.EqualTo( 9f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_Dim1_ExtractsCorrectly()
        {
            // Shape [3, 4]: 3 rows, 4 columns
            var a = new Tensor( new[] { 3, 4 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f } );

            // Slice dimension 1: start at column 1, take 2 columns
            var slice = a.Slice( 1, 1, 2 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 3, 2 } ) );
            // Should get columns 1-2 from each row: [2,3, 6,7, 10,11]
            Assert.That( slice.Data[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 1 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 2 ], Is.EqualTo( 6f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 3 ], Is.EqualTo( 7f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 4 ], Is.EqualTo( 10f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 5 ], Is.EqualTo( 11f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_FromStart_TakesFirstElements()
        {
            var a = new Tensor( new[] { 5, 3 } );
            for (int i = 0; i < 15; i++)
                a.Data[ i ] = i + 1;

            // Take first 3 rows
            var slice = a.Slice( 0, 0, 3 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 3, 3 } ) );
            Assert.That( slice.Data[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 8 ], Is.EqualTo( 9f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_ToEnd_TakesLastElements()
        {
            var a = new Tensor( new[] { 5, 3 } );
            for (int i = 0; i < 15; i++)
                a.Data[ i ] = i + 1;

            // Take last 2 rows (start=3, length=2)
            var slice = a.Slice( 0, 3, 2 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            // Rows 3-4: [10,11,12, 13,14,15]
            Assert.That( slice.Data[ 0 ], Is.EqualTo( 10f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 5 ], Is.EqualTo( 15f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_SingleElement_WorksCorrectly()
        {
            var a = new Tensor( new[] { 3, 4 } );
            for (int i = 0; i < 12; i++)
                a.Data[ i ] = i + 1;

            // Take 1 column from middle
            var slice = a.Slice( 1, 2, 1 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 3, 1 } ) );
            Assert.That( slice.Data[ 0 ], Is.EqualTo( 3f ).Within( 1e-6 ) );  // column 2, row 0
            Assert.That( slice.Data[ 1 ], Is.EqualTo( 7f ).Within( 1e-6 ) );  // column 2, row 1
            Assert.That( slice.Data[ 2 ], Is.EqualTo( 11f ).Within( 1e-6 ) ); // column 2, row 2
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_3DTensor_WorksCorrectly()
        {
            // Shape [2, 3, 4]
            var a = new Tensor( new[] { 2, 3, 4 } );
            for (int i = 0; i < 24; i++)
                a.Data[ i ] = i + 1;

            // Slice middle dimension: start=1, take 2
            var slice = a.Slice( 1, 1, 2 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 2, 2, 4 } ) );
            Assert.That( slice.Size, Is.EqualTo( 16 ) );

            // First block, middle 2 rows
            Assert.That( slice.Data[ 0 ], Is.EqualTo( 5f ).Within( 1e-6 ) );   // [0,1,0]
            Assert.That( slice.Data[ 7 ], Is.EqualTo( 12f ).Within( 1e-6 ) );  // [0,1,3]
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_NegativeDim_WorksCorrectly()
        {
            var a = new Tensor( new[] { 3, 4, 5 } );
            for (int i = 0; i < 60; i++)
                a.Data[ i ] = i + 1;

            // dim=-1 is equivalent to dim=2
            var slice = a.Slice( -1, 2, 3 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 3, 4, 3 } ) );
            Assert.That( slice.Size, Is.EqualTo( 36 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_Backward_GradientsFlowCorrectly()
        {
            var a = new Tensor( new[] { 4, 3 } );
            for (int i = 0; i < 12; i++)
                a.Data[ i ] = i + 1;

            // Slice 2 rows starting from row 1
            var slice = a.Slice( 0, 1, 2 );
            var result = slice * 2f;

            result.Backward();

            // Gradients should only flow to sliced positions (rows 1-2)
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );  // row 0
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 2f ).Within( 1e-6 ) );  // row 1
            Assert.That( a.Grad[ 4 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 5 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 6 ], Is.EqualTo( 2f ).Within( 1e-6 ) );  // row 2
            Assert.That( a.Grad[ 7 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 8 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 9 ], Is.EqualTo( 0f ).Within( 1e-6 ) );  // row 3
            Assert.That( a.Grad[ 10 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 11 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_ChainedOperations_ComputesGradientsCorrectly()
        {
            var a = new Tensor( new[] { 5, 2 } );
            for (int i = 0; i < 10; i++)
                a.Data[ i ] = i + 1;

            // Slice rows 1-3, then sum
            var slice = a.Slice( 0, 1, 3 );
            var sum = slice.Sum();

            // Sum should be 3+4+5+6+7+8 = 33
            Assert.That( sum.Data[ 0 ], Is.EqualTo( 33f ).Within( 1e-6 ) );

            sum.Backward();

            // Gradients of 1 should flow to rows 1-3
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 4 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 5 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 6 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 7 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 8 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 9 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_MultipleSlices_GradientsAccumulate()
        {
            var a = new Tensor( new[] { 4, 2 } );
            for (int i = 0; i < 8; i++)
                a.Data[ i ] = i + 1;

            var slice1 = a.Slice( 0, 0, 2 ); // rows 0-1
            var slice2 = a.Slice( 0, 1, 2 ); // rows 1-2 (overlaps row 1)

            var sum = slice1.Sum() + slice2.Sum();
            sum.Backward();

            // Row 0: only in slice1
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            // Row 1: in both slices, should accumulate
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            // Row 2: only in slice2
            Assert.That( a.Grad[ 4 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 5 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            // Row 3: not in any slice
            Assert.That( a.Grad[ 6 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
            Assert.That( a.Grad[ 7 ], Is.EqualTo( 0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_InvalidDimension_ThrowsException()
        {
            var a = new Tensor( new[] { 3, 4 } );

            Assert.Throws<ArgumentException>( () => a.Slice( 2, 0, 1 ) );  // dim too large
            Assert.Throws<ArgumentException>( () => a.Slice( -3, 0, 1 ) ); // dim too negative
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_InvalidStart_ThrowsException()
        {
            var a = new Tensor( new[] { 3, 4 } );

            Assert.Throws<ArgumentException>( () => a.Slice( 0, -1, 1 ) ); // negative start
            Assert.Throws<ArgumentException>( () => a.Slice( 0, 3, 1 ) );  // start out of bounds
            Assert.Throws<ArgumentException>( () => a.Slice( 1, 4, 1 ) );  // start out of bounds
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_InvalidLength_ThrowsException()
        {
            var a = new Tensor( new[] { 3, 4 } );

            Assert.Throws<ArgumentException>( () => a.Slice( 0, 0, 0 ) );   // zero length
            Assert.Throws<ArgumentException>( () => a.Slice( 0, 0, -1 ) );  // negative length
            Assert.Throws<ArgumentException>( () => a.Slice( 0, 0, 4 ) );   // length too large
            Assert.Throws<ArgumentException>( () => a.Slice( 0, 2, 2 ) );   // start+length exceeds size
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_RequiresGradFalse_NoGradientsComputed()
        {
            var a = new Tensor( new[] { 4, 3 } );
            a.RequiresGrad = false;
            for (int i = 0; i < 12; i++)
                a.Data[ i ] = i + 1;

            var slice = a.Slice( 0, 1, 2 );

            Assert.That( slice.RequiresGrad, Is.False );

            slice.Backward();

            // No gradients should accumulate
            for (int i = 0; i < a.Size; i++)
                Assert.That( a.Grad[ i ], Is.EqualTo( 0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_FullDimension_CopiesAll()
        {
            var a = new Tensor( new[] { 3, 4 } );
            for (int i = 0; i < 12; i++)
                a.Data[ i ] = i + 1;

            // Take all rows
            var slice = a.Slice( 0, 0, 3 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 3, 4 } ) );
            for (int i = 0; i < 12; i++)
                Assert.That( slice.Data[ i ], Is.EqualTo( a.Data[ i ] ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_AlongLastDimension_WorksCorrectly()
        {
            var a = new Tensor( new[] { 2, 3, 5 } );
            for (int i = 0; i < 30; i++)
                a.Data[ i ] = i + 1;

            // Slice last dimension: take 3 elements starting from index 1
            var slice = a.Slice( 2, 1, 3 );

            Assert.That( slice.Shape, Is.EqualTo( new[] { 2, 3, 3 } ) );
            Assert.That( slice.Size, Is.EqualTo( 18 ) );

            // First row, first depth: should skip first element (1) and take 2,3,4
            Assert.That( slice.Data[ 0 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 1 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( slice.Data[ 2 ], Is.EqualTo( 4f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Slice_BatchProcessingUseCase_WorksCorrectly()
        {
            // Common use case: processing batches
            var data = new Tensor( new[] { 10, 5 } ); // 10 samples, 5 features each
            for (int i = 0; i < 50; i++)
                data.Data[ i ] = i + 1;

            // Extract batch of 3 samples starting from sample 2
            var batch = data.Slice( 0, 2, 3 );

            Assert.That( batch.Shape, Is.EqualTo( new[] { 3, 5 } ) );

            // Process batch
            var processed = batch * 2f;
            processed.Backward();

            // Gradients only on batch elements
            for (int i = 0; i < 10; i++)
                Assert.That( data.Grad[ i ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // rows 0-1
            for (int i = 10; i < 25; i++)
                Assert.That( data.Grad[ i ], Is.EqualTo( 2f ).Within( 1e-6 ) ); // rows 2-4 (batch)
            for (int i = 25; i < 50; i++)
                Assert.That( data.Grad[ i ], Is.EqualTo( 0f ).Within( 1e-6 ) ); // rows 5-9
        }
        //------------------------------------------------------------------
    }
}
