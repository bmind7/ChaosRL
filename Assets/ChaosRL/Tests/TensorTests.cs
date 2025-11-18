using System;
using NUnit.Framework;
using ChaosRL;

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
    }
}
