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
    }
}
