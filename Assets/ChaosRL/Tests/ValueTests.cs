using System;
using NUnit.Framework;
using ChaosRL;

namespace ChaosRL.Tests
{
    public class ValueTests
    {
        //------------------------------------------------------------------
        [Test]
        public void Add_TwoValues_ComputesForwardAndBackward()
        {
            // Arrange
            var a = new Value( 2.0f );
            var b = new Value( 3.5f );

            // Act
            var c = a + b;

            // Assert forward value
            Assert.That( c.Data, Is.EqualTo( 5.5f ).Within( 1e-6 ) );

            // Backpropagate and assert gradients
            c.Backward();
            Assert.That( c.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Multiply_TwoValues_ComputesForwardAndBackward()
        {
            var a = new Value( 2.0f );
            var b = new Value( 3.0f );

            var c = a * b;

            Assert.That( c.Data, Is.EqualTo( 6.0f ).Within( 1e-6 ) );

            c.Backward();
            Assert.That( c.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( a.Grad, Is.EqualTo( 3.0f ).Within( 1e-6 ) ); // dc/da = b
            Assert.That( b.Grad, Is.EqualTo( 2.0f ).Within( 1e-6 ) ); // dc/db = a
        }
        //------------------------------------------------------------------
        [Test]
        public void UnaryMinus_ComputesForwardAndBackward()
        {
            var a = new Value( 2.0f );
            var c = -a;

            Assert.That( c.Data, Is.EqualTo( -2.0f ).Within( 1e-6 ) );

            c.Backward();
            Assert.That( a.Grad, Is.EqualTo( -1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Subtract_TwoValues_ComputesForwardAndBackward()
        {
            var a = new Value( 5.0f );
            var b = new Value( 2.0f );
            var c = a - b;

            Assert.That( c.Data, Is.EqualTo( 3.0f ).Within( 1e-6 ) );

            c.Backward();
            Assert.That( a.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad, Is.EqualTo( -1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Divide_TwoValues_ComputesForwardAndBackward()
        {
            var a = new Value( 8.0f );
            var b = new Value( 2.0f );
            var c = a / b; // 8 / 2 = 4

            Assert.That( c.Data, Is.EqualTo( 4.0f ).Within( 1e-6 ) );

            c.Backward();
            // dc/da = 1/b = 0.5
            // dc/db = -a/b^2 = -8/4 = -2
            Assert.That( a.Grad, Is.EqualTo( 0.5f ).Within( 1e-6 ) );
            Assert.That( b.Grad, Is.EqualTo( -2.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Pow_IntegerExponent_ComputesForwardAndBackward()
        {
            var a = new Value( 2.0f );
            var y = a.Pow( 3.0f ); // 2^3 = 8

            Assert.That( y.Data, Is.EqualTo( 8.0f ).Within( 1e-6 ) );

            y.Backward();
            // dy/da = 3 * a^(2) = 12
            Assert.That( a.Grad, Is.EqualTo( 12.0f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Pow_FractionalExponent_ComputesForwardAndBackward()
        {
            var a = new Value( 4.0f );
            var y = a.Pow( 0.5f ); // sqrt(4) = 2

            Assert.That( y.Data, Is.EqualTo( 2.0f ).Within( 1e-6 ) );

            y.Backward();
            // dy/da = 0.5 * a^(-0.5) = 0.5 / sqrt(4) = 0.25
            Assert.That( a.Grad, Is.EqualTo( 0.25f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ReLU_Positive_Negative_And_Zero()
        {
            // Positive
            var a = new Value( 3.0f );
            var ya = a.ReLU();
            Assert.That( ya.Data, Is.EqualTo( 3.0f ).Within( 1e-6 ) );
            ya.Backward();
            Assert.That( a.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );

            // Negative
            var b = new Value( -2.0f );
            var yb = b.ReLU();
            Assert.That( yb.Data, Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            yb.Backward();
            Assert.That( b.Grad, Is.EqualTo( 0.0f ).Within( 1e-6 ) );

            // Zero edge (derivative defined as 0 in implementation)
            var c = new Value( 0.0f );
            var yc = c.ReLU();
            Assert.That( yc.Data, Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            yc.Backward();
            Assert.That( c.Grad, Is.EqualTo( 0.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Exp_ComputesForwardAndBackward()
        {
            var a = new Value( 1.5f );
            var y = a.Exp();

            var expected = (float)Math.Exp( 1.5f );
            Assert.That( y.Data, Is.EqualTo( expected ).Within( 1e-6 ) );

            y.Backward();
            Assert.That( a.Grad, Is.EqualTo( expected ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Tanh_ComputesForwardAndBackward()
        {
            var a = new Value( 0.5f );
            var y = a.Tanh();

            var expected = (float)Math.Tanh( 0.5f );
            Assert.That( y.Data, Is.EqualTo( expected ).Within( 1e-6 ) );

            y.Backward();
            var expectedGrad = 1.0f - expected * expected;
            Assert.That( a.Grad, Is.EqualTo( expectedGrad ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Log_ComputesForwardAndBackward()
        {
            var a = new Value( 2.0f );
            var y = a.Log();

            var expected = (float)Math.Log( 2.0f );
            Assert.That( y.Data, Is.EqualTo( expected ).Within( 1e-6 ) );

            y.Backward();
            Assert.That( a.Grad, Is.EqualTo( 0.5f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Clamp_AllowsGradientWithinBounds_BlocksWhenClamped()
        {
            var within = new Value( 0.5f );
            var clampedWithin = within.Clamp( -1.0f, 1.0f );
            Assert.That( clampedWithin.Data, Is.EqualTo( 0.5f ).Within( 1e-6 ) );
            clampedWithin.Backward();
            Assert.That( within.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );

            var outside = new Value( 2.0f );
            var clampedOutside = outside.Clamp( -1.0f, 1.0f );
            Assert.That( clampedOutside.Data, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
            clampedOutside.Backward();
            Assert.That( outside.Grad, Is.EqualTo( 0.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Max_ComputesForwardAndBackward()
        {
            var a = new Value( 2.0f );
            var b = new Value( 5.0f );
            var m = Value.Max( a, b );

            Assert.That( m.Data, Is.EqualTo( 5.0f ).Within( 1e-6 ) );

            m.Backward();
            Assert.That( a.Grad, Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Min_ComputesForwardAndBackward()
        {
            var a = new Value( 2.0f );
            var b = new Value( -1.0f );
            var m = Value.Min( a, b );

            Assert.That( m.Data, Is.EqualTo( -1.0f ).Within( 1e-6 ) );

            m.Backward();
            Assert.That( a.Grad, Is.EqualTo( 0.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ImplicitConversion_FromFloat_WorksInExpressions()
        {
            // Construct directly from float via implicit conversion
            Value v = 5.0f;
            Assert.That( v.Data, Is.EqualTo( 5.0f ).Within( 1e-6 ) );

            // Mix Value and float in an expression
            var b = new Value( 3.0f );
            var c = b + 2.0f; // 2.0f implicitly converts to Value
            Assert.That( c.Data, Is.EqualTo( 5.0f ).Within( 1e-6 ) );
            c.Backward();
            Assert.That( b.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void BackwardPass_Accumulation_And_TopologicalOrder()
        {
            // f(a,b) = a + b + a*b
            var a = new Value( 2.0f );
            var b = new Value( 3.0f );
            var c = a + b;
            var d = a * b;
            var f = c + d; // = a + b + a*b

            Assert.That( f.Data, Is.EqualTo( 2.0f + 3.0f + 6.0f ).Within( 1e-6 ) );

            f.Backward();
            // df/da = 1 + b = 4
            // df/db = 1 + a = 3
            Assert.That( a.Grad, Is.EqualTo( 4.0f ).Within( 1e-6 ) );
            Assert.That( b.Grad, Is.EqualTo( 3.0f ).Within( 1e-6 ) );
            Assert.That( f.Grad, Is.EqualTo( 1.0f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
    }
}
