using NUnit.Framework;

namespace ChaosRL.Tests
{
    public class L2RegularizerTests
    {
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithPositiveCoefficient_ReturnsExpectedPenaltyAndGradients()
        {
            var w1 = new Value( 1.5f );
            var w2 = new Value( -2.0f );
            var regularizer = new L2Regularizer( new[]
            {
                new[] { w1, w2 },
            } );

            var penalty = regularizer.Compute( 0.1f );

            Assert.That( penalty.Data, Is.EqualTo( 0.3125f ).Within( 1e-6f ) );

            penalty.Backward();

            Assert.That( w1.Grad, Is.EqualTo( 0.15f ).Within( 1e-6f ) );
            Assert.That( w2.Grad, Is.EqualTo( -0.2f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithNonPositiveCoefficient_ReturnsZeroWithoutGradients()
        {
            var weight = new Value( 3.0f );
            var regularizer = new L2Regularizer( new[]
            {
                new[] { weight },
            } );

            var penalty = regularizer.Compute( 0.0f );

            Assert.That( penalty.Data, Is.EqualTo( 0.0f ).Within( 1e-6f ) );

            penalty.Backward();

            Assert.That( weight.Grad, Is.EqualTo( 0.0f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithCustomScale_AppliesScaleFactor()
        {
            var w1 = new Value( 1.0f );
            var w2 = new Value( 2.0f );
            var regularizer = new L2Regularizer( new[]
            {
                new[] { w1, w2 },
            } );

            var penalty = regularizer.Compute( 0.2f, scale: 1.0f );

            Assert.That( penalty.Data, Is.EqualTo( 1.0f ).Within( 1e-6f ) );

            penalty.Backward();

            Assert.That( w1.Grad, Is.EqualTo( 0.4f ).Within( 1e-6f ) );
            Assert.That( w2.Grad, Is.EqualTo( 0.8f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
    }
}
