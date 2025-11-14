using NUnit.Framework;

namespace ChaosRL.Tests
{
    public class AdamOptimizerTests
    {
        //------------------------------------------------------------------
        [Test]
        public void Step_WithSingleParameter_AppliesBiasCorrectedUpdate()
        {
            var parameter = new Value( 1.0f );
            parameter.Grad = 1.0f;

            var optimizer = new AdamOptimizer( new[] { new[] { parameter } } );

            optimizer.Step( 0.1f );

            Assert.That( parameter.Data, Is.EqualTo( 0.9f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ResetState_ZeroesMomentsAndBiasPowers()
        {
            var parameter = new Value( 0.0f );
            var optimizer = new AdamOptimizer( new[] { new[] { parameter } } );

            parameter.Grad = 1.0f;
            optimizer.Step( 0.1f );

            parameter.Grad = 1.0f;
            optimizer.Step( 0.1f );
            Assert.That( parameter.Data, Is.EqualTo( -0.2f ).Within( 1e-5f ) );

            optimizer.ResetState();
            parameter.Data = 0.0f;
            parameter.Grad = 1.0f;
            optimizer.Step( 0.1f );

            Assert.That( parameter.Data, Is.EqualTo( -0.1f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
    }
}
