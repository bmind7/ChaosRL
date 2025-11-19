using NUnit.Framework;
using ChaosRL;

namespace ChaosRL.Tests
{
    public class DistTests
    {
        [Test]
        public void LogProb_StandardNormal_AtZero_MatchesKnownValue()
        {
            var d = Dist.StandardNormal();
            var lp = d.LogProb( 0f ).Data;
            // -0.5 * ln(2π) ≈ -0.91893853
            Assert.That( lp, Is.EqualTo( -0.9189385f ).Within( 1e-4 ) );
        }

        [Test]
        public void Entropy_StandardNormal_MatchesKnownValue()
        {
            var d = Dist.StandardNormal();
            var h = d.Entropy().Data;
            // 0.5 * ln(2πe) ≈ 1.41893853
            Assert.That( h, Is.EqualTo( 1.4189385f ).Within( 1e-4 ) );
        }

        [Test]
        public void Sample_ReturnsFiniteValues()
        {
            RandomHub.SetSeed( 123 );
            var d = new Dist( 1.5f, 0.7f );
            for (int i = 0; i < 10; i++)
            {
                var x = d.Sample();
                Assert.That( float.IsFinite( x.Data ) );
            }
        }

        [Test]
        public void LogProb_GradientsFlowToInput()
        {
            var mean = new Value( 0f );
            var std = new Value( 1f );
            var dist = new Dist( mean, std );

            var x = new Value( 1f );
            var logProb = dist.LogProb( x );

            logProb.Backward();

            // Gradients flow to x (the input) when computing log probability
            Assert.That( x.Grad, Is.Not.EqualTo( 0f ) );
        }

        [Test]
        public void LogProb_GradientsFlowToDistributionParameters()
        {
            // To get gradients w.r.t. distribution parameters,
            // they must be part of the computation graph
            var meanParam = new Value( 0f );
            var stdParam = new Value( 2f );

            // Create distribution parameters through computation
            var mean = meanParam + 0f;  // Identity operation to include in graph
            var std = stdParam + 0f;
            var dist = new Dist( mean, std );

            var x = new Value( 1f );
            var logProb = dist.LogProb( x );

            logProb.Backward();

            // Now gradients should flow to the parameter values
            Assert.That( meanParam.Grad, Is.Not.EqualTo( 0f ) );
            Assert.That( stdParam.Grad, Is.Not.EqualTo( 0f ) );
        }

        [Test]
        public void Sample_SupportsGradientFlow()
        {
            var mean = new Value( 2f );
            var std = new Value( 1f );
            var dist = new Dist( mean, std );

            RandomHub.SetSeed( 789 );
            var sample = dist.Sample();

            // Create a simple loss
            var loss = (sample - 5f) * (sample - 5f);
            loss.Backward();

            // Mean and std should have non-zero gradient (reparameterization trick)
            Assert.That( mean.Grad, Is.Not.EqualTo( 0f ) );
            Assert.That( std.Grad, Is.Not.EqualTo( 0f ) );
        }

        [Test]
        public void Entropy_SupportsGradientFlow()
        {
            var mean = new Value( 0f );
            var std = new Value( 2f );
            var dist = new Dist( mean, std );

            var entropy = dist.Entropy();
            entropy.Backward();

            // Entropy depends only on std, not mean
            Assert.That( mean.Grad, Is.EqualTo( 0f ) );
            Assert.That( std.Grad, Is.Not.EqualTo( 0f ) );
        }
    }
}
