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
    }
}
