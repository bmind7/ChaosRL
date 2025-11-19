using NUnit.Framework;

using System;

namespace ChaosRL.Tests
{
    public class DistTensorTests
    {
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithMatchingShapes_CreatesSuccessfully()
        {
            var mean = new Tensor( new[] { 2, 3 }, new[] { 0f, 1f, 2f, 3f, 4f, 5f } );
            var std = new Tensor( new[] { 2, 3 }, new[] { 1f, 1f, 1f, 1f, 1f, 1f } );

            var dist = new DistTensor( mean, std );

            Assert.That( dist.Mean, Is.EqualTo( mean ) );
            Assert.That( dist.StdDev, Is.EqualTo( std ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithMismatchedShapes_ThrowsException()
        {
            var mean = new Tensor( new[] { 2, 3 } );
            var std = new Tensor( new[] { 3, 2 } );

            Assert.Throws<ArgumentException>( () => new DistTensor( mean, std ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void StandardNormal_CreatesDistributionWithZeroMeanUnitStd()
        {
            var dist = DistTensor.StandardNormal();

            Assert.That( dist.Mean.Data[ 0 ], Is.EqualTo( 0f ).Within( 1e-6f ) );
            Assert.That( dist.StdDev.Data[ 0 ], Is.EqualTo( 1f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void LogProb_StandardNormal_AtZero_MatchesKnownValue()
        {
            var dist = DistTensor.StandardNormal();
            var x = new Tensor( 0f );
            var lp = dist.LogProb( x );

            // -0.5 * ln(2π) ≈ -0.91893853
            Assert.That( lp.Data[ 0 ], Is.EqualTo( -0.9189385f ).Within( 1e-4f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void LogProb_StandardNormal_AtOne_MatchesKnownValue()
        {
            var dist = DistTensor.StandardNormal();
            var x = new Tensor( 1f );
            var lp = dist.LogProb( x );

            // -0.5 * 1^2 - 0.5 * ln(2π) = -0.5 - 0.91893853 ≈ -1.41893853
            Assert.That( lp.Data[ 0 ], Is.EqualTo( -1.4189385f ).Within( 1e-4f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void LogProb_WithBatch_ComputesForAllElements()
        {
            var mean = new Tensor( new[] { 3 }, new[] { 0f, 0f, 0f } );
            var std = new Tensor( new[] { 3 }, new[] { 1f, 1f, 1f } );
            var dist = new DistTensor( mean, std );

            var x = new Tensor( new[] { 3 }, new[] { 0f, 1f, -1f } );
            var lp = dist.LogProb( x );

            Assert.That( lp.Shape, Is.EqualTo( new[] { 3 } ) );
            Assert.That( lp.Data[ 0 ], Is.EqualTo( -0.9189385f ).Within( 1e-4f ) );
            Assert.That( lp.Data[ 1 ], Is.EqualTo( -1.4189385f ).Within( 1e-4f ) );
            Assert.That( lp.Data[ 2 ], Is.EqualTo( -1.4189385f ).Within( 1e-4f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void LogProb_WithMismatchedShape_ThrowsException()
        {
            var mean = new Tensor( new[] { 2 } );
            var std = new Tensor( new[] { 2 } );
            var dist = new DistTensor( mean, std );

            var x = new Tensor( new[] { 3 } );

            Assert.Throws<ArgumentException>( () => dist.LogProb( x ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Entropy_StandardNormal_MatchesKnownValue()
        {
            var dist = DistTensor.StandardNormal();
            var h = dist.Entropy();

            // 0.5 * (1 + ln(2π)) = 0.5 * (1 + 1.8378771) ≈ 1.41893853
            Assert.That( h.Data[ 0 ], Is.EqualTo( 1.4189385f ).Within( 1e-4f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Entropy_WithDifferentStd_ComputesCorrectly()
        {
            var mean = new Tensor( 0f );
            var std = new Tensor( 2f );
            var dist = new DistTensor( mean, std );

            var h = dist.Entropy();

            // 0.5 * (1 + ln(2π*4)) = 0.5 * (1 + ln(8π))
            float expected = 0.5f * (1f + MathF.Log( 2f * MathF.PI * 4f ));
            Assert.That( h.Data[ 0 ], Is.EqualTo( expected ).Within( 1e-4f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Entropy_WithBatch_ComputesForAllElements()
        {
            var mean = new Tensor( new[] { 2 }, new[] { 0f, 0f } );
            var std = new Tensor( new[] { 2 }, new[] { 1f, 2f } );
            var dist = new DistTensor( mean, std );

            var h = dist.Entropy();

            Assert.That( h.Shape, Is.EqualTo( new[] { 2 } ) );
            Assert.That( h.Data[ 0 ], Is.EqualTo( 1.4189385f ).Within( 1e-4f ) );

            float expected1 = 0.5f * (1f + MathF.Log( 2f * MathF.PI * 4f ));
            Assert.That( h.Data[ 1 ], Is.EqualTo( expected1 ).Within( 1e-4f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sample_ReturnsFiniteValues()
        {
            RandomHub.SetSeed( 123 );
            var mean = new Tensor( 1.5f );
            var std = new Tensor( 0.7f );
            var dist = new DistTensor( mean, std );

            for (int i = 0; i < 10; i++)
            {
                var sample = dist.Sample();
                Assert.That( float.IsFinite( sample.Data[ 0 ] ) );
            }
        }
        //------------------------------------------------------------------
        [Test]
        public void Sample_ReturnsCorrectShape()
        {
            var mean = new Tensor( new[] { 2, 3 } );
            var std = new Tensor( new[] { 2, 3 } );
            var dist = new DistTensor( mean, std );

            var sample = dist.Sample();

            Assert.That( sample.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( sample.Size, Is.EqualTo( 6 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sample_WithBatch_ProducesDifferentValues()
        {
            RandomHub.SetSeed( 456 );
            var mean = new Tensor( new[] { 5 }, new[] { 0f, 0f, 0f, 0f, 0f } );
            var std = new Tensor( new[] { 5 }, new[] { 1f, 1f, 1f, 1f, 1f } );
            var dist = new DistTensor( mean, std );

            var sample = dist.Sample();

            // Check that not all values are identical (statistical test)
            bool allSame = true;
            for (int i = 1; i < sample.Size; i++)
            {
                if (Math.Abs( sample.Data[ i ] - sample.Data[ 0 ] ) > 0.01f)
                {
                    allSame = false;
                    break;
                }
            }

            Assert.That( allSame, Is.False );
        }
        //------------------------------------------------------------------
        [Test]
        public void Pdf_StandardNormal_AtZero_MatchesKnownValue()
        {
            var dist = DistTensor.StandardNormal();
            var x = new Tensor( 0f );
            var pdf = dist.Pdf( x );

            // 1/sqrt(2π) ≈ 0.3989423
            float expected = 1f / MathF.Sqrt( 2f * MathF.PI );
            Assert.That( pdf.Data[ 0 ], Is.EqualTo( expected ).Within( 1e-4f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Pdf_WithMismatchedShape_ThrowsException()
        {
            var mean = new Tensor( new[] { 2 } );
            var std = new Tensor( new[] { 2 } );
            var dist = new DistTensor( mean, std );

            var x = new Tensor( new[] { 3 } );

            Assert.Throws<ArgumentException>( () => dist.Pdf( x ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void LogProb_AndPdf_AreConsistent()
        {
            var mean = new Tensor( 0f );
            var std = new Tensor( 1f );
            var dist = new DistTensor( mean, std );

            var x = new Tensor( 0.5f );
            var logProb = dist.LogProb( x );
            var pdf = dist.Pdf( x );

            // log(pdf) should equal logProb
            float logOfPdf = MathF.Log( pdf.Data[ 0 ] );
            Assert.That( logOfPdf, Is.EqualTo( logProb.Data[ 0 ] ).Within( 1e-4f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sample_SupportsGradientFlow()
        {
            var mean = new Tensor( 2f );
            var std = new Tensor( 1f );
            var dist = new DistTensor( mean, std );

            RandomHub.SetSeed( 789 );
            var sample = dist.Sample();

            // Create a simple loss
            var loss = (sample - 5f).Pow( 2 );
            loss.Backward();

            // Mean should have non-zero gradient (reparameterization trick)
            Assert.That( mean.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
            Assert.That( std.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void LogProb_GradientsFlowToInput()
        {
            var mean = new Tensor( 0f );
            var std = new Tensor( 1f );
            var dist = new DistTensor( mean, std );

            var x = new Tensor( 1f );
            var logProb = dist.LogProb( x );

            logProb.Backward();

            // Gradients flow to x (the input) when computing log probability
            Assert.That( x.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void LogProb_GradientsFlowToDistributionParameters()
        {
            // To get gradients w.r.t. distribution parameters,
            // they must be part of the computation graph
            var mean = new Tensor( 0f );
            var std = new Tensor( 2f );

            var dist = new DistTensor( mean, std );

            var x = new Tensor( 1f );
            var logProb = dist.LogProb( x );

            logProb.Backward();

            // Now gradients should flow to the parameter tensors
            Assert.That( mean.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
            Assert.That( std.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Entropy_SupportsGradientFlow()
        {
            var mean = new Tensor( 0f );
            var std = new Tensor( 2f );
            var dist = new DistTensor( mean, std );

            var entropy = dist.Entropy();
            entropy.Backward();

            // Entropy depends only on std, not mean
            Assert.That( mean.Grad[ 0 ], Is.EqualTo( 0f ) );
            Assert.That( std.Grad[ 0 ], Is.Not.EqualTo( 0f ) );
        }
        //------------------------------------------------------------------
    }
}
