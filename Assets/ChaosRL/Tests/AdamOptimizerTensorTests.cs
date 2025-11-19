using NUnit.Framework;

namespace ChaosRL.Tests
{
    public class AdamOptimizerTensorTests
    {
        //------------------------------------------------------------------
        [Test]
        public void Step_WithSingleScalarParameter_AppliesBiasCorrectedUpdate()
        {
            var parameter = new Tensor( new[] { 1 }, new[] { 1.0f } );
            parameter.Grad[ 0 ] = 1.0f;

            var optimizer = new AdamOptimizerTensor( new[] { new[] { parameter } } );

            optimizer.Step( 0.1f );

            Assert.That( parameter.Data[ 0 ], Is.EqualTo( 0.9f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Step_WithVectorParameter_UpdatesAllElements()
        {
            var parameter = new Tensor( new[] { 3 }, new[] { 1.0f, 2.0f, 3.0f } );
            parameter.Grad[ 0 ] = 1.0f;
            parameter.Grad[ 1 ] = 2.0f;
            parameter.Grad[ 2 ] = 3.0f;

            var optimizer = new AdamOptimizerTensor( new[] { new[] { parameter } } );

            optimizer.Step( 0.1f );

            Assert.That( parameter.Data[ 0 ], Is.EqualTo( 0.9f ).Within( 1e-6f ) );
            Assert.That( parameter.Data[ 1 ], Is.EqualTo( 1.9f ).Within( 1e-6f ) );
            Assert.That( parameter.Data[ 2 ], Is.EqualTo( 2.9f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Step_WithMatrixParameter_UpdatesAllElements()
        {
            var parameter = new Tensor( new[] { 2, 2 }, new[] { 1.0f, 2.0f, 3.0f, 4.0f } );
            parameter.Grad[ 0 ] = 0.5f;
            parameter.Grad[ 1 ] = 1.0f;
            parameter.Grad[ 2 ] = 1.5f;
            parameter.Grad[ 3 ] = 2.0f;

            var optimizer = new AdamOptimizerTensor( new[] { new[] { parameter } } );

            optimizer.Step( 0.1f );

            // With bias correction, each element gets update of -learningRate regardless of gradient magnitude
            Assert.That( parameter.Data[ 0 ], Is.EqualTo( 0.9f ).Within( 1e-6f ) );
            Assert.That( parameter.Data[ 1 ], Is.EqualTo( 1.9f ).Within( 1e-6f ) );
            Assert.That( parameter.Data[ 2 ], Is.EqualTo( 2.9f ).Within( 1e-6f ) );
            Assert.That( parameter.Data[ 3 ], Is.EqualTo( 3.9f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Step_WithMultipleParameters_UpdatesAllIndependently()
        {
            var param1 = new Tensor( new[] { 2 }, new[] { 1.0f, 2.0f } );
            var param2 = new Tensor( new[] { 2 }, new[] { 3.0f, 4.0f } );

            param1.Grad[ 0 ] = 1.0f;
            param1.Grad[ 1 ] = 1.0f;
            param2.Grad[ 0 ] = 2.0f;
            param2.Grad[ 1 ] = 2.0f;

            var optimizer = new AdamOptimizerTensor( new[] { new[] { param1, param2 } } );

            optimizer.Step( 0.1f );

            Assert.That( param1.Data[ 0 ], Is.EqualTo( 0.9f ).Within( 1e-6f ) );
            Assert.That( param1.Data[ 1 ], Is.EqualTo( 1.9f ).Within( 1e-6f ) );
            Assert.That( param2.Data[ 0 ], Is.EqualTo( 2.9f ).Within( 1e-6f ) );
            Assert.That( param2.Data[ 1 ], Is.EqualTo( 3.9f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Step_WithMultipleSteps_AccumulatesMomentum()
        {
            var parameter = new Tensor( new[] { 1 }, new[] { 0.0f } );
            var optimizer = new AdamOptimizerTensor( new[] { new[] { parameter } } );

            // First step
            parameter.Grad[ 0 ] = 1.0f;
            optimizer.Step( 0.1f );
            float afterFirstStep = parameter.Data[ 0 ];

            // Second step with same gradient
            parameter.Grad[ 0 ] = 1.0f;
            optimizer.Step( 0.1f );
            float afterSecondStep = parameter.Data[ 0 ];

            // Verify parameter continues to decrease
            Assert.That( afterFirstStep, Is.LessThan( 0.0f ) );
            Assert.That( afterSecondStep, Is.LessThan( afterFirstStep ) );
            Assert.That( parameter.Data[ 0 ], Is.EqualTo( -0.2f ).Within( 1e-5f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ResetState_ZeroesMomentsAndBiasPowers()
        {
            var parameter = new Tensor( new[] { 1 }, new[] { 0.0f } );
            var optimizer = new AdamOptimizerTensor( new[] { new[] { parameter } } );

            // Take two steps to build up momentum
            parameter.Grad[ 0 ] = 1.0f;
            optimizer.Step( 0.1f );

            parameter.Grad[ 0 ] = 1.0f;
            optimizer.Step( 0.1f );
            Assert.That( parameter.Data[ 0 ], Is.EqualTo( -0.2f ).Within( 1e-5f ) );

            // Reset and verify behavior is same as first step
            optimizer.ResetState();
            parameter.Data[ 0 ] = 0.0f;
            parameter.Grad[ 0 ] = 1.0f;
            optimizer.Step( 0.1f );

            Assert.That( parameter.Data[ 0 ], Is.EqualTo( -0.1f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithMultipleParameterGroups_CollectsAllParameters()
        {
            var param1 = new Tensor( new[] { 1 }, new[] { 1.0f } );
            var param2 = new Tensor( new[] { 1 }, new[] { 2.0f } );
            var param3 = new Tensor( new[] { 1 }, new[] { 3.0f } );

            var optimizer = new AdamOptimizerTensor( new[] {
                new[] { param1, param2 },
                new[] { param3 }
            } );

            Assert.That( optimizer.Parameters.Count, Is.EqualTo( 3 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithEmptyParameterGroups_ThrowsException()
        {
            Assert.Throws<System.ArgumentException>( () =>
                new AdamOptimizerTensor( new Tensor[][] { } )
            );
        }
        //------------------------------------------------------------------
        [Test]
        public void Step_WithCustomHyperparameters_UsesProvidedValues()
        {
            // Create separate parameters for each optimizer
            var parameter1 = new Tensor( new[] { 1 }, new[] { 0.0f } );
            var parameter2 = new Tensor( new[] { 1 }, new[] { 0.0f } );

            var optimizer1 = new AdamOptimizerTensor( new[] { new[] { parameter1 } }, beta1: 0.9f, beta2: 0.999f );
            var optimizer2 = new AdamOptimizerTensor( new[] { new[] { parameter2 } }, beta1: 0.5f, beta2: 0.9f );

            // Take multiple steps - first step is the same due to bias correction
            // but subsequent steps will differ due to momentum accumulation
            for (int i = 0; i < 3; i++)
            {
                parameter1.Grad[ 0 ] = 1.0f;
                parameter2.Grad[ 0 ] = 1.0f;

                optimizer1.Step( 0.1f );
                optimizer2.Step( 0.1f );
            }

            // Different hyperparameters should produce different final values
            Assert.That( parameter1.Data[ 0 ], Is.Not.EqualTo( parameter2.Data[ 0 ] ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Step_WithZeroGradient_ParameterUnchanged()
        {
            var parameter = new Tensor( new[] { 2 }, new[] { 1.0f, 2.0f } );
            parameter.Grad[ 0 ] = 0.0f;
            parameter.Grad[ 1 ] = 0.0f;

            var optimizer = new AdamOptimizerTensor( new[] { new[] { parameter } } );
            optimizer.Step( 0.1f );

            Assert.That( parameter.Data[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6f ) );
            Assert.That( parameter.Data[ 1 ], Is.EqualTo( 2.0f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Step_WithMixedShapes_HandlesCorrectly()
        {
            var scalar = new Tensor( new[] { 1 }, new[] { 1.0f } );
            var vector = new Tensor( new[] { 3 }, new[] { 1.0f, 2.0f, 3.0f } );
            var matrix = new Tensor( new[] { 2, 2 }, new[] { 1.0f, 2.0f, 3.0f, 4.0f } );

            scalar.Grad[ 0 ] = 1.0f;
            vector.Grad[ 0 ] = 1.0f;
            vector.Grad[ 1 ] = 1.0f;
            vector.Grad[ 2 ] = 1.0f;
            matrix.Grad[ 0 ] = 1.0f;
            matrix.Grad[ 1 ] = 1.0f;
            matrix.Grad[ 2 ] = 1.0f;
            matrix.Grad[ 3 ] = 1.0f;

            var optimizer = new AdamOptimizerTensor( new[] { new[] { scalar, vector, matrix } } );
            optimizer.Step( 0.1f );

            Assert.That( scalar.Data[ 0 ], Is.EqualTo( 0.9f ).Within( 1e-6f ) );
            Assert.That( vector.Data[ 0 ], Is.EqualTo( 0.9f ).Within( 1e-6f ) );
            Assert.That( matrix.Data[ 0 ], Is.EqualTo( 0.9f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
    }
}
