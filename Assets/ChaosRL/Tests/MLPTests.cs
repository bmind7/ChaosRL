using NUnit.Framework;

using UnityEngine;

namespace ChaosRL.Tests
{
    public class MLPTests
    {
        //------------------------------------------------------------------
        [Test]
        public void MLPTensor_SimpleClassification_LearnsMapping()
        {
            // Create a simple neural network: 4 inputs -> 4 hidden -> 1 output
            var mlp = new MLP( numInputs: 4, layerSizes: new[] { 4, 1 }, lastLayerNonLin: false );

            // Create training data: batch size of 2
            // Sample 1: [1, 2, 3, 4] -> 1
            // Sample 2: [-1, -2, -3, -4] -> -1
            var inputData = new float[]
            {
                1f, 2f, 3f, 4f,      // First sample
                -1f, -2f, -3f, -4f   // Second sample
            };
            var targetData = new float[]
            {
                1f,    // Target for first sample
                -1f    // Target for second sample
            };

            var inputs = new Tensor( new[] { 2, 4 }, inputData, "inputs" );
            var targets = new Tensor( new[] { 2, 1 }, targetData, "targets" );

            // Create optimizer
            var optimizer = new AdamOptimizer( new[] { mlp.Parameters }, beta1: 0.9f, beta2: 0.999f );

            // Training parameters
            const int epochs = 1000;
            const float learningRate = 0.003f;
            float finalLoss = 0f;

            // Training loop
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Zero gradients
                mlp.ZeroGrad();

                // Forward pass
                var predictions = mlp.Forward( inputs );

                // Compute MSE loss: mean((predictions - targets)^2)
                var diff = predictions - targets;
                var squared = diff * diff;
                var loss = squared.Mean();

                // Backward pass
                loss.Backward();

                // Update weights
                optimizer.Step( learningRate );

                // Store final loss
                if (epoch == epochs - 1)
                    finalLoss = loss.Data[ 0 ];

                // Print progress every 200 epochs
                if (epoch % 10 == 0 || epoch == epochs - 1)
                {
                    Debug.Log( $"Epoch {epoch}: Loss = {loss.Data[ 0 ]:F6}" );
                }
            }

            // Test that the network learned the mapping
            mlp.ZeroGrad();
            var finalPredictions = mlp.Forward( inputs );

            Debug.Log( $"\nFinal predictions:" );
            Debug.Log( $"Input [1, 2, 3, 4] -> Prediction: {finalPredictions.Data[ 0 ]:F4}, Target: 1.0" );
            Debug.Log( $"Input [-1, -2, -3, -4] -> Prediction: {finalPredictions.Data[ 1 ]:F4}, Target: -1.0" );

            // Assert that the final loss is small (learning succeeded)
            Assert.That( finalLoss, Is.LessThan( 0.01f ), "Network should learn to minimize loss below 0.01" );

            // Assert that predictions are close to targets
            Assert.That( finalPredictions.Data[ 0 ], Is.EqualTo( 1.0f ).Within( 0.2f ), "First prediction should be close to 1.0" );
            Assert.That( finalPredictions.Data[ 1 ], Is.EqualTo( -1.0f ).Within( 0.2f ), "Second prediction should be close to -1.0" );
        }
        //------------------------------------------------------------------
    }
}
