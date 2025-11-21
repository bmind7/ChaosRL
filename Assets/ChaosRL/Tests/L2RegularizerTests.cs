using NUnit.Framework;

using System;

namespace ChaosRL.Tests
{
    public class L2RegularizerTests
    {
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithPositiveCoefficient_ReturnsExpectedPenalty()
        {
            // Create two parameter tensors
            var w1 = new Tensor( new[] { 2 }, new[] { 1.5f, -2.0f } );
            var w2 = new Tensor( new[] { 3 }, new[] { 0.5f, 1.0f, -1.5f } );

            var regularizer = new L2Regularizer( new[]
            {
                new[] { w1, w2 },
            } );

            // Compute L2 penalty: 0.5 * 0.1 * (1.5^2 + (-2.0)^2 + 0.5^2 + 1.0^2 + (-1.5)^2)
            // = 0.05 * (2.25 + 4.0 + 0.25 + 1.0 + 2.25)
            // = 0.05 * 9.75 = 0.4875
            var penalty = regularizer.Compute( 0.1f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 0.4875f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithZeroCoefficient_ReturnsZero()
        {
            var weight = new Tensor( new[] { 3 }, new[] { 1.0f, 2.0f, 3.0f } );
            var regularizer = new L2Regularizer( new[]
            {
                new[] { weight },
            } );

            var penalty = regularizer.Compute( 0.0f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 0.0f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithNegativeCoefficient_ReturnsZero()
        {
            var weight = new Tensor( new[] { 2 }, new[] { 3.0f, 4.0f } );
            var regularizer = new L2Regularizer( new[]
            {
                new[] { weight },
            } );

            var penalty = regularizer.Compute( -0.5f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 0.0f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithCustomScale_AppliesScaleFactor()
        {
            var w1 = new Tensor( new[] { 1 }, new[] { 1.0f } );
            var w2 = new Tensor( new[] { 1 }, new[] { 2.0f } );

            var regularizer = new L2Regularizer( new[]
            {
                new[] { w1, w2 },
            } );

            // L2 = scale * coef * (1^2 + 2^2) = 1.0 * 0.2 * 5 = 1.0
            var penalty = regularizer.Compute( 0.2f, scale: 1.0f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithMultipleParameterGroups_SumsAllParameters()
        {
            var group1_w1 = new Tensor( new[] { 2 }, new[] { 1.0f, 1.0f } );
            var group2_w1 = new Tensor( new[] { 2 }, new[] { 1.0f, 1.0f } );

            var regularizer = new L2Regularizer( new[]
            {
                new[] { group1_w1 },
                new[] { group2_w1 },
            } );

            // L2 = 0.5 * 0.1 * (1^2 + 1^2 + 1^2 + 1^2) = 0.05 * 4 = 0.2
            var penalty = regularizer.Compute( 0.1f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 0.2f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_With2DTensor_ComputesCorrectly()
        {
            // 2x3 weight matrix
            var weights = new Tensor( new[] { 2, 3 }, new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f } );

            var regularizer = new L2Regularizer( new[]
            {
                new[] { weights },
            } );

            // L2 = 0.5 * 0.01 * (1 + 4 + 9 + 16 + 25 + 36) = 0.005 * 91 = 0.455
            var penalty = regularizer.Compute( 0.01f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 0.455f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithZeroWeights_ReturnsZero()
        {
            var weights = new Tensor( new[] { 3 }, new[] { 0.0f, 0.0f, 0.0f } );

            var regularizer = new L2Regularizer( new[]
            {
                new[] { weights },
            } );

            var penalty = regularizer.Compute( 0.1f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 0.0f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithNullParameterGroups_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>( () => new L2Regularizer( null ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithEmptyParameterGroups_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>( () => new L2Regularizer( new Tensor[][] { } ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_WithGroupsContainingOnlyNulls_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>( () => new L2Regularizer( new[] { new Tensor[] { null, null } } ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithMixedNegativeAndPositiveWeights_ComputesCorrectly()
        {
            var weights = new Tensor( new[] { 4 }, new[] { -2.0f, 3.0f, -1.0f, 4.0f } );

            var regularizer = new L2Regularizer( new[]
            {
                new[] { weights },
            } );

            // L2 = 0.5 * 0.1 * (4 + 9 + 1 + 16) = 0.05 * 30 = 1.5
            var penalty = regularizer.Compute( 0.1f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 1.5f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Compute_WithScalarTensor_WorksCorrectly()
        {
            var scalar = new Tensor( 3.0f );

            var regularizer = new L2Regularizer( new[]
            {
                new[] { scalar },
            } );

            // L2 = 0.5 * 0.1 * 9 = 0.45
            var penalty = regularizer.Compute( 0.1f );

            Assert.That( penalty.Data[ 0 ], Is.EqualTo( 0.45f ).Within( 1e-6f ) );
        }
        //------------------------------------------------------------------
    }
}
