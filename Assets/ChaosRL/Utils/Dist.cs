using System;

namespace ChaosRL
{
    /// <summary>
    /// Normal (Gaussian) distribution utility that operates on Tensor for autodiff.
    /// Supports batch operations where mean and std can be tensors of any compatible shape.
    /// </summary>
    public class Dist
    {
        //------------------------------------------------------------------
        public Tensor Mean { get; }
        public Tensor StdDev { get; }
        //------------------------------------------------------------------
        public static Dist StandardNormal() => new Dist( 0f, 1f );
        //------------------------------------------------------------------
        public Dist( Tensor mean, Tensor std )
        {
            // Note: StdDev should be > 0. Consider parameterizing via a positive transform if needed.
            if (!ShapesMatch( mean.Shape, std.Shape ))
                throw new ArgumentException( "Mean and StdDev must have matching shapes" );

            Mean = mean;
            StdDev = std;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Sample using reparameterization trick: x = mean + std * z, where z ~ N(0,1).
        /// Returns a Tensor enabling gradients w.r.t. Mean/StdDev.
        /// </summary>
        public Tensor Sample()
        {
            var z = new float[ Mean.Size ];
            for (int i = 0; i < z.Length; i++)
            {
                double u1 = 1.0 - RandomHub.NextDouble(); // avoid 0
                double u2 = 1.0 - RandomHub.NextDouble();
                z[ i ] = (float)(Math.Sqrt( -2.0 * Math.Log( u1 ) ) * Math.Cos( 2.0 * Math.PI * u2 ));
            }

            var zTensor = new Tensor( Mean.Shape, z );
            return Mean + StdDev * zTensor; // reparameterization: x = mean + std * z
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Probability density as a Tensor: (1/(sqrt(2π)σ)) * exp(-0.5*z^2)
        /// where z = (x - μ) / σ
        /// </summary>
        public Tensor Pdf( Tensor x )
        {
            if (!ShapesMatch( x.Shape, Mean.Shape ))
                throw new ArgumentException( "Input x must have same shape as Mean" );

            var z = (x - Mean) / StdDev;
            float sqrt2pi = (float)Math.Sqrt( 2.0 * Math.PI );
            var coeff = (1.0f / sqrt2pi) / StdDev;
            var expTerm = (-0.5f * z * z).Exp();
            return coeff * expTerm;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Log probability as a Tensor: -0.5*z^2 - log(std) - 0.5*log(2*pi)
        /// where z = (x - mean) / std
        /// </summary>
        public Tensor LogProb( Tensor x )
        {
            if (!ShapesMatch( x.Shape, Mean.Shape ))
                throw new ArgumentException( "Input x must have same shape as Mean" );

            var z = (x - Mean) / StdDev;
            float halfLog2Pi = 0.5f * (float)Math.Log( 2.0 * Math.PI );

            var logStd = StdDev.Log();
            var result = -0.5f * z * z - logStd - halfLog2Pi;

            return result;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Analytical entropy of Normal(mean, std): 0.5 * (1 + log(2*pi*std^2))
        /// </summary>
        public Tensor Entropy()
        {
            float twoPi = (float)(2.0 * Math.PI);
            var variance = StdDev * StdDev;
            var inside = variance * twoPi; // 2*pi*std^2
            return 0.5f * (1.0f + inside.Log());
        }
        //------------------------------------------------------------------
        private static bool ShapesMatch( int[] shape1, int[] shape2 )
        {
            if (shape1.Length != shape2.Length)
                return false;

            for (int i = 0; i < shape1.Length; i++)
                if (shape1[ i ] != shape2[ i ])
                    return false;

            return true;
        }
        //------------------------------------------------------------------
    }
}
