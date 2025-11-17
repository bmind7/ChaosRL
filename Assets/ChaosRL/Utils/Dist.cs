using System;

namespace ChaosRL
{
    /// <summary>
    /// Normal (Gaussian) distribution utility that operates on Value for autodiff.
    /// </summary>
    public class Dist
    {
        //------------------------------------------------------------------
        public Value Mean { get; }
        public Value StdDev { get; }
        //------------------------------------------------------------------
        public static Dist StandardNormal() => new Dist( 0f, 1f );
        //------------------------------------------------------------------
        public Dist( Value mean, Value std )
        {
            // Note: StdDev should be > 0. Consider parameterizing via a positive transform if needed.
            Mean = mean;
            StdDev = std;
        }
        //------------------------------------------------------------------
        // Sample using Box–Muller with RandomHub for thread safety; returns a Value enabling gradients w.r.t. Mean/StdDev.
        public Value Sample()
        {
            double u1 = 1.0 - RandomHub.NextDouble(); // avoid 0
            double u2 = 1.0 - RandomHub.NextDouble();
            float z = (float)(Math.Sqrt( -2.0 * Math.Log( u1 ) ) * Math.Cos( 2.0 * Math.PI * u2 ));
            return Mean + StdDev * z; // reparameterization: x = mean + std * z
        }
        //------------------------------------------------------------------
        // Probability density as a Value
        public Value Pdf( Value x )
        {
            Value z = (x - Mean) / StdDev;
            float sqrt2pi = (float)Math.Sqrt( 2.0 * Math.PI );
            Value coeff = (1.0f / sqrt2pi) / StdDev;
            Value expTerm = (-0.5f * z * z).Exp();
            return coeff * expTerm;
        }
        //------------------------------------------------------------------
        // Log probability as a Value: -0.5*z^2 - log(std) - 0.5*log(2*pi)
        public Value LogProb( Value x )
        {
            Value z = (x - Mean) / StdDev;
            float halfLog2Pi = 0.5f * (float)Math.Log( 2.0 * Math.PI );
            return -0.5f * z * z - StdDev.Log() - halfLog2Pi;
        }
        //------------------------------------------------------------------
        // Analytical entropy of Normal(mean, std): 0.5 * (1 + log(2*pi*std^2))
        public Value Entropy()
        {
            float twoPi = (float)(2.0 * Math.PI);
            Value inside = twoPi * StdDev * StdDev; // 2*pi*std^2
            return 0.5f * (1.0f + inside.Log());
        }
        //------------------------------------------------------------------
    }
}
