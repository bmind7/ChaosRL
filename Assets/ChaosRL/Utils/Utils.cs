using System;

using UnityEngine;

namespace ChaosRL
{
    public static class Utils
    {
        //------------------------------------------------------------------
        // In-place z-score normalization: x' = (x - μ) / σ
        public static Span<float> Normalize( Span<float> data )
        {
            float mean = 0f;
            // Two-pass approach maintains numerical stability for the variance estimate
            for (int i = 0; i < data.Length; i++)
            {
                mean += data[ i ];
            }
            mean /= data.Length;

            float variance = 0f;
            for (int i = 0; i < data.Length; i++)
            {
                float diff = data[ i ] - mean;
                variance += diff * diff;
            }
            variance /= data.Length;
            // Epsilon prevents division by zero when all entries match the mean
            float std = Mathf.Sqrt( variance + 1e-8f );

            for (int i = 0; i < data.Length; i++)
            {
                data[ i ] = (data[ i ] - mean) / std;
            }

            return data;
        }
        //------------------------------------------------------------------
        public static Value Mean( ReadOnlySpan<Value> values )
        {
            if (values.Length == 0)
                throw new ArgumentException( "Mean requires at least one element", nameof( values ) );

            Value sum = 0f;
            for (int i = 0; i < values.Length; i++)
            {
                sum += values[ i ];
            }

            return sum / values.Length;
        }
        //------------------------------------------------------------------
        // Calculate mean return per environment based on how many done flags were set
        public static float MeanReturn( ReadOnlySpan<float> returns, ReadOnlySpan<float> doneFlags, int numEnvs )
        {
            if (returns.Length != doneFlags.Length)
                throw new ArgumentException( "returns and doneFlags must have equal length" );

            float totalReturn = 0f;
            int episodeEnds = 0;

            for (int i = 0; i < returns.Length; i++)
            {
                totalReturn += returns[ i ];
                if (doneFlags[ i ] > 0.5f)
                    episodeEnds++;
            }

            if (episodeEnds == 0)
                episodeEnds = 1;

            return (totalReturn / episodeEnds) / numEnvs;
        }
        //------------------------------------------------------------------
    }
}
