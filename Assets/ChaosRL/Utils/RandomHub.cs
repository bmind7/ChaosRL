using System;

namespace ChaosRL
{
    // Centralized, seedable RNG for the project.
    // Thread-safe via a simple lock as System.Random isn't thread-safe.
    public static class RandomHub
    {
        //------------------------------------------------------------------

        private static readonly object _lock = new object();
        private static Random _rng = new Random();

        // Set a deterministic seed for all randomness routed through RandomHub.
        //------------------------------------------------------------------
        public static void SetSeed( int seed )
        {
            lock (_lock)
            {
                _rng = new Random( seed );
            }
        }
        //------------------------------------------------------------------
        // Returns a double in [0, 1)
        public static double NextDouble()
        {
            lock (_lock)
            {
                return _rng.NextDouble();
            }
        }
        //------------------------------------------------------------------
        // Returns a float in [minInclusive, maxExclusive)
        public static float NextFloat( float minInclusive, float maxExclusive )
        {
            return (float)(minInclusive + (maxExclusive - minInclusive) * NextDouble());
        }
        //------------------------------------------------------------------
        // Returns an int in [minInclusive, maxExclusive)
        public static int NextInt( int minInclusive, int maxExclusive )
        {
            lock (_lock)
            {
                return _rng.Next( minInclusive, maxExclusive );
            }
        }
        //------------------------------------------------------------------
        // Shuffles an array in-place using Fisher-Yates algorithm
        public static void Shuffle<T>( T[] array )
        {
            for (int i = array.Length - 1; i > 0; i--)
            {
                int j = NextInt( 0, i + 1 );
                T temp = array[ i ];
                array[ i ] = array[ j ];
                array[ j ] = temp;
            }
        }
        //------------------------------------------------------------------
    }
}
