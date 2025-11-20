using System;

using UnityEngine;

namespace ChaosRL
{
    public static class Utils
    {
        //------------------------------------------------------------------
        /// <summary>
        /// Extracts a minibatch from a source tensor based on shuffled indices.
        /// Supports 1D and 2D tensors.
        /// </summary>
        /// <param name="source">Source tensor to extract from</param>
        /// <param name="indices">Shuffled indices array</param>
        /// <param name="start">Starting position in indices array</param>
        /// <param name="batchSize">Desired batch size</param>
        /// <returns>New tensor containing the minibatch</returns>
        public static Tensor GetMinibatch( Tensor source, int[] indices, int start, int batchSize )
        {
            int end = Math.Min( start + batchSize, indices.Length );
            int mbSize = end - start;

            // Determine result shape based on source dimensions
            int[] resultShape;
            if (source.Shape.Length == 2)
            {
                // [N, features] -> [mbSize, features]
                resultShape = new int[] { mbSize, source.Shape[ 1 ] };
            }
            else if (source.Shape.Length == 1)
            {
                // [N] -> [mbSize]
                resultShape = new int[] { mbSize };
            }
            else
            {
                throw new ArgumentException( $"GetMinibatch only supports 1D or 2D tensors, got {source.Shape.Length}D" );
            }

            var result = new Tensor( resultShape, requiresGrad: false );

            if (source.Shape.Length == 2)
            {
                int features = source.Shape[ 1 ];
                for (int i = 0; i < mbSize; i++)
                {
                    int srcIdx = indices[ start + i ];
                    for (int j = 0; j < features; j++)
                        result[ i, j ] = source[ srcIdx, j ];
                }
            }
            else // 1D
            {
                for (int i = 0; i < mbSize; i++)
                {
                    int srcIdx = indices[ start + i ];
                    result[ i ] = source[ srcIdx ];
                }
            }

            return result;
        }
        //------------------------------------------------------------------
    }
}
