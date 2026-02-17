using System;

namespace ChaosRL
{
    public static class Utils
    {
        //------------------------------------------------------------------
        /// <summary>
        /// Extracts a minibatch from a source tensor based on shuffled indices.
        /// Supports 1D and 2D tensors. Delegates element copying to the tensor's backend
        /// so the implementation stays device-agnostic.
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
            int featureSize;
            if (source.Shape.Length == 2)
            {
                // [N, features] -> [mbSize, features]
                featureSize = source.Shape[ 1 ];
                resultShape = new int[] { mbSize, featureSize };
            }
            else if (source.Shape.Length == 1)
            {
                // [N] -> [mbSize]
                featureSize = 1;
                resultShape = new int[] { mbSize };
            }
            else
            {
                throw new ArgumentException( $"GetMinibatch only supports 1D or 2D tensors, got {source.Shape.Length}D" );
            }

            var result = new Tensor( resultShape, requiresGrad: false, device: source.Device );

            // Gather rows via backend - device-agnostic
            source.Backend.Gather( source.Data, result.Data, indices, start, mbSize, featureSize );

            return result;
        }
        //------------------------------------------------------------------
    }
}
