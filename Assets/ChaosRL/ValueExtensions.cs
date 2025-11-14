using System.Text;

namespace ChaosRL
{
    public static class ValueExtensions
    {
        //------------------------------------------------------------------
        // Prints a Value[,] in a readable matrix-like format.
        // Note: calling arr.ToString() will NOT use this due to Object.ToString().
        // ToString -> data-only matrix (just Value.Data)
        public static string ToStringData( this Value[,] self )
        {
            return self.ToMatrixString( full: false );
        }
        //------------------------------------------------------------------
        public static string ToStringFull( this Value[,] self )
        {
            return self.ToMatrixString( full: true );
        }
        //------------------------------------------------------------------
        public static string ToMatrixString( this Value[,] self, bool full = false, string colSeparator = ", ", string rowSeparator = "\n" )
        {
            if (self == null) return "null";

            int rows = self.GetLength( 0 );
            int cols = self.GetLength( 1 );

            var sb = new StringBuilder();
            sb.Append( '[' );
            for (int r = 0; r < rows; r++)
            {
                sb.Append( '[' );
                for (int c = 0; c < cols; c++)
                {
                    var v = self[ r, c ];
                    if (v != null)
                    {
                        sb.Append( full ? v.ToString() : v.Data.ToString() );
                    }
                    else
                    {
                        sb.Append( "null" );
                    }
                    if (c < cols - 1) sb.Append( colSeparator );
                }
                sb.Append( ']' );
                if (r < rows - 1) sb.Append( rowSeparator );
            }
            sb.Append( ']' );
            return sb.ToString();
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Converts a float array to a Value array.
        /// Each float becomes a Value with that float as its Data property.
        /// </summary>
        /// <param name="floats">The float array to convert</param>
        /// <returns>A new Value array</returns>
        public static Value[] ToValues( this float[] floats )
        {
            if (floats == null) return null;

            var values = new Value[ floats.Length ];
            for (int i = 0; i < floats.Length; i++)
            {
                values[ i ] = new Value( floats[ i ] );
            }
            return values;
        }
        //------------------------------------------------------------------
        /// <summary>
        /// Converts a Value array to a float array.
        /// Each Value's Data property becomes a float in the result array.
        /// </summary>
        /// <param name="values">The Value array to convert</param>
        /// <returns>A new float array</returns>
        public static float[] ToFloats( this Value[] values )
        {
            if (values == null) return null;

            var floats = new float[ values.Length ];
            for (int i = 0; i < values.Length; i++)
            {
                floats[ i ] = values[ i ]?.Data ?? 0f; // Use 0f if Value is null
            }
            return floats;
        }
        //------------------------------------------------------------------
    }
}
