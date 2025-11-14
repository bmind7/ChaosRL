using System;
using System.Collections.Generic;

namespace ChaosRL
{
    public class Value
    {
        //------------------------------------------------------------------
        public float Data;
        public HashSet<Value> Children;
        public string Name;
        public float Grad = 0;            // Accumulates gradient during backprop
        private Action _backward = null;  // Stores this node's backprop step
        //------------------------------------------------------------------
        public Value( float data, Value[] children = null, string name = "" )
        {
            this.Data = data;
            this.Children = children != null ?
                                new HashSet<Value>( children ) :
                                new HashSet<Value>();
            this.Name = name;
        }
        //------------------------------------------------------------------
        public static implicit operator Value( float f )
        {
            return new Value( f );
        }
        //------------------------------------------------------------------
        public static Value operator +( Value a, Value b )
        {
            var result = new Value( a.Data + b.Data, new Value[] { a, b }, "+" );
            result._backward = () =>
            {
                a.Grad += 1 * result.Grad;
                b.Grad += 1 * result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Value operator *( Value a, Value b )
        {
            var result = new Value( a.Data * b.Data, new Value[] { a, b }, "*" );
            result._backward = () =>
            {
                a.Grad += b.Data * result.Grad;
                b.Grad += a.Data * result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Value operator -( Value a )
        {
            return a * -1f;
        }
        //------------------------------------------------------------------
        public static Value operator -( Value a, Value b )
        {
            return a + (-b);
        }
        //------------------------------------------------------------------
        public static Value operator /( Value a, Value b )
        {
            return a * b.Pow( -1f );
        }
        //------------------------------------------------------------------
        public Value Pow( float exponent )
        {
            var result = new Value( (float)Math.Pow( this.Data, exponent ), new Value[] { this }, "^" );
            result._backward = () =>
            {
                this.Grad += exponent * (float)Math.Pow( this.Data, exponent - 1 ) * result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public Value Exp()
        {
            var result = new Value( (float)Math.Exp( this.Data ), new Value[] { this }, "exp" );
            result._backward = () =>
            {
                this.Grad += result.Data * result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public Value ReLU()
        {
            var result = new Value( this.Data < 0 ? 0 : this.Data, new Value[] { this }, "ReLU" );
            result._backward = () =>
            {
                this.Grad += (this.Data > 0 ? 1 : 0) * result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public Value Tanh()
        {
            var result = new Value( (float)Math.Tanh( this.Data ), new Value[] { this }, "tanh" );
            result._backward = () =>
            {
                // Derivative of tanh(x) is 1 - tanh²(x)
                this.Grad += (1 - result.Data * result.Data) * result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public Value Log()
        {
            var result = new Value( (float)Math.Log( this.Data ), new Value[] { this }, "log" );
            result._backward = () =>
            {
                this.Grad += (1.0f / this.Data) * result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public Value Clamp( float min, float max )
        {
            float clampedData = Math.Max( min, Math.Min( max, this.Data ) );
            var result = new Value( clampedData, new Value[] { this }, "clamp" );
            result._backward = () =>
            {
                // Pass gradient only where clamp leaves the value untouched.
                if (this.Data >= min && this.Data <= max)
                {
                    this.Grad += result.Grad;
                }
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Value Max( Value a, Value b )
        {
            var result = new Value( Math.Max( a.Data, b.Data ), new Value[] { a, b }, "max" );
            result._backward = () =>
            {
                // Send gradient to the input that produced the forward maximum; ties pick the first for determinism
                if (a.Data >= b.Data)
                    a.Grad += result.Grad;
                else
                    b.Grad += result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public static Value Min( Value a, Value b )
        {
            var result = new Value( Math.Min( a.Data, b.Data ), new Value[] { a, b }, "min" );
            result._backward = () =>
            {
                // Mirror Max: gradient flows to the input that produced the minimum; ties still pick the first
                if (a.Data <= b.Data)
                    a.Grad += result.Grad;
                else
                    b.Grad += result.Grad;
            };
            return result;
        }
        //------------------------------------------------------------------
        public void Backward()
        {
            // Backpropagation: build topological order so parents run after their children
            var topo = new List<Value>();
            var visited = new HashSet<Value>();

            void BuildTopo( Value v )
            {
                if (visited.Contains( v ))
                    return;

                visited.Add( v );
                foreach (var child in v.Children)
                {
                    BuildTopo( child );
                }
                topo.Add( v );
            }

            BuildTopo( this );

            this.Grad = 1.0f; // Seed gradient

            topo.Reverse();
            foreach (var v in topo)
            {
                v._backward?.Invoke();
            }
        }
        //------------------------------------------------------------------
        public override string ToString()
        {
            return $"Value {{ Data={this.Data:G6}, Grad={this.Grad:G6} }}";
        }
        //------------------------------------------------------------------
    }
}
