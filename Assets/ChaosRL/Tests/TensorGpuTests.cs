using System;

using NUnit.Framework;

using UnityEngine;

namespace ChaosRL.Tests
{
    public class TensorGpuTests : TensorScopedTestBase
    {
        //------------------------------------------------------------------
        [SetUp]
        public void EnsureGpuBackend()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore( "Compute shaders are not supported on this platform." );

            Tensor.GpuBackend ??= new GpuBackend();
        }
        //------------------------------------------------------------------
        [Test]
        public void Constructor_GpuTensor_CreatesCorrectly()
        {
            var tensor = new Tensor( new[] { 2, 3 }, device: TensorDevice.GPU );

            Assert.That( tensor.Device, Is.EqualTo( TensorDevice.GPU ) );
            Assert.That( tensor.Shape, Is.EqualTo( new[] { 2, 3 } ) );
            Assert.That( tensor.Size, Is.EqualTo( 6 ) );
            Assert.That( tensor.Data.Length, Is.EqualTo( 6 ) );
            Assert.That( tensor.Grad.Length, Is.EqualTo( 6 ) );
            Assert.That( tensor.Data.IsCreated, Is.True );
            Assert.That( tensor.Grad.IsCreated, Is.True );
        }
        //------------------------------------------------------------------
        [Test]
        public void Add_GpuTensors_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2 }, new[] { 2.0f, 3.0f }, device: TensorDevice.GPU );
            var b = new Tensor( new[] { 2 }, new[] { 3.5f, 1.5f }, device: TensorDevice.GPU );

            var c = a + b;

            Assert.That( c.Device, Is.EqualTo( TensorDevice.GPU ) );
            Assert.That( c.Data[ 0 ], Is.EqualTo( 5.5f ).Within( 1e-5 ) );
            Assert.That( c.Data[ 1 ], Is.EqualTo( 4.5f ).Within( 1e-5 ) );

            c.Backward();
            Assert.That( c.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-5 ) );
            Assert.That( c.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-5 ) );
            Assert.That( a.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-5 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-5 ) );
            Assert.That( b.Grad[ 0 ], Is.EqualTo( 1.0f ).Within( 1e-5 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 1.0f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Mul_GpuScalarBroadcast_ComputesForwardAndBackward()
        {
            var tensor = new Tensor( new[] { 3 }, new[] { 2f, 3f, 4f }, device: TensorDevice.GPU );
            var scalar = new Tensor( new[] { 1 }, new[] { 2f }, device: TensorDevice.GPU );

            var result = tensor * scalar;

            Assert.That( result.Device, Is.EqualTo( TensorDevice.GPU ) );
            Assert.That( result.Data[ 0 ], Is.EqualTo( 4f ).Within( 1e-5 ) );
            Assert.That( result.Data[ 1 ], Is.EqualTo( 6f ).Within( 1e-5 ) );
            Assert.That( result.Data[ 2 ], Is.EqualTo( 8f ).Within( 1e-5 ) );

            result.Backward();

            Assert.That( tensor.Grad[ 0 ], Is.EqualTo( 2f ).Within( 1e-5 ) );
            Assert.That( tensor.Grad[ 1 ], Is.EqualTo( 2f ).Within( 1e-5 ) );
            Assert.That( tensor.Grad[ 2 ], Is.EqualTo( 2f ).Within( 1e-5 ) );
            Assert.That( scalar.Grad[ 0 ], Is.EqualTo( 9f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Unary_Gpu_ReLU_Tanh_Log_Exp_ComputesCorrectly()
        {
            var input = new Tensor( new[] { 3 }, new[] { -1f, 0.5f, 2f }, device: TensorDevice.GPU );

            var relu = input.ReLU();
            Assert.That( relu.Data[ 0 ], Is.EqualTo( 0f ).Within( 1e-5 ) );
            Assert.That( relu.Data[ 1 ], Is.EqualTo( 0.5f ).Within( 1e-5 ) );
            Assert.That( relu.Data[ 2 ], Is.EqualTo( 2f ).Within( 1e-5 ) );

            var tanh = input.Tanh();
            Assert.That( tanh.Data[ 0 ], Is.EqualTo( (float)Math.Tanh( -1f ) ).Within( 1e-5 ) );
            Assert.That( tanh.Data[ 1 ], Is.EqualTo( (float)Math.Tanh( 0.5f ) ).Within( 1e-5 ) );
            Assert.That( tanh.Data[ 2 ], Is.EqualTo( (float)Math.Tanh( 2f ) ).Within( 1e-5 ) );

            var positive = new Tensor( new[] { 3 }, new[] { 1f, 2f, 3f }, device: TensorDevice.GPU );
            var log = positive.Log();
            Assert.That( log.Data[ 0 ], Is.EqualTo( 0f ).Within( 1e-5 ) );
            Assert.That( log.Data[ 1 ], Is.EqualTo( (float)Math.Log( 2f ) ).Within( 1e-5 ) );
            Assert.That( log.Data[ 2 ], Is.EqualTo( (float)Math.Log( 3f ) ).Within( 1e-5 ) );

            var exp = positive.Exp();
            Assert.That( exp.Data[ 0 ], Is.EqualTo( (float)Math.Exp( 1f ) ).Within( 1e-4 ) );
            Assert.That( exp.Data[ 1 ], Is.EqualTo( (float)Math.Exp( 2f ) ).Within( 1e-4 ) );
            Assert.That( exp.Data[ 2 ], Is.EqualTo( (float)Math.Exp( 3f ) ).Within( 1e-4 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void MatMul_GpuTensors_ComputesForwardAndBackward()
        {
            var a = new Tensor( new[] { 2, 3 }, new[]
            {
                1f, 2f, 3f,
                4f, 5f, 6f
            }, device: TensorDevice.GPU );

            var b = new Tensor( new[] { 3, 2 }, new[]
            {
                7f, 8f,
                9f, 10f,
                11f, 12f
            }, device: TensorDevice.GPU );

            var c = a.MatMul( b );

            Assert.That( c.Device, Is.EqualTo( TensorDevice.GPU ) );
            Assert.That( c.Shape, Is.EqualTo( new[] { 2, 2 } ) );
            Assert.That( c.Data[ 0 ], Is.EqualTo( 58f ).Within( 1e-4 ) );
            Assert.That( c.Data[ 1 ], Is.EqualTo( 64f ).Within( 1e-4 ) );
            Assert.That( c.Data[ 2 ], Is.EqualTo( 139f ).Within( 1e-4 ) );
            Assert.That( c.Data[ 3 ], Is.EqualTo( 154f ).Within( 1e-4 ) );

            var loss = c.Sum();
            loss.Backward();

            Assert.That( a.Grad[ 0 ], Is.EqualTo( 15f ).Within( 1e-4 ) );
            Assert.That( a.Grad[ 1 ], Is.EqualTo( 19f ).Within( 1e-4 ) );
            Assert.That( a.Grad[ 2 ], Is.EqualTo( 23f ).Within( 1e-4 ) );
            Assert.That( a.Grad[ 3 ], Is.EqualTo( 15f ).Within( 1e-4 ) );
            Assert.That( a.Grad[ 4 ], Is.EqualTo( 19f ).Within( 1e-4 ) );
            Assert.That( a.Grad[ 5 ], Is.EqualTo( 23f ).Within( 1e-4 ) );

            Assert.That( b.Grad[ 0 ], Is.EqualTo( 5f ).Within( 1e-4 ) );
            Assert.That( b.Grad[ 1 ], Is.EqualTo( 5f ).Within( 1e-4 ) );
            Assert.That( b.Grad[ 2 ], Is.EqualTo( 7f ).Within( 1e-4 ) );
            Assert.That( b.Grad[ 3 ], Is.EqualTo( 7f ).Within( 1e-4 ) );
            Assert.That( b.Grad[ 4 ], Is.EqualTo( 9f ).Within( 1e-4 ) );
            Assert.That( b.Grad[ 5 ], Is.EqualTo( 9f ).Within( 1e-4 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void Sum_GpuTensor_BackwardBroadcastsGradient()
        {
            var tensor = new Tensor( new[] { 4 }, new[] { 1f, 2f, 3f, 4f }, device: TensorDevice.GPU );

            var sum = tensor.Sum();
            Assert.That( sum.Data[ 0 ], Is.EqualTo( 10f ).Within( 1e-5 ) );

            sum.Backward();

            for (int i = 0; i < 4; i++)
                Assert.That( tensor.Grad[ i ], Is.EqualTo( 1f ).Within( 1e-5 ) );
        }
        //------------------------------------------------------------------
        [Test]
        public void ToGpu_ToCpu_RoundTrip_PreservesDataAndGrad()
        {
            var cpu = new Tensor( new[] { 2, 2 }, new[] { 1f, 2f, 3f, 4f }, device: TensorDevice.CPU );
            cpu.Grad[ 0 ] = 0.1f;
            cpu.Grad[ 1 ] = 0.2f;
            cpu.Grad[ 2 ] = 0.3f;
            cpu.Grad[ 3 ] = 0.4f;

            var gpu = cpu.ToGpu();
            Assert.That( gpu.Device, Is.EqualTo( TensorDevice.GPU ) );

            var back = gpu.ToCpu();
            Assert.That( back.Device, Is.EqualTo( TensorDevice.CPU ) );

            Assert.That( back.Data[ 0 ], Is.EqualTo( 1f ).Within( 1e-6 ) );
            Assert.That( back.Data[ 1 ], Is.EqualTo( 2f ).Within( 1e-6 ) );
            Assert.That( back.Data[ 2 ], Is.EqualTo( 3f ).Within( 1e-6 ) );
            Assert.That( back.Data[ 3 ], Is.EqualTo( 4f ).Within( 1e-6 ) );

            Assert.That( back.Grad[ 0 ], Is.EqualTo( 0.1f ).Within( 1e-6 ) );
            Assert.That( back.Grad[ 1 ], Is.EqualTo( 0.2f ).Within( 1e-6 ) );
            Assert.That( back.Grad[ 2 ], Is.EqualTo( 0.3f ).Within( 1e-6 ) );
            Assert.That( back.Grad[ 3 ], Is.EqualTo( 0.4f ).Within( 1e-6 ) );
        }
        //------------------------------------------------------------------
    }
}
