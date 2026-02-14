namespace ChaosRL
{
    /// <summary>
    /// Identifies which compute device a tensor's storage resides on.
    /// Used by <see cref="Tensor"/> to dispatch operations to the correct backend.
    /// </summary>
    public enum TensorDevice
    {
        /// <summary>CPU-side NativeArray storage with Burst job compute.</summary>
        CPU = 0,

        /// <summary>GPU-side ComputeBuffer storage with compute shader dispatch.</summary>
        GPU = 1,
    }
}
