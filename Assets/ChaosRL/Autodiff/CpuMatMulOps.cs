using System;

using Unity.Collections;
using Unity.Jobs;

namespace ChaosRL
{
    /// <summary>
    /// CPU-specific MatMul orchestration for <see cref="CpuBackend"/>.
    /// Schedules Burst job graphs for transpose and matrix multiplication,
    /// auto-selecting the GEBP or naive kernel based on dimension thresholds.
    /// This is an internal implementation detail - not part of the backend abstraction.
    /// </summary>
    internal static class CpuMatMulOps
    {
        static CpuMatMulOps()
        {
            if (PackBPanelScalarParallelJob.NR != MatMulGebpScalarParallelJob.NR)
                throw new InvalidOperationException(
                    $"NR mismatch: PackBPanel.NR={PackBPanelScalarParallelJob.NR} vs GebpKernel.NR={MatMulGebpScalarParallelJob.NR}" );

            if (MatMulGebpScalarParallelJob.MR < 1)
                throw new InvalidOperationException(
                    $"MR must be >= 1, got {MatMulGebpScalarParallelJob.MR}" );
        }

        /// <summary>
        /// K-dimension block size for GEBP L1 residency.
        /// Chosen so one panel slice (KC x NR x 4 = 16 KB) + 6 A-rows (MR x KC x 4 = 6 KB)
        /// fit comfortably in L1 data cache (~32-48 KB).
        /// </summary>
        internal const int KC = 256;

        /// <summary>
        /// Minimum dimension size for using the GEBP kernel.
        /// Below this threshold the naive transpose+dot path is used.
        /// </summary>
        internal const int GebpThreshold = 16;

        //--------------------------------------------------------------
        /// <summary>
        /// Schedules a tiled transpose: Output(cols x rows) = Input(rows x cols).
        /// </summary>
        internal static JobHandle ScheduleTranspose(
            NativeArray<float> input,
            NativeArray<float> output,
            int rows,
            int cols,
            JobHandle dependsOn = default )
        {
            int tileSize = TransposeTiledParallelJob.TILE;
            int tileRows = (rows + tileSize - 1) / tileSize;
            int tileCols = (cols + tileSize - 1) / tileSize;
            int totalTiles = tileRows * tileCols;

            var job = new TransposeTiledParallelJob
            {
                Input = input,
                Output = output,
                Rows = rows,
                Cols = cols
            };
            return job.Schedule( totalTiles, CpuBackend.GetBatchSize( totalTiles ), dependsOn );
        }
        //--------------------------------------------------------------
        /// <summary>
        /// Schedules a MatMul: C(m x n) = A(m x k) @ B(k x n).
        /// B is in original row-major layout.
        /// Automatically picks GEBP (large) or Naive (small) kernel.
        /// Any temporary buffers are auto-disposed when the returned handle completes.
        /// </summary>
        internal static JobHandle ScheduleMatMul(
            NativeArray<float> a,
            NativeArray<float> b,
            NativeArray<float> c,
            int m,
            int k,
            int n,
            bool accumulate,
            JobHandle dependsOn = default )
        {
            if (m >= GebpThreshold && k >= GebpThreshold && n >= GebpThreshold)
                return ScheduleGebpMatMul( a, b, c, m, k, n, accumulate, dependsOn );

            // Small path: transpose B internally, run naive, auto-dispose temp.
            var bt = new NativeArray<float>( k * n, Allocator.TempJob );
            var th = ScheduleTranspose( b, bt, k, n, dependsOn );
            var mmh = ScheduleNaiveMatMul( a, bt, c, m, k, n, accumulate, th );
            return bt.Dispose( mmh );
        }
        //--------------------------------------------------------------
        /// <summary>
        /// Schedules a naive MatMul using pre-transposed B:
        /// C(m x n) = A(m x k) @ BT^T, where BT is (n x k).
        /// </summary>
        private static JobHandle ScheduleNaiveMatMul(
            NativeArray<float> a,
            NativeArray<float> bt,
            NativeArray<float> c,
            int m,
            int k,
            int n,
            bool accumulate,
            JobHandle dependsOn = default )
        {
            int totalElements = m * n;
            var job = new MatMulNaiveParallelJob
            {
                A = a,
                BT = bt,
                C = c,
                M = m,
                K = k,
                N = n,
                Accumulate = accumulate
            };
            return job.Schedule( totalElements, CpuBackend.GetBatchSize( totalElements ), dependsOn );
        }
        //--------------------------------------------------------------
        /// <summary>
        /// Schedules a Kc-blocked GEBP MatMul: C(m x n) = A(m x k) @ B(k x n).
        /// B must be in original row-major layout (NOT transposed).
        /// Packs B into column-panel layout in Kc-thick slices for L1 residency,
        /// then runs GEBP micro-kernels per slice.
        /// The packed buffer is auto-disposed when the returned handle completes.
        /// </summary>
        private static JobHandle ScheduleGebpMatMul(
            NativeArray<float> a,
            NativeArray<float> b,
            NativeArray<float> c,
            int m,
            int k,
            int n,
            bool accumulate,
            JobHandle dependsOn = default )
        {
            const int NR = PackBPanelScalarParallelJob.NR;
            int numPanels = (n + NR - 1) / NR;

            int maxKc = Math.Min( KC, k );
            int packedSize = numPanels * maxKc * NR;
            var packedB = new NativeArray<float>( packedSize, Allocator.TempJob );

            int rowGroups = (m + MatMulGebpScalarParallelJob.MR - 1) / MatMulGebpScalarParallelJob.MR;
            int packBatch = CpuBackend.GetBatchSize( numPanels );
            int gebpBatch = CpuBackend.GetBatchSize( rowGroups );

            JobHandle prevHandle = dependsOn;

            for (int kb = 0; kb < k; kb += KC)
            {
                int thisKc = Math.Min( KC, k - kb );
                bool isFirstBlock = (kb == 0);

                var packJob = new PackBPanelScalarParallelJob
                {
                    B = b,
                    PackedB = packedB,
                    N = n,
                    KOffset = kb,
                    Kc = thisKc
                };
                var packHandle = packJob.Schedule( numPanels, packBatch, prevHandle );

                var gebpJob = new MatMulGebpScalarParallelJob
                {
                    A = a,
                    PackedB = packedB,
                    C = c,
                    M = m,
                    K = k,
                    N = n,
                    KOffset = kb,
                    Kc = thisKc,
                    Accumulate = isFirstBlock ? accumulate : true
                };
                prevHandle = gebpJob.Schedule( rowGroups, gebpBatch, packHandle );
            }

            return packedB.Dispose( prevHandle );
        }
        //--------------------------------------------------------------
    }
}