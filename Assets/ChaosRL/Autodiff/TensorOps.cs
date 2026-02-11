using System;

using Unity.Collections;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;

namespace ChaosRL
{
    /// <summary>
    /// Static helpers that schedule Burst job graphs for linear-algebra operations.
    /// Keeps scheduling orchestration separate from job definitions (TensorJobs.cs)
    /// and autograd logic (Tensor.cs).
    /// </summary>
    public static class TensorOps
    {
        /// <summary>
        /// K-dimension block size for GEBP L1 residency.
        /// Chosen so one panel slice (KC×NR×4 = 16 KB) + 6 A-rows (MR×KC×4 = 6 KB)
        /// fit comfortably in L1 data cache (~32-48 KB).
        /// </summary>
        public const int KC = 256;

        /// <summary>
        /// Minimum dimension size for using the GEBP kernel.
        /// Below this threshold the naive transpose+dot path is used.
        /// </summary>
        public const int GebpThreshold = 16;

        //--------------------------------------------------------------
        public static int GetBatchSize( int totalWorkItems )
        {
            int workerCount = Math.Max( 1, JobsUtility.JobWorkerCount + 1 );
            int batch = totalWorkItems / (workerCount * 4);
            return Math.Max( 1, batch );
        }
        //--------------------------------------------------------------
        /// <summary>
        /// Schedules a tiled transpose: Output(cols×rows) = Input(rows×cols).
        /// </summary>
        public static JobHandle ScheduleTranspose(
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
            return job.Schedule( totalTiles, GetBatchSize( totalTiles ), dependsOn );
        }
        //--------------------------------------------------------------
        /// <summary>
        /// Schedules a MatMul: C(m×n) = A(m×k) @ B(k×n).
        /// B is in original row-major layout.
        /// Automatically picks GEBP (large) or Naive (small) kernel.
        /// Any temporary buffers are auto-disposed when the returned handle completes.
        /// </summary>
        public static JobHandle ScheduleMatMul(
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
        /// C(m×n) = A(m×k) @ BT^T, where BT is (n×k).
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
            return job.Schedule( totalElements, GetBatchSize( totalElements ), dependsOn );
        }
        //--------------------------------------------------------------
        /// <summary>
        /// Schedules a Kc-blocked GEBP MatMul: C(m×n) = A(m×k) @ B(k×n).
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
            int packBatch = GetBatchSize( numPanels );
            int gebpBatch = GetBatchSize( rowGroups );

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
