using System;
using System.Linq;

using UnityEngine;

namespace ChaosRL
{
    public class AcademyTensor : MonoBehaviour
    {
        //------------------------------------------------------------------
        public static AcademyTensor Instance { get; private set; }

        public int NumEnvs => _numEnvs;

        // --- Standard PPO implementation config with continuous action spaces
        // https://arxiv.org/abs/1707.06347
        [Header( "PPO Config" )]
        [SerializeField] private int _numEnvs = 1;
        [SerializeField] private int _updateEpochs = 3;
        [SerializeField] private int _minibatchSize = 1024;
        [SerializeField] private int _approxBufferSize = 10240;
        [SerializeField] private float _gamma = 0.95f;
        [SerializeField] private float _gaeLambda = 0.95f;
        [SerializeField] private float _clipCoef = 0.2f;
        [SerializeField] private float _betaCoef = 0.01f;     // set to 0 to disable entropy bonus
        [SerializeField] private float _vCoef = 0.5f;
        [SerializeField] private float _learningRate = 2e-5f;
        [SerializeField] private float _l2Coef = 1e-4f;
        [SerializeField] private int[] _hiddenLayers = new int[] { 64, 64 };
        // Offset added to log-std to output to avoid too high std values
        [SerializeField] private float _logStdOffset = 0.6f;

        [Header( "Agent Config" )]
        [SerializeField] private int _inputSize = 7;
        [SerializeField] private int _actionSize = 2;

        // --- Neural Networks and Optimizers ---
        private MLPTensor _policyNetwork;
        private MLPTensor _valueNetwork;
        private AdamOptimizerTensor _optimizer;
        private L2RegularizerTensor _l2Regularizer;

        // --- Experience Replay Buffers ---
        private int _bufferSize;
        private int _currentStep = 0;
        private int _bufferIdx = 0;
        private Tensor _observationBuffer;
        private Tensor _valBuffer;
        private Tensor _actionBuffer;
        private Tensor _logProbBuffer;
        private Tensor _doneBuffer;
        private Tensor _rewardBuffer;
        //------------------------------------------------------------------
        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy( this.gameObject );
            }
            else
            {
                Instance = this;
                DontDestroyOnLoad( this.gameObject );
            }

            Debug.Log( "Academy started." );

            // --- To insure consisten simullation even durin frame drops while Networks are updating
            Application.targetFrameRate = 60;
            Time.captureFramerate = 60;

            // --- Policy outputs mean and log-std for each action dimension in a single head
            var layerSizes = _hiddenLayers.Append( _actionSize * 2 ).ToArray();
            _policyNetwork = new MLPTensor( _inputSize, layerSizes, lastLayerNonLin: true );

            // --- Value network predicts a single scalar state value
            layerSizes = _hiddenLayers.Append( 1 ).ToArray();
            _valueNetwork = new MLPTensor( _inputSize, layerSizes );
            _optimizer = new AdamOptimizerTensor( new[]
                {
                    _policyNetwork.Parameters,
                    _valueNetwork.Parameters,
                } );

            _l2Regularizer = new L2RegularizerTensor( new[]
                {
                    _policyNetwork.Parameters,
                    _valueNetwork.Parameters,
                } );

            // --- Experience Replay Buffers ---
            _bufferSize = (_approxBufferSize / _numEnvs);

            _observationBuffer = new Tensor( new[] { _bufferSize, _numEnvs, _inputSize }, requiresGrad: false );
            _valBuffer = new Tensor( new[] { _bufferSize, _numEnvs }, requiresGrad: false );
            _actionBuffer = new Tensor( new[] { _bufferSize, _numEnvs, _actionSize }, requiresGrad: false );
            _logProbBuffer = new Tensor( new[] { _bufferSize, _numEnvs }, requiresGrad: false );
            _doneBuffer = new Tensor( new[] { _bufferSize, _numEnvs }, requiresGrad: false );
            _rewardBuffer = new Tensor( new[] { _bufferSize, _numEnvs }, requiresGrad: false );
        }
        //------------------------------------------------------------------
        public float[] RequestDecision( int agentIdx, float[] observation, bool done, float reward )
        {
            var obs = new Tensor( new[] { 1, observation.Length }, observation, requiresGrad: false );
            var policyOutput = _policyNetwork.Forward( obs );
            var val = _valueNetwork.Forward( obs );
            var outputActions = new float[ _actionSize ];

            // Sample Action Policy (Gaussian policy): 
            // mean from first half, log-std from second half of network output

            var mean = policyOutput.Slice( 1, 0, _actionSize );
            var logStd = policyOutput.Slice( 1, _actionSize, _actionSize );
            var std = (logStd + _logStdOffset).Exp();
            var dist = new DistTensor( mean, std );
            var action = dist.Sample();

            // Clipped action only send to environment
            for (int i = 0; i < _actionSize; i++)
            {
                // always take first raw because batch size is 1
                outputActions[ i ] = Mathf.Clamp( action[ 0, i ], -1f, 1f );
                _actionBuffer[ _bufferIdx, agentIdx, i ] = action[ 0, i ];
            }

            _logProbBuffer[ _bufferIdx, agentIdx ] = dist.LogProb( action ).Sum().Data[ 0 ];

            for (int i = 0; i < _inputSize; i++)
                _observationBuffer[ _bufferIdx, agentIdx, i ] = observation[ i ];

            _valBuffer[ _bufferIdx, agentIdx ] = val[ 0, 0 ];
            _doneBuffer[ _bufferIdx, agentIdx ] = done ? 1f : 0f;
            _rewardBuffer[ _bufferIdx, agentIdx ] = reward;


            // Update networks if buffer is full or step limit reached
            if (_bufferIdx == _bufferSize - 1 && _currentStep % _numEnvs == _numEnvs - 1)
                UpdateNetworks();

            _currentStep++;
            // Advance buffer index in ring-buffer fashion over time and across envs
            _bufferIdx = (_currentStep / _numEnvs) % _bufferSize;

            return outputActions;
        }
        //------------------------------------------------------------------
        public (Tensor advantages, Tensor returns) ComputeGAE()
        {
            int T = _bufferSize;    // time steps per env
            int N = _numEnvs;       // number of envs

            var advantages = new Tensor( new[] { T, N }, requiresGrad: false );
            var returns = new Tensor( new[] { T, N }, requiresGrad: false );

            // Maintain separate GAE accumulators per environment to avoid mixing signals between envs
            for (int env = 0; env < N; env++)
            {
                float gae = 0f;

                // Start from the element before last so the last slot is only used for bootstrapping
                for (int t = T - 2; t >= 0; t--)
                {
                    float reward = _rewardBuffer[ t, env ];
                    float value = _valBuffer[ t, env ];
                    float done = _doneBuffer[ t, env ];
                    float nonTerminal = 1f - done;
                    float nextValue = _valBuffer[ t + 1, env ];

                    float delta = reward + _gamma * nonTerminal * nextValue - value;
                    gae = delta + _gamma * _gaeLambda * nonTerminal * gae;

                    advantages[ t, env ] = gae;
                    // Removes predicted value from advantages to get full return from environment
                    returns[ t, env ] = gae + value;
                }
            }

            return (advantages, returns);
        }
        //------------------------------------------------------------------
        public void UpdateNetworks()
        {
            (Tensor advantages, Tensor returns) = ComputeGAE();

            // Excluding the last row from experience buffer (which is only used for bootstrapping)
            int trimBatchSize = _bufferSize - 1;
            int totalElements = trimBatchSize * _numEnvs;

            // Flatten time and env dimensions for minibatch sampling (just slice - data already contiguous)
            var batch_observations = _observationBuffer.Slice( 0, 0, trimBatchSize ).Reshape( new[] { totalElements, _inputSize } );
            var batch_actions = _actionBuffer.Slice( 0, 0, trimBatchSize ).Reshape( new[] { totalElements, _actionSize } );
            var batch_log_probs = _logProbBuffer.Slice( 0, 0, trimBatchSize ).Reshape( new[] { totalElements, 1 } );
            var batch_advantages = advantages.Slice( 0, 0, trimBatchSize ).Reshape( new[] { totalElements, 1 } );
            var batch_returns = returns.Slice( 0, 0, trimBatchSize ).Reshape( new[] { totalElements, 1 } );
            var batch_doneFlags = _doneBuffer.Slice( 0, 0, trimBatchSize ).Reshape( new[] { totalElements, 1 } );






            // // Normalize advantages to keep the policy gradient scale well behaved
            // Span<float> advantages_span = batch_advantages.Data;
            // advantages_span = Utils.Normalize( advantages_span );

            batch_advantages = batch_advantages.Normalize().Reshape( new[] { totalElements, 1 } );






            int[] indices = Enumerable.Range( 0, totalElements ).ToArray();

            float averageLoss = 0f, totalPgLossSum = 0f, totalEntropyLossSum = 0f, totalValueLossSum = 0f;
            int batchCount = 0;

            for (int epoch = 0; epoch < _updateEpochs; epoch++)
            {
                // Make sure we sample random indices for each epoch
                RandomHub.Shuffle( indices );

                for (int start = 0; start < totalElements; start += _minibatchSize)
                {
                    // Extract minibatch using helper
                    var mb_observations = UtilsTensor.GetMinibatch( batch_observations, indices, start, _minibatchSize );
                    var mb_actions = UtilsTensor.GetMinibatch( batch_actions, indices, start, _minibatchSize );
                    var mb_log_probs = UtilsTensor.GetMinibatch( batch_log_probs, indices, start, _minibatchSize );
                    var mb_advantages = UtilsTensor.GetMinibatch( batch_advantages, indices, start, _minibatchSize );
                    var mb_returns = UtilsTensor.GetMinibatch( batch_returns, indices, start, _minibatchSize );

                    int mbSize = mb_observations.Shape[ 0 ];

                    // Forward pass through networks for entire minibatch
                    var mb_policyOutput = _policyNetwork.Forward( mb_observations );
                    var mb_values = _valueNetwork.Forward( mb_observations );

                    // Extract mean and log-std from policy output
                    var mb_mean = mb_policyOutput.Slice( 1, 0, _actionSize );
                    var mb_logStd = mb_policyOutput.Slice( 1, _actionSize, _actionSize );
                    var mb_std = (mb_logStd + _logStdOffset).Exp();

                    // Create distribution and compute log probabilities
                    var mb_dist = new DistTensor( mb_mean, mb_std );
                    var mb_newLogProbs = mb_dist.LogProb( mb_actions ).Sum( 1 );
                    var mb_entropy = mb_dist.Entropy().Mean( 1 );

                    // Importance sampling ratio
                    var mb_ratio = (mb_newLogProbs - mb_log_probs).Exp();

                    // PPO clipped objective
                    var mb_pgLoss1 = -mb_advantages * mb_ratio;
                    var mb_pgLoss2 = -mb_advantages * mb_ratio.Clamp( 1f - _clipCoef, 1f + _clipCoef );
                    var mb_pgLoss = Tensor.Max( mb_pgLoss1, mb_pgLoss2 );

                    // Value loss
                    var mb_valueLoss = ((mb_values.Squeeze() - mb_returns).Pow( 2 )) * 0.5f;

                    // Compute mean losses
                    var totalPgLoss = mb_pgLoss.Mean();
                    var totalEntropyLoss = mb_entropy.Mean();
                    var totalValueLoss = mb_valueLoss.Mean();
                    var l2Penalty = _l2Regularizer.Compute( _l2Coef );

                    // Accumulate loss statistics
                    totalPgLossSum += totalPgLoss.Data[ 0 ];
                    totalEntropyLossSum += totalEntropyLoss.Data[ 0 ];
                    totalValueLossSum += totalValueLoss.Data[ 0 ];
                    batchCount++;

                    // Standard PPO loss: policy term + value term + entropy bonus + weight decay
                    var totalLoss = totalPgLoss - _betaCoef * totalEntropyLoss + _vCoef * totalValueLoss + l2Penalty;
                    averageLoss += totalLoss.Data[ 0 ];

                    _policyNetwork.ZeroGrad();
                    _valueNetwork.ZeroGrad();

                    totalLoss.Backward();

                    _optimizer.Step( _learningRate );
                }
            }
            averageLoss /= batchCount;
            float meanPgLoss = totalPgLossSum / batchCount;
            float meanEntropyLoss = totalEntropyLossSum / batchCount;
            float meanValueLoss = totalValueLossSum / batchCount;

            float meanReturn = Utils.MeanReturn( batch_returns.Data, batch_doneFlags.Data, _numEnvs );
            Debug.Log( $"Step count: {_currentStep}, Mean Return: {meanReturn}, Average Loss: {averageLoss}, PG Loss: {meanPgLoss}, Entropy Loss: {meanEntropyLoss}, Value Loss: {meanValueLoss}" );
        }
        //------------------------------------------------------------------
    }
}
