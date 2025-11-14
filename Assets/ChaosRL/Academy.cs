using System;
using System.Linq;
using System.Runtime.InteropServices;

using UnityEngine;

namespace ChaosRL
{
    public class Academy : MonoBehaviour
    {
        //------------------------------------------------------------------
        public static Academy Instance { get; private set; }

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
        private MLP _policyNetwork;
        private MLP _valueNetwork;
        private AdamOptimizer _optimizer;
        private L2Regularizer _l2Regularizer;

        // --- Experience Replay Buffers ---
        private int _bufferSize;
        private int _currentStep = 0;
        private int _bufferIdx = 0;
        private float[,,] _observationBuffer;
        private float[,] _valBuffer;
        private float[,,] _actionBuffer;
        private float[,] _logProbBuffer;
        private float[,] _doneBuffer;
        private float[,] _rewardBuffer;

        // --- Temporary arrays to avoid repeated allocations during Network Update---
        private Value[] _pgLoss;
        private Value[] _entropyLoss;
        private Value[] _valueLoss;
        private Value[] _obsArray;
        private Dist[] _distArray;
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
            _policyNetwork = new MLP( _inputSize, layerSizes, lastLayerNonLin: true );

            // --- Value network predicts a single scalar state value
            layerSizes = _hiddenLayers.Append( 1 ).ToArray();
            _valueNetwork = new MLP( _inputSize, layerSizes );

            _optimizer = new AdamOptimizer( new[]
                {
                    _policyNetwork.Parameters,
                    _valueNetwork.Parameters,
                } );

            _l2Regularizer = new L2Regularizer( new[]
                {
                    _policyNetwork.Parameters,
                    _valueNetwork.Parameters,
                } );


            // --- Experience Replay Buffers ---
            _bufferSize = (_approxBufferSize / _numEnvs);
            _observationBuffer = new float[ _bufferSize, _numEnvs, _inputSize ];
            _valBuffer = new float[ _bufferSize, _numEnvs ];
            _actionBuffer = new float[ _bufferSize, _numEnvs, _actionSize ];
            _logProbBuffer = new float[ _bufferSize, _numEnvs ];
            _doneBuffer = new float[ _bufferSize, _numEnvs ];
            _rewardBuffer = new float[ _bufferSize, _numEnvs ];

            // --- Network update temporary arrays ---
            int trimmedBatchSize = Math.Max( 1, _bufferSize - 1 );
            int totalElements = trimmedBatchSize * _numEnvs;

            _pgLoss = new Value[ totalElements ];
            _entropyLoss = new Value[ totalElements ];
            _valueLoss = new Value[ totalElements ];

            _obsArray = new Value[ _inputSize ];
            _distArray = new Dist[ _actionSize ];
        }
        //------------------------------------------------------------------
        public float[] RequestDecision( int agentIdx, float[] observation, bool done, float reward )
        {
            var obs = observation.ToValues();
            var policyOutput = _policyNetwork.Forward( obs );
            var val = _valueNetwork.Forward( obs );
            var outputActions = new float[ _actionSize ];

            // Avoids invinite accumulation of log-probs over time
            _logProbBuffer[ _bufferIdx, agentIdx ] = 0f;
            // Sample Action Policy (Gaussian policy): 
            // mean from first half, log-std from second half of network output
            for (int i = 0; i < _actionSize; i++)
            {
                // Offset added to log-std to output to avoid too high std values
                var logStd = policyOutput[ _actionSize + i ] + _logStdOffset;
                var std = logStd.Exp();
                var dist = new Dist( policyOutput[ i ], std );
                var action = dist.Sample();
                // Clipped action only send to environment
                float clipped = Mathf.Clamp( action.Data, -1f, 1f );
                outputActions[ i ] = clipped;
                // Stored sampled action for later use in PPO update
                _actionBuffer[ _bufferIdx, agentIdx, i ] = action.Data;
                // Store log-prob of sampled action so PPO can later compare new vs old policy
                _logProbBuffer[ _bufferIdx, agentIdx ] += dist.LogProb( action.Data ).Data;
            }

            for (int i = 0; i < _inputSize; i++)
                _observationBuffer[ _bufferIdx, agentIdx, i ] = observation[ i ];

            _valBuffer[ _bufferIdx, agentIdx ] = val[ 0 ].Data;
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
        public (float[,] advantages, float[,] returns) ComputeGAE()
        {
            int T = _bufferSize;    // time steps per env
            int N = _numEnvs;       // number of envs

            var advantages = new float[ T, N ];
            var returns = new float[ T, N ];

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
            (float[,] advantages, float[,] returns) = ComputeGAE();

            // Excluding the last row from experience buffer (which is only used for bootstrapping)
            int trimBatchSize = _bufferSize - 1;
            int totalElements = trimBatchSize * _numEnvs;

            // Flattened views over the buffers for easier minibatch sampling
            Span<float> batch_observations = MemoryMarshal.CreateSpan( ref _observationBuffer[ 0, 0, 0 ], totalElements * _inputSize );
            Span<float> batch_actions = MemoryMarshal.CreateSpan( ref _actionBuffer[ 0, 0, 0 ], totalElements * _actionSize );
            Span<float> batch_log_probs = MemoryMarshal.CreateSpan( ref _logProbBuffer[ 0, 0 ], totalElements );
            Span<float> batch_advantages = MemoryMarshal.CreateSpan( ref advantages[ 0, 0 ], totalElements );
            Span<float> batch_returns = MemoryMarshal.CreateSpan( ref returns[ 0, 0 ], totalElements );
            Span<float> doneFlags = MemoryMarshal.CreateSpan( ref _doneBuffer[ 0, 0 ], totalElements );

            // Normalize advantages to keep the policy gradient scale well behaved
            batch_advantages = Utils.Normalize( batch_advantages );

            int[] indices = Enumerable.Range( 0, totalElements ).ToArray();

            float averageLoss = 0f;
            for (int epoch = 0; epoch < _updateEpochs; epoch++)
            {
                // Make sure we sample random indices for each epoch
                RandomHub.Shuffle( indices );

                for (int start = 0; start < totalElements; start += _minibatchSize)
                {
                    // Avoid overflow on last minibatch
                    int end = Math.Min( start + _minibatchSize, totalElements );
                    int mbSize = end - start;

                    for (int mbIdx = start; mbIdx < end; mbIdx++)
                    {
                        int idx = indices[ mbIdx ];

                        for (int i = 0; i < _inputSize; i++)
                            _obsArray[ i ] = new Value( batch_observations[ idx * _inputSize + i ] );
                        var policyOutput = _policyNetwork.Forward( _obsArray );
                        var val = _valueNetwork.Forward( _obsArray );

                        // Squared error between predicted value and simulated return
                        _valueLoss[ mbIdx ] = 0.5f * (val[ 0 ] - new Value( batch_returns[ idx ] )).Pow( 2 );

                        // Calculate log-prob and entropy under current policy and based on stored actions
                        Value entropySum = 0;
                        Value newLogProb = 0;
                        for (int i = 0; i < _actionSize; i++)
                        {
                            var logStd = policyOutput[ _actionSize + i ] + _logStdOffset;
                            var std = logStd.Exp();
                            _distArray[ i ] = new Dist( policyOutput[ i ], std );
                            newLogProb += _distArray[ i ].LogProb( batch_actions[ idx * _actionSize + i ] );
                            entropySum += _distArray[ i ].Entropy();
                        }

                        // Importance sampling ratio between new and old policy
                        var ratio = (newLogProb - batch_log_probs[ idx ]).Exp();
                        // Unclipped and clipped objectives from PPO
                        var pgLoss1 = -batch_advantages[ idx ] * ratio;
                        var pgLoss2 = -batch_advantages[ idx ] * ratio.Clamp( 1f - _clipCoef, 1f + _clipCoef );
                        _pgLoss[ mbIdx ] = Value.Max( pgLoss1, pgLoss2 );

                        // Encourage exploration by maximizing entropy of the policy distribution
                        _entropyLoss[ mbIdx ] = entropySum / _actionSize;
                    }

                    var totalPgLoss = Utils.Mean( _pgLoss.AsSpan( start, mbSize ) );
                    var totalEntropyLoss = Utils.Mean( _entropyLoss.AsSpan( start, mbSize ) );
                    var totalValueLoss = Utils.Mean( _valueLoss.AsSpan( start, mbSize ) );

                    Value l2Penalty = _l2Regularizer.Compute( _l2Coef );

                    // Standard PPO loss: policy term + value term + entropy bonus + weight decay
                    var totalLoss = totalPgLoss - _betaCoef * totalEntropyLoss + _vCoef * totalValueLoss + l2Penalty;
                    averageLoss += totalLoss.Data;

                    _policyNetwork.ZeroGrad();
                    _valueNetwork.ZeroGrad();

                    totalLoss.Backward();

                    _optimizer.Step( _learningRate );
                }
            }
            averageLoss /= (_updateEpochs * (totalElements / (float)_minibatchSize));

            var meanPgLossValue = Utils.Mean( _pgLoss );
            var meanEntropyLossValue = Utils.Mean( _entropyLoss );
            var meanValueLossValue = Utils.Mean( _valueLoss );

            float meanReturn = Utils.MeanReturn( batch_returns, doneFlags, _numEnvs );
            Debug.Log( $"Step count: {_currentStep}, Mean Return: {meanReturn}, Average Loss: {averageLoss}, PG Loss: {meanPgLossValue.Data}, Entropy Loss: {meanEntropyLossValue.Data}, Value Loss: {meanValueLossValue.Data}" );
        }
        //------------------------------------------------------------------
    }
}
