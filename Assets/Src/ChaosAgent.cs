using UnityEngine;
using ChaosRL;

public class ChaosAgent : MonoBehaviour
{
    //------------------------------------------------------------------
    private const float TiltDamper = 0.1f;

    private static int _nextAgentIdx = 0;

    [Header( "Scene References" )]
    [Tooltip( "The ball to track and reset when out of bounds." )]
    [SerializeField] private Transform _ball;

    [Header( "Episode Settings" )]
    [Tooltip( "If the ball is further than this distance from the platform center, the episode ends." )]
    [SerializeField] private float _resetDistance = 1f;
    [SerializeField] private int _maxEpisodeSteps = 300;
    [SerializeField] private int _stepInterval = 5;

    [Header( "Control Settings" )]
    [Tooltip( "Maximum absolute tilt in degrees applied from actions (-1..1 → -max..+max). Expect 2 continuous actions: pitch (X), roll (Z)." )]
    [Range( 1f, 45f )]
    [SerializeField] private float _maxTiltDegrees = 20f;

    private Vector3 _platformCenterPosition;

    private Rigidbody _platformRB;
    private Quaternion _targetLocalRotation;
    private float _accumulatedReward = 0f;
    private bool _isEpisodeEnded = false;
    private int _currentStep = 0;
    private int _fixedUpdateCount = 0;
    private int _agentIdx = 0;
    //------------------------------------------------------------------

    private void Start()
    {
        _agentIdx = _nextAgentIdx;
        _nextAgentIdx++;

        _platformCenterPosition = transform.position;
        _platformRB = GetComponent<Rigidbody>();
        if (_platformRB != null)
        {
            _platformRB.isKinematic = true; // ensure kinematic for MoveRotation control
        }
        _targetLocalRotation = transform.localRotation;

        ResetState();
    }
    //------------------------------------------------------------------
    public void ResetState()
    {
        // Debug.Log( $"Resetting agent {_agentIdx} state." );

        // Reset ball velocities if it has a Rigidbody
        if (_ball.TryGetComponent<Rigidbody>( out var ballRb ))
        {
            ballRb.linearVelocity = Vector3.zero;
            ballRb.angularVelocity = Vector3.zero;
        }

        // Randomize initial tilt so the policy sees varied starting states instead of a single posture
        float randPitch = Random.Range( -_maxTiltDegrees, _maxTiltDegrees );
        float randRoll = Random.Range( -_maxTiltDegrees, _maxTiltDegrees );
        var startRot = Quaternion.Euler( randPitch, 0, randRoll );
        transform.localRotation = startRot;
        _targetLocalRotation = startRot;

        // Place ball at/near the platform center (slightly above to avoid clipping)
        Vector3 offsetUp = Vector3.up * 0.5f;
        _ball.position = _platformCenterPosition + offsetUp;

        //---
        _currentStep = 0;
        _accumulatedReward = 0f;
        _fixedUpdateCount = 0;
    }
    //------------------------------------------------------------------
    private void CheckAndMaybeEndEpisode()
    {
        float dist = Vector3.Distance( _ball.position, _platformCenterPosition );
        if (dist > Mathf.Max( 0.001f, _resetDistance ) || _currentStep >= _maxEpisodeSteps)
        {
            // End episode when ball drifts too far or we hit the step cap to keep trajectories bounded
            _isEpisodeEnded = true;
        }
        else
        {
            _isEpisodeEnded = false;
        }
    }
    //------------------------------------------------------------------
    private void AddProximityReward()
    {
        // Per-step proximity reward: 1 when d<=0.1m, 0 when d>=0.5m, linear in-between
        float d = Vector3.Distance( _ball.position, _platformCenterPosition );
        float proximityReward = Mathf.InverseLerp( 0.5f, 0.1f, d );
        _accumulatedReward += proximityReward;
    }
    //------------------------------------------------------------------
    public float[] CollectObservations()
    {
        float[] observations = new float[ 7 ];

        observations[ 0 ] = transform.rotation.x;
        observations[ 1 ] = transform.rotation.y;
        observations[ 2 ] = transform.rotation.z;
        observations[ 3 ] = transform.rotation.w;

        Vector3 rel = _ball.position - _platformCenterPosition;
        // Encode ball position relative to platform center so policy can reason in local space
        observations[ 4 ] = rel.x;
        observations[ 5 ] = rel.y;
        observations[ 6 ] = rel.z;

        return observations;
    }
    //------------------------------------------------------------------
    public void ApplyActions( float[] continuous )
    {
        Vector3 euler = transform.localEulerAngles;
        float pitch = SmoothTilt( euler.x, continuous[ 0 ] );
        float roll = SmoothTilt( euler.z, continuous[ 1 ] );

        _targetLocalRotation = Quaternion.Euler( pitch, euler.y, roll );
    }
    //------------------------------------------------------------------
    private float SmoothTilt( float eulerComponent, float actionValue )
    {
        float currentAngle = Mathf.DeltaAngle( 0f, eulerComponent );
        float targetAngle = Mathf.Clamp( actionValue, -1f, 1f ) * _maxTiltDegrees;
        float nextAngle = currentAngle + TiltDamper * Mathf.DeltaAngle( currentAngle, targetAngle );
        return nextAngle;
    }
    //------------------------------------------------------------------
    private void UpdateRotation()
    {
        _platformRB.MoveRotation( _targetLocalRotation );
    }
    //------------------------------------------------------------------
    private void FixedUpdate()
    {
        AddProximityReward();

        if (_fixedUpdateCount % _stepInterval == 0)
        {
            // Only query the policy every few physics ticks to reduce decision overhead and smooth control
            CheckAndMaybeEndEpisode();
            float[] obs = CollectObservations();
            float[] actions = AcademyTensor.Instance.RequestDecision( _agentIdx, obs, _isEpisodeEnded, _accumulatedReward );
            ApplyActions( actions );

            if (_isEpisodeEnded)
                ResetState();

            _accumulatedReward = 0f;
            _currentStep++;
        }
        _fixedUpdateCount++;

        UpdateRotation();
    }
    //------------------------------------------------------------------
}
