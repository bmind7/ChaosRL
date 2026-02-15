using System.Runtime.CompilerServices;

// Allow the test assembly to access internal members (e.g. TensorStorage.Buffer)
// so tests can validate low-level behaviour while keeping the public API clean.
[assembly: InternalsVisibleTo( "ChaosRLTests" )]
