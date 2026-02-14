using NUnit.Framework;

namespace ChaosRL.Tests
{
    public abstract class TensorScopedTestBase
    {
        private TensorScope _tensorScope;

        //------------------------------------------------------------------
        [SetUp]
        public void TensorScopeSetUp()
        {
            _tensorScope = new TensorScope();
        }
        //------------------------------------------------------------------
        [TearDown]
        public void TensorScopeTearDown()
        {
            _tensorScope?.Dispose();
            _tensorScope = null;
        }
        //------------------------------------------------------------------
    }
}
