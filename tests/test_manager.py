import unittest
import sys
import importlib

class TestClassifierEngineMockMLFlowError(unittest.TestCase):
    def test_mlflow_import_error(self):
        # Ensure the module is unloaded so that importing it re-runs top-level code.
        if 'pyquantflow.model.manager' in sys.modules:
            del sys.modules['pyquantflow.model.manager']

        # Temporarily inject None for mlflow in sys.modules to simulate an ImportError
        old_mlflow = sys.modules.get('mlflow', None)
        sys.modules['mlflow'] = None
        
        try:
            # We capture logs from pyquantflow.model.manager
            with self.assertLogs('pyquantflow.model.manager', level='WARNING') as cm:
                import pyquantflow.model.manager
            
            # Verify the specific error message is present
            self.assertTrue(any('mlflow not found' in log for log in cm.output))
        finally:
            # Restore the previous state of sys.modules['mlflow']
            if old_mlflow is not None:
                sys.modules['mlflow'] = old_mlflow
            else:
                del sys.modules['mlflow']
                
            # Reload module to clean up
            if 'pyquantflow.model.manager' in sys.modules:
                importlib.reload(sys.modules['pyquantflow.model.manager'])

if __name__ == '__main__':
    unittest.main()
