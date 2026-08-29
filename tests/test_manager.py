import unittest
import sys
import importlib

from pyquantflow.model.manager import BaseModelEngine


class TestBaseModelEngine(unittest.TestCase):
    def test_abstract_method_validate_enforcement(self):
        class DummyModelEngine(BaseModelEngine):
            def register_mlflow_evaluation(
                self,
                model,
                params,
                metrics,
                experiment_name=None,
                run_name=None,
            ):
                pass

            # Intentionally omit `validate` to test abstract method enforcement

        with self.assertRaisesRegex(
            TypeError, "Can't instantiate abstract class DummyModelEngine.*validate"
        ):
            DummyModelEngine()


class TestClassifierEngineMockMLFlowError(unittest.TestCase):
    def test_mlflow_import_error(self):
        # Ensure the module is unloaded so that importing it re-runs top-level code.
        if "pyquantflow.model.manager" in sys.modules:
            del sys.modules["pyquantflow.model.manager"]

        # Temporarily inject None for mlflow in sys.modules to simulate an ImportError
        old_mlflow = sys.modules.get("mlflow", None)
        sys.modules["mlflow"] = None

        try:
            # We capture logs from pyquantflow.model.manager
            with self.assertLogs("pyquantflow.model.manager", level="WARNING") as cm:
                # Reimport the module inside the context so the top-level
                # try/except ImportError block runs and emits the warning.
                importlib.import_module("pyquantflow.model.manager")

            # Verify the specific error message is present
            self.assertTrue(any("mlflow not found" in log for log in cm.output))
        finally:
            # Restore the previous state of sys.modules['mlflow']
            if old_mlflow is not None:
                sys.modules["mlflow"] = old_mlflow
            else:
                del sys.modules["mlflow"]

            # Reload module to clean up
            if "pyquantflow.model.manager" in sys.modules:
                importlib.reload(sys.modules["pyquantflow.model.manager"])


if __name__ == "__main__":
    unittest.main()
