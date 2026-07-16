import os
import unittest
from unittest import mock

from causal.llm import factory
from causal.llm.factory import resolve_provider


@mock.patch.object(factory, "load_dotenv_if_available", lambda: None)
class ResolveProviderTests(unittest.TestCase):
    """`.env` on disk carries a real OPENAI_API_KEY; these tests must not depend
    on it, so dotenv loading is disabled and only the in-process environment
    (as set up by each test) is consulted."""

    def test_dev_default_uses_openrouter(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CAUSALOMOP_ENV", None)
            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertEqual(resolve_provider(), "openrouter")

    def test_explicit_dev_uses_openrouter(self):
        with mock.patch.dict(os.environ, {"CAUSALOMOP_ENV": "dev"}, clear=False):
            self.assertEqual(resolve_provider(), "openrouter")

    def test_prod_without_cloud_keys_uses_local(self):
        with mock.patch.dict(os.environ, {"CAUSALOMOP_ENV": "prod"}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertEqual(resolve_provider(), "local")

    def test_prod_rejects_openrouter_key(self):
        with mock.patch.dict(
            os.environ,
            {"CAUSALOMOP_ENV": "prod", "OPENROUTER_API_KEY": "sk-or-fake"},
            clear=False,
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaises(RuntimeError):
                resolve_provider()

    def test_prod_rejects_openai_compat_key(self):
        with mock.patch.dict(
            os.environ,
            {"CAUSALOMOP_ENV": "prod", "OPENAI_API_KEY": "sk-fake"},
            clear=False,
        ):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with self.assertRaises(RuntimeError):
                resolve_provider()

    def test_rejects_unknown_environment(self):
        with mock.patch.dict(os.environ, {"CAUSALOMOP_ENV": "staging"}, clear=False):
            with self.assertRaises(ValueError):
                resolve_provider()

    def test_explicit_argument_overrides_environment_variable(self):
        with mock.patch.dict(os.environ, {"CAUSALOMOP_ENV": "prod"}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertEqual(resolve_provider("dev"), "openrouter")


if __name__ == "__main__":
    unittest.main()
