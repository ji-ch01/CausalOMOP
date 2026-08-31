import os
import unittest
from unittest import mock

from causal.llm.base import strip_json_code_fence
from causal.llm.openrouter import OpenRouterGPTConfig, build_request_kwargs


class OpenRouterGPTConfigTests(unittest.TestCase):
    def test_defaults_to_gpt_5(self):
        config = OpenRouterGPTConfig()
        self.assertEqual(config.model, "openai/gpt-5")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.max_completion_tokens, 20000)
        self.assertIsNone(config.reasoning_effort)

    def test_accepts_any_openrouter_vendor_slug(self):
        for model in (
            "openai/gpt-5",
            "openai/o3",
            "anthropic/claude-opus-5",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-r1:free",
        ):
            with self.subTest(model=model):
                self.assertEqual(OpenRouterGPTConfig(model=model).validated().model, model)

    def test_rejects_slug_without_vendor_prefix(self):
        with self.assertRaises(ValueError):
            OpenRouterGPTConfig(model="gpt-5").validated()

    def test_rejects_non_https_base_url(self):
        with self.assertRaises(ValueError):
            OpenRouterGPTConfig(base_url="http://openrouter.ai/api/v1").validated()

    def test_rejects_unknown_reasoning_effort(self):
        with self.assertRaises(ValueError):
            OpenRouterGPTConfig(reasoning_effort="turbo").validated()

    def test_rejects_non_positive_token_ceiling(self):
        with self.assertRaises(ValueError):
            OpenRouterGPTConfig(max_completion_tokens=0).validated()

    @mock.patch("causal.llm.openrouter.load_dotenv_if_available", lambda: None)
    def test_reads_overrides_from_env(self):
        env = {
            "CAUSALOMOP_LLM_MODEL": "anthropic/claude-opus-5",
            "CAUSALOMOP_LLM_MAX_TOKENS": "8000",
            "CAUSALOMOP_LLM_TIMEOUT": "45",
            "CAUSALOMOP_LLM_REASONING_EFFORT": "Low",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            config = OpenRouterGPTConfig.from_env()
        self.assertEqual(config.model, "anthropic/claude-opus-5")
        self.assertEqual(config.max_completion_tokens, 8000)
        self.assertEqual(config.timeout_seconds, 45.0)
        self.assertEqual(config.reasoning_effort, "low")

    @mock.patch("causal.llm.openrouter.load_dotenv_if_available", lambda: None)
    def test_explicit_model_argument_wins_over_env(self):
        with mock.patch.dict(
            os.environ, {"CAUSALOMOP_LLM_MODEL": "openai/gpt-5"}, clear=False
        ):
            config = OpenRouterGPTConfig.from_env("anthropic/claude-sonnet-5")
        self.assertEqual(config.model, "anthropic/claude-sonnet-5")


class RequestKwargsTests(unittest.TestCase):
    def test_requests_json_mode_and_token_ceiling(self):
        kwargs = build_request_kwargs(OpenRouterGPTConfig(model="openai/gpt-5"), "hi")
        self.assertEqual(kwargs["model"], "openai/gpt-5")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["max_completion_tokens"], 20000)
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("extra_body", kwargs)

    def test_reasoning_effort_is_passed_through_extra_body(self):
        config = OpenRouterGPTConfig(model="openai/o3", reasoning_effort="low")
        kwargs = build_request_kwargs(config, "hi")
        self.assertEqual(kwargs["extra_body"], {"reasoning": {"effort": "low"}})


class StripJsonCodeFenceTests(unittest.TestCase):
    def test_unwraps_json_fence(self):
        self.assertEqual(strip_json_code_fence('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_unwraps_bare_fence(self):
        self.assertEqual(strip_json_code_fence('```\n{"a": 1}\n```'), '{"a": 1}')

    def test_leaves_bare_payload_untouched(self):
        self.assertEqual(strip_json_code_fence('{"a": 1}'), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()
