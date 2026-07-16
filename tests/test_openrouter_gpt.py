import unittest

from causal.llm.openrouter import OpenRouterGPTConfig


class OpenRouterGPTConfigTests(unittest.TestCase):
    def test_defaults_to_gpt_oss_120b(self):
        config = OpenRouterGPTConfig()
        self.assertEqual(config.model, "openai/gpt-oss-120b")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")

    def test_rejects_non_gpt_model(self):
        with self.assertRaises(ValueError):
            OpenRouterGPTConfig(model="anthropic/claude").validated()


if __name__ == "__main__":
    unittest.main()
