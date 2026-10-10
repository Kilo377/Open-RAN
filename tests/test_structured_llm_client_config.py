import unittest

from ric.non_rt.ollama_gptoss_client import StructuredJsonLlmClient


class TestStructuredJsonLlmClientConfig(unittest.TestCase):
    def test_deepseek_provider_defaults(self) -> None:
        client = StructuredJsonLlmClient(provider="deepseek", enabled=False)

        self.assertEqual(client.provider, "deepseek")
        self.assertEqual(client.base_url, "https://api.deepseek.com")
        self.assertEqual(client.model_name, "deepseek-v4-pro")
        self.assertEqual(client.api_key_env_var, "DEEPSEEK_API_KEY")
        self.assertIn("deepseek", client.model_label)

    def test_deepseek_request_kwargs_do_not_use_dashscope_thinking_body(self) -> None:
        client = StructuredJsonLlmClient(
            provider="deepseek",
            enabled=False,
            enable_thinking=True,
            reasoning_effort="high",
            extra_body={"custom_flag": True},
        )

        self.assertTrue(client.enable_thinking)
        self.assertEqual(client.reasoning_effort, "high")
        self.assertEqual(client.extra_body["custom_flag"], True)


if __name__ == "__main__":
    unittest.main()
