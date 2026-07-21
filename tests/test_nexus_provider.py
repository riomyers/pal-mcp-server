"""Tests for Nexus AI Gateway provider implementation.

These tests cover the gaps flagged by the CDMD code review when the Nexus
provider was first introduced:

- Cross-provider alias collisions (P0): native Gemini model names must keep
  routing to GoogleProvider when both providers are registered.
- DEFAULT_HEADERS propagation (P0): the Cloudflare WAF workaround must reach
  the OpenAI client on both the primary and minimal-fallback init paths.
- _registry None-guard (P0): lookup methods must degrade gracefully when the
  registry singleton failed to construct.
- Alias resolution / capability lookup contract.
- Provider priority routing for Anthropic/Nexus-owned models.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from providers.nexus import NexusProvider
from providers.shared import ProviderType


class TestNexusProvider:
    """Core unit tests for NexusProvider."""

    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None
        # Drop the registry singleton between tests so a failing-construction
        # test cannot poison subsequent ones.
        NexusProvider._registry = None

    def test_initialization_with_explicit_key(self):
        provider = NexusProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.NEXUS
        assert provider.base_url == "https://nexus.subatomic.pro/v1"

    def test_initialization_with_custom_base_url(self):
        provider = NexusProvider("test-key", base_url="https://nexus.local/v1")
        assert provider.base_url == "https://nexus.local/v1"

    def test_initialization_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEXUS_API_KEY", None)
            with pytest.raises(ValueError, match="Nexus API key must be provided"):
                NexusProvider("")

    def test_provider_type(self):
        provider = NexusProvider("test-key")
        assert provider.get_provider_type() == ProviderType.NEXUS

    def test_friendly_name(self):
        provider = NexusProvider("test-key")
        assert provider.FRIENDLY_NAME == "Nexus AI Gateway"

    def test_default_headers_set(self):
        """Cloudflare WAF block bypass — UA override must be declared on the class."""
        assert NexusProvider.DEFAULT_HEADERS["User-Agent"] == "pal-mcp-server/nexus"


class TestNexusRegistry:
    """Verify the JSON-backed model registry loads as expected."""

    def setup_method(self):
        NexusProvider._registry = None

    def teardown_method(self):
        NexusProvider._registry = None

    def test_registry_loads_canonical_models(self):
        provider = NexusProvider("test-key")
        models = provider._registry.list_models()
        # Six canonicals: opus, sonnet, haiku, gemini, deepseek, gemma.
        for canonical in ("opus", "sonnet", "haiku", "gemini", "deepseek", "gemma"):
            assert canonical in models

    def test_resolve_canonical_passthrough(self):
        provider = NexusProvider("test-key")
        assert provider._resolve_model_name("sonnet") == "sonnet"
        assert provider._resolve_model_name("opus") == "opus"
        assert provider._resolve_model_name("haiku") == "haiku"

    def test_resolve_anthropic_aliases(self):
        provider = NexusProvider("test-key")
        assert provider._resolve_model_name("claude-sonnet-4-6") == "sonnet"
        assert provider._resolve_model_name("claude-opus-4-6") == "opus"
        assert provider._resolve_model_name("claude-haiku-4-5") == "haiku"

    def test_resolve_namespaced_gemini_aliases(self):
        """Nexus gemini canonical exposes namespaced aliases only — no collision."""
        provider = NexusProvider("test-key")
        assert provider._resolve_model_name("nexus-gemini") == "gemini"
        assert provider._resolve_model_name("nexus-gemini-pro") == "gemini"

    def test_resolve_unknown_returns_input(self):
        """Locks in current behaviour: unknown names pass through verbatim."""
        provider = NexusProvider("test-key")
        assert provider._resolve_model_name("not-a-model") == "not-a-model"

    def test_alias_resolution_is_case_insensitive(self):
        provider = NexusProvider("test-key")
        assert provider._resolve_model_name("SONNET") == "sonnet"
        assert provider._resolve_model_name("Claude-Opus-4-6") == "opus"

    def test_get_all_model_capabilities(self):
        provider = NexusProvider("test-key")
        caps = provider.get_all_model_capabilities()
        assert "sonnet" in caps
        assert caps["sonnet"].provider == ProviderType.NEXUS
        assert caps["sonnet"].context_window == 200_000

    def test_lookup_capabilities_returns_none_for_unknown(self):
        provider = NexusProvider("test-key")
        # Bypass the registry lookup entirely — returns None instead of raising.
        assert provider._lookup_capabilities("totally-fake-model") is None

    def test_registry_none_guard_lookup(self):
        """If _registry is None, lookup must degrade rather than AttributeError."""
        provider = NexusProvider("test-key")
        NexusProvider._registry = None
        assert provider._lookup_capabilities("sonnet") is None

    def test_registry_none_guard_resolve(self):
        provider = NexusProvider("test-key")
        provider._alias_cache.clear()
        NexusProvider._registry = None
        # Should pass the input through instead of crashing.
        assert provider._resolve_model_name("sonnet") == "sonnet"


class TestNexusCrossProviderRouting:
    """P0-1 regression tests: native Gemini names must NOT route through Nexus."""

    def setup_method(self):
        from providers.registry import ModelProviderRegistry

        ModelProviderRegistry.reset_for_testing()
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None
        NexusProvider._registry = None

    def teardown_method(self):
        from providers.registry import ModelProviderRegistry

        ModelProviderRegistry.reset_for_testing()
        NexusProvider._registry = None

    def test_native_gemini_names_route_to_google(self):
        """gemini-2.5-pro / gemini-2.5-flash must keep going to GoogleProvider."""
        from providers.gemini import GeminiModelProvider
        from providers.registry import ModelProviderRegistry

        with patch.dict(
            os.environ,
            {
                "NEXUS_API_KEY": "nexus-key",
                "GEMINI_API_KEY": "gemini-key",
            },
        ):
            ModelProviderRegistry.register_provider(ProviderType.NEXUS, NexusProvider)
            ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

            for name in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-pro"):
                provider = ModelProviderRegistry.get_provider_for_model(name)
                assert provider is not None, f"No provider found for {name}"
                assert provider.get_provider_type() == ProviderType.GOOGLE, (
                    f"{name} routed via {provider.get_provider_type().value}, "
                    f"expected google. Cross-provider alias collision regressed."
                )

    def test_anthropic_models_route_to_nexus(self):
        """opus/sonnet/haiku have no native provider — must go through Nexus."""
        from providers.registry import ModelProviderRegistry

        with patch.dict(os.environ, {"NEXUS_API_KEY": "nexus-key"}):
            ModelProviderRegistry.register_provider(ProviderType.NEXUS, NexusProvider)

            for name in ("opus", "sonnet", "haiku", "claude-sonnet-4-6"):
                provider = ModelProviderRegistry.get_provider_for_model(name)
                assert provider is not None, f"No provider found for {name}"
                assert provider.get_provider_type() == ProviderType.NEXUS

    def test_namespaced_gemini_aliases_route_to_nexus(self):
        """Intentional Nexus passthrough for the gemini canonical via namespaced alias."""
        from providers.gemini import GeminiModelProvider
        from providers.registry import ModelProviderRegistry

        with patch.dict(
            os.environ,
            {
                "NEXUS_API_KEY": "nexus-key",
                "GEMINI_API_KEY": "gemini-key",
            },
        ):
            ModelProviderRegistry.register_provider(ProviderType.NEXUS, NexusProvider)
            ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

            provider = ModelProviderRegistry.get_provider_for_model("nexus-gemini-pro")
            assert provider is not None
            assert provider.get_provider_type() == ProviderType.NEXUS


class TestNexusDefaultHeaders:
    """P0-2 regression tests: User-Agent override must reach the OpenAI client."""

    def setup_method(self):
        NexusProvider._registry = None

    def teardown_method(self):
        NexusProvider._registry = None

    @patch("providers.openai_compatible.OpenAI")
    def test_primary_client_init_includes_default_headers(self, mock_openai_class):
        """Primary httpx-wrapped client init must pass DEFAULT_HEADERS to OpenAI()."""
        mock_openai_class.return_value = MagicMock()
        provider = NexusProvider("test-key")
        _ = provider.client  # trigger lazy init

        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        assert "default_headers" in call_kwargs
        assert call_kwargs["default_headers"]["User-Agent"] == "pal-mcp-server/nexus"

    @patch("providers.openai_compatible.OpenAI")
    def test_minimal_fallback_client_preserves_default_headers(self, mock_openai_class):
        """If primary init raises, the minimal-kwargs fallback must STILL include
        DEFAULT_HEADERS — otherwise Cloudflare WAF will 403 every Nexus request."""
        # First call (primary path) raises, second call (minimal fallback) succeeds.
        mock_openai_class.side_effect = [
            Exception("primary client init failed"),
            MagicMock(),
        ]
        provider = NexusProvider("test-key")
        _ = provider.client

        assert mock_openai_class.call_count == 2
        fallback_kwargs = mock_openai_class.call_args_list[1][1]
        assert (
            "default_headers" in fallback_kwargs
        ), "Minimal-kwargs fallback dropped default_headers — CF WAF will block."
        assert fallback_kwargs["default_headers"]["User-Agent"] == "pal-mcp-server/nexus"


class TestNexusGenerateContent:
    """End-to-end (mocked) tests for the request path."""

    def setup_method(self):
        NexusProvider._registry = None

    def teardown_method(self):
        NexusProvider._registry = None

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_resolves_alias_before_api_call(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "PONG"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "sonnet"
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 6
        mock_client.chat.completions.create.return_value = mock_response

        provider = NexusProvider("test-key")
        result = provider.generate_content(prompt="ping", model_name="claude-sonnet-4-6", temperature=0.0)

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # CRITICAL: alias gets resolved to short canonical before hitting Nexus.
        assert call_kwargs["model"] == "sonnet"
        assert result.content == "PONG"
        assert result.model_name == "sonnet"
        assert result.provider == ProviderType.NEXUS
