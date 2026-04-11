"""Nexus AI Gateway provider implementation.

Routes all PAL requests through the Nexus AI Gateway, which exposes an
OpenAI-compatible ``/v1/chat/completions`` endpoint and handles multi-provider
fallback (Claude → Gemini → DeepSeek), priority queueing, response caching,
and circuit breakers automatically.

Why this provider exists:
    Native provider direct calls (Gemini, Anthropic) fail when individual API
    keys hit quota / 429s / 503s. Nexus fronts every supported model behind a
    single endpoint with its own fallback chain, so a single transient failure
    on one provider does not break a PAL tool invocation. By registering Nexus
    as the highest-priority provider in PAL's registry, every model name owned
    by Nexus (opus, sonnet, haiku, gemini, deepseek, gemma) is intercepted
    before falling back to native providers.
"""

import logging

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .registries.nexus import NexusModelRegistry
from .shared import ModelCapabilities, ProviderType

DEFAULT_NEXUS_BASE_URL = "https://nexus.subatomic.pro/v1"


class NexusProvider(OpenAICompatibleProvider):
    """OpenAI-compatible client for the Nexus AI Gateway."""

    FRIENDLY_NAME = "Nexus AI Gateway"

    # Cloudflare WAF in front of nexus.subatomic.pro blocks the default
    # ``User-Agent: OpenAI/Python ...`` header used by the openai SDK. Override
    # with a neutral identifier so the bot-management rules don't reject every
    # request before it reaches the gateway.
    DEFAULT_HEADERS = {"User-Agent": "pal-mcp-server/nexus"}

    _registry: NexusModelRegistry | None = None

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        if not base_url:
            base_url = get_env("NEXUS_BASE_URL", "") or DEFAULT_NEXUS_BASE_URL
        if not api_key:
            api_key = get_env("NEXUS_API_KEY", "") or ""

        if not api_key:
            raise ValueError(
                "Nexus API key must be provided via api_key parameter or NEXUS_API_KEY environment variable"
            )

        logging.info(f"Initializing Nexus provider with endpoint: {base_url}")

        self._alias_cache: dict[str, str] = {}

        super().__init__(api_key, base_url=base_url, **kwargs)

        if NexusProvider._registry is None:
            NexusProvider._registry = NexusModelRegistry()
            models = self._registry.list_models()
            aliases = self._registry.list_aliases()
            logging.info(f"Nexus provider loaded {len(models)} models with {len(aliases)} aliases")

    def get_provider_type(self) -> ProviderType:
        return ProviderType.NEXUS

    def _lookup_capabilities(
        self,
        canonical_name: str,
        requested_name: str | None = None,
    ) -> ModelCapabilities | None:
        builtin = super()._lookup_capabilities(canonical_name, requested_name)
        if builtin is not None:
            return builtin

        registry_entry = self._registry.resolve(canonical_name)
        if registry_entry:
            registry_entry.provider = ProviderType.NEXUS
            return registry_entry

        return None

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve PAL model names/aliases to Nexus short names."""

        cache_key = model_name.lower()
        if cache_key in self._alias_cache:
            return self._alias_cache[cache_key]

        config = self._registry.resolve(model_name)
        if config:
            if config.model_name != model_name:
                logging.debug("Resolved model alias '%s' to Nexus '%s'", model_name, config.model_name)
            resolved = config.model_name
            self._alias_cache[cache_key] = resolved
            self._alias_cache.setdefault(resolved.lower(), resolved)
            return resolved

        self._alias_cache[cache_key] = model_name
        return model_name

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        if not self._registry:
            return {}

        capabilities = {}
        for model in self._registry.list_models():
            config = self._registry.resolve(model)
            if config:
                capabilities[model] = config
        return capabilities
