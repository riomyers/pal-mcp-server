"""Registry loader for the Nexus AI Gateway provider."""

from __future__ import annotations

from ..shared import ProviderType
from .base import CapabilityModelRegistry


class NexusModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/nexus_models.json``."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__(
            env_var_name="NEXUS_MODELS_CONFIG_PATH",
            default_filename="nexus_models.json",
            provider=ProviderType.NEXUS,
            friendly_prefix="Nexus ({model})",
            config_path=config_path,
        )
