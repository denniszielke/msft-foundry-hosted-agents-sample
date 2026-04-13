"""Agent registry with auto-discovery of hosted agents."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Lightweight descriptor for a hosted agent."""

    name: str
    description: str = ""
    cpu: str = "1"
    memory: str = "2Gi"
    env_vars: dict[str, str] = field(default_factory=dict)
    path: Path | None = None


def discover_hosted_agents() -> list[AgentConfig]:
    """Scan src/agents/*/ for __init__.py files exporting AGENT_CONFIG."""
    agents_dir = Path(__file__).parent
    configs: list[AgentConfig] = []

    for child in sorted(agents_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue

        init_file = child / "__init__.py"
        if not init_file.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            f"_agent_cfg_{child.name.replace('-', '_')}", init_file
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        raw = getattr(mod, "AGENT_CONFIG", None)
        if raw is None:
            continue

        config = AgentConfig(**raw, path=child)
        configs.append(config)

    return configs
