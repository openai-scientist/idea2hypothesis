"""Start the idea2hypothesis HTTP API with uvicorn (developer helper, not a research CLI).

Environment::

    I2H_CONFIG    path to the YAML configuration, default configs/example.yaml
    I2H_ENV_FILE  optional KEY=VALUE file loaded before start, default .env when it exists;
                  variables already set in the environment are never overwritten
    PORT          overrides ``api.port`` from the configuration
    HOST          overrides ``api.host`` from the configuration

Requires the ``api`` extra: ``pip install -e ".[api]"``. Swagger UI is served at ``/docs``.
The package itself never reads env files; this helper only fills ``os.environ`` for local runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from idea2hypothesis.api.app import create_app
from idea2hypothesis.config import ConfigError, load_config


def load_env_file(path: Path) -> list[str]:
    """Set variables from ``path`` that are not already set; returns their names."""
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and value and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def main() -> int:
    env_file = os.environ.get("I2H_ENV_FILE")
    path = Path(env_file) if env_file else Path(".env")
    if env_file and not path.is_file():
        print(f"env file not found: {path}", file=sys.stderr)
        return 2
    if path.is_file():
        names = load_env_file(path)
        print(f"loaded {len(names)} variable(s) from {path}: {', '.join(names) or '-'}")

    config_path = os.environ.get("I2H_CONFIG", "configs/example.yaml")
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    host = os.environ.get("HOST") or config.api.host
    port = int(os.environ.get("PORT") or config.api.port)
    uvicorn.run(create_app(config), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
