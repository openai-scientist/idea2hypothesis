"""Prompt loading, domain/user overrides, strict rendering and snapshots.

Prompt assets ship inside the package and are read through :mod:`importlib.resources`, so
they work from an installed wheel as well as from a source checkout.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DOMAINS = ("ml", "hep", "biology")
_ENTRY_FIELDS = {"system", "user", "json_mode", "max_tokens", "guidance"}
_VAR_RE = re.compile(r"\{\{(\w+)\}\}")
_INCLUDE_RE = re.compile(r"\{\{>\s*(\w+)\s*\}\}")
_MAX_INCLUDE_DEPTH = 5
RESERVED_VARIABLES = frozenset({"domain_guidance"})


class PromptError(Exception):
    """Invalid prompt file, unknown key or missing template variable."""


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    json_mode: bool = False
    max_tokens: int | None = None


def _read_package_yaml(*parts: str) -> dict[str, Any]:
    node = resources.files("idea2hypothesis.prompts")
    for part in parts:
        node = node / part
    try:
        data = yaml.safe_load(node.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PromptError(f"cannot load packaged prompt file {'/'.join(parts)}: {exc}") from exc
    if not isinstance(data, dict):
        raise PromptError(f"packaged prompt file {'/'.join(parts)} must contain a mapping")
    return data


def _validate_entry(key: str, entry: Any, origin: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise PromptError(f"{origin}: prompt {key!r} must be a mapping")
    unknown = set(entry) - _ENTRY_FIELDS
    if unknown:
        raise PromptError(f"{origin}: prompt {key!r} has unknown fields {sorted(unknown)}")
    return dict(entry)


class PromptLoader:
    """Resolves prompts for one domain, with optional user override file."""

    def __init__(self, domain: str = "ml", override_file: str | Path | None = None) -> None:
        if domain not in DOMAINS:
            raise PromptError(f"unknown prompt domain {domain!r}; expected one of {list(DOMAINS)}")
        self.domain = domain
        self.override_file = str(override_file) if override_file else ""

        base = _read_package_yaml("stages.yaml")
        self._prompts: dict[str, dict[str, Any]] = {}
        for section in ("stages", "sub_prompts"):
            for key, entry in (base.get(section) or {}).items():
                self._prompts[key] = _validate_entry(key, entry, "stages.yaml")
        self._blocks: dict[str, str] = {
            str(k): str(v) for k, v in (base.get("blocks") or {}).items()
        }

        roles_file = _read_package_yaml("hypothesis_roles.yaml")
        roles = (roles_file.get("roles") or {}).get(domain)
        if not roles:
            raise PromptError(f"hypothesis_roles.yaml has no roles for domain {domain!r}")
        self._roles: dict[str, dict[str, Any]] = {
            name: _validate_entry(name, entry, "hypothesis_roles.yaml")
            for name, entry in roles.items()
        }

        if domain != "ml":
            self._apply(_read_package_yaml("domains", f"{domain}.yaml"), f"domains/{domain}.yaml")
        if override_file:
            self._apply(self._read_override(Path(override_file)), str(override_file))

    @staticmethod
    def _read_override(path: Path) -> dict[str, Any]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise PromptError(f"cannot read prompt override file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise PromptError(f"prompt override file {path} must contain a mapping")
        return data

    def _apply(self, data: dict[str, Any], origin: str) -> None:
        allowed = {"version", "stages", "sub_prompts", "blocks", "roles"}
        unknown = set(data) - allowed
        if unknown:
            raise PromptError(f"{origin}: unknown top-level keys {sorted(unknown)}")
        for section in ("stages", "sub_prompts"):
            for key, entry in (data.get(section) or {}).items():
                if key not in self._prompts:
                    raise PromptError(f"{origin}: unknown prompt key {key!r}")
                self._prompts[key].update(_validate_entry(key, entry, origin))
        for name, text in (data.get("blocks") or {}).items():
            if not isinstance(text, str):
                raise PromptError(f"{origin}: block {name!r} must be a string")
            self._blocks[str(name)] = text
        for name, entry in (data.get("roles") or {}).items():
            entry = _validate_entry(name, entry, origin)
            self._roles.setdefault(name, {}).update(entry)

    # -- introspection ----------------------------------------------------

    def keys(self) -> list[str]:
        return sorted(self._prompts)

    def role_names(self) -> list[str]:
        return list(self._roles)

    def required_variables(self, key: str) -> frozenset[str]:
        entry = self._entry(key)
        text = self._expand(str(entry.get("system", "")) + "\n" + str(entry.get("user", "")))
        return frozenset(_VAR_RE.findall(text)) - RESERVED_VARIABLES

    def _entry(self, key: str) -> dict[str, Any]:
        try:
            return self._prompts[key]
        except KeyError:
            raise PromptError(f"unknown prompt key {key!r}") from None

    # -- rendering --------------------------------------------------------

    def _expand(self, template: str, depth: int = 0) -> str:
        if depth > _MAX_INCLUDE_DEPTH:
            raise PromptError("prompt blocks are nested too deeply (cycle?)")

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in self._blocks:
                raise PromptError(f"unknown prompt block {name!r}")
            return self._expand(self._blocks[name].rstrip("\n"), depth + 1)

        return _INCLUDE_RE.sub(replace, template)

    def _substitute(self, template: str, variables: dict[str, str], where: str) -> str:
        expanded = self._expand(template)
        missing = sorted({n for n in _VAR_RE.findall(expanded) if n not in variables})
        if missing:
            raise PromptError(f"{where}: missing template variables {missing}")
        return _VAR_RE.sub(lambda m: variables[m.group(1)], expanded)

    def _render_entry(
        self, entry: dict[str, Any], variables: dict[str, Any], where: str
    ) -> RenderedPrompt:
        values = {k: str(v) for k, v in variables.items()}
        guidance = str(entry.get("guidance", "")).strip()
        values["domain_guidance"] = (
            self._substitute(guidance, {**values, "domain_guidance": ""}, f"{where}.guidance")
            if guidance
            else ""
        )
        return RenderedPrompt(
            system=self._substitute(
                str(entry.get("system", "")), values, f"{where}.system"
            ).strip(),
            user=self._substitute(str(entry.get("user", "")), values, f"{where}.user").strip(),
            json_mode=bool(entry.get("json_mode", False)),
            max_tokens=entry.get("max_tokens"),
        )

    def render(self, key: str, **variables: Any) -> RenderedPrompt:
        """Render a stage or sub-prompt; every template variable must be supplied."""
        return self._render_entry(self._entry(key), variables, key)

    def render_role(self, role: str, **variables: Any) -> RenderedPrompt:
        """Render one hypothesis-perspective role prompt (always requests JSON)."""
        try:
            entry = self._roles[role]
        except KeyError:
            raise PromptError(f"unknown hypothesis role {role!r}") from None
        rendered = self._render_entry(entry, variables, f"role:{role}")
        return RenderedPrompt(rendered.system, rendered.user, True, entry.get("max_tokens", 6144))

    # -- snapshot ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Template texts and a content hash, stored with each run for reproducibility."""
        content = {
            "domain": self.domain,
            "override_file": self.override_file,
            "prompts": copy.deepcopy(self._prompts),
            "blocks": dict(self._blocks),
            "roles": copy.deepcopy(self._roles),
        }
        canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
        content["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return content
