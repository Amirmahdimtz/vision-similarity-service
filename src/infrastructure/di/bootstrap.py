from __future__ import annotations

import importlib
from pathlib import Path


_BOOTSTRAPPED = False
_EXCLUDED_PARTS = {
    "__pycache__",
    "alembic",
    "di",
    "dtos",
    "models",
    "res",
    "scripts",
    "sso",
    "versions",
}


def _import_package_modules(package_name: str) -> None:
    package = importlib.import_module(package_name)
    package_paths = list(getattr(package, "__path__", []))

    for package_path in package_paths:
        root = Path(package_path)

        for module_file in sorted(root.rglob("*.py")):
            relative_path = module_file.relative_to(root)

            if set(relative_path.parts) & _EXCLUDED_PARTS:
                continue

            if module_file.name == "__init__.py":
                module_parts = relative_path.parent.parts
            else:
                module_parts = relative_path.with_suffix("").parts

            if not module_parts:
                continue

            module_name = ".".join((package_name, *module_parts))
            importlib.import_module(module_name)


def bootstrap_di() -> None:
    global _BOOTSTRAPPED

    if _BOOTSTRAPPED:
        return

    for package_name in ("src.infrastructure", "src.core", "src.application"):
        _import_package_modules(package_name)

    _BOOTSTRAPPED = True
