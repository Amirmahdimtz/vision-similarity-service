from __future__ import annotations

import importlib
import pkgutil


_BOOTSTRAPPED = False
_EXCLUDED_PARTS = {
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

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return

    for module in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        module_parts = set(module.name.split("."))
        if module_parts & _EXCLUDED_PARTS:
            continue

        importlib.import_module(module.name)


def bootstrap_di() -> None:
    global _BOOTSTRAPPED

    if _BOOTSTRAPPED:
        return

    for package_name in ("src.infrastructure", "src.core", "src.application"):
        _import_package_modules(package_name)

    _BOOTSTRAPPED = True
