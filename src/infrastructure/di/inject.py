from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, TypeVar, get_type_hints


T = TypeVar("T")
_PROVIDERS: dict[type[Any], type[Any]] = {}
_SINGLETONS: dict[type[Any], Any] = {}


def register_provider(provider_type: type[Any]) -> None:
    _PROVIDERS[provider_type] = provider_type


def resolve(dependency_type: type[T]) -> T:
    provider_type = _PROVIDERS.get(dependency_type)

    if provider_type is None:
        raise RuntimeError(
            f"No DI provider registered for {dependency_type!r}. "
            "Check @inject and bootstrap discovery."
        )

    if getattr(provider_type, "__di_singleton__", False):
        existing = _SINGLETONS.get(provider_type)
        if existing is not None:
            return existing

        instance = provider_type()
        _SINGLETONS[provider_type] = instance
        return instance

    return provider_type()


def inject(cls: type[T]) -> type[T]:
    original_init = cls.__init__
    signature = inspect.signature(original_init)

    @wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind_partial(self, *args, **kwargs)
        type_hints = get_type_hints(original_init)

        for name, parameter in signature.parameters.items():
            if name == "self" or name in bound.arguments:
                continue

            if parameter.default is not inspect.Parameter.empty:
                continue

            dependency_type = type_hints.get(name)
            if dependency_type is None:
                raise RuntimeError(
                    f"Dependency '{name}' in {cls.__name__}.__init__ "
                    "must have a resolvable type annotation."
                )

            kwargs[name] = resolve(dependency_type)

        original_init(self, *args, **kwargs)

    cls.__init__ = wrapped_init  # type: ignore[method-assign]
    register_provider(cls)
    return cls
