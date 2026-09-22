from __future__ import annotations

import importlib
import inspect
import pkgutil

from fastapi import APIRouter, FastAPI

import src.application as application_package
from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


@inject
class WebService:
    def __init__(self, config_reader: ConfigReader):
        self._config_reader = config_reader

    def create_app(self) -> FastAPI:
        app = FastAPI(
            title=str(
                self._config_reader.get("app.name", "Vision Similarity Service")
            ),
            version=str(self._config_reader.get("app.version", "0.1.0")),
        )

        api_prefix = self._config_reader.get_api_prefix()

        for feature_name, controller_class in self._discover_controllers():
            controller = controller_class()
            router = controller.api()

            if not isinstance(router, APIRouter):
                raise TypeError(
                    f"{controller_class.__name__}.api() must return APIRouter."
                )

            app.include_router(
                router,
                prefix=f"{api_prefix}/{feature_name}",
            )

        @app.get("/health", tags=["Health"])
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return app

    @staticmethod
    def _discover_controllers() -> list[tuple[str, type]]:
        controllers: list[tuple[str, type]] = []
        application_paths = getattr(application_package, "__path__", [])

        for module in pkgutil.iter_modules(application_paths):
            if not module.ispkg or module.name.startswith("_"):
                continue

            feature_name = module.name
            controller_module_name = (
                f"src.application.{feature_name}.{feature_name}_controller"
            )

            try:
                controller_module = importlib.import_module(controller_module_name)
            except ModuleNotFoundError as exc:
                if exc.name == controller_module_name:
                    continue
                raise

            controller_classes = [
                obj
                for _, obj in inspect.getmembers(controller_module, inspect.isclass)
                if obj.__module__ == controller_module.__name__
                and obj.__name__.endswith("Controller")
                and hasattr(obj, "api")
            ]

            if len(controller_classes) != 1:
                raise RuntimeError(
                    f"Expected exactly one controller in {controller_module_name}, "
                    f"found {len(controller_classes)}."
                )

            controllers.append((feature_name, controller_classes[0]))

        return controllers
