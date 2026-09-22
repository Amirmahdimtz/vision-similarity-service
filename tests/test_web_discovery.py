from pathlib import Path

from src.application import web


def test_controller_discovery_deduplicates_namespace_paths(monkeypatch) -> None:
    application_path = str(Path(next(iter(web.application_package.__path__))).resolve())

    monkeypatch.setattr(
        web.application_package,
        "__path__",
        [application_path, application_path],
    )

    controllers = web.WebService._discover_controllers()
    feature_names = [feature_name for feature_name, _ in controllers]

    assert feature_names.count("image_similarity") == 1
