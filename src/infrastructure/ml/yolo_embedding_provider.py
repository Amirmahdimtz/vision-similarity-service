from __future__ import annotations

from io import BytesIO
from threading import Lock

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


class InvalidImageError(ValueError):
    pass


@inject
class YoloEmbeddingProvider:
    __di_singleton__ = True

    def __init__(self, config_reader: ConfigReader):
        self._model_path = str(config_reader.get("ml.model_path", "yolo26n-cls.pt"))
        self._model = YOLO(self._model_path)
        self._inference_lock = Lock()

    def embed(self, image_bytes: bytes) -> tuple[float, ...]:
        if not image_bytes:
            raise InvalidImageError("Image is empty.")

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()

            with Image.open(BytesIO(image_bytes)) as image:
                rgb_image = image.convert("RGB")

                # Guard access to the shared model instance. This is conservative
                # and prevents concurrent requests from racing through one model.
                with self._inference_lock:
                    results = self._model.embed(source=rgb_image, verbose=False)
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("Uploaded file is not a valid image.") from exc

        if not results:
            raise RuntimeError("YOLO returned no embedding.")

        embedding = results[0]

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)

        embedding = embedding.detach().float().cpu().flatten()

        if embedding.numel() == 0:
            raise RuntimeError("YOLO returned an empty embedding.")

        normalized = F.normalize(embedding, dim=0)
        return tuple(float(value) for value in normalized.tolist())
