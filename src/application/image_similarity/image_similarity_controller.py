from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.application.image_similarity.dtos.image_similarity_dto import (
    ImageSimilarityResponseDto,
)
from src.core.services.image_similarity.image_similarity_service import (
    ImageSimilarityService,
)
from src.infrastructure.di.inject import inject
from src.infrastructure.ml.yolo_embedding_provider import InvalidImageError
from src.infrastructure.utils.config_reader import ConfigReader


@inject
class ImageSimilarityController:
    def __init__(
        self,
        image_similarity_service: ImageSimilarityService,
        config_reader: ConfigReader,
    ):
        self._image_similarity_service = image_similarity_service
        max_upload_size_mb = int(config_reader.get("ml.max_upload_size_mb", 10))
        self._max_upload_size_bytes = max_upload_size_mb * 1024 * 1024

    def api(self) -> APIRouter:
        router = APIRouter(
            prefix="",
            tags=["Image Similarity"],
        )

        @router.post(
            "/",
            response_model=ImageSimilarityResponseDto,
            status_code=status.HTTP_200_OK,
            summary="Compare semantic similarity of two images",
        )
        async def compare_images(
            image1: UploadFile = File(...),
            image2: UploadFile = File(...),
        ) -> ImageSimilarityResponseDto:
            image1_bytes = await self._read_image_async(image1)
            image2_bytes = await self._read_image_async(image2)

            try:
                similarity = await self._image_similarity_service.compare_async(
                    image1=image1_bytes,
                    image2=image2_bytes,
                )
            except InvalidImageError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc

            return ImageSimilarityResponseDto(
                similarity=round(similarity, 6),
            )

        return router

    async def _read_image_async(self, upload: UploadFile) -> bytes:
        content_type = (upload.content_type or "").lower()

        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"'{upload.filename or 'file'}' must have an image content type.",
            )

        content = await upload.read(self._max_upload_size_bytes + 1)

        if not content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"'{upload.filename or 'file'}' is empty.",
            )

        if len(content) > self._max_upload_size_bytes:
            max_size_mb = self._max_upload_size_bytes // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size must not exceed {max_size_mb} MB.",
            )

        return content
