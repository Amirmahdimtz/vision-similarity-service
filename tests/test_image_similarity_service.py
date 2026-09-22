import pytest

from src.core.services.image_similarity.image_similarity_service import (
    ImageSimilarityService,
)


class FakeEmbeddingProvider:
    def __init__(self, embeddings: list[tuple[float, ...]]):
        self._embeddings = iter(embeddings)

    def embed(self, image_bytes: bytes) -> tuple[float, ...]:
        return next(self._embeddings)


@pytest.mark.asyncio
async def test_compare_async_returns_one_for_identical_embeddings() -> None:
    provider = FakeEmbeddingProvider(
        embeddings=[
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ]
    )
    service = ImageSimilarityService(embedding_provider=provider)  # type: ignore[arg-type]

    similarity = await service.compare_async(b"image-1", b"image-2")

    assert similarity == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_compare_async_returns_zero_for_orthogonal_embeddings() -> None:
    provider = FakeEmbeddingProvider(
        embeddings=[
            (1.0, 0.0),
            (0.0, 1.0),
        ]
    )
    service = ImageSimilarityService(embedding_provider=provider)  # type: ignore[arg-type]

    similarity = await service.compare_async(b"image-1", b"image-2")

    assert similarity == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_compare_async_rejects_mismatched_dimensions() -> None:
    provider = FakeEmbeddingProvider(
        embeddings=[
            (1.0, 0.0),
            (1.0, 0.0, 0.0),
        ]
    )
    service = ImageSimilarityService(embedding_provider=provider)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="dimensions"):
        await service.compare_async(b"image-1", b"image-2")
