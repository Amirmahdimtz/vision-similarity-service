from __future__ import annotations

import asyncio
import math

from src.infrastructure.di.inject import inject
from src.infrastructure.ml.yolo_embedding_provider import YoloEmbeddingProvider


@inject
class ImageSimilarityService:
    def __init__(self, embedding_provider: YoloEmbeddingProvider):
        self._embedding_provider = embedding_provider

    async def compare_async(self, image1: bytes, image2: bytes) -> float:
        embedding1, embedding2 = await asyncio.gather(
            asyncio.to_thread(self._embedding_provider.embed, image1),
            asyncio.to_thread(self._embedding_provider.embed, image2),
        )

        if len(embedding1) != len(embedding2):
            raise RuntimeError("Embedding dimensions do not match.")

        dot_product = math.fsum(
            value1 * value2
            for value1, value2 in zip(embedding1, embedding2, strict=True)
        )

        # Embeddings are normalized by the provider. Clamp tiny floating-point
        # overshoots so the API remains inside the cosine similarity range.
        return max(-1.0, min(1.0, dot_product))
