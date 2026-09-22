from pydantic import BaseModel, Field


class ImageSimilarityResponseDto(BaseModel):
    similarity: float = Field(..., ge=-1.0, le=1.0)
