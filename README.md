# Vision Similarity Service

A lightweight, layered FastAPI service for comparing two images using Ultralytics YOLO26 Classification embeddings, L2 normalization, and Cosine Similarity.

> Project status: MVP — version 0.1.0-dev  
> Default model: yolo26n-cls.pt  
> API framework: FastAPI  
> Output: similarity score in the range [-1, 1]

---

## Overview

The service accepts two images and returns a numerical score representing how close their learned visual feature representations are.

~~~text
Image 1 ──┐
          ├──► YOLO Embeddings ──► Cosine Similarity ──► Score
Image 2 ──┘
~~~

Example response:

~~~json
{
  "similarity": 0.924308
}
~~~

This value is not a probability.

It should not be interpreted as:

~~~text
The images are 92.43% likely to be similar.
~~~

A more accurate interpretation is:

~~~text
The cosine similarity between the two extracted image embeddings is 0.924308.
~~~

---

## Model Used: YOLO26n-cls

The default model used by this project is:

~~~text
yolo26n-cls.pt
~~~

This is the Nano variant of the YOLO26 Classification family provided by Ultralytics.

The -cls suffix indicates that the model is designed for Image Classification rather than Object Detection.

According to the official Ultralytics documentation, YOLO26 classification models are pretrained on ImageNet.

### Official YOLO26n-cls Characteristics

| Property | YOLO26n-cls |
|---|---:|
| Task | Image Classification |
| Scale | Nano |
| Pretraining dataset | ImageNet |
| Input size | 224 px |
| Top-1 accuracy | 71.4% |
| Top-5 accuracy | 90.1% |
| Parameters | ~2.8M |
| FLOPs | ~0.4B |

These values are Ultralytics classification benchmark figures on ImageNet. They are not API latency measurements for this project. Runtime performance depends on CPU/GPU, PyTorch, operating system, worker configuration, and deployment strategy.

Official documentation:

- [Ultralytics Image Classification](https://docs.ultralytics.com/tasks/classify/)
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)

---

## Why Classification Instead of Detection?

YOLO is widely known for Object Detection, but the YOLO26 family supports multiple computer vision tasks, including classification.

If this project used a Detection model, its primary output would conceptually look like this:

~~~text
person -> bounding box
car    -> bounding box
dog    -> bounding box
~~~

That is not the problem this service is solving.

The goal is:

~~~text
How close are these two entire images
in the feature space learned by the model?
~~~

For this reason, the project uses a Classification model and extracts an internal feature representation instead of comparing final predicted classes.

Conceptually:

~~~text
Image
  │
  ▼
YOLO26 Classification Model
  │
  ▼
Learned Visual Features
  │
  ▼
Embedding Vector
~~~

In this service, YOLO is used primarily as a feature extractor.

---

## What Is an Embedding?

A raw image contains a large number of pixel values.

Direct pixel-by-pixel comparison is usually not useful for visual similarity because changes in lighting, crop, scale, background, or camera angle can produce large pixel differences even when images contain related visual content.

A neural network progressively transforms the input image into higher-level learned features.

The resulting representation can be expressed as a numerical vector:

~~~text
Image A
   │
   ▼
[0.13, -0.44, 0.71, ..., 0.09]
~~~

This vector is called an embedding.

A second image produces another vector:

~~~text
Image B
   │
   ▼
[0.11, -0.39, 0.68, ..., 0.12]
~~~

If the model extracts similar learned features from both images, their embedding vectors should point in similar directions in the embedding space.

That is what this project measures.

---

## How Ultralytics Extracts the Embedding

The project uses the official Ultralytics embedding API:

~~~python
results = self._model.embed(
    source=rgb_image,
    verbose=False,
)
~~~

According to the Ultralytics API documentation, model.embed() wraps the prediction pipeline and returns feature embeddings.

By default, embeddings are extracted from the second-to-last model layer.

Ultralytics also supports explicitly selecting embedding layers, but this MVP intentionally uses the default behavior.

Official reference:

- [Ultralytics Model.embed API](https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model.embed)

---

# How the Model Is Used in This Project

The ML integration is implemented in:

~~~text
src/infrastructure/ml/yolo_embedding_provider.py
~~~

The processing pipeline has five main steps.

## 1. Load the Model

~~~python
self._model = YOLO(self._model_path)
~~~

The model path comes from configuration:

~~~yaml
ml:
  model_path: "yolo26n-cls.pt"
~~~

On the first run, Ultralytics may automatically download the model weights if they are not available locally.

---

## 2. Validate the Image

The service does not rely only on the file extension or HTTP MIME type.

The uploaded bytes are verified with Pillow:

~~~python
with Image.open(BytesIO(image_bytes)) as image:
    image.verify()
~~~

If the file is not a valid image, the provider raises InvalidImageError.

This prevents a renamed non-image file from being processed as a valid image.

---

## 3. Convert the Image to RGB

~~~python
rgb_image = image.convert("RGB")
~~~

This produces a consistent input format for images that may originally be grayscale, RGBA, palette-based, or stored in another Pillow-supported mode.

---

## 4. Extract the Feature Embedding

~~~python
results = self._model.embed(
    source=rgb_image,
    verbose=False,
)
~~~

The output is then converted into a flat CPU tensor:

~~~python
embedding = embedding.detach().float().cpu().flatten()
~~~

The embedding dimension is intentionally not hard-coded.

This keeps the service independent from the exact embedding size produced by a compatible model configuration.

---

## 5. L2-Normalize the Embedding

~~~python
normalized = F.normalize(embedding, dim=0)
~~~

For an embedding vector x:

~~~text
x_normalized = x / ||x||₂
~~~

After L2 normalization:

~~~text
||x_normalized||₂ ≈ 1
~~~

This simplifies cosine similarity.

---

# Cosine Similarity

For two vectors A and B:

~~~text
cosine_similarity(A, B) =
    (A · B) / (||A||₂ × ||B||₂)
~~~

Because both embeddings are already L2-normalized:

~~~text
||A||₂ = 1
||B||₂ = 1
~~~

the equation becomes:

~~~text
cosine_similarity(A, B) = A · B
~~~

The service therefore computes a dot product:

~~~python
dot_product = math.fsum(
    value1 * value2
    for value1, value2 in zip(
        embedding1,
        embedding2,
        strict=True,
    )
)
~~~

The final value is clamped to the mathematically valid range to protect against tiny floating-point overshoots:

~~~python
return max(-1.0, min(1.0, dot_product))
~~~

The API contract therefore remains:

~~~text
-1.0 <= similarity <= 1.0
~~~

## Mathematical Interpretation

| Score | Geometric meaning |
|---:|---|
| 1 | Vectors point in the same direction |
| 0 | Vectors are orthogonal |
| -1 | Vectors point in opposite directions |

This table describes cosine similarity mathematically. It does not define an application-level similarity threshold.

---

# Does a Score of 0.92 Mean the Images Are Similar?

Not necessarily.

For example:

~~~json
{
  "similarity": 0.924308
}
~~~

means that the two embeddings are close in the learned feature space.

It does not yet justify a rule such as:

~~~text
similarity >= 0.80 => similar
~~~

A threshold should be calibrated using a labeled dataset of real image pairs.

A proper evaluation flow would be:

~~~text
Labeled Image Pairs
        │
        ▼
Calculate Similarity Scores
        │
        ▼
Analyze Positive / Negative Distributions
        │
        ▼
Choose a Threshold
        │
        ▼
Evaluate Precision / Recall / F1 / ROC
~~~

Only after calibration should the API return a binary decision such as:

~~~json
{
  "similarity": 0.924308,
  "threshold": 0.81,
  "is_similar": true
}
~~~

---

# Is This True Semantic Similarity?

This distinction is important.

yolo26n-cls is a classifier pretrained on ImageNet. It was not specifically trained as a generic human-level semantic similarity model or as an image-text alignment model.

Therefore, the current output is more precisely described as:

> Similarity between learned visual feature embeddings extracted by YOLO26n-cls.

The model compares images in the visual feature space learned during classification training.

For stronger semantic-similarity use cases, this project can later benchmark YOLO embeddings against embedding-oriented models such as CLIP.

---

# End-to-End Processing Flow

~~~mermaid
flowchart TD
    A[Image 1] --> C[ImageSimilarityController]
    B[Image 2] --> C

    C --> D[HTTP Validation]
    D --> E[ImageSimilarityService]

    E --> F[YoloEmbeddingProvider]
    E --> G[YoloEmbeddingProvider]

    F --> H[Verify Image 1]
    G --> I[Verify Image 2]

    H --> J[Convert to RGB]
    I --> K[Convert to RGB]

    J --> L[YOLO26n-cls model.embed]
    K --> M[YOLO26n-cls model.embed]

    L --> N[L2 Normalize Embedding 1]
    M --> O[L2 Normalize Embedding 2]

    N --> P[Cosine Similarity]
    O --> P

    P --> Q[JSON Response]
~~~

---

# Architecture

The service follows a lightweight layered architecture:

~~~text
Host
  ↓
Application
  ↓
Core
  ↓
Infrastructure
~~~

The main dependency flow is:

~~~mermaid
flowchart TD
    HTTP[HTTP Request] --> Controller[ImageSimilarityController]
    Controller --> Service[ImageSimilarityService]
    Service --> Provider[YoloEmbeddingProvider]
    Provider --> Model[Ultralytics YOLO26n-cls]
~~~

## Application Layer

Location:

~~~text
src/application/image_similarity/
~~~

Responsibilities:

- HTTP routing
- UploadFile handling
- request validation
- HTTP status codes
- HTTPException mapping
- response DTO mapping

The Controller does not load the YOLO model and does not calculate cosine similarity.

## Core Layer

Location:

~~~text
src/core/services/image_similarity/
~~~

Responsibilities:

- orchestrating embedding extraction
- validating equal embedding dimensions
- calculating similarity
- returning the final score

The Core layer does not own FastAPI-specific HTTP behavior.

## Infrastructure Layer

Location:

~~~text
src/infrastructure/ml/
~~~

Responsibilities:

- Ultralytics integration
- YOLO model loading
- Pillow image verification
- RGB conversion
- tensor conversion
- embedding normalization

This separation keeps HTTP concerns away from ML implementation details.

---

# Constructor Dependency Injection

Dependencies are supplied through constructors.

Example:

~~~python
@inject
class ImageSimilarityService:
    def __init__(
        self,
        embedding_provider: YoloEmbeddingProvider,
    ):
        self._embedding_provider = embedding_provider
~~~

The service does not directly construct its dependency.

This reduces coupling and makes unit testing easier.

---

# Model Lifetime and Singleton Behavior

Loading an ML model for every HTTP request would be inefficient.

A poor lifecycle would be:

~~~text
Request
   ↓
Load Model
   ↓
Inference
   ↓
Destroy Model
~~~

Instead, the YOLO provider is registered as a singleton:

~~~python
@inject
class YoloEmbeddingProvider:
    __di_singleton__ = True
~~~

One model instance is therefore reused inside each application process.

Conceptually:

~~~text
Application Process
       │
       ▼
Load YOLO Once
       │
       ▼
Shared Model Instance
   ┌────┼────┐
   ▼    ▼    ▼
 Req1  Req2  Req3
~~~

This reduces repeated model-loading cost and memory churn.

---

# Thread Safety

Because the model instance is shared, the MVP uses a conservative lock:

~~~python
with self._inference_lock:
    results = self._model.embed(...)
~~~

This prevents multiple threads from entering the same model instance simultaneously.

The current design prioritizes correctness and predictable behavior over maximum throughput.

For a higher-throughput production deployment, concurrency should be benchmarked and may require strategies such as:

- multiple worker processes
- model replication
- dedicated GPU inference workers
- request queues
- batching
- GPU-aware scheduling

---

# FastAPI and Blocking Inference

YOLO inference is synchronous.

Running it directly on the FastAPI event-loop thread would block the event loop while inference executes.

The service therefore uses asyncio.to_thread():

~~~python
embedding1, embedding2 = await asyncio.gather(
    asyncio.to_thread(self._embedding_provider.embed, image1),
    asyncio.to_thread(self._embedding_provider.embed, image2),
)
~~~

This prevents synchronous inference work from running directly on the event-loop thread.

Because the shared model provider currently uses a lock, model inference is effectively serialized in this MVP.

---

# Project Structure

~~~text
vision-similarity-service/
├── src/
│   ├── host/
│   │   ├── app.py
│   │   └── res/
│   │       ├── appsettings.yaml
│   │       └── appsettings.development.yaml
│   │
│   ├── application/
│   │   ├── web.py
│   │   └── image_similarity/
│   │       ├── image_similarity_controller.py
│   │       └── dtos/
│   │           └── image_similarity_dto.py
│   │
│   ├── core/
│   │   └── services/
│   │       └── image_similarity/
│   │           └── image_similarity_service.py
│   │
│   └── infrastructure/
│       ├── di/
│       │   ├── bootstrap.py
│       │   └── inject.py
│       ├── ml/
│       │   └── yolo_embedding_provider.py
│       └── utils/
│           └── config_reader.py
│
├── tests/
│   ├── test_image_similarity_service.py
│   └── test_web_discovery.py
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
~~~

There is no Database or Repository layer in the MVP because this use case does not persist data.

---

# API

## Compare Two Images

~~~http
POST /api/v1/image_similarity/
Content-Type: multipart/form-data
~~~

Required fields:

| Field | Type | Required |
|---|---|---|
| image1 | Image file | Yes |
| image2 | Image file | Yes |

Example:

~~~bash
curl -X POST "http://localhost:5000/api/v1/image_similarity/" \
  -F "image1=@image1.jpg" \
  -F "image2=@image2.jpg"
~~~

Example response:

~~~json
{
  "similarity": 0.924308
}
~~~

## Health Check

~~~http
GET /health
~~~

Response:

~~~json
{
  "status": "ok"
}
~~~

## Swagger UI

~~~text
http://localhost:5000/docs
~~~

OpenAPI schema:

~~~text
http://localhost:5000/openapi.json
~~~

---

# Validation and Error Handling

The service performs validation at multiple levels.

## MIME Type

The upload must have an image MIME type:

~~~text
image/*
~~~

Otherwise:

~~~text
415 Unsupported Media Type
~~~

## Empty File

An empty upload returns:

~~~text
422 Unprocessable Entity
~~~

## Maximum Upload Size

The default limit per image is:

~~~text
10 MB
~~~

A larger image returns:

~~~text
413 Request Entity Too Large
~~~

## Real Image Verification

After HTTP validation, the actual bytes are verified with Pillow.

Renaming a non-image file to .jpg is therefore not enough to pass validation.

---

# Configuration

Development configuration:

~~~text
src/host/res/appsettings.development.yaml
~~~

Current values:

~~~yaml
app:
  name: "Vision Similarity Service"
  version: "0.1.0-dev"

server:
  host: "0.0.0.0"
  port: 5000

api:
  prefix: "/api/v1"

ml:
  model_path: "yolo26n-cls.pt"
  max_upload_size_mb: 10
~~~

A different compatible classification model can be configured, for example:

~~~yaml
ml:
  model_path: "yolo26s-cls.pt"
~~~

Changing model scale introduces trade-offs in latency, memory usage, and model quality, so it should be benchmarked.

---

# Installation and Local Development

## Requirements

Recommended development setup:

- Python 3.12
- pip
- Git
- Internet access for the first model download

## Windows PowerShell

Clone:

~~~powershell
git clone https://github.com/Amirmahdimtz/vision-similarity-service.git
cd vision-similarity-service
~~~

Create a virtual environment:

~~~powershell
py -3.12 -m venv .venv
~~~

If PowerShell blocks activation scripts:

~~~powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
~~~

Activate:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Install dependencies:

~~~powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Run tests:

~~~powershell
python -m pytest
~~~

Start the API:

~~~powershell
python -m uvicorn src.host.app:app --host 0.0.0.0 --port 5000 --reload
~~~

## Linux / macOS

~~~bash
git clone https://github.com/Amirmahdimtz/vision-similarity-service.git
cd vision-similarity-service

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pytest
python -m uvicorn src.host.app:app --host 0.0.0.0 --port 5000 --reload
~~~

On the first inference run, Ultralytics may download the configured model weights.

---

# Testing

Run:

~~~bash
python -m pytest
~~~

The current unit tests use a fake embedding provider, so they do not need to load the real YOLO model.

Current scenarios include:

### Identical Embeddings

~~~text
A = [1, 0, 0]
B = [1, 0, 0]

Similarity = 1
~~~

### Orthogonal Embeddings

~~~text
A = [1, 0]
B = [0, 1]

Similarity = 0
~~~

### Invalid Embedding Dimensions

If the vectors have different dimensions, the Service raises an error.

### Controller Discovery Regression

The tests also verify that duplicate namespace paths do not register the same Controller twice.

---

# Current MVP Limitations

Version 0.1.0-dev is intentionally limited in scope.

Current limitations:

- no calibrated similarity threshold
- no is_similar decision
- no project-specific evaluation dataset
- no YOLO-vs-CLIP benchmark
- no batch similarity endpoint
- no embedding cache
- no GPU-specific optimization
- no load/performance testing
- no production metrics or observability
- no persistence/database

These are not bugs. They are intentionally outside the current MVP scope.

---

# Roadmap

## Phase 1 — MVP ✅

- [x] FastAPI service
- [x] Two-image upload
- [x] YOLO26n-cls embedding extraction
- [x] L2 normalization
- [x] Cosine similarity
- [x] Image validation
- [x] Constructor Dependency Injection
- [x] Singleton model provider
- [x] Unit tests
- [x] Swagger / OpenAPI

## Phase 2 — Evaluation

- [ ] Build a labeled image-pair dataset
- [ ] Measure similarity-score distributions
- [ ] Select a threshold
- [ ] Evaluate Precision / Recall / F1 / ROC
- [ ] Add is_similar

## Phase 3 — Model Benchmarking

- [ ] Benchmark YOLO26n-cls
- [ ] Benchmark larger YOLO26 classification variants
- [ ] Benchmark CLIP
- [ ] Compare quality / latency / memory

## Phase 4 — Production Hardening

- [ ] Integration tests
- [ ] Structured logging
- [ ] Metrics
- [ ] Load testing
- [ ] Deployment optimization
- [ ] GPU strategy if required
- [ ] Batch endpoint if required

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| FastAPI | HTTP API |
| Uvicorn | ASGI server |
| Ultralytics | YOLO26 model API |
| PyTorch | Tensor operations and normalization |
| Pillow | Image validation and RGB conversion |
| Pydantic | API response validation |
| PyYAML | Configuration |
| pytest | Testing |
| pytest-asyncio | Async tests |

---

# Why This MVP Architecture?

The project intentionally avoids unnecessary complexity.

The current dependency chain is:

~~~text
Controller
   ↓
Service
   ↓
YOLO Provider
~~~

There is no database, repository, extra domain layer, CQRS layer, or unnecessary interface because the current use case does not require them.

At the same time, the major boundaries are preserved:

- HTTP logic stays in the Application layer
- workflow logic stays in the Core layer
- model-specific code stays in Infrastructure
- dependencies are provided through constructor injection

This keeps the implementation small, testable, and extensible.

---

# Technical References

- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics Image Classification](https://docs.ultralytics.com/tasks/classify/)
- [Ultralytics Model.embed API](https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model.embed)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLO26 Paper — arXiv:2606.03748](https://arxiv.org/abs/2606.03748)

---

# License Notice

This repository currently does not define a separate project license.

Ultralytics has its own licensing terms. Before production or commercial use, review the current Ultralytics license conditions:

- [Ultralytics Licensing](https://www.ultralytics.com/license)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

---

# Current Version

~~~text
0.1.0-dev
~~~

Default model:

~~~text
yolo26n-cls.pt
~~~
