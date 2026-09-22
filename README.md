# Vision Similarity Service

FastAPI service for comparing the semantic similarity of two images using YOLO image embeddings and cosine similarity.

## Architecture

The project follows the service conventions defined for this project:

```text
HTTP Request
    ↓
ImageSimilarityController
    ↓
ImageSimilarityService
    ↓
YoloEmbeddingProvider
    ↓
Ultralytics YOLO
```

There is no database or repository layer in the MVP because this feature does not persist data.

## Requirements

- Python 3.11+
- pip
- Internet access on the first run so Ultralytics can download the configured pretrained model

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
uvicorn src.host.app:app --host 0.0.0.0 --port 5000 --reload
```

Open Swagger UI at:

```text
http://localhost:5000/docs
```

## Compare two images

```bash
curl -X POST "http://localhost:5000/api/v1/image_similarity/" \
  -F "image1=@/path/to/image1.jpg" \
  -F "image2=@/path/to/image2.jpg"
```

Example response:

```json
{
  "similarity": 0.817423
}
```

The value is cosine similarity between the normalized YOLO embeddings. It must not be interpreted directly as a probability or percentage.

## Configuration

Development configuration is in:

```text
src/host/res/appsettings.development.yaml
```

The default model is:

```text
yolo26n-cls.pt
```

You can change it through `ml.model_path`.

## Tests

```bash
pytest
```

Unit tests use a fake embedding provider and therefore do not download or load the YOLO model.
