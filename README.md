# Vision Similarity Service

سرویس مقایسه‌ی دو تصویر با استفاده از **Ultralytics YOLO26 Classification embeddings**، **L2 normalization** و **Cosine Similarity**.

> **وضعیت پروژه:** MVP — نسخه‌ی 0.1.0-dev  
> **مدل پیش‌فرض:** yolo26n-cls.pt  
> **API:** FastAPI  
> **خروجی:** similarity score در بازه‌ی -1 تا 1

---

## مسئله چیست؟

ورودی سرویس دو تصویر است و خروجی یک عدد است که نشان می‌دهد representationهای استخراج‌شده از دو تصویر توسط مدل چقدر به یکدیگر نزدیک‌اند.

~~~text
Image 1 ──┐
          ├──► YOLO Embeddings ──► Cosine Similarity ──► Score
Image 2 ──┘
~~~

نمونه‌ی خروجی:

~~~json
{
  "similarity": 0.924308
}
~~~

این مقدار **احتمال نیست**. یعنی نباید آن را «92.43 درصد احتمال مشابه بودن تصاویر» تفسیر کرد.

تعبیر دقیق‌تر این است:

> Cosine similarity بین دو embedding استخراج‌شده از تصاویر برابر 0.924308 است.

---

## مدل مورد استفاده: YOLO26n-cls

مدل فعلی پروژه:

~~~text
yolo26n-cls.pt
~~~

است.

این مدل نسخه‌ی **Nano** از خانواده‌ی **YOLO26 Classification** در Ultralytics است. پسوند <code>-cls</code> مشخص می‌کند که مدل برای **Image Classification** ساخته شده است، نه Object Detection.

طبق مستندات رسمی Ultralytics، مدل‌های YOLO26 Classification به‌صورت pretrained روی **ImageNet** ارائه می‌شوند.

### مشخصات رسمی مدل

| مشخصه | YOLO26n-cls |
|---|---:|
| Task | Image Classification |
| Scale | Nano |
| Pretraining dataset | ImageNet |
| Input size | 224 px |
| Top-1 accuracy | 71.4% |
| Top-5 accuracy | 90.1% |
| Parameters | ~2.8M |
| FLOPs | ~0.4B |

این اعداد benchmark رسمی Ultralytics روی ImageNet هستند و latency واقعی این API را نشان نمی‌دهند. سرعت سرویس به CPU/GPU، PyTorch، سیستم‌عامل و deployment بستگی دارد.

منبع: [Ultralytics Classification Documentation](https://docs.ultralytics.com/tasks/classify/)

---

## چرا Classification و نه Detection؟

YOLO بیشتر با Object Detection شناخته می‌شود، اما خانواده‌ی YOLO26 چند task مختلف از جمله Classification را پشتیبانی می‌کند.

اگر از مدل Detection استفاده می‌کردیم، هدف اصلی مدل چیزی شبیه این بود:

~~~text
person -> bounding box
car    -> bounding box
dog    -> bounding box
~~~

اما مسئله‌ی این پروژه این است:

~~~text
کل تصویر از نظر featureهایی که مدل یاد گرفته،
چقدر به تصویر دوم نزدیک است؟
~~~

بنابراین به یک representation از **کل تصویر** نیاز داریم. به همین دلیل از مدل Classification استفاده می‌کنیم و به‌جای class prediction نهایی، feature embedding آن را استخراج می‌کنیم.

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

در این پروژه YOLO عملاً نقش **feature extractor** را دارد.

---

## Embedding چیست؟

تصویر خام مجموعه‌ی بزرگی از pixelها است. مقایسه‌ی مستقیم pixelها برای similarity مناسب نیست؛ چون تغییر نور، زاویه، crop، background یا scale می‌تواند مقدار pixelها را کاملاً تغییر دهد.

یک Neural Network در لایه‌های مختلف featureهای تصویر را استخراج می‌کند؛ از الگوهای ساده‌تر تا representationهای سطح بالاتر.

در نهایت می‌توان تصویر را با یک بردار عددی نمایش داد:

~~~text
Image A
   │
   ▼
[0.13, -0.44, 0.71, ..., 0.09]
~~~

این بردار یک **Embedding** است.

برای تصویر دوم:

~~~text
Image B
   │
   ▼
[0.11, -0.39, 0.68, ..., 0.12]
~~~

اگر دو تصویر از دید featureهایی که مدل یاد گرفته است نزدیک باشند، embeddingهای آن‌ها نیز در فضای برداری نزدیک‌تر خواهند بود.

---

## Ultralytics چگونه Embedding را استخراج می‌کند؟

در پروژه از API رسمی زیر استفاده می‌شود:

~~~python
results = self._model.embed(
    source=rgb_image,
    verbose=False,
)
~~~

طبق مستندات رسمی Ultralytics، <code>model.embed()</code> یک wrapper روی prediction pipeline است و به‌صورت پیش‌فرض feature embedding را از **second-to-last model layer** استخراج می‌کند.

در صورت نیاز Ultralytics اجازه می‌دهد layer مشخصی برای embedding انتخاب شود، اما MVP فعلی از رفتار پیش‌فرض استفاده می‌کند.

منبع: [Ultralytics Model.embed API](https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model.embed)

---

## Pipeline دقیق مدل در این پروژه

پیاده‌سازی ML در فایل زیر قرار دارد:

~~~text
src/infrastructure/ml/yolo_embedding_provider.py
~~~

### 1. Load مدل

~~~python
self._model = YOLO(self._model_path)
~~~

نام model از configuration خوانده می‌شود:

~~~yaml
ml:
  model_path: "yolo26n-cls.pt"
~~~

در اولین اجرا، اگر weight روی سیستم موجود نباشد، Ultralytics آن را download می‌کند.

### 2. بررسی واقعی بودن تصویر

فقط به filename یا Content-Type اعتماد نمی‌کنیم. bytes تصویر با Pillow بررسی می‌شود:

~~~python
with Image.open(BytesIO(image_bytes)) as image:
    image.verify()
~~~

اگر فایل واقعاً image معتبر نباشد، <code>InvalidImageError</code> ایجاد می‌شود.

### 3. تبدیل به RGB

~~~python
rgb_image = image.convert("RGB")
~~~

با این کار ورودی‌هایی مثل Grayscale یا RGBA قبل از inference به format یکسان تبدیل می‌شوند.

### 4. استخراج Feature Embedding

~~~python
results = self._model.embed(
    source=rgb_image,
    verbose=False,
)
~~~

سپس خروجی به یک Tensor یک‌بعدی روی CPU تبدیل می‌شود:

~~~python
embedding = embedding.detach().float().cpu().flatten()
~~~

dimension بردار در کد hard-code نشده است؛ بنابراین Service به یک embedding size ثابت وابسته نیست.

### 5. L2 Normalization

~~~python
normalized = F.normalize(embedding, dim=0)
~~~

اگر embedding را x در نظر بگیریم:

~~~text
x_normalized = x / ||x||₂
~~~

پس طول بردار normalize‌شده تقریباً 1 می‌شود.

---

## Cosine Similarity چگونه محاسبه می‌شود؟

فرمول اصلی:

~~~text
cosine_similarity(A, B) =
    (A · B) / (||A||₂ × ||B||₂)
~~~

اما در این پروژه هر دو embedding از قبل L2-normalized هستند:

~~~text
||A||₂ = 1
||B||₂ = 1
~~~

پس فرمول به این تبدیل می‌شود:

~~~text
cosine_similarity(A, B) = A · B
~~~

به همین دلیل Service dot product را محاسبه می‌کند:

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

و در انتها برای جلوگیری از floating-point overshoot مقدار clamp می‌شود:

~~~python
return max(-1.0, min(1.0, dot_product))
~~~

### معنی ریاضی Score

| Score | مفهوم هندسی |
|---:|---|
| 1 | دو بردار در یک جهت |
| 0 | دو بردار orthogonal |
| -1 | دو بردار در جهت مخالف |

این جدول فقط معنی ریاضی Cosine Similarity است و **threshold پروژه** نیست.

---

## آیا 0.92 یعنی تصاویر Similar هستند؟

نه لزوماً.

مثلاً این خروجی:

~~~json
{
  "similarity": 0.924308
}
~~~

نشان می‌دهد embeddingهای دو تصویر در فضای feature مدل به هم نزدیک هستند.

اما نمی‌توان بدون evaluation گفت:

~~~text
similarity >= 0.80 => similar
~~~

Threshold باید با یک dataset واقعی از pairهای برچسب‌خورده تعیین شود:

~~~text
Labeled Image Pairs
        │
        ▼
Calculate Similarities
        │
        ▼
Positive / Negative Score Distributions
        │
        ▼
Choose Threshold
        │
        ▼
Precision / Recall / F1 / ROC Evaluation
~~~

بعد از calibration می‌توان پاسخ API را توسعه داد:

~~~json
{
  "similarity": 0.924308,
  "threshold": 0.81,
  "is_similar": true
}
~~~

---

## آیا این واقعاً Semantic Similarity است؟

در توضیح علمی پروژه باید این بخش دقیق بیان شود.

<code>yolo26n-cls</code> یک classifier pretrained روی ImageNet است. این مدل به‌طور اختصاصی برای generic human-level semantic similarity یا image-text alignment train نشده است.

بنابراین خروجی فعلی دقیق‌تر است که این‌طور تعریف شود:

> **Similarity between learned visual feature embeddings extracted by YOLO26n-cls.**

یعنی مدل similarity را در فضای visual featureهایی اندازه می‌گیرد که هنگام classification یاد گرفته است.

برای semantic similarity قوی‌تر، در مراحل بعد می‌توان YOLO embedding را با مدل‌های embedding-oriented مانند **CLIP** روی dataset واقعی پروژه benchmark کرد.

---

## جریان کامل Request

~~~mermaid
flowchart TD
    A[Image 1] --> C[ImageSimilarityController]
    B[Image 2] --> C
    C --> D[HTTP Validation]
    D --> E[ImageSimilarityService]
    E --> F[YoloEmbeddingProvider]
    F --> G[Validate Image]
    G --> H[Convert RGB]
    H --> I[YOLO26n-cls model.embed]
    I --> J[L2 Normalize]
    J --> K[Cosine Similarity]
    K --> L[JSON Response]
~~~

برای هر دو تصویر embedding ساخته می‌شود و سپس Service آن‌ها را مقایسه می‌کند.

---

# معماری پروژه

معماری سرویس:

~~~text
Host
  ↓
Application
  ↓
Core
  ↓
Infrastructure
~~~

Dependency flow اصلی:

~~~mermaid
flowchart TD
    HTTP[HTTP Request] --> Controller[ImageSimilarityController]
    Controller --> Service[ImageSimilarityService]
    Service --> Provider[YoloEmbeddingProvider]
    Provider --> Model[Ultralytics YOLO26n-cls]
~~~

### Application Layer

مسیر:

~~~text
src/application/image_similarity/
~~~

مسئول:

- HTTP routing
- UploadFile handling
- HTTP validation
- Status codes
- HTTPException
- Response DTO

Controller مدل YOLO نمی‌سازد و similarity calculation را انجام نمی‌دهد.

### Core Layer

مسیر:

~~~text
src/core/services/image_similarity/
~~~

مسئول workflow:

- استخراج embedding تصویر اول
- استخراج embedding تصویر دوم
- بررسی برابر بودن dimension
- محاسبه‌ی similarity
- برگرداندن score

### Infrastructure Layer

مسیر:

~~~text
src/infrastructure/ml/
~~~

مسئول:

- Ultralytics
- YOLO model loading
- Pillow image validation
- Tensor processing
- Embedding normalization

---

## Constructor Dependency Injection

Dependencyها توسط constructor دریافت می‌شوند:

~~~python
@inject
class ImageSimilarityService:
    def __init__(
        self,
        embedding_provider: YoloEmbeddingProvider,
    ):
        self._embedding_provider = embedding_provider
~~~

Service خودش dependency را ایجاد نمی‌کند.

این design باعث کاهش coupling و ساده‌تر شدن testing می‌شود.

---

## Model Lifetime و Singleton

Load کردن مدل ML برای هر request پرهزینه است. به همین دلیل <code>YoloEmbeddingProvider</code> Singleton است:

~~~python
@inject
class YoloEmbeddingProvider:
    __di_singleton__ = True
~~~

در نتیجه در هر process مدل یک‌بار ساخته می‌شود و requestهای بعدی همان instance را استفاده می‌کنند.

### Thread Safety

دسترسی به model shared با Lock محافظت شده است:

~~~python
with self._inference_lock:
    results = self._model.embed(...)
~~~

در MVP این تصمیم محافظه‌کارانه است تا چند thread همزمان وارد یک model instance نشوند.

### FastAPI Event Loop

Inference مدل synchronous است. برای اینکه مستقیماً event loop وب‌سرور را block نکند، Service از <code>asyncio.to_thread()</code> استفاده می‌کند.

~~~python
embedding1, embedding2 = await asyncio.gather(
    asyncio.to_thread(self._embedding_provider.embed, image1),
    asyncio.to_thread(self._embedding_provider.embed, image2),
)
~~~

به دلیل Lock، inferenceهای model shared فعلاً عملاً serialize می‌شوند. برای throughput بالاتر باید worker strategy، GPU scheduling یا model replication جداگانه benchmark شود.

---

# ساختار Repository

~~~text
vision-similarity-service/
├── src/
│   ├── host/
│   │   ├── app.py
│   │   └── res/
│   │       ├── appsettings.yaml
│   │       └── appsettings.development.yaml
│   ├── application/
│   │   ├── web.py
│   │   └── image_similarity/
│   │       ├── image_similarity_controller.py
│   │       └── dtos/
│   │           └── image_similarity_dto.py
│   ├── core/
│   │   └── services/
│   │       └── image_similarity/
│   │           └── image_similarity_service.py
│   └── infrastructure/
│       ├── di/
│       │   ├── bootstrap.py
│       │   └── inject.py
│       ├── ml/
│       │   └── yolo_embedding_provider.py
│       └── utils/
│           └── config_reader.py
├── tests/
│   ├── test_image_similarity_service.py
│   └── test_web_discovery.py
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
~~~

در MVP دیتابیس و Repository نداریم، چون این use-case هیچ داده‌ای را persist نمی‌کند.

---

# API

## Compare two images

~~~http
POST /api/v1/image_similarity/
Content-Type: multipart/form-data
~~~

دو field لازم است:

| Field | Type | Required |
|---|---|---|
| image1 | Image file | Yes |
| image2 | Image file | Yes |

نمونه:

~~~bash
curl -X POST "http://localhost:5000/api/v1/image_similarity/" \
  -F "image1=@image1.jpg" \
  -F "image2=@image2.jpg"
~~~

Response:

~~~json
{
  "similarity": 0.924308
}
~~~

## Health Check

~~~http
GET /health
~~~

~~~json
{
  "status": "ok"
}
~~~

## Swagger

~~~text
http://localhost:5000/docs
~~~

OpenAPI:

~~~text
http://localhost:5000/openapi.json
~~~

---

# Validation و Error Handling

### MIME Type

ورودی باید Content-Type تصویری داشته باشد:

~~~text
image/*
~~~

در غیر این صورت:

~~~text
415 Unsupported Media Type
~~~

### Empty File

فایل خالی:

~~~text
422 Unprocessable Entity
~~~

### Maximum Size

محدودیت پیش‌فرض هر تصویر:

~~~text
10 MB
~~~

فایل بزرگ‌تر:

~~~text
413 Request Entity Too Large
~~~

### Image Verification

بعد از HTTP validation، خود bytes توسط Pillow verify می‌شوند. بنابراین صرفاً تغییر extension یک فایل غیرتصویری به jpg برای عبور از validation کافی نیست.

---

# Configuration

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

برای آزمایش model scale دیگر:

~~~yaml
ml:
  model_path: "yolo26s-cls.pt"
~~~

تغییر مدل باید همراه benchmark accuracy، latency و memory انجام شود.

---

# نصب و اجرا

## Windows PowerShell

~~~powershell
git clone https://github.com/Amirmahdimtz/vision-similarity-service.git
cd vision-similarity-service

py -3.12 -m venv .venv
~~~

اگر اجرای activation script توسط PowerShell بسته است:

~~~powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
~~~

سپس:

~~~powershell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pytest

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

در اولین استفاده از مدل ممکن است weight مدل توسط Ultralytics دانلود شود.

---

# Tests

اجرای تست‌ها:

~~~powershell
python -m pytest
~~~

تست‌های Service با fake embedding provider اجرا می‌شوند و برای unit test نیازی به load کردن YOLO واقعی ندارند.

سناریوهای فعلی:

~~~text
Identical embeddings:
A = [1, 0, 0]
B = [1, 0, 0]
Similarity = 1
~~~

~~~text
Orthogonal embeddings:
A = [1, 0]
B = [0, 1]
Similarity = 0
~~~

همچنین mismatch شدن dimension بردارها و duplicate controller discovery تست می‌شود.

---

# محدودیت‌های MVP

نسخه‌ی فعلی عمداً محدود است:

- threshold کالیبره‌شده ندارد
- is_similar برنمی‌گرداند
- evaluation dataset اختصاصی ندارد
- YOLO در برابر CLIP benchmark نشده است
- batch similarity ندارد
- embedding cache ندارد
- GPU-specific optimization ندارد
- load/performance test ندارد
- production metrics و observability ندارد
- persistence/database ندارد

این موارد bug نیستند؛ خارج از scope نسخه‌ی MVP هستند.

---

# Roadmap

### Phase 1 — MVP ✅

- [x] FastAPI API
- [x] Two-image upload
- [x] YOLO26n-cls embeddings
- [x] L2 normalization
- [x] Cosine similarity
- [x] Image validation
- [x] Constructor Dependency Injection
- [x] Singleton model provider
- [x] Unit tests
- [x] Swagger/OpenAPI

### Phase 2 — Evaluation

- [ ] ساخت labeled image-pair dataset
- [ ] محاسبه‌ی score distribution
- [ ] انتخاب threshold
- [ ] Precision / Recall / F1 evaluation
- [ ] اضافه کردن is_similar

### Phase 3 — Model Benchmark

- [ ] Benchmark YOLO26n-cls
- [ ] Benchmark مدل‌های بزرگ‌تر YOLO26-cls
- [ ] Benchmark CLIP
- [ ] مقایسه‌ی accuracy / latency / memory

### Phase 4 — Production Hardening

- [ ] Integration tests
- [ ] Structured logging
- [ ] Metrics
- [ ] Load testing
- [ ] Deployment optimization
- [ ] GPU strategy در صورت نیاز
- [ ] Batch endpoint در صورت نیاز

---

# Technology Stack

| Technology | Usage |
|---|---|
| Python | Core language |
| FastAPI | HTTP API |
| Uvicorn | ASGI server |
| Ultralytics | YOLO26 API |
| PyTorch | Tensor operations and normalization |
| Pillow | Image validation and RGB conversion |
| Pydantic | API DTO validation |
| PyYAML | Configuration |
| pytest | Testing |
| pytest-asyncio | Async tests |

---

# منابع فنی

- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics Image Classification](https://docs.ultralytics.com/tasks/classify/)
- [Ultralytics Model.embed API](https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model.embed)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLO26 Paper — arXiv:2606.03748](https://arxiv.org/abs/2606.03748)

---

## License Notice

این repository در حال حاضر License مستقل تعریف نکرده است.

Ultralytics شرایط مجوز خودش را دارد. قبل از استفاده‌ی production یا commercial، شرایط فعلی آن را بررسی کنید:

- [Ultralytics Licensing](https://www.ultralytics.com/license)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

---

## Current Version

~~~text
0.1.0-dev
~~~

Default model:

~~~text
yolo26n-cls.pt
~~~
