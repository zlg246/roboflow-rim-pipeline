# Vehicle Panel Damage Annotation Pipeline

Automatically annotates vehicle panel damage images from S3 using a local YOLO model
and a vision model (Claude or OpenAI), then uploads only relevant images to Roboflow for human review.

---

## Production Bucket Safety

**The production bucket is READ ONLY.**

| Operation | Where |
|-----------|-------|
| `s3:ListBucket` | prod bucket (read) |
| `s3:GetObject` | prod bucket (read) |
| `s3:PutObject` | `LOG_BUCKET` only (separate bucket) |

The pipeline **never writes to the production bucket.** Log uploads go to `LOG_BUCKET`
(set in `.env`). If `LOG_BUCKET` is empty, logs are kept locally only.

---

## S3 Structure

```
prod-castle-hill-toyota/
└── prior_condition/
    └── SCANNER_A_{device}_{YYYY-MM-DD}_{time}/
        ├── 3/   ← included (left camera)
        └── 7/   ← included (right camera)
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in .env
cp .env.example .env
# Edit .env with your API keys

# 3. Place your YOLO model
cp /path/to/best.pt ./best.pt

# 4. Run tests (no API keys needed — everything is mocked)
pytest test/ -v
```

---

## Usage

```bash
# Current week Monday–Friday (default)
python pipeline.py

# Specific date range
python pipeline.py --from 2026-03-16 --to 2026-03-20

# Single day
python pipeline.py --from 2026-03-16 --to 2026-03-16

# Named shortcuts
python pipeline.py --from last-friday --to last-friday
python pipeline.py --from monday --to friday

# Dry run — YOLO + vision model but NO Roboflow upload
python pipeline.py --from 2026-03-16 --to 2026-03-16 --dry-run

# Test on 5 images only
python pipeline.py --from 2026-03-16 --to 2026-03-16 --limit 5

# Override camera subfolders
python pipeline.py --from 2026-03-16 --to 2026-03-16 --subfolders 3,7
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--from DATE` | Monday this week | Inclusive start date |
| `--to DATE` | Friday this week | Inclusive end date |
| `--subfolders N,N` | `3,7` | Camera subfolder IDs to include |
| `--bucket BUCKET` | `S3_BUCKET` env var | Override which S3 bucket to scan |
| `--limit N` | none | Process at most N images (useful for testing) |
| `--dry-run` | false | Run YOLO + vision model but skip Roboflow upload |

`--from` / `--to` accept ISO dates (`2026-03-16`) or keywords:
`today`, `yesterday`, `monday`…`sunday`, `last-monday`…`last-sunday`.

---

## Architecture

```
S3 (read-only)
    │  list_image_keys()   →  filtered list of S3 keys
    │  load_image()        →  PIL image + base64 string + MIME type
    ▼
YOLO (local — no network)
    │  run_inference()     →  [{class, x, y, w, h, confidence}, ...]
    ▼
Vision model (Claude / OpenAI)
    │  verify_image()      →  {decision, confidence, reasoning, corrections:[]}
    ▼
Roboflow
    │  upload()            →  image + VOC XML annotation uploaded
    ▼
JSONL log  +  optional SNS email  +  optional EC2 self-stop
```

---

## Pipeline Loop

Each image goes through these steps:

1. **Load** — download from S3 via `GetObject`; returns a PIL image, base64 string, and MIME type
2. **YOLO inference** — runs the trained `.pt` model locally; returns normalised bounding-box dicts or `[]`
3. **Short-circuit** — if YOLO finds nothing, decision is set to `null` immediately and the vision call is skipped
4. **Vision verification** — sends the image and YOLO predictions to the configured model; returns a structured JSON decision with corrected bounding boxes
5. **Upload gate** — only `scratch` and `other` proceed to upload; `null` is skipped
6. **Dry-run check** — if `--dry-run` was passed, the upload is printed but not executed
7. **Roboflow upload** — saves image as temp JPEG + Pascal VOC XML; uploaded with `is_prediction=True` so annotations appear as model suggestions, keeping images in the human review queue
8. **Log record** — appended after every image regardless of outcome

---

## Decisions

| Decision | Meaning | Uploaded to Roboflow? |
|----------|---------|----------------------|
| `scratch` | Confirmed surface scratch damage (YOLO verified or added by vision model) | ✅ Yes |
| `other` | Detection exists but is not surface damage (dent, shadow, dirt, wrong class) | ✅ Yes |
| `null` | No annotatable damage — all predictions clearly off the vehicle panel | ❌ No |

---

## Deduplication and Resumption

Before processing begins, the pipeline reads the existing JSONL log and collects every
`s3_key` already recorded. Re-running with the same date range resumes from where it
left off — already-processed images are skipped automatically.

---

## Batch Naming

The Roboflow batch name is derived from the bucket and date range:

```
prod-castle-hill-toyota  +  2026-05-12 → 2026-05-16
→  castle_hill_toyota_2026-05-12_2026-05-16
```

The `prod-` prefix is stripped and hyphens become underscores.

---

## Configuration

All settings live in `config.py`, loaded from `.env` via `python-dotenv`.

| Variable | Purpose |
|----------|---------|
| `S3_BUCKET` | Production bucket to scan (read-only) |
| `S3_ROOT_PREFIX` | Top-level folder prefix inside the bucket |
| `S3_TARGET_SUBFOLDERS` | Camera angle subfolder IDs, e.g. `3,7` |
| `LOG_BUCKET` | Separate bucket for log uploads (never the production bucket) |
| `YOLO_MODEL_PATH` | Path to the trained `.pt` model file |
| `YOLO_CONF_THRESHOLD` | Minimum confidence for a YOLO detection (default `0.25`) |
| `VISION_PROVIDER` | `claude` (default) or `openai` |
| `ANTHROPIC_API_KEY` | Required when `VISION_PROVIDER=claude` |
| `ANTHROPIC_MODEL` | Claude model name (default `claude-sonnet-4-6`) |
| `OPENAI_API_KEY` | Required when `VISION_PROVIDER=openai` |
| `OPENAI_MODEL` | OpenAI model name (default `gpt-4o`) |
| `SAVE_NULL_REVIEW` | `true` to save YOLO-detected but vision-rejected images to `null_review/` |
| `UPLOAD_JPEG_QUALITY` | JPEG quality for Roboflow uploads (default `100`) |
| `NULL_REVIEW_JPEG_QUALITY` | JPEG quality for null-review saves (default `90`) |
| `ROBOFLOW_API_KEY` | Roboflow API key |
| `ROBOFLOW_WORKSPACE` | Roboflow workspace slug |
| `ROBOFLOW_PROJECT` | Roboflow project slug |
| `EC2_SELF_STOP` | `true` to stop the EC2 instance after the run |
| `EC2_INSTANCE_ID` | Instance ID for self-stop |
| `AWS_REGION` | AWS region for EC2/SNS (default `ap-southeast-2`) |
| `SNS_TOPIC_ARN` | ARN for completion email (optional) |

---

## Log Format

One JSON object per line in `logs/pipeline_<timestamp>.jsonl`:

| Field | Type | Description |
|-------|------|-------------|
| `s3_key` | string | Full S3 object key |
| `timestamp` | string | UTC ISO-8601 time of processing |
| `batch` | string | Roboflow batch name |
| `dry_run` | bool | Whether this was a dry run |
| `decision` | string | Vision model decision (see taxonomy above) |
| `vision_provider` | string | `claude` or `openai` |
| `vision_confidence` | float | Confidence score 0–1 |
| `reasoning` | string | One/two sentence explanation |
| `yolo_count` | int | Number of YOLO predictions |
| `yolo_predictions` | list | Raw YOLO boxes |
| `corrections` | list | Corrected/added/deleted boxes |
| `uploaded` | bool \| `"dry_run"` | Upload outcome |
| `error` | string | Present only when an exception was caught |

---

## Post-Run Steps

After the loop completes:

1. **Log upload** — `upload_log_to_s3()` copies the local JSONL to `LOG_BUCKET`
2. **SNS notification** — `_notify()` emails a summary to the configured SNS topic
3. **EC2 self-stop** — `_stop_self()` shuts down the instance if `EC2_SELF_STOP=true`

---

## Running Tests

```bash
# All tests (no AWS/Claude/Roboflow needed)
pytest test/ -v

# With coverage
pytest test/ -v --cov=. --cov-report=term-missing

# Single module
pytest test/test_s3_loader.py -v
```

---

## Required AWS IAM Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::prod-castle-hill-toyota",
        "arn:aws:s3:::prod-castle-hill-toyota/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::your-logging-bucket/pipeline_logs/*"
    }
  ]
}
```

---

## Claude Code Agents

Project-specific agents live in [.claude/agents/](.claude/agents/). Claude Code loads them automatically.

| Agent | Purpose | Modifies files? |
|-------|---------|----------------|
| `error-handling` | Audits exception handling — silent failures, wrong catch scope, missing cleanup | No |
| `language-convention` | Audits code style — naming, type hints, docstrings, logging, imports | No |

Both agents produce a severity-ranked report. You approve individual fixes before any changes are made.

```
# Audit the whole codebase
> use the language-convention agent on this project
> use the error-handling agent on this project

# Target a specific file
> use the error-handling agent on vision_verifier.py
> check naming conventions in roboflow_uploader.py
```
