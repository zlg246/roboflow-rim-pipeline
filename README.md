# Vehicle Panel Damage Annotation Pipeline

Automatically annotates vehicle panel damage images from S3 using a local YOLO model
and Claude Vision, then uploads only relevant images to Roboflow for human review.

## Production Bucket Safety

**The production bucket `prod-castle-hill-toyota` is READ ONLY.**

| Operation | Where |
|-----------|-------|
| `s3:ListBucket` | prod bucket (read) |
| `s3:GetObject` | prod bucket (read) |
| `s3:PutObject` | LOG_BUCKET only (separate bucket) |

The pipeline **never writes to the production bucket.** Log uploads go to `LOG_BUCKET`
(set in `.env`). If `LOG_BUCKET` is empty, logs are kept locally only.

## S3 Structure

```
prod-castle-hill-toyota/
└── prior_condition/
    └── SCANNER_A_{device}_{YYYY-MM-DD}_{time}/
        ├── 3/   ← included (left camera)
        └── 7/   ← included (right camera)
```

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

# Dry run — YOLO + Claude but NO Roboflow upload
python pipeline.py --from 2026-03-16 --to 2026-03-16 --dry-run

# Test on 5 images only
python pipeline.py --from 2026-03-16 --to 2026-03-16 --limit 5

# Override camera subfolders
python pipeline.py --from 2026-03-16 --to 2026-03-16 --subfolders 3,7
```

## Pipeline Flow

```
S3 (read-only) → list images in subfolders 3 & 7 within date range
      ↓
Local YOLO → raw predictions per image
      ↓
Claude Vision → verifies image + predictions → one of 6 decisions
      ↓
  correct / partial / missed / other → upload to Roboflow with annotations
  null / bad_quality                 → skip (not uploaded)
      ↓
Annotators review pre-labelled images in Roboflow
```

## Decisions

| Decision | Meaning | Uploaded? |
|----------|---------|-----------|
| `correct` | YOLO predictions accurate | ✅ Yes |
| `partial` | Some boxes need adjustment | ✅ Yes |
| `missed` | Damage visible, YOLO missed it | ✅ Yes |
| `other` | Detections are wrong class | ✅ Yes |
| `null` | No damage present | ❌ No |
| `bad_quality` | Image too dark/blurry | ❌ No |

## Claude Code Agents

Project-specific agents live in [.claude/agents/](.claude/agents/). Claude Code loads them automatically — no registration needed.

| Agent | Purpose | Modifies files? |
|---|---|---|
| `error-handling` | Audits exception handling — finds silent failures, wrong catch scope, missing cleanup | No |
| `language-convention` | Audits code style — naming, type hints, docstrings, logging, magic literals, imports | No |

Both agents produce a structured report with severity-ranked issues and recommended fixes. You approve individual fixes before any changes are made.

### Usage examples

```
# Audit the whole codebase
> use the language-convention agent on this project
> use the error-handling agent on this project

# Target a specific file
> use the language-convention agent on pipeline.py
> review error handling in vision_verifier.py

# Natural language — Claude picks the right agent automatically
> check naming conventions across the codebase
> audit type hints in roboflow_uploader.py
> are there any silent exceptions in s3_loader.py?
```

### Typical workflow

1. Run an agent → read the report
2. Choose which issues to fix: *"Fix all CRITICAL issues in pipeline.py"*
3. Review the diff, confirm

See [.claude/agents/README.md](.claude/agents/README.md) for full documentation on each agent.

## Running Tests

```bash
# All tests (no AWS/Claude/Roboflow needed)
pytest test/ -v

# With coverage
pytest test/ -v --cov=. --cov-report=term-missing

# Single module
pytest test/test_s3_loader.py -v
```

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
