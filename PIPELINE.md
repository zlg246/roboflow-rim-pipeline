# How pipeline.py Works

`pipeline.py` is the main orchestration script for the Vehicle Panel Damage Annotation Pipeline.
It reads images from AWS S3 (read-only), runs a local YOLO model for initial damage detection,
verifies results with GPT-4o Vision, and uploads pre-annotated images to Roboflow for human review.

---

## Architecture

```
S3 (read-only)
    │  list_image_keys()        →  filtered list of S3 keys
    │  load_image()             →  PIL image + base64 string + MIME type
    ▼
YOLO (local GPU — no network)
    │  run_inference()          →  [{class, x, y, w, h, confidence}, ...]
    ▼
GPT-4o Vision (OpenAI API)
    │  verify_image()           →  {decision, confidence, reasoning, corrections:[]}
    ▼
Roboflow
    │  upload()                 →  image + VOC XML annotation uploaded
    │  create_annotation_job()  →  job appears in Review queue
    ▼
JSONL log  +  optional SNS email  +  optional EC2 self-stop
```

---

## Entry Point

```
python pipeline.py [flags]
```

`__main__` (line 357) calls `_parse_args()` then `run()`.
`run()` can also be imported and called directly (used by integration tests).

---

## Key Functions

| Function | Line | Purpose |
|---|---|---|
| `run()` | 178 | Main orchestration loop |
| `_parse_args()` | 88 | Build argparse CLI |
| `_parse_date_arg()` | 45 | Accept ISO dates, keywords (`today`, `last-friday`, etc.) |
| `_make_log_path()` | 134 | Create `logs/` directory; return timestamped JSONL path |
| `_load_processed()` | 139 | Read existing log → set of already-processed S3 keys |
| `_log()` | 150 | Append one JSON record to the JSONL log |
| `_notify()` | 155 | Publish SNS email summary (no-op if `SNS_TOPIC_ARN` unset) |
| `_stop_self()` | 169 | Stop the EC2 instance after completion (no-op if disabled) |

---

## Pipeline Loop — Step by Step

The core work happens in the `for i, key in enumerate(pending)` loop (lines 235–315).
Each iteration processes one S3 image:

1. **Load image** (`load_image`)
   Downloads from S3 via `GetObject`. Returns a PIL image, a base64 string, and the MIME type.

2. **YOLO inference** (`run_inference`)
   Runs the trained `.pt` model locally. Returns a list of normalised bounding-box dicts
   `{class, x, y, width, height, confidence}`, or `[]` if nothing detected above the threshold.

3. **Short-circuit when YOLO finds nothing**
   If predictions is empty, decision is set to `null` immediately — the GPT-4o call is
   skipped entirely to save cost and latency.

4. **GPT-4o verification** (`verify_image`)
   Sends the image (base64, `detail="high"`) and YOLO predictions to GPT-4o.
   Returns a structured JSON decision: one of six outcomes plus corrected bounding boxes.

5. **Upload gate**
   Only damage-related decisions (`correct`, `partial`, `missed`, `other`) proceed.
   `null` and `bad_quality` are logged and skipped.

6. **Dry-run check**
   If `--dry-run` was passed, the upload step is printed but not executed.
   `record["uploaded"]` is set to `"dry_run"` instead of `True/False`.

7. **Roboflow upload** (`upload`)
   Saves image as a temp JPEG and, when corrections exist, writes a Pascal VOC XML file.
   Calls `project.upload()` with `is_prediction=True` so annotations appear as model
   suggestions (not ground truth), keeping images in the human review queue.

8. **Log record** (`_log`)
   Appended after every image regardless of outcome. Includes all fields listed in
   [Log Format](#log-format) below.

---

## Decision Taxonomy

GPT-4o returns exactly one decision per image:

| Decision | Meaning | Uploaded to Roboflow? |
|---|---|---|
| `correct` | YOLO boxes are accurate; surface damage clearly visible | Yes |
| `partial` | Some boxes wrong or missing; some damage missed | Yes |
| `missed` | Damage visible but YOLO produced no predictions | Yes |
| `other` | Detections exist but are wrong class | Yes |
| `null` | No annotatable surface damage (clean panel, dent-only, shadow) | No |
| `bad_quality` | Image too dark, blurry, or overexposed | No |

The sets `UPLOAD_DECISIONS` and `SKIP_DECISIONS` in `config.py` (line 61–62) enforce this gate.

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--from DATE` | Monday this week | Inclusive start date |
| `--to DATE` | Friday this week | Inclusive end date |
| `--subfolders N,N` | `3,7` | Camera subfolder IDs to include |
| `--bucket BUCKET` | `S3_BUCKET` env var | Override which S3 bucket to scan |
| `--limit N` | none | Process at most N images (useful for testing) |
| `--dry-run` | false | Run YOLO + GPT-4o but skip Roboflow upload |

`--from` / `--to` accept ISO dates (`2026-03-16`) or keywords:
`today`, `yesterday`, `monday`…`sunday`, `last-monday`…`last-sunday`.

---

## Deduplication and Resumption

Before processing begins, `_load_processed()` reads the existing JSONL log for the current
run and collects every `s3_key` that already has a record:

```python
processed = _load_processed(log_path)
pending   = [k for k in keys if k not in processed]
```

Re-running the script with the same date range resumes from where it left off —
already-processed images are skipped automatically.

---

## Batch Naming

The Roboflow batch name is derived from the bucket and date range:

```
prod-castle-hill-toyota  +  2026-05-12  →  2026-05-16
→  castle_hill_toyota_2026-05-12_2026-05-16
```

The `prod-` prefix is stripped and hyphens become underscores so the name is
valid for Roboflow job names and readable in the UI.

---

## Configuration Surface

All runtime settings live in `config.py`, loaded from `.env` via `python-dotenv`.

| Variable | Purpose |
|---|---|
| `S3_BUCKET` | Production bucket to scan (read-only) |
| `S3_ROOT_PREFIX` | Top-level folder prefix inside the bucket |
| `S3_TARGET_SUBFOLDERS` | Camera angle subfolder IDs, e.g. `3,7` |
| `LOG_BUCKET` | Separate bucket for log uploads (never the production bucket) |
| `YOLO_MODEL_PATH` | Path to the trained `.pt` model file |
| `YOLO_CONF_THRESHOLD` | Minimum confidence for a YOLO detection (default `0.25`) |
| `OPENAI_API_KEY` | GPT-4o API key |
| `OPENAI_MODEL` | Model name (default `gpt-4o`) |
| `ROBOFLOW_API_KEY` | Roboflow API key |
| `ROBOFLOW_WORKSPACE` | Roboflow workspace slug |
| `ROBOFLOW_PROJECT` | Roboflow project slug |
| `ROBOFLOW_LABELER_EMAIL` | Annotator email for job creation |
| `ROBOFLOW_REVIEWER_EMAIL` | Reviewer email (falls back to labeler) |
| `EC2_SELF_STOP` | `true` to stop the EC2 instance after the run |
| `EC2_INSTANCE_ID` | Instance ID for self-stop |
| `SNS_TOPIC_ARN` | ARN for completion email (optional) |

---

## Log Format

One JSON object per line in `logs/pipeline_<timestamp>.jsonl`:

| Field | Type | Description |
|---|---|---|
| `s3_key` | string | Full S3 object key |
| `timestamp` | string | UTC ISO-8601 time of processing |
| `batch` | string | Roboflow batch / job name |
| `dry_run` | bool | Whether this was a dry run |
| `decision` | string | GPT-4o decision (see taxonomy above) |
| `claude_confidence` | float | GPT-4o confidence score 0–1 |
| `reasoning` | string | GPT-4o one/two sentence explanation |
| `yolo_count` | int | Number of YOLO predictions |
| `yolo_predictions` | list | Raw YOLO boxes |
| `corrections` | list | GPT-4o corrected/added/deleted boxes |
| `uploaded` | bool \| `"dry_run"` | Upload outcome |
| `error` | string | Present only when an exception was caught |

---

## Post-Run Steps

After the loop completes:

1. **Annotation job** — if at least one image was uploaded and `ROBOFLOW_LABELER_EMAIL` is set,
   `create_annotation_job()` creates a named job so images appear in Roboflow's Review column.
2. **Log upload** — `upload_log_to_s3()` copies the local JSONL to `LOG_BUCKET` (never to the
   production bucket).
3. **SNS notification** — `_notify()` emails a summary to the configured SNS topic.
4. **EC2 self-stop** — `_stop_self()` shuts down the instance if `EC2_SELF_STOP=true`.
