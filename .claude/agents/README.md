# Project Agents

Custom sub-agents for this project. Place each agent definition as a `.md` file in this
directory. Claude Code loads them automatically — no registration step needed.

---

## error-handling

**File**: `error-handling.md`

Audits Python files for error-handling defects and produces a detailed remediation plan.
**Never modifies any file** — all output is analysis and recommended fixes only.

### What it checks

| Severity | Examples |
|---|---|
| 🔴 CRITICAL | Silent `except: pass`, swallowed exceptions, missing resource cleanup (`finally`/context manager), catching `SystemExit` |
| 🟡 BUG | Lost exception context (`raise X` vs `raise X from e`), wrong exception type caught, sentinel return values that mask errors |
| 🔵 NICE-TO-HAVE | `print()` used instead of `logging`, missing `exc_info=True`, no custom exception hierarchy, overly wide `try` blocks |

It also checks **error propagation**: does every exception eventually surface to the
top-level caller, or does it disappear silently?

### How to invoke

**Scan all Python files in the project:**
```
> use the error-handling agent on this project
```

**Scan a specific file:**
```
> use the error-handling agent on pipeline.py
```
```
> use the error-handling agent on vision_verifier.py and roboflow_uploader.py
```

You can also trigger it by describing the task naturally:
```
> review error handling in s3_loader.py
> audit exceptions across the whole codebase
> check if errors bubble up correctly in pipeline.py
```

### Output format

The agent produces a Markdown report structured as:

```
# Error Handling Audit — <date>

## Summary table  (file × severity counts)

## Per-file issues
  Each issue has:
  - Severity label
  - Exact file + line number
  - Current (broken) code snippet
  - Problem explanation
  - Whether the error bubbles up
  - Recommended fix snippet
  - Best-practice reference

## Remediation priority  (ordered action list)

## What NOT to change  (intentional patterns flagged as safe)
```

### Workflow — approval before changes

The agent **only analyses** — it will never edit a file on its own.

Typical workflow:
1. Run the agent → read the report
2. Pick the issues you want fixed
3. Tell Claude: *"Fix issue 2 and issue 5 from the report"* or *"Fix all CRITICAL issues in pipeline.py"*
4. Review the diff before confirming

---

## language-convention

**File**: `language-convention.md`

Audits Python files for adherence to industrial best practices.
**Never modifies any file** — all output is analysis and recommended fixes only.

### What it checks

| Category | Examples |
|---|---|
| Naming (PEP 8) | `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, no shadowed built-ins |
| Type hints | Missing parameter/return annotations, bare `dict`/`list`, `Optional` vs `X \| None` |
| Docstrings | Missing module/function/class docstrings, stale descriptions, Google-style format |
| Logging | `print()` in production code, wrong log levels, missing `exc_info=True` on exception logs |
| Magic literals | Repeated string/numeric literals that should be named constants |
| Data structures | Raw `dict` with a known schema that should be `TypedDict` or `dataclass` |
| Imports | PEP 8 grouping (stdlib → third-party → local), alphabetical order, unused imports |
| Misc | Mutable defaults, `os.path` vs `pathlib`, missing context managers |

### How to invoke

**Scan all Python files in the project:**
```
> use the language-convention agent on this project
```

**Scan a specific file:**
```
> use the language-convention agent on pipeline.py
```
```
> review code style in vision_verifier.py and roboflow_uploader.py
> check naming conventions across the codebase
> audit type hints in config.py
```

### Output format

```
# Language Convention Audit — <date>

## Summary table  (file × severity counts)

## Per-file issues
  Each issue has:
  - Severity and category label
  - Exact file + line number
  - Current code snippet
  - Problem explanation
  - Recommended fix snippet
  - PEP / best-practice reference

## Remediation priority  (ordered action list)

## What NOT to change  (intentional patterns flagged as safe)
```

### Workflow — approval before changes

The agent **only analyses** — it will never edit a file on its own.

Typical workflow:
1. Run the agent → read the report
2. Pick the issues you want fixed
3. Tell Claude: *"Fix all type-hint issues in pipeline.py"* or *"Apply issue 3 and issue 7"*
4. Review the diff before confirming

---

### Adding more agents

Add a new `.md` file to this directory with frontmatter:

```markdown
---
name: my-agent
description: When Claude Code should invoke this agent automatically.
tools: Read, Bash   # comma-separated list of allowed tools
---

Agent system prompt here...
```

Claude Code picks it up on the next session without any further configuration.
