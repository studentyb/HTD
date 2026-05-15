# A-Group (Full Tracer) Run Guide

## Overview

A-Group runs the complete tracer pipeline (Stage 1-7), including:
- **Stage 4a**: `FineGrainedTracer` collects fine-grained execution traces of failed pytest tests
- **Stage 4b**: `Integration Probe` injects `_MockAttr` mocks and executes the target function

This is the **treatment group** in the paper, designed to validate the gain of dynamic execution tracing for LLM bug fixing.

---

## Environment Requirements

| Item | Version / Notes |
|------|-----------------|
| Python | 3.10+ |
| Docker | Installed and running, supports `sweb.eval.*` images |
| WSL2 | Project runs on WSL2 + overlayfs |
| LLM | `kimi-k2.5`, called via Kimi API (`https://api.kimi.com/coding`) |
| Disk | Stabilizes at ~5% after each batch cleanup; reserve 50GB+ |

---

## File Descriptions

| File | Purpose |
|------|---------|
| `ab_tracer_experiment_verified.py` | Main experiment script defining Stage 1-7 |
| `a_only_batch_runner.py` | A-group batch entry point, calls `run_group_a()` |
| `run_batches_4_to_10_micro.sh` | Micro-batch scheduler for Batch 4-10 (50 micro-batches x 7 instances) |
| `run_batches_2_to_10.sh` | Batch scheduler for Batch 2-10 (backup) |
| `run_batches_4_to_10_resume.sh` | Resume-from-checkpoint script |
| `manual_tracer_llm_debug_verified.py` | LLM calling, prompt templates, core tracer logic |
| `verified_docker_executor.py` | Docker executor, handles pytest + trace injection |
| `verified_data_loader.py` | SWE-bench instance loader |
| `config.py` | LLM config, temperature, token tracking, tracer parameters |
| `llm_client.py` | Anthropic API client with exponential backoff retry |
| `prepare_images.py` | Docker image pre-build |
| `batch_results/` | Report output directory |

---

## Configuration

Edit `../config.py` (or `./config.py`):

```python
MODEL = "kimi-k2.5"
LLM_BASE_URL = "https://api.kimi.com/coding"
LLM_API_KEY = "sk-kimi-..."

TEMPERATURE = 0.8
MAX_VLLM_RETRIES = 10
```

**Tracer parameters** (in the same file):
```python
TRACER_MAX_DEPTH = 1000          # Maximum trace steps
TRACER_LOOP_SAMPLING = 5         # Loop sampling frequency
TRACER_MAX_VAR_SIZE = 100        # Max variable repr length
TRACER_ENABLE_DU_CHAIN = True    # Track def-use chain
```

---

## Running Steps

### Method 1: Manual Single Instance (for debugging)

```bash
python3 a_only_batch_runner.py \
    --no-pause \
    --instances django__django-11790 \
    --output /tmp/a_single_report.json
```

### Method 2: Run a Batch (10-50 instances)

```bash
python3 a_only_batch_runner.py \
    --no-pause \
    --instances django__django-11790 django__django-11821 sympy__sympy-13031 \
    --output batch_results/batch_x_report.json \
    --cleanup
```

### Method 3: Fully Automatic Batch 4-10 (recommended)

```bash
bash run_batches_4_to_10_micro.sh
```

This script will:
1. Read lines 151-500 of `verified_all_500_instances.txt` (350 instances)
2. Split into 50 micro-batches (7 instances each)
3. Invoke `a_only_batch_runner.py` one by one
4. Output `batch_results/micro_batch_{1..50}_report.json`
5. Auto-clean Docker images after each batch
6. Log to `batch_results/micro_batch_runner.log`

### Method 4: Resume from Checkpoint

If interrupted, resume from a specific micro-batch:

```bash
bash run_batches_4_to_10_resume.sh
```

---

## Report Format

`batch_results/micro_batch_{N}_report.json`:

```json
{
    "timestamp": "2026-04-20T10:00:00",
    "total_instances": 7,
    "a_resolved": 6,
    "batch_time_seconds": 3600,
    "total_tokens": {"prompt": 5000, "completion": 8000},
    "details": [
        {
            "instance_id": "django__django-11790",
            "resolved": true,
            "ftp_passed": 1,
            "ftp_total": 1,
            "ptp_passed": 18,
            "ptp_total": 18
        }
    ]
}
```

- `resolved`: Whether all validations passed (fail-to-pass + pass-to-pass)
- `ftp_passed/total`: Failed test patch fix status
- `ptp_passed/total`: Pass-to-pass regression test status

---

## Experiment Results (This Version)

| Batch Range | Instances | Resolved | Success Rate |
|-------------|-----------|----------|--------------|
| Batch 2-3 | 100 | 99 | 99.0% |
| Batch 4-10 (micro) | 350 | 308 | 88.0% |
| **Total** | **~450** | **~407** | **~90.4%** |

> Note: Batch 1 report is missing, estimated ~48/50. Full 500-instance A-group is counted as **436/500 = 87.2%**.

---

## Troubleshooting

### pytest Not Executed (pytest_seconds=0)
- Check if Docker images are built: `docker images | grep sweb.eval`
- Check if `verified_all_500_instances.txt` path is correct
- Check Docker errors in `batch_results/micro_batch_runner.log`

### Rate Limit
- `llm_client.py` has built-in exponential backoff (30s->300s)
- If persistent, reduce concurrency or switch API key

### Disk Full
- Images are auto-cleaned after each batch
- Manual cleanup: `python3 /tmp/cleanup_batch_images.py --instances <instance_id>`

---

## Key Code Entry Points

| Functionality | Entry Function | File |
|---------------|----------------|------|
| Full pipeline | `run_group_a()` | `ab_tracer_experiment_verified.py` |
| Trace collection | `stage_4a_pytest_traces()` | `ab_tracer_experiment_verified.py` |
| Integration Probe | `stage_4b_integration_probe()` | `ab_tracer_experiment_verified.py` |
| LLM calling | `get_completion_with_retry()` | `llm_client.py` |
| Prompt assembly | `stage_5_analyze_traces()` | `manual_tracer_llm_debug_verified.py` |
| Docker execution | `validate_with_traces()` | `verified_docker_executor.py` |
