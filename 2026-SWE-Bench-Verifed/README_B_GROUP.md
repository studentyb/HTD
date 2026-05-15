# B-Group (No-Trace Baseline) Run Guide

## Overview

B-Group is the **control group (ablation)** of the A/B experiment, running a **trace-free** baseline pipeline. The only difference from A-Group is:

- **Skip Stage 4a**: Do not collect pytest execution traces (`collect_traces=False`)
- **Skip Stage 4b**: Do not run Integration Probe
- **Prompt excludes**: Any runtime trace data or probe results

The LLM only receives: problem statement + buggy code + pytest failure summary.

---

## Environment Requirements

Identical to A-Group:

| Item | Version / Notes |
|------|-----------------|
| Python | 3.10+ |
| Docker | Installed and running |
| WSL2 | Project runs on WSL2 + overlayfs |
| LLM | `kimi-k2.5`, `temperature=0.8` |
| Disk | Stabilizes at ~5% after each batch cleanup |

---

## File Descriptions

| File | Purpose |
|------|---------|
| `no_trace_batch_runner.py` | B-group main script, skips Stage 4a/4b |
| `run_no_trace_50batches.sh` | 50-batch fully automatic scheduler |
| `ab_tracer_experiment_verified.py` | Reused Stage 1-3, 5-7 functions |
| `manual_tracer_llm_debug_verified.py` | Prompt template (minimal version) |
| `verified_docker_executor.py` | Docker executor (no trace injection) |
| `config.py` / `llm_client.py` | Shared with A-Group |
| `batch_results/no_trace/` | Report output directory |

---

## Configuration

Shares the same configuration file `config.py` with A-Group:

```python
MODEL = "kimi-k2.5"
LLM_BASE_URL = "https://api.kimi.com/coding"
LLM_API_KEY = "sk-kimi-..."
TEMPERATURE = 0.8
```

> B-group does **not** modify tracer-related configs because the tracer is directly skipped.

---

## Running Steps

### Method 1: Manual Single Instance (for debugging)

```bash
python3 no_trace_batch_runner.py \
    --no-pause \
    --instances django__django-11790 \
    --output /tmp/b_single_report.json
```

### Method 2: Run a Batch (10 instances)

```bash
python3 no_trace_batch_runner.py \
    --no-pause \
    --instances django__django-11790 django__django-11821 sympy__sympy-13031 \
    --output batch_results/no_trace/no_trace_batch_x_report.json \
    --cleanup
```

### Method 3: Fully Automatic 50 Batches (recommended)

```bash
bash run_no_trace_50batches.sh
```

This script will:
1. Read `verified_all_500_instances.txt` (500 instances)
2. Split into 50 batches (10 instances each)
3. Invoke `no_trace_batch_runner.py` one by one
4. Output `batch_results/no_trace/no_trace_batch_{1..50}_report.json`
5. Auto-clean Docker images after each batch
6. Log to `batch_results/no_trace/no_trace_runner.log`
7. Auto-skip already completed batches (resume support)

**Run in background**:
```bash
nohup bash run_no_trace_50batches.sh > batch_results/no_trace/nohup.out 2>&1 &
```

---

## Report Format

`batch_results/no_trace/no_trace_batch_{N}_report.json`:

```json
{
    "timestamp": "2026-04-29T10:00:00",
    "total_instances": 10,
    "resolved": 9,
    "batch_time_seconds": 5400,
    "total_tokens": {"prompt": 10000, "completion": 20000},
    "details": [
        {
            "instance_id": "django__django-11790",
            "resolved": true,
            "ftp_passed": 1,
            "ftp_total": 1,
            "ptp_passed": 18,
            "ptp_total": 18,
            "llm_success": true,
            "llm_error": null,
            "timing": {
                "total_seconds": 300.0,
                "pytest_seconds": 120.0,
                "llm_seconds": 30.0,
                "validate_seconds": 150.0
            },
            "tokens": {
                "prompt_tokens": 1500,
                "completion_tokens": 2000,
                "total_tokens": 3500
            }
        }
    ]
}
```

**Key field descriptions**:
- `resolved`: Whether the fix passed validation
- `llm_success`: Whether the LLM call succeeded (different from fix success)
- `timing.pytest_seconds`: **Diagnostic metric**. If `0.0`, pytest did not execute (environment failure)
- `timing.validate_seconds`: Validation phase duration

---

## Experiment Results (This Version)

| Metric | Value |
|--------|-------|
| Total instances | 500 |
| Successfully fixed | **371** |
| Overall success rate | **74.2%** |
| Total time | 179.7 hours (~7.5 days) |
| Total tokens | ~3.11 million |
| Fastest batch | Batch 43: 1.1 hours (10/10) |
| Slowest batch | Batch 31: 36 hours (10/10, xarray) |

### Per-Project Statistics

| Project | Instances | Resolved | Success Rate |
|---------|-----------|----------|--------------|
| sympy | 75 | 74 | **98.7%** |
| sphinx-doc | 44 | 43 | **97.7%** |
| pydata | 22 | 21 | **95.5%** |
| matplotlib | 34 | 30 | **88.2%** |
| pylint-dev | 10 | 9 | **90.0%** |
| psf | 8 | 7 | **87.5%** |
| pytest-dev | 19 | 15 | **78.9%** |
| scikit-learn | 32 | 24 | **75.0%** |
| django | 231 | 145 | **62.8%** |
| astropy | 22 | 0 | **0.0%** |

> **Note**: All astropy instances failed, and early batches (Batch 1-6, 10-13) had extremely low success rates (0-10%). The root cause is `pytest_seconds=0` (pytest did not execute due to environment configuration failures), not LLM capability limitations.

---

## Comparison with A-Group

| Dimension | A-Group (Full Tracer) | B-Group (No Trace) |
|-----------|----------------------|-------------------|
| Instances | 500 | 500 |
| Success rate | **87.2%** (436/500) | **74.2%** (371/500) |
| Includes Stage 4a | Yes (pytest execution trace) | No (skipped) |
| Includes Stage 4b | Yes (Integration Probe) | No (skipped) |
| Prompt content | Code + behavioral trace | Code + test summary |
| Avg time / instance | ~21 minutes | ~21 minutes |
| LLM temperature | 0.8 | 0.8 |

**Tracer incremental contribution: ~+13.0 percentage points** (87.2% vs 74.2%).

---

## Troubleshooting

### pytest_seconds = 0 (Early Batch Problem)

**Symptom**: Batch 1-6, 10-13 all failed, `pytest_seconds=0.0`, `validate_seconds=0.1s`.

**Root cause**: Docker environment not ready or pytest command execution failed.

**Solution**:
1. Check Docker daemon: `docker ps`
2. Check if images exist: `docker images | grep sweb.eval`
3. Manually run a single instance to verify the environment
4. Re-run the failed batches

### Batch Hanging for Hours

**Symptom**: A batch runs for more than 10 hours with no output.

**Possible causes**:
- Specific project instances are extremely slow (e.g., xarray, Batch 31 took 36 hours)
- Docker container is hung

**Solution**:
```bash
# Check current containers
docker ps

# If process is dead, kill and re-run
kill -9 <PID>
# The script will auto-skip completed batches
bash run_no_trace_50batches.sh
```

### Rate Limit / API Errors
- Same as A-Group, built-in exponential backoff
- Check errors in `batch_results/no_trace/no_trace_runner.log`

---

## Key Code Entry Points

| Functionality | Entry Function | File |
|---------------|----------------|------|
| No-trace full pipeline | `run_group_no_trace()` | `no_trace_batch_runner.py` |
| Trace-skipping pytest | `validate_with_traces(collect_traces=False)` | `verified_docker_executor.py` |
| Minimal Prompt | `run_group_no_trace()` internal prompt assembly | `no_trace_batch_runner.py` |
| Batch scheduling | `main()` loop | `run_no_trace_50batches.sh` |

---

## Re-running Failed Batches

To re-run Batch 1-6 for a clean baseline:

```bash
# Delete old reports for failed batches
rm batch_results/no_trace/no_trace_batch_{1,2,3,4,5,6,10,11,12,13}_report.json

# Re-run
bash run_no_trace_50batches.sh
```

The script will automatically skip existing reports and only re-run the deleted batches.
