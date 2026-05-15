#!/usr/bin/env python3
"""Run No-Trace baseline for a batch of instances.

Skips Stage 4a (pytest trace collection) and Stage 4b (integration probe).
Uses only: problem description + buggy code + pytest failure messages (no traces).
Records: tokens consumed, LLM iterations, timing per instance.
"""
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ab_tracer_experiment_verified import (
    OUTPUT_BASE_DIR,
    print_section,
    get_output_dir,
    stage_1_load_instance,
    stage_2_prepare_repo,
    stage_3_extract_target,
    stage_7_validate_fix,
)
from manual_tracer_llm_debug_verified import get_completion_with_retry
from verified_docker_executor import VerifiedDockerTestExecutor

try:
    from config import TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS
    import config as _config
except ImportError:
    import src.config as _config


def run_no_trace(instance_id, pause_between_stages=False):
    """Run a single instance without any tracing (baseline)."""
    output_dir = get_output_dir(instance_id, "NO_TRACE")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section(f"No-Trace Batch: {instance_id}")
    instance_start = time.time()

    # Stage 1-3
    common_dir = OUTPUT_BASE_DIR / "_common" / instance_id.replace("/", "_")
    common_dir.mkdir(parents=True, exist_ok=True)

    try:
        instance, fail_to_pass, pass_to_pass = stage_1_load_instance(instance_id, common_dir)
    except Exception as e:
        print(f"\n✗ Stage 1 failed: {e}")
        return {"error": f"stage 1 failed: {e}"}

    try:
        repo_mgr = stage_2_prepare_repo(instance)
    except Exception as e:
        print(f"\n✗ Stage 2 failed: {e}")
        return {"error": f"stage 2 failed: {e}"}

    try:
        result = stage_3_extract_target(repo_mgr, instance, common_dir)
    except Exception as e:
        print(f"\n✗ Stage 3 failed: {e}")
        return {"error": f"stage 3 failed: {e}"}

    if not result[0]:
        print("\n✗ 无法继续，提取目标失败")
        return {"error": "stage 3 extract target failed"}
    target_file, target_func, buggy_code = result

    # Skip Stage 4a and 4b
    print("\n>>> [No-Trace] 跳过 Stage 4a (pytest trace) 和 Stage 4b (integration probe)")

    # Run pytest WITHOUT traces to get failure messages
    docker_executor = VerifiedDockerTestExecutor(instance, timeout=600)
    docker_executor.target_file = target_file
    docker_executor.target_func = target_func

    pytest_start = time.time()
    pytest_report, _ = docker_executor.validate_with_traces(
        buggy_code, collect_traces=False, trace_failed_only=True
    )
    pytest_elapsed = time.time() - pytest_start

    # Build failure message summary
    failure_msgs = []
    if pytest_report.fail_to_pass_passed < pytest_report.fail_to_pass_total:
        failure_msgs.append(
            f"FAIL_TO_PASS: {pytest_report.fail_to_pass_passed}/{pytest_report.fail_to_pass_total} tests passed."
        )
    if pytest_report.pass_to_pass_passed < pytest_report.pass_to_pass_total:
        failure_msgs.append(
            f"PASS_TO_PASS regress: {pytest_report.pass_to_pass_passed}/{pytest_report.pass_to_pass_total} tests passed."
        )
    failure_summary = "\n".join(failure_msgs) if failure_msgs else "All tests passed (unexpected)."

    problem_statement = instance.get('problem_statement', '')
    prompt = f"""You are an expert Python debugger. Fix the following bug based on the problem description and the buggy code.

## Problem Statement
{problem_statement if problem_statement else "Fix the bug in the provided code."}

## Buggy Code
```python
{buggy_code}
```

## Test Results (no execution traces available)
{failure_summary}

## Instructions

1. Analyze the problem statement and buggy code to identify the root cause.
2. Provide a fixed version of the buggy code that resolves the issue.
3. Make minimal changes to the existing code.
4. Explain what was wrong and how you fixed it.

## Output Format

Please provide your response in the following format:

### Explanation
<Explain the bug and your fix>

### Fixed Code
```python
<Your fixed code here>
```
"""

    # Record token usage before LLM call
    prompt_tokens_before = getattr(_config, 'TOTAL_PROMPT_TOKENS', 0)
    completion_tokens_before = getattr(_config, 'TOTAL_COMPLETION_TOKENS', 0)

    print("\n[1/2] 构建 No-Trace LLM Prompt ...")
    print(f"    Prompt 长度: {len(prompt)} 字符")
    with open(output_dir / "llm_prompt.txt", 'w') as f:
        f.write(prompt)

    print("\n[2/2] 调用 LLM (get_completion_with_retry) ...")
    llm_start = time.time()
    messages = [
        {"role": "system", "content": "You are an expert Python debugger."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = get_completion_with_retry(messages)
        llm_success = True
        llm_error = None
    except Exception as e:
        response = ""
        llm_success = False
        llm_error = str(e)
    llm_elapsed = time.time() - llm_start

    # Record token usage after LLM call
    prompt_tokens_after = getattr(_config, 'TOTAL_PROMPT_TOKENS', 0)
    completion_tokens_after = getattr(_config, 'TOTAL_COMPLETION_TOKENS', 0)
    prompt_tokens_used = prompt_tokens_after - prompt_tokens_before
    completion_tokens_used = completion_tokens_after - completion_tokens_before
    total_tokens_used = prompt_tokens_used + completion_tokens_used

    print(f"\n✓ LLM 响应完成，耗时 {llm_elapsed:.1f} 秒")
    print(f"    Tokens: prompt={prompt_tokens_used}, completion={completion_tokens_used}, total={total_tokens_used}")
    with open(output_dir / "llm_response.txt", 'w') as f:
        f.write(response)

    # Extract fixed code
    fixed_code = ""
    if llm_success and '### Fixed Code' in response:
        parts = response.split('### Fixed Code')
        if len(parts) > 1:
            code_part = parts[1]
            if '```python' in code_part:
                code = code_part.split('```python')[1]
                if '```' in code:
                    fixed_code = code.split('```')[0].strip()
                else:
                    fixed_code = code.strip()
            elif '```' in code_part:
                code = code_part.split('```')[1]
                fixed_code = code.strip()
            else:
                fixed_code = code_part.strip()

    if not fixed_code:
        print("\n⚠ 无法从 LLM 响应中提取 Fixed Code，使用原始代码")
        fixed_code = buggy_code
    else:
        with open(output_dir / "fixed_code.py", 'w') as f:
            f.write(fixed_code)
        print(f"\n✓ 提取到 fixed code，已保存到 {output_dir / 'fixed_code.py'}")

    # Stage 7: validate
    validate_start = time.time()
    report = stage_7_validate_fix(
        instance, fixed_code, target_file, target_func, output_dir
    )
    validate_elapsed = time.time() - validate_start
    total_elapsed = time.time() - instance_start

    print("\n" + "=" * 80)
    status = "✅ 已解决" if report.is_resolved else "❌ 未解决"
    print(f" No-Trace 结果: {status}")
    print(f"   FAIL_TO_PASS: {report.fail_to_pass_passed}/{report.fail_to_pass_total}")
    print(f"   PASS_TO_PASS: {report.pass_to_pass_passed}/{report.pass_to_pass_total}")
    print(f"   Timing: total={total_elapsed:.1f}s, pytest={pytest_elapsed:.1f}s, llm={llm_elapsed:.1f}s, validate={validate_elapsed:.1f}s")
    print(f"   Tokens: prompt={prompt_tokens_used}, completion={completion_tokens_used}, total={total_tokens_used}")
    print("=" * 80)

    return {
        "resolved": report.is_resolved,
        "ftp_passed": report.fail_to_pass_passed,
        "ftp_total": report.fail_to_pass_total,
        "ptp_passed": report.pass_to_pass_passed,
        "ptp_total": report.pass_to_pass_total,
        "timing": {
            "total_seconds": round(total_elapsed, 1),
            "pytest_seconds": round(pytest_elapsed, 1),
            "llm_seconds": round(llm_elapsed, 1),
            "validate_seconds": round(validate_elapsed, 1),
        },
        "tokens": {
            "prompt_tokens": prompt_tokens_used,
            "completion_tokens": completion_tokens_used,
            "total_tokens": total_tokens_used,
        },
        "llm_success": llm_success,
        "llm_error": llm_error,
    }


def cleanup_batch_images(instances):
    print("\n>>> Cleaning up batch instance images to free disk space...")
    cmd = [sys.executable, "/tmp/cleanup_batch_images.py", "--instances"] + instances
    try:
        subprocess.run(cmd, check=False, timeout=300)
    except Exception as e:
        print(f"Cleanup warning: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pause", action="store_true", help="不暂停")
    parser.add_argument("--instances", nargs='+', required=True)
    parser.add_argument("--output", type=str, default="/tmp/ab_tracer_experiment_verified/no_trace_batch_report.json")
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup after batch finishes")
    args = parser.parse_args()

    batch_start = time.time()
    results = []
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}

    for instance_id in args.instances:
        result = run_no_trace(instance_id, pause_between_stages=not args.no_pause)
        if result is None or "error" in result:
            results.append({
                "instance_id": instance_id,
                "resolved": False,
                "error": result.get("error", "unknown error") if result else "unknown error",
                "ftp_passed": 0, "ftp_total": 0,
                "ptp_passed": 0, "ptp_total": 0,
                "timing": {}, "tokens": {},
            })
        else:
            results.append({
                "instance_id": instance_id,
                "resolved": result["resolved"],
                "ftp_passed": result["ftp_passed"],
                "ftp_total": result["ftp_total"],
                "ptp_passed": result["ptp_passed"],
                "ptp_total": result["ptp_total"],
                "timing": result.get("timing", {}),
                "tokens": result.get("tokens", {}),
                "llm_success": result.get("llm_success", True),
                "llm_error": result.get("llm_error"),
            })
            tok = result.get("tokens", {})
            total_tokens["prompt"] += tok.get("prompt_tokens", 0)
            total_tokens["completion"] += tok.get("completion_tokens", 0)
            total_tokens["total"] += tok.get("total_tokens", 0)

    resolved = sum(1 for r in results if r["resolved"])
    total = len(results)
    batch_elapsed = time.time() - batch_start

    print("\n" + "=" * 80)
    print(f" No-Trace Batch Report: {resolved}/{total} resolved")
    print(f" Batch time: {batch_elapsed:.1f}s")
    print(f" Total tokens: prompt={total_tokens['prompt']}, completion={total_tokens['completion']}, total={total_tokens['total']}")
    print("=" * 80)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_instances": total,
        "resolved": resolved,
        "batch_time_seconds": round(batch_elapsed, 1),
        "total_tokens": total_tokens,
        "details": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to: {out_path}")

    if args.cleanup:
        cleanup_batch_images(args.instances)


if __name__ == "__main__":
    main()
