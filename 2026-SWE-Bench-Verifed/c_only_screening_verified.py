#!/usr/bin/env python3
"""
C-only Screening for Verified instances
=======================================
Run a "zero-info" baseline: only problem statement + buggy code.
Skips both stage 4a (pytest traces) and 4b (tracer probe).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ab_tracer_experiment_verified import (
    stage_1_load_instance,
    stage_2_prepare_repo,
    stage_3_extract_target,
    stage_5_analyze_traces,
    stage_6_llm_fix,
    stage_7_validate_fix,
    OUTPUT_BASE_DIR,
    get_output_dir,
    print_section,
)
import argparse
import json
from datetime import datetime


def run_group_c(instance_id):
    common_dir = OUTPUT_BASE_DIR / "_common" / instance_id.replace("/", "_")
    common_dir.mkdir(parents=True, exist_ok=True)

    instance, fail_to_pass, pass_to_pass = stage_1_load_instance(instance_id, common_dir)
    repo_mgr = stage_2_prepare_repo(instance)
    result = stage_3_extract_target(repo_mgr, instance, common_dir)
    if not result[0]:
        return None
    target_file, target_func, buggy_code = result

    output_dir = get_output_dir(instance_id, "C")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip 4a and 4b entirely
    pytest_traces = []
    probe_data = {"trace_count": 0, "traces": [], "exception_info": None, "import_error": "Skipped in group C"}

    analysis = stage_5_analyze_traces(pytest_traces, probe_data, target_func, output_dir)
    fixed_code, explanation = stage_6_llm_fix(buggy_code, target_func, analysis, instance, output_dir)
    report = stage_7_validate_fix(instance, fixed_code, target_file, target_func, output_dir)

    return {
        "instance_id": instance_id,
        "resolved": report.is_resolved,
        "ftp_passed": report.fail_to_pass_passed,
        "ftp_total": report.fail_to_pass_total,
        "ptp_passed": report.pass_to_pass_passed,
        "ptp_total": report.pass_to_pass_total,
    }


def main():
    parser = argparse.ArgumentParser(description="C-only screening (zero execution info)")
    parser.add_argument("--instances", nargs='+', required=True, help="实例 ID 列表")
    args = parser.parse_args()

    results = []

    for instance_id in args.instances:
        print_section(f"C-only Screening: {instance_id}")
        result_c = run_group_c(instance_id)
        if result_c is None:
            results.append({
                "instance_id": instance_id,
                "resolved": False,
                "error": "stage 1-3 failed",
                "ftp_passed": 0, "ftp_total": 0,
                "ptp_passed": 0, "ptp_total": 0,
            })
            continue

        results.append(result_c)

    resolved = sum(1 for r in results if r["resolved"])
    total = len(results)

    print("\n" + "=" * 80)
    print(" C-only Screening 汇总报告")
    print("=" * 80)
    print(f"总计实例数: {total}")
    print(f"C 组（纯文本，无执行信息）解决数: {resolved}/{total}  ({resolved/total*100:.1f}%)")
    print("\n--- 逐实例明细 ---")
    for r in results:
        status = "✅" if r["resolved"] else "❌"
        print(f"  {status} {r['instance_id']}: FTP {r['ftp_passed']}/{r['ftp_total']}, PTP {r['ptp_passed']}/{r['ptp_total']}")

    report_path = OUTPUT_BASE_DIR / "c_only_screening_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_instances": total,
            "c_resolved": resolved,
            "details": results,
        }, f, indent=2, default=str)
    print(f"\n✓ 报告已保存到: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
