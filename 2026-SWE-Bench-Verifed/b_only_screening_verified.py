#!/usr/bin/env python3
"""
B-only Screening for Verified instances
=======================================
Run only Group B (no tracer probe) on a batch of instances to identify
those where LLM fails without execution trace information.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ab_tracer_experiment_verified import (
    run_common_stages,
    run_group_b,
    OUTPUT_BASE_DIR,
    print_section,
)
import argparse
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="B-only screening (no tracer)")
    parser.add_argument("--no-pause", action="store_true", help="不暂停")
    parser.add_argument("--instances", nargs='+', required=True, help="实例 ID 列表")
    args = parser.parse_args()
    pause_between_stages = not args.no_pause

    results = []
    start_all = datetime.now()

    for instance_id in args.instances:
        print_section(f"B-only Screening: {instance_id}")
        common = run_common_stages(instance_id)
        if common is None:
            results.append({
                "instance_id": instance_id,
                "resolved": False,
                "error": "stage 1-3 failed",
                "ftp_passed": 0,
                "ftp_total": 0,
                "ptp_passed": 0,
                "ptp_total": 0,
            })
            continue

        result_b = run_group_b(common, instance_id, pause_between_stages)
        results.append({
            "instance_id": instance_id,
            "resolved": result_b["resolved"],
            "ftp_passed": result_b["ftp_passed"],
            "ftp_total": result_b["ftp_total"],
            "ptp_passed": result_b["ptp_passed"],
            "ptp_total": result_b["ptp_total"],
        })

    resolved = sum(1 for r in results if r["resolved"])
    total = len(results)

    print("\n" + "=" * 80)
    print(" B-only Screening 汇总报告")
    print("=" * 80)
    print(f"总计实例数: {total}")
    print(f"B 组（无 tracer）解决数: {resolved}/{total}  ({resolved/total*100:.1f}%)")
    print("\n--- 逐实例明细 ---")
    for r in results:
        status = "✅" if r["resolved"] else "❌"
        print(f"  {status} {r['instance_id']}: FTP {r['ftp_passed']}/{r['ftp_total']}, PTP {r['ptp_passed']}/{r['ptp_total']}")

    report_path = OUTPUT_BASE_DIR / "b_only_screening_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_instances": total,
            "b_resolved": resolved,
            "details": results,
        }, f, indent=2, default=str)
    print(f"\n✓ 报告已保存到: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
