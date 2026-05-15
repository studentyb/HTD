#!/usr/bin/env python3
"""Run Group A (full tracer) for a batch of instances and collect results."""
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import subprocess

sys.path.insert(0, str(Path(__file__).parent))

from ab_tracer_experiment_verified import (
    run_common_stages,
    run_group_a,
    OUTPUT_BASE_DIR,
    print_section,
)

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
    parser.add_argument("--output", type=str, default="/tmp/ab_tracer_experiment_verified/a_only_batch_report.json")
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup after batch finishes")
    args = parser.parse_args()
    pause_between_stages = not args.no_pause

    results = []
    for instance_id in args.instances:
        print_section(f"A-only Batch: {instance_id}")
        common = run_common_stages(instance_id)
        if common is None:
            results.append({
                "instance_id": instance_id,
                "resolved": False,
                "error": "stage 1-3 failed",
                "ftp_passed": 0, "ftp_total": 0,
                "ptp_passed": 0, "ptp_total": 0,
            })
            continue

        result_a = run_group_a(common, instance_id, pause_between_stages)
        results.append({
            "instance_id": instance_id,
            "resolved": result_a["resolved"],
            "ftp_passed": result_a["ftp_passed"],
            "ftp_total": result_a["ftp_total"],
            "ptp_passed": result_a["ptp_passed"],
            "ptp_total": result_a["ptp_total"],
        })

    resolved = sum(1 for r in results if r["resolved"])
    total = len(results)
    print("\n" + "=" * 80)
    print(f" A-only Batch Report: {resolved}/{total} resolved")
    print("=" * 80)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_instances": total,
        "a_resolved": resolved,
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
    sys.exit(main())
