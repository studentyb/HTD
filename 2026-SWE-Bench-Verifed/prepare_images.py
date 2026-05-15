#!/usr/bin/env python3
"""
镜像预构建脚本
==============
为 SWE-bench Verified 实例预先构建 Docker 镜像（env + instance）。
"""

import argparse
import sys
import docker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verified_data_loader import VerifiedDataLoader
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images, build_instance_images


def main():
    parser = argparse.ArgumentParser(description="Pre-build swebench images for Verified instances")
    parser.add_argument("--instance-id", type=str, action="append", help="Single instance ID to prepare (can be used multiple times)")
    parser.add_argument("--instance-ids-file", type=str, help="File with one instance ID per line")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers for building")
    args = parser.parse_args()

    loader = VerifiedDataLoader()

    instance_ids = []
    if args.instance_id:
        instance_ids.extend(args.instance_id)
    elif args.instance_ids_file:
        with open(args.instance_ids_file, 'r') as f:
            instance_ids = [line.strip() for line in f if line.strip()]
    else:
        print("Please provide --instance-id or --instance-ids-file")
        sys.exit(1)

    # Validate instance IDs
    dataset = []
    for iid in instance_ids:
        inst = loader.get_instance(iid)
        if not inst:
            print(f"⚠ Unknown instance ID: {iid}")
            continue
        dataset.append(inst)
        print(f"✓ Found instance: {iid} ({inst['repo']})")

    if not dataset:
        print("No valid instances to prepare.")
        sys.exit(1)

    client = docker.from_env()

    print("\n[1/2] Building environment images...")
    build_env_images(
        client=client,
        dataset=dataset,
        force_rebuild=False,
        max_workers=args.max_workers,
        namespace=None,
        env_image_tag='latest',
        instance_image_tag='latest',
    )
    print("✓ Environment images ready")

    print("\n[2/2] Building instance images...")
    build_instance_images(
        client=client,
        dataset=dataset,
        force_rebuild=False,
        max_workers=args.max_workers,
        namespace=None,
        tag='latest',
        env_image_tag='latest',
    )
    print("✓ Instance images ready")

    print("\n--- Image Check ---")
    for inst in dataset:
        spec = make_test_spec(inst)
        env_exists = False
        inst_exists = False
        try:
            client.images.get(spec.env_image_key)
            env_exists = True
        except docker.errors.ImageNotFound:
            pass
        try:
            client.images.get(spec.instance_image_key)
            inst_exists = True
        except docker.errors.ImageNotFound:
            pass
        print(f"  {inst['instance_id']}: env={env_exists}, instance={inst_exists}")


if __name__ == "__main__":
    main()
