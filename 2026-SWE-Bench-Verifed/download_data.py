#!/usr/bin/env python3
"""
SWE-bench Verified 数据集下载脚本
==================================
从 HuggingFace 下载 Verified 测试集（500 实例）并保存为 JSONL。
"""

from datasets import load_dataset
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "verified_test.jsonl"
IMAGE_LIST_FILE = DATA_DIR / "docker_images.txt"

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading SWE-bench Verified dataset...")
dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')

print(f"Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in dataset:
        f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')

# Verified 使用本地构建镜像，不依赖远程 dockerhub_tag
# 生成空镜像列表文件占位
with open(IMAGE_LIST_FILE, 'w', encoding='utf-8') as f:
    f.write("# SWE-bench Verified uses locally built images via swebench\n")
    f.write("# No pre-built remote images to pull\n")

print(f"\n✅ 下载完成!")
print(f"   实例总数: {len(dataset)}")
print(f"   数据文件: {OUTPUT_FILE}")
