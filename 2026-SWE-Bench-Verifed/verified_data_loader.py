#!/usr/bin/env python3
"""
Verified 数据加载器
==================
适配 SWE-bench Verified 的字段格式（FAIL_TO_PASS / PASS_TO_PASS / test_patch）。
"""

import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional


class VerifiedDataLoader:
    """SWE-bench Verified 数据加载器"""

    def __init__(self, data_file: Optional[Path] = None):
        if data_file is None:
            data_file = Path(__file__).parent / "data" / "verified_test.jsonl"
        self.data_file = data_file
        self.problems = self._load_data()
        self.problem_map = {p['instance_id']: p for p in self.problems}

    def _load_data(self) -> List[Dict]:
        problems = []
        if not self.data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problem = json.loads(line)
                    # 标准化字段：大写 -> 小写
                    for src, dst in [('FAIL_TO_PASS', 'fail_to_pass'), ('PASS_TO_PASS', 'pass_to_pass')]:
                        if src in problem:
                            val = problem[src]
                            if isinstance(val, str):
                                try:
                                    val = json.loads(val)
                                except json.JSONDecodeError:
                                    try:
                                        val = ast.literal_eval(val)
                                    except (ValueError, SyntaxError):
                                        val = []
                            problem[dst] = val if isinstance(val, list) else []
                        else:
                            problem[dst] = []
                    # 保留 test_patch
                    problems.append(problem)
        return problems

    def get_instance(self, instance_id: str) -> Optional[Dict]:
        return self.problem_map.get(instance_id)

    def list_instances(self) -> List[str]:
        return list(self.problem_map.keys())

    def analyze_patch(self, patch: str) -> Dict:
        """分析补丁，提取修改的文件和函数"""
        files_changed = set()
        functions_modified = []

        for line in patch.split('\n'):
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[-1][2:] if parts[-1].startswith('b/') else parts[-1]
                    files_changed.add(file_path)

        for line in patch.split('\n'):
            if line.startswith('@@') and 'def ' in line:
                match = re.search(r'def\s+(\w+)', line)
                if match:
                    functions_modified.append(match.group(1))

        if not functions_modified:
            for line in patch.split('\n'):
                if line.startswith('+def ') or line.startswith('-def '):
                    match = re.search(r'def\s+(\w+)', line)
                    if match:
                        functions_modified.append(match.group(1))

        target_file = None
        for f in files_changed:
            if 'test' not in f.lower() and f.endswith('.py'):
                target_file = f
                break

        return {
            'files_changed': list(files_changed),
            'target_file': target_file,
            'functions_modified': list(set(functions_modified)),
            'is_simple': len(files_changed) == 1
        }


if __name__ == "__main__":
    loader = VerifiedDataLoader()
    print(f"Loaded {len(loader.problems)} instances")
    print(f"First instance: {loader.list_instances()[0]}")
    inst = loader.get_instance(loader.list_instances()[0])
    print(f"fail_to_pass count: {len(inst.get('fail_to_pass', []))}")
    print(f"pass_to_pass count: {len(inst.get('pass_to_pass', []))}")
    print(f"Has test_patch: {bool(inst.get('test_patch'))}")
