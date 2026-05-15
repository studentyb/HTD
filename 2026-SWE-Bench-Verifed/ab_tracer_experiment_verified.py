#!/usr/bin/env python3
"""
Tracer A/B 对比实验 (Verified 适配版)
======================================
对同一组 SWE-bench Verified 实例，分别运行：
  A 组（Full Tracer）: 完整 7 阶段，含 4b 集成探测真实 trace
  B 组（No Probe）   : 跳过 4b，仅使用问题描述 + buggy code + pytest 失败信息
"""

import sys
import os
import json
import time
import re
import tempfile
import shutil
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "swe_bench_pro"))

from verified_data_loader import VerifiedDataLoader
from verified_docker_executor import VerifiedDockerTestExecutor
from manual_tracer_llm_debug_verified import (
    stage_1_load_instance,
    stage_2_prepare_repo,
    stage_3_extract_target,
    stage_5_analyze_traces,
    stage_6_llm_fix,
    stage_7_validate_fix,
)
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_container, cleanup_container
from swebench.harness.docker_utils import copy_to_container
import docker
import logging

OUTPUT_BASE_DIR = Path("/tmp/ab_tracer_experiment_verified")


def get_output_dir(instance_id: str, group: str) -> Path:
    safe_id = instance_id.replace("/", "_").replace(":", "_")
    d = OUTPUT_BASE_DIR / safe_id / group
    d.mkdir(parents=True, exist_ok=True)
    return d


def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def pause():
    input("\n按 Enter 继续...")


def stage_4a_pytest_traces(instance, buggy_code, target_func, target_file, fail_to_pass, output_dir):
    print_section("阶段 4a: pytest Trace 收集（仅失败测试）")
    docker_executor = VerifiedDockerTestExecutor(instance, timeout=600)
    docker_executor.target_file = target_file
    docker_executor.target_func = target_func
    print(f"    镜像: {docker_executor.image_name}")
    print(f"    目标文件: {docker_executor.target_file}")
    print(f"    目标函数: {docker_executor.target_func}")

    print("\n    开始运行 pytest 并收集 traces ...")
    print("    -" * 30)
    start = time.time()
    pytest_report, pytest_traces = docker_executor.validate_with_traces(
        buggy_code, collect_traces=True, trace_failed_only=True
    )
    elapsed = time.time() - start
    print("    -" * 30)
    print(f"\n    ✓ pytest trace 收集完成，耗时 {elapsed:.1f} 秒")
    print(f"    FAIL_TO_PASS: {pytest_report.fail_to_pass_passed}/{pytest_report.fail_to_pass_total}")
    print(f"    PASS_TO_PASS: {pytest_report.pass_to_pass_passed}/{pytest_report.pass_to_pass_total}")

    with open(output_dir / "docker_trace_report_pytest.json", 'w') as f:
        json.dump(pytest_report.to_dict(), f, indent=2, default=str)
    with open(output_dir / "docker_traces_pytest.json", 'w') as f:
        json.dump(pytest_traces, f, indent=2, default=str)

    print("\n    --- pytest Trace 摘要 ---")
    for t in pytest_traces[:3]:
        test_name = t.get('test_name', 'unknown')
        trace_data = t.get('trace', {})
        test_passed = trace_data.get('test_passed', False)
        trace_summary = trace_data.get('trace_summary', '')[:300]
        print(f"\n    >>> {test_name}: {'通过' if test_passed else '失败'}")
        print(f"        {trace_summary}")

    return pytest_report, pytest_traces


def stage_4b_integration_probe(instance, target_func, target_file, output_dir):
    print_section("阶段 4b: 无 mock 集成探测 Trace 收集")

    old_probe = output_dir / "probe_trace_result.json"
    if old_probe.exists():
        old_probe.unlink()

    tracer_lib_dir = Path(tempfile.mkdtemp(prefix='tracer_lib_'))
    src_dir = Path(__file__).parent.parent
    for fname in ['tracer.py', 'utils.py', 'config.py', 'timeout_utils.py']:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, tracer_lib_dir)
    (tracer_lib_dir / 'loguru.py').write_text(
        "class Logger:\n"
        "    def info(self, *a, **k): pass\n"
        "    def debug(self, *a, **k): pass\n"
        "    def warning(self, *a, **k): pass\n"
        "    def error(self, *a, **k): pass\n"
        "    def trace(self, *a, **k): pass\n"
        "    def remove(self, *a, **k): pass\n"
        "    def add(self, *a, **k): pass\n"
        "logger = Logger()\n"
    )

    target_module = target_file.replace('/', '.').replace('.py', '')
    if target_module.startswith('lib.'):
        target_module = target_module[4:]

    probe_script = (
        '#!/usr/bin/env python3\n'
        'import sys, os, json, importlib.util, inspect\n'
        "sys.path.insert(0, '/tmp/tracer_lib')\n"
        "sys.path.insert(0, '/testbed')\n"
        '\n'
        'class _MockOpenAI:\n'
        '    pass\n'
        "sys.modules['openai'] = _MockOpenAI()\n"
        '\n'
        'class _MockSix:\n'
        '    PY2 = False\n'
        '    PY3 = True\n'
        '    string_types = (str,)\n'
        '    text_type = str\n'
        '    binary_type = bytes\n'
        '    integer_types = (int,)\n'
        '    class moves:\n'
        '        pass\n'
        "sys.modules['six'] = _MockSix()\n"
        '\n'
        'class _MockPytz:\n'
        '    class UTC:\n'
        '        pass\n'
        '    utc = UTC()\n'
        '    @classmethod\n'
        '    def timezone(cls, name):\n'
        '        return cls.UTC()\n'
        "sys.modules['pytz'] = _MockPytz()\n"
        '\n'
        'class _MockSqlparse:\n'
        '    def parse(self, *a, **k):\n'
        '        return []\n'
        '    def format(self, *a, **k):\n'
        '        return ""\n'
        "sys.modules['sqlparse'] = _MockSqlparse()\n"
        '\n'
        'class _MockAttr:\n'
        '    def __getattr__(self, name):\n'
        '        return _MockAttr()\n'
        '    def __iter__(self):\n'
        '        return iter([])\n'
        '    def __getitem__(self, key):\n'
        '        return _MockAttr()\n'
        '    def __len__(self):\n'
        '        return 0\n'
        '    def __call__(self, *a, **k):\n'
        '        return _MockAttr()\n'
        '    def __enter__(self):\n'
        '        return self\n'
        '    def __exit__(self, *a, **k):\n'
        '        pass\n'
        '    def __bool__(self):\n'
        '        return True\n'
        '    __nonzero__ = __bool__\n'
        '_MockAttr.__version__ = "1.0.0"\n'
        'for _mock_mod in ("asgiref", "asgiref.sync", "asgiref.local", "yaml", "toml", "certifi", "cryptography"):\n'
        '    sys.modules[_mock_mod] = _MockAttr()\n'
        '\n'
        'try:\n'
        '    import django\n'
        '    from django.conf import settings\n'
        '    if not settings.configured:\n'
        '        settings.configure()\n'
        'except Exception:\n'
        '    pass\n'
        '\n'
        'import config as tracer_config\n'
        'tracer_config.TRACER_LOOP_SAMPLING = 1\n'
        '\n'
        'from tracer import FineGrainedTracer, TraceStrategy\n'
        '\n'
        f"target_abs_path = os.path.join('/testbed', {repr(target_file)})\n"
        f"target_module_name = {repr(target_module)}\n"
        f"target_func_name = {repr(target_func)}\n"
        '\n'
        'def _patched_is_user_code(self, frame) -> bool:\n'
        '    filename = frame.f_code.co_filename\n'
        '    func_name = frame.f_code.co_name\n'
        "    if func_name in ('<lambda>', '__create_fn__'):\n"
        '        return False\n'
        '    if filename == target_abs_path:\n'
        '        return True\n'
        '    return False\n'
        '\n'
        'FineGrainedTracer._is_user_code = _patched_is_user_code\n'
        '\n'
        'strategy = TraceStrategy.MINIMAL.copy()\n'
        "strategy.pop('max_trace_depth', None)\n"
        'tracer = FineGrainedTracer(max_trace_depth=50000, **strategy)\n'
        '\n'
        'def _heuristic_value(name):\n'
        "    lname = name.lower()\n"
        "    if lname in ('method',):\n"
        "        return 'GET'\n"
        "    if lname in ('url', 'uri', 'path', 'host', 'endpoint'):\n"
        "        return 'http://example.com'\n"
        "    if 'header' in lname:\n"
        '        return {}\n'
        "    if lname in ('data', 'body', 'content', 'payload'):\n"
        '        return b""\n'
        "    if 'timeout' in lname:\n"
        '        return 30\n'
        "    if lname in ('port',):\n"
        '        return 80\n'
        "    if 'use_' in lname or lname in ('force', 'validate', 'decompress', 'use_proxy', 'use_gssapi'):\n"
        '        return True\n'
        "    if lname in ('headers', 'cookies', 'params', 'query'):\n"
        '        return {}\n'
        "    if lname in ('qid', 'id'):\n"
        "        return 'Q12345'\n"
        "    if lname in ('seed',):\n"
        "        return 'test_seed'\n"
        "    if lname in ('bust_cache', 'fetch_missing'):\n"
        '        return False\n'
        '    return None\n'
        '\n'
        'def _call_with_heuristics(obj):\n'
        '    try:\n'
        '        sig = inspect.signature(obj)\n'
        '    except Exception:\n'
        '        sig = None\n'
        '    if sig:\n'
        '        args = []\n'
        '        kwargs = {}\n'
        '        for i, (name, param) in enumerate(sig.parameters.items()):\n'
        "            if i == 0 and name in ('self', 'cls'):\n"
        '                continue\n'
        '            val = param.default if param.default is not inspect.Parameter.empty else _heuristic_value(name)\n'
        '            if param.kind == inspect.Parameter.POSITIONAL_ONLY:\n'
        '                args.append(val)\n'
        '            elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:\n'
        '                args.append(val)\n'
        '            elif param.kind == inspect.Parameter.KEYWORD_ONLY:\n'
        '                kwargs[name] = val\n'
        '            elif param.kind == inspect.Parameter.VAR_POSITIONAL:\n'
        '                args.append(val)\n'
        '        return obj(*args, **kwargs)\n'
        '    return obj()\n'
        '\n'
        'mod = None\n'
        'import_error = None\n'
        '\n'
        'try:\n'
        '    mod = importlib.import_module(target_module_name)\n'
        'except Exception as e:\n'
        '    import_error = str(e)\n'
        '\n'
        'if mod is None:\n'
        '    try:\n'
        '        spec = importlib.util.spec_from_file_location(target_module_name, target_abs_path)\n'
        '        mod = importlib.util.module_from_spec(spec)\n'
        '        sys.modules[target_module_name] = mod\n'
        '        spec.loader.exec_module(mod)\n'
        '    except Exception as e:\n'
        '        import_error = (import_error or "") + "; " + str(e)\n'
        '\n'
        'obj = None\n'
        'owner_class = None\n'
        '\n'
        'if mod is not None:\n'
        '    obj = getattr(mod, target_func_name, None)\n'
        '    if obj is None:\n'
        '        for name in dir(mod):\n'
        '            attr = getattr(mod, name)\n'
        '            if isinstance(attr, type) and hasattr(attr, target_func_name):\n'
        '                obj = getattr(attr, target_func_name, None)\n'
        '                owner_class = attr\n'
        '                break\n'
        '\n'
        'if callable(obj):\n'
        '    tracer.start(user_func_name=target_func_name)\n'
        '    try:\n'
        '        if isinstance(obj, type):\n'
        '            instance = obj()\n'
        '            if hasattr(instance, target_func_name):\n'
        '                method = getattr(instance, target_func_name)\n'
        '                _call_with_heuristics(method)\n'
        '            else:\n'
        '                _call_with_heuristics(instance)\n'
        '        elif owner_class is not None:\n'
        '            try:\n'
        '                class_bound = getattr(owner_class, target_func_name)\n'
        '                _call_with_heuristics(class_bound)\n'
        '            except Exception:\n'
        '                instance = owner_class()\n'
        '                method = getattr(instance, target_func_name)\n'
        '                _call_with_heuristics(method)\n'
        '        else:\n'
        '            _call_with_heuristics(obj)\n'
        '    except Exception as e:\n'
        '        pass\n'
        '    tracer.stop()\n'
        'else:\n'
        '    print(f"Target not callable or not found: {target_func_name} (import_error={import_error})", file=sys.stderr)\n'
        '\n'
        'result = {\n'
        "    'trace_count': len(tracer.traces),\n"
        "    'traces': tracer.traces,\n"
        "    'exception_info': tracer.exception_info,\n"
        "    'import_error': import_error,\n"
        "    'target_module': target_module_name,\n"
        "    'target_func': target_func_name,\n"
        '}\n'
        "with open('/tmp/probe_out/probe_trace_result.json', 'w') as f:\n"
        '    json.dump(result, f, indent=2, default=str)\n'
        '\n'
        'print(f"Probe completed: {len(tracer.traces)} traces captured", file=sys.stderr)\n'
    )

    probe_path = output_dir / "integration_probe.py"
    with open(probe_path, 'w') as f:
        f.write(probe_script)

    print(f"\n    ✓ 集成探测脚本已生成: {probe_path}")
    print(f"    目标模块: {target_module}")
    print(f"    目标函数: {target_func}")

    client = docker.from_env()
    test_spec = make_test_spec(instance)
    logger = logging.getLogger("probe_ab")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    print(f"\n    在 Docker 中运行集成探测脚本 ...")
    container = None
    try:
        import uuid
        container = build_container(test_spec, client, run_id=f"probe_ab_{uuid.uuid4().hex[:8]}", logger=logger, nocache=False)
        container.start()
        copy_to_container(container, tracer_lib_dir, Path("/tmp/tracer_lib"))
        copy_to_container(container, probe_path, Path("/tmp/integration_probe.py"))
        container.exec_run("mkdir -p /tmp/probe_out", workdir="/testbed")

        val = container.exec_run(
            "sh -c 'cd /testbed && /opt/miniconda3/envs/testbed/bin/python3 /tmp/integration_probe.py'",
            workdir="/testbed",
            user="root"
        )
        logs = val.output.decode('utf-8', errors='replace')
        print(f"    容器退出码: {val.exit_code}")
        if logs.strip():
            print(f"    日志输出:\n{logs[:800]}")

        import tarfile, io
        bits, stat = container.get_archive("/tmp/probe_out/probe_trace_result.json")
        tar = tarfile.open(fileobj=io.BytesIO(b''.join(bits)), mode='r:*')
        member = tar.getmember("probe_trace_result.json")
        f = tar.extractfile(member)
        probe_data = json.loads(f.read().decode('utf-8'))
        tar.close()
        with open(output_dir / "probe_trace_result.json", 'w') as f2:
            json.dump(probe_data, f2, indent=2, default=str)
    except Exception as e:
        print(f"    ✗ 探测执行失败: {e}")
        traceback.print_exc()
        probe_data = {"trace_count": 0, "traces": [], "exception_info": None, "import_error": str(e)}
    finally:
        if container:
            cleanup_container(client, container, logger)

    traces = probe_data.get('traces', [])
    import_err = probe_data.get('import_error')
    print(f"\n    ✓ 探测完成，收集到 {len(traces)} 条真实 trace")
    if import_err:
        print(f"    导入警告: {import_err}")

    target_traces = [t for t in traces if t.get('function') == target_func]
    print(f"\n    --- 真实业务逻辑 Trace（{target_func} 方法/类内）---")
    print(f"    共 {len(target_traces)} 条 trace\n")
    for t in target_traces[:25]:
        vars_str = str(t.get('variables', {}))[:100]
        print(f"      L{t['line']:4d} | {t['event']:6s} | {vars_str}")
    if len(target_traces) > 25:
        print(f"      ... 还有 {len(target_traces) - 25} 条 trace")

    return probe_data


# ============================================================
# 公共部分：阶段 1-3 + 4a（对每个实例只跑一次）
# ============================================================
def run_common_stages(instance_id):
    common_dir = OUTPUT_BASE_DIR / "_common" / instance_id.replace("/", "_")
    common_dir.mkdir(parents=True, exist_ok=True)

    instance, fail_to_pass, pass_to_pass = stage_1_load_instance(instance_id, common_dir)
    repo_mgr = stage_2_prepare_repo(instance)
    result = stage_3_extract_target(repo_mgr, instance, common_dir)
    if not result[0]:
        print("\n✗ 无法继续，提取目标失败")
        return None
    target_file, target_func, buggy_code = result

    pytest_report, pytest_traces = stage_4a_pytest_traces(
        instance, buggy_code, target_func, target_file, fail_to_pass, common_dir
    )

    return {
        "instance": instance,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "target_file": target_file,
        "target_func": target_func,
        "buggy_code": buggy_code,
        "pytest_report": pytest_report,
        "pytest_traces": pytest_traces,
        "common_dir": common_dir,
    }


# ============================================================
# A 组：完整 tracer（含 4b 探测）
# ============================================================
def run_group_a(common, instance_id, pause_between_stages):
    output_dir = get_output_dir(instance_id, "A")
    print("\n" + ">>>" * 20)
    print(" 开始 A 组实验：完整 Tracer（含 4b 集成探测）")
    print(">>>" * 20)

    for f in common["common_dir"].iterdir():
        shutil.copy2(f, output_dir / f.name)

    probe_data = stage_4b_integration_probe(
        common["instance"], common["target_func"], common["target_file"], output_dir
    )
    if pause_between_stages:
        pause()

    analysis = stage_5_analyze_traces(
        common["pytest_traces"], probe_data, common["target_func"], output_dir
    )
    if pause_between_stages:
        pause()

    fixed_code, explanation = stage_6_llm_fix(
        common["buggy_code"], common["target_func"], analysis, common["instance"], output_dir
    )
    if pause_between_stages:
        pause()

    report = stage_7_validate_fix(
        common["instance"], fixed_code, common["target_file"], common["target_func"], output_dir
    )

    print("\n" + "=" * 80)
    status = "✅ 已解决" if report.is_resolved else "❌ 未解决"
    print(f" A 组结果: {status}")
    print(f"   FAIL_TO_PASS: {report.fail_to_pass_passed}/{report.fail_to_pass_total}")
    print(f"   PASS_TO_PASS: {report.pass_to_pass_passed}/{report.pass_to_pass_total}")
    print("=" * 80)

    return {
        "group": "A",
        "resolved": report.is_resolved,
        "ftp_passed": report.fail_to_pass_passed,
        "ftp_total": report.fail_to_pass_total,
        "ptp_passed": report.pass_to_pass_passed,
        "ptp_total": report.pass_to_pass_total,
        "probe_trace_count": probe_data.get("trace_count", 0),
        "probe_exception": probe_data.get("exception_info"),
    }


# ============================================================
# B 组：跳过 4b 探测
# ============================================================
def run_group_b(common, instance_id, pause_between_stages):
    output_dir = get_output_dir(instance_id, "B")
    print("\n" + ">>>" * 20)
    print(" 开始 B 组实验：跳过 4b 探测（仅 pytest + 问题描述）")
    print(">>>" * 20)

    for f in common["common_dir"].iterdir():
        shutil.copy2(f, output_dir / f.name)

    probe_data = {"trace_count": 0, "traces": [], "exception_info": None, "import_error": "Skipped in group B"}
    print("\n    [B 组] 跳过阶段 4b，probe_data 为空")
    if pause_between_stages:
        pause()

    analysis = stage_5_analyze_traces(
        common["pytest_traces"], probe_data, common["target_func"], output_dir
    )
    if pause_between_stages:
        pause()

    fixed_code, explanation = stage_6_llm_fix(
        common["buggy_code"], common["target_func"], analysis, common["instance"], output_dir
    )
    if pause_between_stages:
        pause()

    report = stage_7_validate_fix(
        common["instance"], fixed_code, common["target_file"], common["target_func"], output_dir
    )

    print("\n" + "=" * 80)
    status = "✅ 已解决" if report.is_resolved else "❌ 未解决"
    print(f" B 组结果: {status}")
    print(f"   FAIL_TO_PASS: {report.fail_to_pass_passed}/{report.fail_to_pass_total}")
    print(f"   PASS_TO_PASS: {report.pass_to_pass_passed}/{report.pass_to_pass_total}")
    print("=" * 80)

    return {
        "group": "B",
        "resolved": report.is_resolved,
        "ftp_passed": report.fail_to_pass_passed,
        "ftp_total": report.fail_to_pass_total,
        "ptp_passed": report.pass_to_pass_passed,
        "ptp_total": report.pass_to_pass_total,
        "probe_trace_count": 0,
        "probe_exception": None,
    }


# ============================================================
# 实验入口
# ============================================================
EXPERIMENT_INSTANCES = [
    "django__django-11532",
    "pytest-dev__pytest-5809",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tracer A/B 对比实验 (Verified)")
    parser.add_argument("--no-pause", action="store_true", help="不暂停")
    parser.add_argument("--instances", nargs='+', default=EXPERIMENT_INSTANCES, help="实例 ID 列表")
    args = parser.parse_args()
    pause_between_stages = not args.no_pause

    results = []
    start_all = time.time()

    for instance_id in args.instances:
        print("\n" + "#" * 80)
        print(f"# 实验实例: {instance_id}")
        print("#" * 80)

        common = run_common_stages(instance_id)
        if common is None:
            results.append({"instance_id": instance_id, "A": None, "B": None, "error": "stage 1-3 failed"})
            continue
        if pause_between_stages:
            pause()

        result_a = run_group_a(common, instance_id, pause_between_stages)
        result_b = run_group_b(common, instance_id, pause_between_stages)

        results.append({
            "instance_id": instance_id,
            "A": result_a,
            "B": result_b,
        })

    total_elapsed = time.time() - start_all

    print("\n" + "=" * 80)
    print(" A/B 实验最终汇总报告 (Verified)")
    print("=" * 80)

    a_resolved = sum(1 for r in results if r["A"] and r["A"]["resolved"])
    b_resolved = sum(1 for r in results if r["B"] and r["B"]["resolved"])
    total = len(results)

    print(f"\n总计实例数: {total}")
    print(f"A 组（含 tracer）解决数: {a_resolved}/{total}  ({a_resolved/total*100:.1f}%)")
    print(f"B 组（无 tracer）解决数: {b_resolved}/{total}  ({b_resolved/total*100:.1f}%)")
    print(f"Tracer 提升: {a_resolved - b_resolved} 个实例")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")

    print("\n--- 逐实例明细 ---")
    for r in results:
        iid = r["instance_id"]
        a_ok = "✅" if r["A"] and r["A"]["resolved"] else "❌"
        b_ok = "✅" if r["B"] and r["B"]["resolved"] else "❌"
        a_probe = r["A"]["probe_trace_count"] if r["A"] else "N/A"
        print(f"  {a_ok} A | {b_ok} B | probe={str(a_probe):>4} | {iid}")

    report_path = OUTPUT_BASE_DIR / "ab_experiment_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_instances": total,
            "a_resolved": a_resolved,
            "b_resolved": b_resolved,
            "total_seconds": total_elapsed,
            "details": results,
        }, f, indent=2, default=str)
    print(f"\n✓ 完整报告已保存到: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
