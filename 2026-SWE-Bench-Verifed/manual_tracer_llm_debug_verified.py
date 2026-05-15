#!/usr/bin/env python3
"""
手动 Tracer + LLM 完整调试脚本（Verified 适配版）
=================================================

阶段 1-7 与 Pro 版本一致，但适配 SWE-bench Verified 的字段和镜像体系。

Usage:
    python manual_tracer_llm_debug_verified.py --instances astropy__astropy-12907 --no-pause
"""

import sys
import os
import json
import time
import textwrap
import traceback
import tempfile
import shutil
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "swe_bench_pro"))

from complete_debug_workflow import RepoManager, CodeExtractor
from llm_client import get_completion_with_retry
from verified_data_loader import VerifiedDataLoader
from verified_docker_executor import VerifiedDockerTestExecutor

DEFAULT_INSTANCE = "astropy__astropy-12907"
OUTPUT_BASE_DIR = Path("/tmp/manual_tracer_llm_verified")


def get_output_dir(instance_id: str) -> Path:
    safe_id = instance_id.replace("/", "_").replace(":", "_")
    d = OUTPUT_BASE_DIR / safe_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def pause():
    input("\n按 Enter 继续...")


def stage_1_load_instance(instance_id, output_dir):
    print_section(f"阶段 1: 加载实例数据")

    loader = VerifiedDataLoader()
    instance = loader.get_instance(instance_id)

    print(f"✓ 实例 ID: {instance['instance_id']}")
    print(f"✓ 仓库:    {instance['repo']}")
    print(f"✓ Commit:  {instance['base_commit']}")

    fail_to_pass = instance.get('fail_to_pass', [])
    pass_to_pass = instance.get('pass_to_pass', [])

    print(f"\n✓ FAIL_TO_PASS: {len(fail_to_pass)} 个")
    for t in fail_to_pass[:5]:
        print(f"    - {t}")
    if len(fail_to_pass) > 5:
        print(f"    ... 还有 {len(fail_to_pass) - 5} 个")
    print(f"✓ PASS_TO_PASS: {len(pass_to_pass) if isinstance(pass_to_pass, list) else 'N/A'}")
    print(f"✓ Has test_patch: {bool(instance.get('test_patch'))}")

    with open(output_dir / "instance.json", 'w') as f:
        json.dump(instance, f, indent=2, default=str)
    print(f"\n✓ 已保存到 {output_dir / 'instance.json'}")

    return instance, fail_to_pass, pass_to_pass


def stage_2_prepare_repo(instance):
    print_section("阶段 2: 准备仓库")

    repo_mgr = RepoManager(instance['repo'])
    print(f"[1/2] 克隆/更新仓库到 {repo_mgr.repo_path} ...")
    repo_mgr.clone_or_update()
    print(f"    ✓ 完成")

    print(f"[2/2] 检出到 base_commit {instance['base_commit'][:12]} ...")
    import subprocess
    subprocess.run(['git', 'reset', '--hard'], cwd=repo_mgr.repo_path, capture_output=True)
    subprocess.run(['git', 'clean', '-fd'], cwd=repo_mgr.repo_path, capture_output=True)
    repo_mgr.checkout(instance['base_commit'])
    print(f"    ✓ 完成")

    return repo_mgr


def _infer_target_func_from_patch(patch: str, target_file: str) -> str:
    import re
    from collections import Counter
    blocks = re.split(r'diff --git a/(.+?) b/\1', patch)
    target_block = ""
    for i in range(1, len(blocks), 2):
        if i < len(blocks) and blocks[i] == target_file:
            target_block = blocks[i+1] if i+1 < len(blocks) else ""
            break
    if not target_block:
        target_block = patch
    func_candidates = []
    func_candidates.extend(re.findall(r'^[+-]\s*def\s+(\w+)\s*\(', target_block, re.MULTILINE))
    func_candidates.extend(re.findall(r'^[+-]\s*class\s+(\w+)\b', target_block, re.MULTILINE))
    ctx_matches = re.findall(r'@@ .* @@\s*(?:(def|class)\s+(\w+))', target_block)
    for _, name in ctx_matches:
        func_candidates.append(name)
    if func_candidates:
        return Counter(func_candidates).most_common(1)[0][0]
    return None


def stage_3_extract_target(repo_mgr, instance, output_dir):
    print_section("阶段 3: 提取目标函数和 buggy code")

    patch = instance.get('patch', '')

    py_files = re.findall(r'diff --git a/(.+\.py) b/\1', patch)
    if not py_files:
        print("✗ 无法从 patch 提取目标文件")
        return None, None, None
    target_file = py_files[0]
    print(f"✓ 目标文件: {target_file}")

    code = repo_mgr.get_file_content(target_file)
    if not code:
        print(f"✗ 无法读取文件: {target_file}")
        return None, None, None

    target_func = _infer_target_func_from_patch(patch, target_file)
    if target_func:
        print(f"✓ 从 patch 推断目标函数/类: {target_func}")

    if target_func and f"class {target_func}" in code:
        lines = code.split('\n')
        start = None
        for i, line in enumerate(lines):
            if re.match(rf"^class\s+{re.escape(target_func)}\b", line):
                start = i
                break
        if start is None:
            print("✗ 无法定位类起始行")
            return None, None, None
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if re.match(r"^(class|def)\s+", lines[i]) and not lines[i].startswith(" ") and not lines[i].startswith("\t"):
                end = i
                break
        buggy_code = '\n'.join(lines[start:end])
    elif target_func and f"def {target_func}" in code:
        func_result = CodeExtractor.extract_function_with_context(code, target_func)
        if not func_result:
            print("✗ 无法提取函数代码")
            return None, None, None
        buggy_code = func_result['standalone_code']
    else:
        print("⚠ 无法精确提取目标代码，使用文件前 150 行")
        buggy_code = '\n'.join(code.split('\n')[:150])
        target_func = target_func or "unknown"

    print(f"\n✓ 提取到 {len(buggy_code)} 字符的 buggy code")
    print("-" * 60)
    print(buggy_code[:1000])
    if len(buggy_code) > 1000:
        print("...")
    print("-" * 60)

    with open(output_dir / "buggy_code.py", 'w') as f:
        f.write(buggy_code)
    print(f"\n✓ 已保存到 {output_dir / 'buggy_code.py'}")

    return target_file, target_func, buggy_code


def stage_4_docker_trace_collection(instance, buggy_code, target_func, target_file, fail_to_pass, output_dir):
    print_section("阶段 4: 在 Docker 中插桩收集 trace")
    print("""
    说明:
    [4a] 运行 pytest 测试并收集 trace（作为参考）
         注意: pytest 测试可能包含大量 mock，trace 可能不完整
    [4b] 无 mock 集成探测 trace 收集（核心真实路径）
         只 mock 最底层 I/O，收集真实业务逻辑 trace
    """)

    # ---------- 4a: pytest trace ----------
    print(f"\n>>> [4a] pytest Trace 收集")
    docker_executor = VerifiedDockerTestExecutor(instance, timeout=600)
    docker_executor.target_file = target_file
    docker_executor.target_func = target_func
    print(f"    DockerTestExecutor 初始化完成")
    print(f"    镜像: {docker_executor.image_name}")
    print(f"    目标文件: {docker_executor.target_file}")
    print(f"    目标函数: {docker_executor.target_func}")

    print("\n    开始运行 pytest 并收集 traces ...")
    print("    -" * 30)
    start = time.time()
    pytest_report, pytest_traces = docker_executor.validate_with_traces(
        buggy_code, collect_traces=True, trace_failed_only=False
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

    # ---------- 4b: Integration Probe ----------
    print(f"\n>>> [4b] 无 mock 集成探测 Trace 收集")

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

    # Build probe script with raw string + format to avoid triple-quote nesting
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
        '_MockAttr.__version__ = "1.0.0"\n'
        'for _mock_mod in ("asgiref", "asgiref.sync", "yaml", "toml", "certifi", "cryptography"):\n'
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

    print(f"\n    >>> tracer lib 准备情况")
    print(f"    临时目录: {tracer_lib_dir}")
    for f in sorted(tracer_lib_dir.iterdir()):
        print(f"      - {f.name}")

    print(f"\n    >>> Docker 卷挂载映射")
    print(f"      {tracer_lib_dir}  ->  /tmp/tracer_lib  (ro)")
    print(f"      {probe_path}      ->  /tmp/integration_probe.py  (ro)")
    print(f"      {output_dir}      ->  /tmp/probe_out  (rw)")

    # For Verified, we need to run probe inside the swebench container
    import docker
    client = docker.from_env()
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.docker_build import build_container, cleanup_container
    import logging

    test_spec = make_test_spec(instance)
    logger = logging.getLogger("probe")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    print(f"\n    在 Docker 中运行集成探测脚本 ...")
    container = None
    try:
        import uuid
        container = build_container(test_spec, client, run_id=f"probe_{uuid.uuid4().hex[:8]}", logger=logger, nocache=False)
        container.start()
        # Copy probe script and tracer lib into container (or use volumes on run)
        # Since container is already created, we can use docker cp or exec with mounted volumes.
        # Easier: run a one-off exec that mounts volumes is not possible after creation.
        # Alternative: copy files into container using put_archive or docker cp API.
        # docker-py has copy_to_container in swebench harness.
        from swebench.harness.docker_utils import copy_to_container
        copy_to_container(container, tracer_lib_dir, Path("/tmp/tracer_lib"))
        copy_to_container(container, probe_path, Path("/tmp/integration_probe.py"))
        # Also need output dir writable
        from swebench.harness.docker_utils import copy_to_container
        # Create output dir in container
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

        # Copy result back
        from swebench.harness.docker_utils import copy_from_container
        bits, stat = container.get_archive("/tmp/probe_out/probe_trace_result.json")
        import tarfile, io
        tar = tarfile.open(fileobj=io.BytesIO(b''.join(bits)), mode='r:*')
        member = tar.getmember("probe_trace_result.json")
        f = tar.extractfile(member)
        probe_data = json.loads(f.read().decode('utf-8'))
        tar.close()
        # Save to output dir
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

    return pytest_traces, probe_data


def stage_5_analyze_traces(pytest_traces, probe_data, target_func, output_dir):
    print_section("阶段 5: 分析 trace")

    error_types = {}
    for t in pytest_traces:
        trace_data = t.get('trace', {})
        exc = trace_data.get('exception_info')
        if exc:
            et = exc.get('type', 'Exception')
            error_types[et] = error_types.get(et, 0) + 1
        elif not trace_data.get('test_passed', False):
            error_types['LogicMismatch'] = error_types.get('LogicMismatch', 0) + 1

    error_summary = "\n".join([f"- {k}: {v} 次" for k, v in error_types.items()])

    trace_summary = []
    for i, t in enumerate(pytest_traces[:3], 1):
        trace_data = t.get('trace', {})
        passed = trace_data.get('test_passed', False)
        summary = trace_data.get('trace_summary', '')
        status = "通过" if passed else "失败"
        trace_summary.append(f"pytest 测试 {i} ({t.get('test_name', 'unknown')}): {status}")
        if summary:
            trace_summary.append(f"  Trace:\n{summary[:400]}")

    probe_traces = probe_data.get('traces', [])
    open_traces = [t for t in probe_traces if t.get('function') == target_func]
    probe_summary_lines = [
        f"\n=== 真实业务逻辑执行路径（无 mock 探测）===",
        f"在 {target_func} 方法内共收集到 {len(open_traces)} 条 trace：",
    ]
    for t in open_traces[:15]:
        vars_str = str(t.get('variables', {}))[:80]
        probe_summary_lines.append(f"  L{t['line']:4d} {t['event']:8s} | {vars_str}")
    if len(open_traces) > 15:
        probe_summary_lines.append(f"  ... 还有 {len(open_traces) - 15} 条")

    if probe_data.get('exception_info'):
        exc = probe_data['exception_info']
        probe_summary_lines.append(
            f"\n探测过程中捕获异常: {exc.get('type', 'Unknown')}: {exc.get('message', '')} at L{exc.get('line', '?')}"
        )

    probe_summary = "\n".join(probe_summary_lines)

    analysis = {
        "pytest_error_summary": error_types,
        "pytest_trace_summary": trace_summary,
        "probe_trace_count": len(open_traces),
        "probe_exception": probe_data.get('exception_info'),
        "probe_summary": probe_summary_lines,
        "combined_text": {
            "error_summary": error_summary,
            "trace_summary": "\n".join(trace_summary),
            "probe_summary": probe_summary,
        }
    }

    with open(output_dir / "trace_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print("\n>>> pytest 错误分析")
    print(error_summary if error_summary else "    未检测到特定错误类型")

    print(f"\n>>> 真实业务逻辑 trace 分析")
    print(f"    在 {target_func} 内收集到 {len(open_traces)} 条 trace")
    if probe_data.get('exception_info'):
        exc = probe_data['exception_info']
        print(f"    探测中捕获异常: {exc.get('type', 'Unknown')}: {exc.get('message', '')} at L{exc.get('line', '?')}")
    else:
        print("    探测中未捕获异常")

    print(f"\n✓ 分析报告已保存到 {output_dir / 'trace_analysis.json'}")

    return analysis


def stage_6_llm_fix(buggy_code, target_func, analysis, instance, output_dir):
    print_section("阶段 6: 调用 LLM 生成修复")

    problem_statement = instance.get('problem_statement', '')
    error_summary = analysis['combined_text']['error_summary']
    trace_summary = analysis['combined_text']['trace_summary']
    probe_summary = analysis['combined_text']['probe_summary']

    prompt = f"""You are an expert Python debugger. Analyze the following buggy code and execution traces to fix the bug.

## Problem Statement
{problem_statement if problem_statement else "Fix the bug in the provided code."}

## Buggy Code
```python
{buggy_code}
```

## Function/Class Context
Target method for trace: {target_func}

## Execution Trace Analysis

### pytest Tests (with mocks - limited trace coverage)
{error_summary if error_summary else "No specific error types detected."}

{trace_summary}

{probe_summary}

## Instructions

1. Analyze the real execution path above to identify the root cause of the bug.
2. Provide a fixed version of the buggy code that resolves the issue described in the problem statement.
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

    print("\n[1/2] 构建 LLM Prompt ...")
    print(f"    Prompt 长度: {len(prompt)} 字符")
    with open(output_dir / "llm_prompt.txt", 'w') as f:
        f.write(prompt)
    print(f"    ✓ Prompt 已保存到 {output_dir / 'llm_prompt.txt'}")

    print("\n[2/2] 调用 LLM (get_completion_with_retry) ...")
    print("    请耐心等待（通常 10-60 秒）...")
    start = time.time()
    messages = [
        {"role": "system", "content": "You are an expert Python debugger specializing in fixing bugs based on execution traces."},
        {"role": "user", "content": prompt}
    ]
    response = get_completion_with_retry(messages)
    elapsed = time.time() - start
    print(f"\n✓ LLM 响应完成，耗时 {elapsed:.1f} 秒")

    explanation = ""
    fixed_code = ""

    if '### Explanation' in response:
        parts = response.split('### Explanation')
        if len(parts) > 1:
            rest = parts[1]
            if '### Fixed Code' in rest:
                explanation = rest.split('### Fixed Code')[0].strip()
            else:
                explanation = rest.strip()

    if '```python' in response:
        code_parts = response.split('```python')
        if len(code_parts) > 1:
            fixed_code = code_parts[-1].split('```')[0].strip()
    elif '```' in response:
        code_parts = response.split('```')
        if len(code_parts) > 1:
            fixed_code = code_parts[-1].strip()

    print(f"\n✓ 解析到 Fixed Code: {len(fixed_code)} 字符")
    print(f"✓ 解析到 Explanation: {len(explanation)} 字符")

    with open(output_dir / "llm_response.txt", 'w') as f:
        f.write(response)
    with open(output_dir / "fixed_code.py", 'w') as f:
        f.write(fixed_code)
    print(f"\n✓ LLM 原始响应保存到 {output_dir / 'llm_response.txt'}")
    print(f"✓ 修复代码保存到 {output_dir / 'fixed_code.py'}")

    print("\n--- 修复代码预览 ---")
    print("-" * 60)
    print(fixed_code[:1200])
    if len(fixed_code) > 1200:
        print("...")
    print("-" * 60)

    return fixed_code, explanation


def stage_7_validate_fix(instance, fixed_code, target_file, target_func, output_dir):
    print_section("阶段 7: 在 Docker 中验证修复")

    print("\n>>> Docker 镜像与执行器准备")
    docker_executor = VerifiedDockerTestExecutor(instance, timeout=600)
    docker_executor.target_file = target_file
    docker_executor.target_func = target_func
    print(f"    镜像: {docker_executor.image_name}")
    print(f"    目标文件: {docker_executor.target_file}")
    print(f"    目标函数: {docker_executor.target_func}")

    print("\n>>> 开始 Docker 验证（应用 fixed_code）")
    print("    参数: collect_traces=True, trace_failed_only=True")
    print("-" * 60)

    start = time.time()
    report, traces = docker_executor.validate_with_traces(
        fixed_code, collect_traces=True, trace_failed_only=True
    )
    elapsed = time.time() - start

    print("-" * 60)
    print(f"\n✓ Docker 验证完成，耗时 {elapsed:.1f} 秒")

    print(f"\n--- 验证结果 ---")
    print(f"    FAIL_TO_PASS: {report.fail_to_pass_passed}/{report.fail_to_pass_total}")
    print(f"    PASS_TO_PASS: {report.pass_to_pass_passed}/{report.pass_to_pass_total}")
    print(f"    resolved:     {report.is_resolved}")

    with open(output_dir / "docker_validation_report.json", 'w') as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    with open(output_dir / "docker_validation_traces.json", 'w') as f:
        json.dump(traces, f, indent=2, default=str)
    print(f"\n✓ 验证报告保存到 {output_dir / 'docker_validation_report.json'}")

    return report


def run_single_instance(instance_id, pause_between_stages):
    output_dir = get_output_dir(instance_id)

    print("\n" + "=" * 80)
    print(f" 开始处理实例: {instance_id}")
    print(f" 输出目录: {output_dir}")
    print("=" * 80)

    instance, fail_to_pass, pass_to_pass = stage_1_load_instance(instance_id, output_dir)
    if pause_between_stages:
        pause()

    repo_mgr = stage_2_prepare_repo(instance)
    if pause_between_stages:
        pause()

    result = stage_3_extract_target(repo_mgr, instance, output_dir)
    if not result[0]:
        print("\n✗ 无法继续，提取目标失败")
        return False
    target_file, target_func, buggy_code = result
    if pause_between_stages:
        pause()

    pytest_traces, probe_data = stage_4_docker_trace_collection(
        instance, buggy_code, target_func, target_file, fail_to_pass, output_dir
    )
    if pause_between_stages:
        pause()

    analysis = stage_5_analyze_traces(pytest_traces, probe_data, target_func, output_dir)
    if pause_between_stages:
        pause()

    fixed_code, explanation = stage_6_llm_fix(buggy_code, target_func, analysis, instance, output_dir)
    if pause_between_stages:
        pause()

    report = stage_7_validate_fix(instance, fixed_code, target_file, target_func, output_dir)

    print("\n" + "=" * 80)
    if report.is_resolved:
        print(" 🎉🎉🎉 修复成功！所有测试通过！ 🎉🎉🎉")
    else:
        print(" ❌ 修复未完全通过，请查看上述报告和日志")
    print("=" * 80)

    print(f"\n输出文件汇总 ({output_dir}):")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")

    return report.is_resolved


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manual Tracer + LLM Debug for Verified (7 stages)")
    parser.add_argument("--instances", nargs='+', default=[DEFAULT_INSTANCE],
                        help="One or more Verified instance IDs to debug")
    parser.add_argument("--no-pause", action="store_true",
                        help="Do not pause between stages")
    args = parser.parse_args()

    pause_between_stages = not args.no_pause
    all_results = {}

    for instance_id in args.instances:
        try:
            resolved = run_single_instance(instance_id, pause_between_stages)
            all_results[instance_id] = resolved
        except Exception as e:
            print(f"\n✗ 实例 {instance_id} 处理失败: {e}")
            traceback.print_exc()
            all_results[instance_id] = False

    print("\n" + "=" * 80)
    print(" 全部实例处理完成")
    print("=" * 80)
    for iid, resolved in all_results.items():
        status = "✅ 已解决" if resolved else "❌ 未解决"
        print(f"  {status}  {iid}")

    return 0 if all(all_results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
