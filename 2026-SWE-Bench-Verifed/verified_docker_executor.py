#!/usr/bin/env python3
"""
Verified Docker 测试执行器
==========================
适配 SWE-bench Verified 的 swebench 镜像体系：
  - 使用本地构建的 sweb.eval.x86_64.<instance_id> 镜像
  - 容器内工作目录 /testbed
  - conda 环境 testbed
  - 需要先注入 test_patch，再注入修复代码
"""

import sys
import json
import re
import base64
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "swe_bench_pro"))

import docker

from verified_data_loader import VerifiedDataLoader
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.docker_build import build_container, cleanup_container, BuildImageError, setup_logger
from swebench.harness.docker_utils import copy_to_container


@dataclass
class TestResult:
    test_name: str
    passed: bool
    output: str = ""
    error: Optional[str] = None
    trace: Optional[Dict] = None


@dataclass
class ValidationReport:
    instance_id: str
    status: str
    fail_to_pass_results: List[TestResult]
    pass_to_pass_results: List[TestResult]
    details: str

    @property
    def fail_to_pass_passed(self) -> int:
        return sum(1 for r in self.fail_to_pass_results if r.passed)

    @property
    def fail_to_pass_total(self) -> int:
        return len(self.fail_to_pass_results)

    @property
    def pass_to_pass_passed(self) -> int:
        return sum(1 for r in self.pass_to_pass_results if r.passed)

    @property
    def pass_to_pass_total(self) -> int:
        return len(self.pass_to_pass_results)

    @property
    def is_resolved(self) -> bool:
        return (
            self.fail_to_pass_total > 0 and
            self.fail_to_pass_passed == self.fail_to_pass_total and
            self.pass_to_pass_passed == self.pass_to_pass_total
        )

    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "status": self.status,
            "is_resolved": self.is_resolved,
            "fail_to_pass": {
                "passed": self.fail_to_pass_passed,
                "total": self.fail_to_pass_total,
            },
            "pass_to_pass": {
                "passed": self.pass_to_pass_passed,
                "total": self.pass_to_pass_total,
            },
            "details": self.details,
            "fail_to_pass_tests": [
                {"name": r.test_name, "passed": r.passed, "error": r.error}
                for r in self.fail_to_pass_results
            ],
            "pass_to_pass_tests": [
                {"name": r.test_name, "passed": r.passed, "error": r.error}
                for r in self.pass_to_pass_results
            ],
        }


def _setup_logger(name: str) -> logging.Logger:
    log_dir = Path("/tmp/swebench_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    return setup_logger(name, log_file, add_stdout=True)


class VerifiedDockerTestExecutor:
    """适配 SWE-bench Verified 的 Docker 测试执行器"""

    CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate && conda activate testbed"
    WORKDIR = "/testbed"
    DOCKER_USER = "root"

    def __init__(self, instance_data: Dict, timeout: int = 600):
        self.instance = instance_data
        self.timeout = timeout
        self.test_spec: TestSpec = make_test_spec(instance_data)
        self.image_name = self.test_spec.instance_image_key

        self.fail_to_pass_tests = instance_data.get("fail_to_pass", [])
        self.pass_to_pass_tests = instance_data.get("pass_to_pass", [])

        analysis = self._analyze_patch()
        self.target_file = analysis.get("target_file")
        self.target_func = analysis.get("target_func")

        print(f"VerifiedDockerTestExecutor initialized")
        print(f"  Image: {self.image_name}")
        print(f"  Target: {self.target_file}::{self.target_func}")

    def _analyze_patch(self) -> Dict:
        loader = VerifiedDataLoader()
        analysis = loader.analyze_patch(self.instance.get("patch", ""))
        return {
            "target_file": analysis.get("target_file"),
            "target_func": analysis.get("functions_modified")[0] if analysis.get("functions_modified") else None,
        }

    def validate(self, fix_code: str) -> ValidationReport:
        return self.validate_with_traces(fix_code, collect_traces=False)[0]

    def validate_with_traces(
        self, fix_code: str, collect_traces: bool = True, trace_failed_only: bool = False
    ) -> Tuple[ValidationReport, List[Dict]]:
        instance_id = self.instance.get("instance_id", "unknown")
        print(f"\n{'='*80}")
        print(f"Verified Docker Test Execution")
        print(f"Instance: {instance_id}")
        print(f"Collect traces: {collect_traces} (failed_only={trace_failed_only})")
        print(f"{'='*80}")

        traces: List[Dict] = []
        client = docker.from_env()
        logger = _setup_logger(f"vde-{instance_id}")
        container = None

        try:
            # 1. Build & start container
            print(f"\n[1/5] Creating container from {self.image_name}...")
            import uuid
            container = build_container(
                self.test_spec, client, run_id=f"verified_run_{uuid.uuid4().hex[:8]}", logger=logger, nocache=False
            )
            container.start()
            print(f"  ✓ Container started: {container.id[:12]}")

            # 2. Apply test_patch
            print(f"\n[2/5] Applying test patch...")
            self._apply_test_patch(container)
            print(f"  ✓ Test patch applied")

            # 3. Prepare fix
            print(f"\n[3/5] Preparing fix...")
            fix_info = self._prepare_fix(fix_code)
            if not fix_info:
                return self._error_report("Failed to prepare fix"), traces
            print(f"  ✓ Fix prepared (method={fix_info['method']})")

            # 4. Run FAIL_TO_PASS tests
            print(f"\n[4/5] Running FAIL_TO_PASS tests ({len(self.fail_to_pass_tests)} tests)...")
            ftp_results = []
            for test in self.fail_to_pass_tests:
                if collect_traces and trace_failed_only:
                    result = self._run_test_plain(container, test, fix_info)
                    if not result.passed:
                        result = self._run_test_with_trace(container, test, fix_info)
                elif collect_traces:
                    result = self._run_test_with_trace(container, test, fix_info)
                else:
                    result = self._run_test_plain(container, test, fix_info)
                ftp_results.append(result)
                status = "✓" if result.passed else "✗"
                print(f"  {status} {test}")
                if result.trace:
                    traces.append({"test_name": test, "test_type": "fail_to_pass", "trace": result.trace})
            print(f"  Result: {sum(1 for r in ftp_results if r.passed)}/{len(ftp_results)} passed")

            # 5. Run PASS_TO_PASS tests
            print(f"\n[5/5] Running PASS_TO_PASS tests ({len(self.pass_to_pass_tests)} tests)...")
            ptp_results = []
            for test in self.pass_to_pass_tests:
                if collect_traces and trace_failed_only:
                    result = self._run_test_plain(container, test, fix_info)
                    if not result.passed:
                        result = self._run_test_with_trace(container, test, fix_info)
                elif collect_traces:
                    result = self._run_test_with_trace(container, test, fix_info)
                else:
                    result = self._run_test_plain(container, test, fix_info)
                ptp_results.append(result)
                status = "✓" if result.passed else "✗"
                print(f"  {status} {test}")
                if result.trace:
                    traces.append({"test_name": test, "test_type": "pass_to_pass", "trace": result.trace})
            print(f"  Result: {sum(1 for r in ptp_results if r.passed)}/{len(ptp_results)} passed")

            return self._generate_report(instance_id, ftp_results, ptp_results), traces

        except Exception as e:
            import traceback
            print(f"Validation error: {e}")
            print(traceback.format_exc())
            return self._error_report(str(e)), traces
        finally:
            if container:
                cleanup_container(client, container, logger)

    # ------------------------------------------------------------------
    # Patch & fix application
    # ------------------------------------------------------------------
    def _apply_test_patch(self, container) -> bool:
        test_patch = self.instance.get("test_patch", "")
        if not test_patch:
            return True

        # Write patch to temp file and copy to container
        tmpdir = Path(tempfile.mkdtemp())
        patch_file = tmpdir / "test_patch.diff"
        patch_file.write_text(test_patch)
        copy_to_container(container, patch_file, Path("/tmp/test_patch.diff"))

        # Try git apply
        git_cmds = [
            "git apply --verbose /tmp/test_patch.diff",
            "git apply --verbose --reject /tmp/test_patch.diff",
            "patch --batch --fuzz=5 -p1 -i /tmp/test_patch.diff",
        ]
        for cmd in git_cmds:
            val = container.exec_run(cmd, workdir=self.WORKDIR, user=self.DOCKER_USER)
            if val.exit_code == 0:
                return True
        # If all fail, warn but don't raise (some test_patches are additive)
        print("  ⚠ Warning: test patch apply failed or partial failure")
        return False

    def _prepare_fix(self, fix_code: str) -> Optional[Dict]:
        if not self.target_file:
            return None
        is_function_fix = self.target_func and f"def {self.target_func}" in fix_code
        is_class_fix = self.target_func and f"class {self.target_func}" in fix_code
        if self.target_func and (is_function_fix or is_class_fix):
            return {
                "method": "function_replace",
                "target_file": self.target_file,
                "target_func": self.target_func,
                "code": fix_code,
            }
        return {
            "method": "file_replace",
            "target_file": self.target_file,
            "code": fix_code,
        }

    def _apply_fix_in_container(self, container, fix_info: Dict) -> bool:
        target_path = f"{self.WORKDIR}/{fix_info['target_file']}"

        # Write Python script to temp file and copy into container
        tmpdir = Path(tempfile.mkdtemp())
        script_path = tmpdir / "apply_fix.py"

        if fix_info["method"] == "function_replace":
            script_content = self._build_function_replace_script(fix_info, target_path)
        else:
            script_content = self._build_file_replace_script(fix_info, target_path)

        script_path.write_text(script_content)
        copy_to_container(container, script_path, Path("/tmp/apply_fix.py"))

        cmd = f"{self.CONDA_ACTIVATE} && python /tmp/apply_fix.py"
        val = container.exec_run(
            ["/bin/bash", "-c", cmd],
            workdir=self.WORKDIR,
            user=self.DOCKER_USER,
        )
        # If function replace failed, fall back to file replace
        if val.exit_code != 0 and fix_info["method"] == "function_replace":
            print("  Function replace failed, trying file replace...")
            script_content = self._build_file_replace_script(fix_info, target_path)
            script_path.write_text(script_content)
            copy_to_container(container, script_path, Path("/tmp/apply_fix.py"))
            val = container.exec_run(
                ["/bin/bash", "-c", cmd],
                workdir=self.WORKDIR,
                user=self.DOCKER_USER,
            )
        return val.exit_code == 0

    def _build_file_replace_script(self, fix_info: Dict, target_path: str) -> str:
        code_b64 = base64.b64encode(fix_info["code"].encode("utf-8")).decode("ascii")
        return (
            "import base64\n"
            f"with open('{target_path}', 'w') as f:\n"
            f"    f.write(base64.b64decode('{code_b64}').decode('utf-8'))\n"
            "print('Fix applied (file replace)')\n"
        )

    def _build_function_replace_script(self, fix_info: Dict, target_path: str) -> str:
        func_name = fix_info["target_func"]
        code_b64 = base64.b64encode(fix_info["code"].encode("utf-8")).decode("ascii")
        return (
            "import base64\n"
            f"with open('{target_path}', 'r') as f:\n"
            "    content = f.read()\n"
            f"fix_code = base64.b64decode('{code_b64}').decode('utf-8')\n"
            "lines = content.split('\\n')\n"
            "func_start = None\n"
            "func_end = None\n"
            "base_indent = None\n"
            "for i, line in enumerate(lines):\n"
            f"    stripped = line.strip()\n"
            f"    if stripped.startswith('def {func_name}(') or stripped.startswith('class {func_name}'):\n"
            "        func_start = i\n"
            "        base_indent = len(line) - len(line.lstrip())\n"
            "    elif func_start is not None and func_end is None:\n"
            "        curr_indent = len(line) - len(line.lstrip())\n"
            "        if stripped and curr_indent <= base_indent:\n"
            "            if stripped.startswith(('def ', 'class ', '@')):\n"
            "                func_end = i\n"
            "                break\n"
            "if func_end is None:\n"
            "    func_end = len(lines)\n"
            "if func_start is not None:\n"
            "    new_lines = lines[:func_start] + [fix_code] + lines[func_end:]\n"
            "    new_content = '\\n'.join(new_lines)\n"
            f"    with open('{target_path}', 'w') as f:\n"
            "        f.write(new_content)\n"
            f"    print('Replaced function: {func_name}')\n"
            "else:\n"
            f"    print('Function not found: {func_name}')\n"
            "    exit(1)\n"
        )

    # ------------------------------------------------------------------
    # Test runners
    # ------------------------------------------------------------------
    def _ensure_pytest(self, container) -> bool:
        """Ensure pytest is installed in the testbed environment."""
        val = container.exec_run(
            ["/bin/bash", "-c", f"{self.CONDA_ACTIVATE} && python -m pytest --version"],
            workdir=self.WORKDIR, user=self.DOCKER_USER
        )
        if val.exit_code != 0:
            print("    Installing pytest in testbed environment...")
            install_val = container.exec_run(
                ["/bin/bash", "-c", f"{self.CONDA_ACTIVATE} && python -m pip install pytest"],
                workdir=self.WORKDIR, user=self.DOCKER_USER
            )
            if install_val.exit_code != 0:
                print(f"    Warning: failed to install pytest: {install_val.output.decode()[:200]}")
                return False
        return True

    def _run_test_plain(self, container, test_name: str, fix_info: Dict) -> TestResult:
        try:
            if not self._apply_fix_in_container(container, fix_info):
                return TestResult(test_name=test_name, passed=False, error="Fix application failed")

            self._ensure_pytest(container)
            cmd = f"{self.CONDA_ACTIVATE} && cd {self.WORKDIR} && python -m pytest {test_name} -v --tb=short 2>&1"
            val = container.exec_run(["/bin/bash", "-c", cmd], workdir=self.WORKDIR, user=self.DOCKER_USER)
            output = val.output.decode("utf-8", errors="replace")
            passed = val.exit_code == 0
            error = None if passed else self._parse_error(output)
            return TestResult(test_name=test_name, passed=passed, output=output, error=error)
        except Exception as e:
            return TestResult(test_name=test_name, passed=False, error=str(e))

    def _run_test_with_trace(self, container, test_name: str, fix_info: Dict) -> TestResult:
        tracer_lib_dir = self._prepare_tracer_lib_dir()
        trace_out_dir = Path(tempfile.mkdtemp(prefix="trace_out_"))
        collector_script = Path(__file__).parent.parent / "swe_bench_pro" / "docker_trace_collector.py"

        try:
            if not self._apply_fix_in_container(container, fix_info):
                return TestResult(test_name=test_name, passed=False, error="Fix application failed")

            target_module = fix_info["target_file"].replace("/", ".").replace(".py", "")
            if target_module.startswith("lib."):
                target_module = target_module[4:]

            # Copy tracer lib and collector script into container
            copy_to_container(container, tracer_lib_dir, Path("/tmp/tracer_lib"))
            copy_to_container(container, collector_script, Path("/tmp/docker_trace_collector.py"))

            # Ensure trace output directory exists in container
            container.exec_run("mkdir -p /tmp/trace_out", workdir=self.WORKDIR)

            self._ensure_pytest(container)

            trace_cmd = f"""
{self.CONDA_ACTIVATE}
export PYTHONPATH=/tmp/tracer_lib:$PYTHONPATH
python /tmp/docker_trace_collector.py \
    --target-module "{target_module}" \
    --target-func "{self.target_func}" \
    --target-file "{fix_info['target_file']}" \
    --test "{test_name}" \
    --output /tmp/trace_out/result.json \
    --max-trace-depth 50000
2>&1
"""
            val = container.exec_run(
                ["/bin/bash", "-c", trace_cmd],
                workdir=self.WORKDIR,
                user=self.DOCKER_USER,
            )
            output = val.output.decode("utf-8", errors="replace")
            passed = val.exit_code == 0
            error = None if passed else self._parse_error(output)

            # Retrieve trace result from container using get_archive
            trace_data = None
            try:
                import tarfile, io
                bits, stat = container.get_archive("/tmp/trace_out/result.json")
                tar = tarfile.open(fileobj=io.BytesIO(b"".join(bits)), mode="r:*")
                member = tar.getmember("result.json")
                f = tar.extractfile(member)
                trace_data = json.loads(f.read().decode("utf-8"))
                tar.close()
            except Exception as e:
                print(f"Warning: failed to retrieve trace result from container: {e}")

            return TestResult(test_name=test_name, passed=passed, output=output, error=error, trace=trace_data)
        except Exception as e:
            return TestResult(test_name=test_name, passed=False, error=str(e))
        finally:
            for d in (tracer_lib_dir, trace_out_dir):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass

    def _prepare_tracer_lib_dir(self) -> Path:
        tmpdir = Path(tempfile.mkdtemp(prefix="tracer_lib_"))
        src_dir = Path(__file__).parent.parent / "swe_bench_pro"
        for fname in ["tracer.py", "utils.py", "config.py", "timeout_utils.py"]:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, tmpdir)
        # Mock loguru
        (tmpdir / "loguru.py").write_text(
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
        # Mock openai (required by config.py)
        (tmpdir / "openai.py").write_text(
            "class OpenAI:\n"
            "    def __init__(self, *a, **k): pass\n"
            "class ChatCompletion:\n"
            "    @staticmethod\n"
            "    def create(*a, **k):\n"
            "        return {'choices': [{'message': {'content': ''}}]}\n"
        )
        return tmpdir

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _parse_error(self, logs: str) -> str:
        for line in logs.split("\n"):
            if "FAILED" in line or "ERROR" in line:
                return line.strip()
        for line in reversed(logs.split("\n")):
            if line.strip():
                return line.strip()
        return "Unknown error"

    def _generate_report(self, instance_id: str, ftp_results: List[TestResult], ptp_results: List[TestResult]) -> ValidationReport:
        ftp_passed = sum(1 for r in ftp_results if r.passed)
        ftp_total = len(ftp_results)
        ptp_passed = sum(1 for r in ptp_results if r.passed)
        ptp_total = len(ptp_results)

        if ftp_passed == ftp_total and ptp_passed == ptp_total and ftp_total > 0:
            status = "RESOLVED"
            details = f"✓ All tests passed ({ftp_passed}/{ftp_total} FTP, {ptp_passed}/{ptp_total} PTP)"
        elif ftp_passed > 0:
            status = "PARTIAL"
            details = f"~ Partial fix ({ftp_passed}/{ftp_total} FTP, {ptp_passed}/{ptp_total} PTP)"
        else:
            status = "FAILED"
            details = f"✗ No improvement ({ftp_passed}/{ftp_total} FTP, {ptp_passed}/{ptp_total} PTP)"

        return ValidationReport(
            instance_id=instance_id,
            status=status,
            fail_to_pass_results=ftp_results,
            pass_to_pass_results=ptp_results,
            details=details,
        )

    def _error_report(self, error_msg: str) -> ValidationReport:
        return ValidationReport(
            instance_id=self.instance.get("instance_id", "unknown"),
            status="ERROR",
            fail_to_pass_results=[],
            pass_to_pass_results=[],
            details=f"Error: {error_msg}",
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick test for VerifiedDockerTestExecutor")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--fix-file", required=True)
    args = parser.parse_args()

    loader = VerifiedDataLoader()
    inst = loader.get_instance(args.instance_id)
    if not inst:
        print(f"Instance not found: {args.instance_id}")
        sys.exit(1)

    fix_code = Path(args.fix_file).read_text()
    executor = VerifiedDockerTestExecutor(inst)
    report = executor.validate(fix_code)
    print(json.dumps(report.to_dict(), indent=2))
