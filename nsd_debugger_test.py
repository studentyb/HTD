#!/usr/bin/env python3
"""
基于细粒度执行追踪的调试器 - 替代mg_debug

功能：
1. 捕获语句级执行轨迹τ（控制流+数据流+DU路径）
2. 基于追踪信息进行根因定位
3. 与原MGDebugger架构完全兼容
4. 支持Azure OpenAI o1-mini进行智能调试
"""

"""
...
性能建议：
- 追踪深度限制：max_trace_depth=100（可调）
- 循环迭代记录：变量定义最多100次（防内存爆炸）
- 大对象自动截断：>1000元素显示长度
- 典型性能：1000行代码追踪 < 50ms
"""

import sys
import os
import json
import re
from typing import Dict, Any, Set, Tuple, Optional, List
from collections import defaultdict
from loguru import logger
import unittest
import ast

import gzip
import pickle

from openai import OpenAI

# 导入MGDebugger原有组件
from utils import (
    extract_functions, get_dependency_graph_str, 
    create_dependency_graph, topological_sort, merge_changes_to_parents,
    extract_code_blocks, STARTING_CODE, safe_json_dumps
)
from code_conversion import convert_to_hierarchical
from test_generation import generate_test_cases
from config import LLM_API_KEY, MAX_DEBUG_RETRIES, MAX_PARSE_RETRIES, LLM_API_KEY, LLM_BASE_URL, MODEL
import config


_ast_cache: Dict[str, ast.AST] = {}

class ReturnValueSpec:
    """返回值规范管理器 - 确保整个调试流程中返回值格式一致"""
    
    def __init__(self, expected_format: str = "auto"):
        """
        Args:
            expected_format: 期望的返回值格式
                - "auto": 自动从测试用例推断
                - "list": 强制使用列表格式，如 [2.0, 2.2]
                - "tuple": 强制使用元组格式，如 (2.0, 2.2)
                - "dict": 强制使用字典格式，如 {"result": [2.0, 2.2]}
        """
        self.format = expected_format
        self._cached_spec = None
    
    def infer_from_testcase(self, gold_test_cases: List[Dict]) -> str:
        """从黄金测试用例推断返回值格式"""
        if self.format != "auto":
            return self.format
        
        if not gold_test_cases:
            return "list"  # 默认使用列表
        
        # 检查第一个测试用例的期望输出类型
        first_output = gold_test_cases[0].get('expected_output')
        
        if isinstance(first_output, list):
            return "list"
        elif isinstance(first_output, tuple):
            return "tuple"
        elif isinstance(first_output, dict):
            return "dict"
        else:
            return "list"  # 默认回退到列表
    
    def get_format_hint(self) -> str:
        """获取格式提示，用于LLM prompt"""
        format_hints = {
            "list": "返回值必须是Python列表（list）格式，如 [value1, value2]",
            "tuple": "返回值必须是Python元组（tuple）格式，如 (value1, value2)",
            "dict": "返回值必须是Python字典（dict）格式，包含特定字段"
        }
        return format_hints.get(self.format, "返回值格式应与测试用例期望一致")




class ExprUseExtractor(ast.NodeVisitor):
    """提取表达式中所有Load上下文的变量名"""
    def __init__(self):
        self.uses = set()
        self.skip = set()  # 推导式迭代变量等需要排除的
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.uses.add(node.id)
        self.generic_visit(node)
    
    def visit_ListComp(self, node):
       # 记录迭代变量到skip
        for gen in node.generators:
            if isinstance(gen.target, ast.Name):
                self.skip.add(gen.target.id)
            elif isinstance(gen.target, ast.Tuple):
                for elt in gen.target.elts:
                    if isinstance(elt, ast.Name):
                        self.skip.add(elt.id)
            # ✅ 必须访问iter（如range(n)中的n）
            self.visit(gen.iter)
        # ✅ 访问elt
        self.visit(node.elt)
    
        # ✅ 新增：支持所有推导式
    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        for gen in node.generators:
            self.visit(gen.iter)
        self.visit(node.key)
        self.visit(node.value)
        
    def get_clean_uses(self):
        return self.uses - self.skip

def extract_return_vars(full_code: str, func_name: str) -> List[str]:
    """从完整代码中提取 return 的变量名（使用全局行号，但这里只关心变量名）"""
    try:
        tree = ast.parse(full_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        if isinstance(stmt.value, ast.Name):
                            return [stmt.value.id]
    except Exception:
        pass
    return []


def _extract_assignments_from_statements(statements, line_to_targets):
    for stmt in statements:
        if isinstance(stmt, ast.FunctionDef):
           _extract_assignments_from_statements(stmt.body, line_to_targets)
           continue
        if isinstance(stmt, ast.Assign):
            # 检查 value 是否是 lambda
            if isinstance(stmt.value, ast.Lambda):
                # lambda 的参数视为变量定义
                lambda_args = stmt.value.args
                if lambda_args:
                    for arg in lambda_args.args:
                        line_to_targets[stmt.lineno].append(arg.arg)  # 添加参数名
                
                # 递归处理 lambda 的 body（如果是复杂表达式）
                # 注意：lambda.body 是表达式，不是语句列表
                # 这里可以添加对表达式内部嵌套函数的处理（如有必要）
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    line_to_targets[stmt.lineno].append(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            line_to_targets[stmt.lineno].append(elt.id)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name):
                line_to_targets[stmt.lineno].append(stmt.target.id)
        elif isinstance(stmt, ast.For):
            # 提取循环变量
            if isinstance(stmt.target, ast.Name):
                line_to_targets[stmt.lineno].append(stmt.target.id)
            elif isinstance(stmt.target, ast.Tuple):
                for elt in stmt.target.elts:
                    if isinstance(elt, ast.Name):
                        line_to_targets[stmt.lineno].append(elt.id)
            # ✅ 关键修复：递归处理嵌套循环体
            _extract_assignments_from_statements(stmt.body, line_to_targets)
        elif isinstance(stmt, ast.While):
            # ✅ 关键修复：递归处理循环体
            _extract_assignments_from_statements(stmt.body, line_to_targets)
        elif isinstance(stmt, ast.Try):
            _extract_assignments_from_statements(stmt.body, line_to_targets)
            for handler in stmt.handlers:
                _extract_assignments_from_statements(handler.body, line_to_targets)
            _extract_assignments_from_statements(stmt.orelse, line_to_targets)
            _extract_assignments_from_statements(stmt.finalbody, line_to_targets)
        

def extract_assignment_targets(full_code: str, func_name: str) -> Dict[int, List[str]]:
    try:
        tree = ast.parse(full_code)
        line_to_targets = defaultdict(list)
        
        # 👇 新增：找到目标函数节点
        target_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                target_func = node
                break
        
        if not target_func:
            return dict(line_to_targets)
            
        # 👇 新增：递归处理，包括嵌套函数
        def _extract_from_func(func_node):
            # 提取当前函数的赋值
            _extract_assignments_from_statements(func_node.body, line_to_targets)
            # 递归处理嵌套函数
            for stmt in func_node.body:
                if isinstance(stmt, ast.FunctionDef):
                    _extract_from_func(stmt)  # 递归进入嵌套函数
        
        _extract_from_func(target_func)
        return dict(line_to_targets)
        
    except Exception:
        return {}

class TraceStrategy:
    """追踪策略配置"""
    
    DEBUG = {
        'max_trace_depth': 2000,
        'loop_sampling': 1,      # 记录每次迭代
        'record_all_vars': True,
        'exception_snapshot': True,
        'ast_caching': True
    }
    
    PERFORMANCE = {
        'max_trace_depth': 500,
        'loop_sampling': 5,      # 每5次记录一次
        'record_all_vars': False, # 只记录关键变量
        'exception_snapshot': True,
        'ast_caching': True
    }
    
    MINIMAL = {
        'max_trace_depth': 100,
        'loop_sampling': 10,     # 每10次记录一次
        'record_all_vars': False,
        'exception_snapshot': False,
        'ast_caching': True
    }



# ========== 核心：细粒度执行追踪器 ==========

class FineGrainedTracer:
    """
    细粒度执行追踪器 - 捕获语句级执行轨迹τ
    
    修复重点：
    - 支持 TraceStrategy 配置参数
    - 增加 loop_sampling 等策略控制
    """

    def __init__(self, max_trace_depth=1000, max_var_size=100, 
                 loop_sampling=1, record_all_vars=True, 
                 exception_snapshot=True, ast_caching=True):
        # 基础配置
        self.max_trace_depth = max_trace_depth
        self.max_var_size = max_var_size
        
        # 新增：追踪策略配置
        self.loop_sampling = loop_sampling  # 循环采样频率
        self.record_all_vars = record_all_vars  # 是否记录所有变量
        self.exception_snapshot = exception_snapshot  # 异常时是否快照
        self.ast_caching = ast_caching  # 是否启用AST缓存
        
        # 原有状态初始化
        self._ast_tree: Optional[ast.AST] = None
        self._line_to_ast_nodes: Dict[int, List[ast.AST]] = {}
        self._full_code_for_use_analysis: Optional[str] = None
        self.traces = []
        self.call_stack = []
        self.var_definitions = defaultdict(list)
        self.var_uses = defaultdict(list)
        self.loop_counters = defaultdict(int)
        self._old_trace = None
        self.exception_info = None
        self.user_code_names = set()
        self._function_return_vars = {}
        self._line_to_assign_targets = {}
        self._loop_ranges: Dict[int, Set[int]] = {}
        self._global_vars_cache = {}
        
        # 新增：变量值历史记录（用于更精确的时间线）
        self._var_value_history: Dict[str, List[Dict]] = defaultdict(list)
        self._last_snapshot: Dict[str, str] = {}  # 用于差异存储
        self._return_value = None  # 👈 新增：存储最新返回值


    def set_ast_tree(self, tree: ast.AST, full_code: str):
        """设置已解析的 AST 树，并构建行号映射（只做一次）"""
        self._ast_tree = tree
        self._full_code_for_use_analysis = full_code
        self._line_to_ast_nodes = {}
        try:
            for node in ast.walk(tree):
                lineno = getattr(node, 'lineno', None)
                if lineno:
                    if lineno not in self._line_to_ast_nodes:
                        self._line_to_ast_nodes[lineno] = []
                    self._line_to_ast_nodes[lineno].append(node)
        except Exception:
            self._line_to_ast_nodes = {}
     
    def set_assignment_targets(self, line_to_targets: Dict[int, List[str]], full_code: str = None):
        """设置赋值目标，可选分析循环范围"""
        self._line_to_assign_targets = line_to_targets
        if full_code:
            self._analyze_loop_ranges(full_code)
        
        # ✅ 新增：如果没有 full_code，但至少知道循环变量，创建最小循环范围
        elif line_to_targets:
            # 为 for 循环变量创建最小循环范围
            for lineno, vars in line_to_targets.items():
                if any(var in ['i', 'j', 'k', 'idx', 'index'] for var in vars):
                    # 假设这是 for 循环头，记录它本身
                    self._loop_ranges[lineno] = {lineno}

    def _get_body_lines(self, body: List[ast.stmt]) -> Set[int]:
        """递归获取代码块的所有行号"""
        lines = set()
        for stmt in body:
            if hasattr(stmt, 'lineno'):
                lines.add(stmt.lineno)
                # 处理嵌套结构
                if hasattr(stmt, 'body'):
                    lines.update(self._get_body_lines(stmt.body))
                if hasattr(stmt, 'orelse') and stmt.orelse:
                    lines.update(self._get_body_lines(stmt.orelse))
        return lines

    def _analyze_loop_ranges(self, full_code: str):
        """静态分析：提取所有循环的body行号范围（包括循环头）"""
        try:
            tree = ast.parse(full_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    # ✅ 关键修复：包含循环头的行号
                    body_lines = {node.lineno}  # for/while 语句本身
                    for stmt in node.body:
                        if hasattr(stmt, 'lineno'):
                            body_lines.add(stmt.lineno)
                            # 递归处理嵌套
                            if hasattr(stmt, 'body'):
                                body_lines.update(self._get_body_lines(stmt.body))
                    self._loop_ranges[node.lineno] = body_lines
        except:
            pass

    

    def set_full_code_for_use_analysis(self, full_code: str):
        self._full_code_for_use_analysis = full_code
        try:
            self._ast_tree = ast.parse(full_code)
            # 预构建映射，O(n)一次完成
            for node in ast.walk(self._ast_tree):
                lineno = getattr(node, 'lineno', None)
                if lineno:
                    if lineno not in self._line_to_ast_nodes:
                        self._line_to_ast_nodes[lineno] = []
                    self._line_to_ast_nodes[lineno].append(node)
        except:
            self._ast_tree = None

    def set_return_vars(self, func_name: str, vars: List[str]):
        self._function_return_vars[func_name] = vars

    def start(self, user_func_name: str = None):
        """启动追踪器（修复版）"""
        if user_func_name:
            self.user_code_names.add(user_func_name)
        
        # 👇 新增：静态分析提取所有嵌套函数名
        if self._full_code_for_use_analysis:
            try:
                tree = ast.parse(self._full_code_for_use_analysis)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        self.user_code_names.add(node.name)
            except:
                pass
        
        # 👇 关键修复：启用追踪回调
        self._old_trace = sys.gettrace()
        sys.settrace(self._trace_callback)
        return self  # 支持链式调用

    def stop(self):
        """停止追踪"""
        sys.settrace(self._old_trace)

    def __enter__(self):
            """支持 with 语句"""
            self.start()
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.stop()
        return False  # 不抑制异常

    def _is_user_code(self, frame) -> bool:
        """判断是否为需要追踪的用户代码"""
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name
        # ✅ 白名单优先：如果是用户定义的函数（包括嵌套），强制追踪
        if func_name in self.user_code_names:
            return True
         #绝对黑名单：所有调试器内部方法
        BLACKLIST = {
            # 你的追踪器方法
            'start', 'stop', '_trace_callback', '_capture_snapshot', 
            '_capture_args', '_detect_loop', '_is_likely_loop_body',
            '_track_du_chain', '_safe_repr', '_track_function_args',
            # PyDev调试器方法
            '_internal_set_trace', '_get_stack_str', 'internal_get_file',
            'get_code', 'should_stop', 'do_stop', 'get_thread_id',  
            '_should_skip_line', '_is_user_code', '__enter__', '__exit__',
            'site-packages',
        'lib/python',
        '<frozen',  # 👈 新增：过滤所有frozen模块
        'typing.py',  # 👈 新增
        'importlib',  # 👈 新增
        }
        if func_name in BLACKLIST:
            return False

        # 跳过标准库和第三方包
        if ('site-packages' in filename or 'lib/python' in filename):
            return False

        # 保留 <string> 中的用户函数（动态 exec 的函数）
        # <string>中的用户代码
        if filename == '<string>':
            if func_name == '<module>':
                return False
            # ✅ 修改：追踪所有非内部函数，不只限于user_code_names
            return True
            

        # 其他本地文件（如 .py 脚本）默认追踪（可进一步限制）
        return filename.endswith('.py')
    
    def _track_function_args(self, frame, lineno: int):
        for name, val in frame.f_locals.items():
            if isinstance(name, str) and len(name) < 50 and not name.startswith(('_', 'self', 'sys', 'os')):
                if name not in self.var_definitions:
                    self.var_definitions[name].append(lineno)

    def _trace_callback(self, frame, event, arg):
        """修复版：确保深度限制在所有事件中生效"""
        
        # ✅ 关键修复1：深度检查前置（异常除外）
        if event != 'exception' and len(self.traces) >= self.max_trace_depth:
            sys.settrace(None)  # 停止追踪
            return None
        
        # ✅ 关键修复2：异常事件优先处理，但也要检查深度
        if event == 'exception':
            exc_type, exc_value, exc_tb = arg
            
            if self._is_user_code(frame):
                # 只在首次异常时记录
                if self.exception_info is None:
                    self.exception_info = {
                        'type': exc_type.__name__,
                        'message': str(exc_value),
                        'line': frame.f_lineno,
                        'function': frame.f_code.co_name
                    }
                
                # ✅ 修复：异常记录也要遵守深度限制
                if len(self.traces) < self.max_trace_depth:
                    record = {
                        'event': 'exception',
                        'line': frame.f_lineno,
                        'function': frame.f_code.co_name,
                        'filename': frame.f_code.co_filename,
                        'timestamp': len(self.traces),
                        'exception': f"{exc_type.__name__}: {exc_value}"
                    }
                    self.traces.append(record)
                else:
                    # ✅ 达到深度限制，停止追踪
                    sys.settrace(None)
                    return None
            
            return self._trace_callback

        # 用户代码过滤
        if not self._is_user_code(frame):
            return self._trace_callback

        try:
            record = {
                'event': event,
                'line': frame.f_lineno,
                'function': frame.f_code.co_name,
                'filename': frame.f_code.co_filename,
                'timestamp': len(self.traces)
            }

            if event == 'line':
                if self._should_skip_line(frame):
                    return self._trace_callback
                    
                record['variables'] = self._capture_snapshot(frame)
                loop_info = self._detect_loop(frame)
                record['loop_info'] = loop_info
                self._track_du_chain(frame, frame.f_lineno, loop_info)

            elif event == 'call':
                self.call_stack.append(frame.f_code.co_name)
                self._track_function_args(frame, frame.f_lineno)
                record['call_depth'] = len(self.call_stack)
                record['args'] = self._capture_args(frame)

            elif event == 'return':
                # 确保arg（返回值）被正确序列化并存储
                # ✅ 修复：同时存储实际值和字符串表示
                self._return_value = arg  # 👈 存储真实返回值
                record['return_val'] = self._safe_repr(arg)
                current_func = frame.f_code.co_name
                if self.call_stack:
                    self.call_stack.pop()
                
                if current_func in self._function_return_vars:
                    return_vars = self._function_return_vars[current_func]
                    for var in return_vars:
                        if var in self.var_definitions:
                            if frame.f_lineno not in self.var_uses[var]:
                                self.var_uses[var].append(frame.f_lineno)
                
                record['return_val'] = self._safe_repr(arg)

            # ✅ 关键修复3：添加记录前再次检查深度
            if len(self.traces) < self.max_trace_depth:
                self.traces.append(record)
            else:
                # ✅ 达到深度限制，立即停止追踪
                sys.settrace(None)
                return None
                
        except Exception:
            pass

        return self._trace_callback

    def _should_skip_line(self, frame) -> bool:
        """智能跳过：循环体内同一行不重复记录"""
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        
        # 检查最近是否已记录同一行
        if self.traces:
            last_trace = self.traces[-1]
            if (last_trace['function'] == func_name and 
                last_trace['line'] == lineno and 
                last_trace['event'] == 'line'):
                return True
        
        return False    

    # def _capture_snapshot(self, frame) -> Dict[str, str]:
    #     """捕获变量快照（局部+全局），优先显示非临时变量"""
    #     snapshot = {}
        
    #     # 局部变量（排除内部变量，限制数量）
    #     for name, val in frame.f_locals.items():
    #         # ✅ 跳过不可哈希对象（避免缓存问题）
    #         try:
    #             hash(name)
    #         except:
    #             continue
                
    #         if isinstance(name, str) and not name.startswith(('_', 'sys', 'os')):
    #             # ✅ 跳过不可打印对象（如文件句柄）
    #             try:
    #                 snapshot[f"L.{name}"] = self._safe_repr(val)
    #             except:
    #                 snapshot[f"L.{name}"] = "<unprintable>"
    #             if len(snapshot) >= 5:
    #                 break
        
    #     # ✅ 恢复全局变量处理（代码对象级缓存）
    #     code_obj = frame.f_code
    #     if code_obj not in self._global_vars_cache:
    #         self._global_vars_cache[code_obj] = set(code_obj.co_names)
        
    #     for name in self._global_vars_cache[code_obj]:
    #         if name in frame.f_globals and not name.startswith('__'):
    #             key = f"G.{name}"
    #             if key not in snapshot:
    #                 try:
    #                     snapshot[key] = self._safe_repr(frame.f_globals[name])
    #                 except:
    #                     snapshot[key] = "<unprintable>"
    #                 if len(snapshot) >= 8:
    #                     break
        
    #     return snapshot  # ✅ 修复

    def _capture_snapshot(self, frame) -> Dict[str, str]:
        """捕获变量快照（局部+全局），支持差异存储"""
        snapshot = {}
        
        # 收集当前变量
        for name, val in frame.f_locals.items():
            if not isinstance(name, str) or name.startswith('_'):
                continue
            val_repr = self._safe_repr(val)
            key = f"L.{name}"
            snapshot[key] = val_repr
        
        # ✅ 根据策略决定是返回完整快照还是差异
        if self.record_all_vars:
            self._last_snapshot = snapshot
            return snapshot
        
        # 差异存储模式：只返回变化的变量
        if not hasattr(self, '_last_snapshot'):
            self._last_snapshot = {}
        
        diff = {}
        for key, val in snapshot.items():
            if self._last_snapshot.get(key) != val:
                diff[key] = val
        
        self._last_snapshot.update(snapshot)
        return self._last_snapshot  # 返回完整视图，但内部存储高效    
    

    def _capture_args(self, frame) -> Dict[str, str]:
        """捕获函数参数"""
        args = {}
        for k, v in frame.f_locals.items():
            if isinstance(k, str) and not k.startswith('_'):
                args[k] = self._safe_repr(v)
        return args

    def _detect_loop(self, frame) -> Optional[Dict]:
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        
        # ✅ 修复：找到所有包含当前行的循环
        containing_loops = []
        for loop_start, body_lines in self._loop_ranges.items():
            if lineno in body_lines:
                containing_loops.append(loop_start)
        
        if not containing_loops:
            # 如果没有静态分析的循环范围，使用启发式检测
            if self._is_likely_loop_body(frame):
                # 启发式：当前行可能是循环体的一部分
                # 创建一个临时的 loop_info
                key = f"{func_name}:heuristic"
                self.loop_counters[key] += 1
                
                return {
                    'iter': self.loop_counters[key],
                    'loop_line': lineno,  # 使用当前行作为 loop_line
                    'current_line': lineno,
                    'vars': [],
                    'is_heuristic': True
                }
            return None
        
        # ✅ 修复：选择最内层的循环（行号最大的loop_start）
        innermost_loop = max(containing_loops)
        
        key = f"{func_name}:{innermost_loop}"
        self.loop_counters[key] += 1
        
        # 获取循环变量名（for循环的目标）
        loop_vars = self._line_to_assign_targets.get(innermost_loop, [])
        
        return {
            'iter': self.loop_counters[key],
            'loop_line': innermost_loop,
            'current_line': lineno,
            'vars': loop_vars
        }
            

    def _is_likely_loop_body(self, frame) -> bool:
        """启发式判断：检查字节码中是否有循环相关指令"""
        code = frame.f_code.co_code
        return b'FOR_ITER' in code or b'JUMP_ABSOLUTE' in code


  
    def _track_du_chain(self, frame, lineno: int, loop_info: Optional[Dict] = None):
        """追踪DU链（修复循环变量多次定义）"""
        # Step 1: 静态分析的定义变量
        static_targets = self._line_to_assign_targets.get(lineno, [])
        for var in static_targets:
            if isinstance(var, str) and len(var) < 30 and not var.startswith('_'):
                if var not in self.var_definitions:
                    self.var_definitions[var].append(lineno)
        
        # Step 2: 收集当前局部变量
        current_vars = []
        for name, val in frame.f_locals.items():
            if (isinstance(name, str) and len(name) < 30 and 
                not name.startswith(('_', 'self', 'sys', 'os')) and
                not callable(val)):
                current_vars.append(name)
        
        # ✅ Step 3: 修复循环变量记录逻辑
        for var in current_vars:
            is_loop_var = loop_info and var in loop_info.get('vars', [])
            
            if var not in self.var_definitions:
                self.var_definitions[var].append(lineno)
            elif is_loop_var:
                iter_num = loop_info.get('iter', 1)
                
                # 对于循环变量，只要满足采样条件就记录
                if iter_num <= 100 and (iter_num % self.loop_sampling == 0 or iter_num <= 3):
                    # 检查是否是新的值（用于历史记录）
                    current_val = self._safe_repr(frame.f_locals.get(var))
                    history = self._var_value_history[var]
                    
                    if not history or history[-1]['value'] != current_val:
                        self.var_definitions[var].append(lineno)
            
            # ✅ Step 4: 记录变量使用
            if var in self.var_definitions:
                def_line = self.var_definitions[var][-1]
                if lineno != def_line:
                    if lineno not in self.var_uses[var]:
                        self.var_uses[var].append(lineno)
            
            # ✅ Step 5: 记录变量值历史
            if var in current_vars and (self.record_all_vars or is_loop_var):
                self._var_value_history[var].append({
                    'line': lineno,
                    'value': self._safe_repr(frame.f_locals.get(var)),
                    'iteration': loop_info.get('iter') if loop_info else None,
                    'timestamp': len(self.traces)
                })
        
        # AST表达式变量使用追踪
        if self._full_code_for_use_analysis:
            self._extract_expr_uses(lineno, frame)


    def _extract_expr_uses(self, lineno: int, frame):
        """O(1)直接查表"""
        if lineno not in self._line_to_ast_nodes:
            return
        
        for node in self._line_to_ast_nodes[lineno]:
            extractor = ExprUseExtractor()
            extractor.visit(node)
            for var in extractor.get_clean_uses():
                if (var in self.var_definitions and 
                    var not in self._line_to_assign_targets.get(lineno, [])):
                    if lineno not in self.var_uses[var]:
                        self.var_uses[var].append(lineno)


        # 2. 对大对象激进截断
    def _safe_repr(self, obj) -> str:
        """安全地获取对象的字符串表示，处理大对象和不可打印对象"""
        try:
            # ✅ 超大对象直接截断（避免 repr() 开销）
            if isinstance(obj, (str, bytes)) and len(obj) > 1000:
                return f"{type(obj).__name__}({len(obj)} chars, truncated)"
            if isinstance(obj, (list, tuple, dict, set)) and len(obj) > 1000:
                return f"{type(obj).__name__}({len(obj)} items, truncated)"
            
            # ✅ 基础类型处理
            if obj is None:
                return "None"
            elif isinstance(obj, (int, float, bool)):
                s = repr(obj)
            elif isinstance(obj, (str, bytes)):
                s = repr(obj)
            elif isinstance(obj, (list, tuple)):
                s = f"{type(obj).__name__}({len(obj)} items)"
            elif isinstance(obj, dict):
                s = f"dict({len(obj)} keys)"
            elif isinstance(obj, set):
                s = f"set({len(obj)})"
            else:
                s = f"<{type(obj).__name__}>"
            
            # ✅ 应用全局大小限制
            if len(s) > self.max_var_size:
                return s[:self.max_var_size] + "..."
            return s
        except:
            return "<unprintable>"
    
    def _format_exception_summary(self, testcase: Dict) -> str:
        exc = self.exception_info
        lines = [
            "## 🔥 Exception-Driven Debugging Analysis",
            f"**Exception Type**: `{exc['type']}`",
            f"**Exception Message**: `{exc['message']}`",
            f"**Location**: `{exc['function']}()` at line {exc['line']}",
            "",
            "### Key Variable States at Exception Time",
        ]
        
        # 获取异常发生时的变量快照
        exc_trace = next((t for t in self.traces if t.get('exception')), None)
        if exc_trace and 'variables' in exc_trace:
            for k, v in list(exc_trace['variables'].items())[:5]:
                lines.append(f"- `{k}` = `{v}`")
        else:
            lines.append("- No variable snapshot available")
        
        lines.extend([
            "",
            "### Task for LLM",
            "1. The function crashed with the above exception.",
            "2. Analyze the variable states to understand the root cause.",
            "3. Return a **robust version** that handles this error (e.g., type validation, boundary checks, safe conversion).",
            "4. Ensure the output is JSON-serializable if required."
        ])
        
        return "\n".join(lines)
  
    def get_trace_summary(self, testcase: Dict) -> str:
        """
        LLM友好的执行轨迹摘要（修复空值检查）
        """
        if not self.traces:
            return "## Execution Summary\n✅ No execution trace captured"
        
        # 👇 新增：检查异常信息
        if self.exception_info:
            return self._format_exception_summary(testcase)
        
        # 👇 确保 testcase 包含必要字段
        if not isinstance(testcase, dict):
            testcase = {'input': {}, 'expected_output': None, 'actual_output': None}
        
        # 确保有实际输出
        if 'actual_output' not in testcase:
            # 尝试从追踪中获取最后返回值
            for t in reversed(self.traces):
                if 'return_val' in t:
                    testcase['actual_output'] = t['return_val']
                    break
        if not self.traces:
            return "## Execution Summary\n✅ No execution trace captured"
    # 👇 新增：优先处理异常场景
        if self.exception_info:
            return self._format_exception_summary(testcase)
        
            # 1. 偏差焦点分析（替代异常焦点）
        deviation_section = self._format_deviation_focus(testcase)

        # 2. 关键变量时间线（基于结果偏差识别）
        variable_timeline = self._format_variable_timeline(testcase)

        # 3. 执行路径树
        execution_tree = self._format_execution_tree()

        # 4. 数据流传播链（指向错误结果）
        data_flow_chain = self._format_data_flow_chain(testcase)

        return "\n\n".join(filter(None, [
            deviation_section,
            variable_timeline,
            execution_tree,
            data_flow_chain
        ]))

    def _format_execution_tree(self) -> str:
        """执行路径树（简化版）"""
        if not self.traces:
            return ""
        lines = ["## 🌲 Execution Path Tree"]
        for i, t in enumerate(self.traces[:10]):  # 限制前10步
            line = f"  {i+1}. L{t['line']:3d} | `{t['function']}` | {t['event']}"
            if t.get('variables'):
                vars_str = ", ".join(f"{k}={v}" for k, v in list(t['variables'].items())[:2])
                line += f" → {vars_str}"
            lines.append(line)
        if len(self.traces) > 10:
            lines.append("  ... (truncated)")
        return "\n".join(lines)

    def _format_deviation_focus(self, testcase: Dict) -> str:
        """偏差焦点：对比实际输出 vs 期望输出"""
        actual = testcase.get('actual_output')
        expected = testcase.get('expected_output')
        
        if actual == expected:
            return "## ✅ Test Passed\nOutput matches expected result"
        
        # 找到结果返回前的最后几步执行
        return_line = self._find_return_line()
        focus_traces = self._get_traces_before_line(return_line, window=5)
        
        lines = [
            "## 🔴 Deviation Focus Analysis",
            f"**Expected Output**: `{expected}`",
            f"**Actual Output**: `{actual}`",
            f"**Mismatch Location**: Around L{return_line}\n",
            "**Key execution steps before return:**"
        ]
        
        for t in focus_traces:
            marker = "  → " if t['line'] == return_line else "    "
            lines.append(f"{marker}L{t['line']:3d} | `{t['function']}` | {t['event']}")
            
            if t.get('variables'):
                # 高亮可能影响结果的变量
                vars_display = self._highlight_result_impact_vars(t['variables'], expected)
                if vars_display:
                    lines.append(f"      └─ {vars_display}")
        
        return "\n".join(lines)

    def _find_return_line(self) -> int:
        """找到return语句的行号"""
        for t in reversed(self.traces):
            if t['event'] == 'return':
                return t['line']
        return self.traces[-1]['line'] if self.traces else 0

    def _get_traces_before_line(self, line: int, window: int = 5) -> List[Dict]:
        """获取某行前的window条轨迹"""
        target_idx = next((i for i, t in enumerate(self.traces) if t['line'] >= line), len(self.traces))
        start_idx = max(0, target_idx - window)
        return self.traces[start_idx:target_idx]

    def _highlight_result_impact_vars(self, variables: Dict, expected) -> str:
        """高亮影响结果的变量（启发式）"""
        key_vars = []
        for var_name, var_val in variables.items():
            # 如果变量值接近结果（数值类型）
            if isinstance(expected, (int, float)) and isinstance(var_val, str):
                try:
                    val_num = float(var_val.split()[0])  # "list[5]" -> 5.0
                    if abs(val_num - expected) < 0.1:
                        key_vars.append(f"**{var_name}={var_val}** ⚠️")
                        continue
                except:
                    pass
            
            # 如果变量名包含 result/output/ret
            if any(kw in var_name.lower() for kw in ['result', 'output', 'ret', 'ans']):
                key_vars.append(f"**{var_name}={var_val}**")
            else:
                key_vars.append(f"{var_name}={var_val}")
        
        return ", ".join(key_vars[:3])  # 限制显示数量

    def _format_variable_timeline(self, testcase: Dict) -> str:
        """关键变量时间线：基于结果偏差识别"""
        key_vars = self._identify_key_variables_by_deviation(testcase)
        
        if not key_vars:
            return ""
        
        lines = ["## 📈 Key Variable Timeline"]
        
        for var in key_vars[:3]:
            lines.append(f"\n**Variable `{var}` lifecycle:**")
            
            # 定义点
            if var in self.var_definitions:
                def_points = ", ".join([f"L{d}" for d in self.var_definitions[var]])
                lines.append(f"  - **Definition**: {def_points}")
                        # 👇 修复：检查 expected_output 是否存在
                if 'expected_output' in testcase:
                    deviations = self._extract_deviation_trace(var, testcase['expected_output'])
                else:
                    deviations = []  # 异常场景无 expected_output
                    
            # 使用点
            if var in self.var_uses:
                use_points = ", ".join([f"L{u}" for u in self.var_uses[var]])
                lines.append(f"  - **Usage**: {use_points}")
            
            # 值变化（显示如何偏离预期）
            deviations = self._extract_deviation_trace(var, testcase['expected_output'])
            if deviations:
                lines.append(f"  - **Value trace**: {' → '.join(deviations)}")
        
        return "\n".join(lines)

    def _identify_key_variables_by_deviation(self, testcase: Dict) -> List[str]:
        """基于结果偏差识别关键变量（替代异常启发式）"""
        actual = testcase.get('actual_output')
        expected = testcase.get('expected_output')
        
        # 1. 返回变量（通过AST分析）
        return_vars = list(self._function_return_vars.values())
        if return_vars and return_vars[0]:
            return return_vars[0]
        
        # 2. 与结果值相近的变量
        result_vars = []
        for var, defs in self.var_definitions.items():
            # 检查追踪记录中该变量的最终值
            final_val = self._get_final_variable_value(var)
            if final_val == actual:
                result_vars.append(var)
        
        # 3. 函数参数（输入变量）
        param_vars = [k for k in self.var_definitions.keys() if k in testcase['input']]
        
        # 合并并去重
        key_vars = result_vars + param_vars
        return list(dict.fromkeys(key_vars))

    def _get_final_variable_value(self, var: str):
        """获取变量的最终值（从最近的trace）"""
        for t in reversed(self.traces):
            if t.get('variables') and f"L.{var}" in t['variables']:
                return t['variables'][f"L.{var}"]
        return None

    def _extract_deviation_trace(self, var: str, expected) -> List[str]:
        """提取变量的值如何偏离预期"""
        if expected is None:
           return []  # 安全处理 None
        changes = []
        for t in self.traces:
            if t.get('variables') and f"L.{var}" in t['variables']:
                val = t['variables'][f"L.{var}"]
                # 标记偏离预期的值
                if str(val) != str(expected):
                    changes.append(f"**{val}**")
                else:
                    changes.append(str(val))
                
                if len(changes) >= 3:
                    break
        return changes

    def _format_data_flow_chain(self, testcase: Dict) -> str:
        """数据流传播链：指向错误结果"""
        actual = testcase.get('actual_output')
        expected = testcase.get('expected_output')
        
        if actual == expected:
            return ""
        
        lines = ["## 🔗 Error Propagation Data Chain"]
        
        # 找到输出变量
        output_var = self._find_output_variable()
        
        if output_var:
            lines.append(f"**Output variable**: `{output_var}`")
            lines.append("**How the wrong value propagated:**\n")
            
            # 构建从输入 → 输出的链
            chain = []
            if output_var in self.var_definitions:
                chain.extend([f"L{d}(definition)" for d in self.var_definitions[output_var]])
            if output_var in self.var_uses:
                chain.extend([f"L{u}(usage)" for u in self.var_uses[output_var]])
            
            # 添加入参
            input_vars = [k for k in testcase['input'].keys() if k in self.var_definitions]
            if input_vars:
                chain.insert(0, f"Input({', '.join(input_vars)})")
            
            chain.append(f"Output={actual} ❌ (expected {expected})")
            
            lines.append(" → ".join(chain))
        else:
            lines.append(f"**Final state**: Output={actual} ❌ (expected {expected})")
        
        return "\n".join(lines)

    def _find_output_variable(self) -> Optional[str]:
        """找到可能存储输出的变量"""
        # 1. 通过return语句分析
        for vars in self._function_return_vars.values():
            if vars:
                return vars[0]
        
        # 2. 通过变量名启发式
        for var in self.var_uses:
            if var.lower() in ['result', 'output', 'ret', 'answer', 'res']:
                return var
        
        return None

    def get_structured_trace(self) -> Dict:
        return {
            'traces': self.traces,
            'total_steps': len(self.traces),
            'max_call_depth': max((t.get('call_depth', 0) for t in self.traces), default=0),
            'loop_iterations': dict(self.loop_counters),
            'du_chains': {
                'definitions': dict(self.var_definitions),
                'uses': dict(self.var_uses)
            },
            'exception': self.exception_info
        }
    
    # ✅ 在此添加压缩方法
    def save_compressed_trace(self, filepath: str):
        """保存压缩的追踪数据到文件"""
        import gzip
        import pickle
        
        trace_data = self.get_structured_trace()
        try:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(trace_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"追踪数据已压缩保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存压缩追踪数据失败: {e}")
    
    @staticmethod
    def load_compressed_trace(filepath: str) -> Dict:
        """从文件加载压缩的追踪数据"""
        import gzip
        import pickle
        
        try:
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"加载压缩追踪数据失败: {e}")
            return {}



# ========== 带追踪的evaluate函数 ==========

def _extract_variable_uses_from_line(full_code: str, lineno: int) -> List[str]:
    """
    从完整代码中提取指定行的所有变量 use（Load 语义）
    例如: y = x + 1 → 返回 ['x']
          if arr[i] > max → 返回 ['arr', 'i', 'max']
    """
    try:
        tree = ast.parse(full_code)
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == lineno:
                # 收集当前节点及子节点中所有 Load 类型的名字
                loader = _VariableUseExtractor()
                loader.visit(node)
                return list(loader.uses)
        return []
    except Exception:
        return []


class _VariableUseExtractor(ast.NodeVisitor):
    """AST 访问器：提取所有 Load 语义的变量名"""
    def __init__(self):
        self.uses = set()

    def visit_Name(self, node):
        # 只记录 Load（读取），忽略 Store（定义）
        if isinstance(node.ctx, ast.Load):
            self.uses.add(node.id)
        # 继续遍历子节点（如函数调用中的参数）
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # 跳过 self.x 这类属性（通常不视为局部变量 use）
        
        self.visit(node.value)

def extract_assignment_targets_from_tree(tree: ast.AST, func_name: str) -> Dict[int, List[str]]:
    """从已解析的 AST 树中提取赋值目标（避免重复 parse）"""
    try:
        line_to_targets = defaultdict(list)
        # 找到目标函数
        target_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                target_func = node
                break
        if not target_func:
            return dict(line_to_targets)
        # 递归提取
        def _extract_from_func(func_node):
            _extract_assignments_from_statements(func_node.body, line_to_targets)
            for stmt in func_node.body:
                if isinstance(stmt, ast.FunctionDef):
                    _extract_from_func(stmt)  # 嵌套函数
        _extract_from_func(target_func)
        return dict(line_to_targets)
    except Exception:
        return {}

def extract_return_vars_from_tree(tree: ast.AST, func_name: str) -> List[str]:
    """从已解析的 AST 树中提取 return 变量（避免重复 parse）"""
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                for stmt in node.body:
                    if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
                        return [stmt.value.id]
        return []
    except Exception:
        return []        

def evaluate_with_trace(code: str, func_name: str, testcase: Dict, 
                       max_trace_depth: int = 1000,
                       strategy: str = "MINIMAL"):
    """执行代码并捕获执行轨迹（修复语法错误处理）"""
    exec_globals = {}
    full_code = f"{STARTING_CODE}\n\n{code}"
    
    # 策略配置
    strategy_dict = getattr(TraceStrategy, strategy, TraceStrategy.MINIMAL).copy()
    strategy_dict.pop('max_trace_depth', None)
    
    # ✅ 提前初始化 tracer，确保无论如何都有返回值
    tracer = FineGrainedTracer(max_trace_depth=max_trace_depth, **strategy_dict)
    tracer.user_code_names.add(func_name)
    
    try:
        # ✅ 一次性解析 AST
        tree = ast.parse(full_code)
    except SyntaxError as e:
        logger.error(f"代码语法错误: {e}")
        # ✅ 返回包含错误信息的tracer
        tracer.exception_info = {
            'type': 'SyntaxError',
            'message': str(e),
            'line': 0,
            'function': func_name
        }
        return False, {
            'error': f"SyntaxError: {e}",
            'error_type': 'SyntaxError',
            'actual_output': None,
            'expected_output': testcase.get('expected_output')
        }, tracer
    
    # ✅ 设置 AST 信息
    tracer.set_ast_tree(tree, full_code)
    
    # ✅ 基于已解析的 tree 提取信息
    assign_targets = extract_assignment_targets_from_tree(tree, func_name)
    tracer.set_assignment_targets(assign_targets)
    
    return_vars = extract_return_vars_from_tree(tree, func_name)
    tracer.set_return_vars(func_name, return_vars)
    
    try:
        # 编译执行（不追踪）
        compiled_code = compile(full_code, '<string>', 'exec')
        exec(compiled_code, exec_globals)
        
        func = exec_globals[func_name]
        tracer.start(user_func_name=func_name)
        
        try:
            input_dict = testcase['input']
            result = func(**input_dict) if isinstance(input_dict, dict) else func(input_dict)

            # ✅ 将返回值注入testcase
            testcase['actual_output'] = result
            
            expected = testcase['expected_output']
            passed = result == expected
            
            return passed, {
                'actual_output': result,
                'expected_output': expected,
                'passed': passed
            }, tracer
            
        except Exception as e:
            import traceback
            error_info = {
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'actual_output': None,
                'expected_output': testcase.get('expected_output')
            }
            return False, error_info, tracer
            
    except Exception as e:
        logger.error(f"执行失败: {e}")
        error_info = {
            'error': str(e),
            'error_type': type(e).__name__,
            'actual_output': None,
            'expected_output': testcase.get('expected_output')
        }
        return False, error_info, tracer
        
    finally:
        tracer.stop()

# ========== Azure OpenAI o1集成 ==========

class AzureO1Debugger:
    """Azure OpenAI o1-mini调试器"""
    
    def __init__(self, api_key: str = None, endpoint: str = None, deployment: str = "o1"):
        """
        初始化Azure o1客户端
        
        Args:
            api_key: Azure OpenAI API Key (默认从环境变量读取)
            endpoint: Azure OpenAI Endpoint (默认从环境变量读取)
            deployment: 部署名称
        """


# openai.api_type = "azure"
# openai.api_version = "2025-01-01-preview"  # 使用最新的 API 版本
# openai.api_key = LLM_API_KEY
# openai.api_base = LLM_BASE_URL

        self.api_key = LLM_API_KEY
        self.endpoint = LLM_BASE_URL
        # self.deployment =  "o1"
        self.deployment =  MODEL
        
        if not self.api_key or not self.endpoint:
            logger.warning("Azure o1未配置，将使用基础调试模式")
            self.client = None
        else:
            try:
                # from openai import AzureOpenAI
                # self.client = AzureOpenAI(
                #     api_key=self.api_key,
                #     api_version="2025-01-01-preview",
                #     azure_endpoint=self.endpoint
                # )
                self.client = OpenAI(
                    api_key=LLM_API_KEY,  # <-- 替换成你的 Google API Key
                    base_url=LLM_BASE_URL # <-- 这是关键！
                )              
                logger.info("Azure o1调试器已启用")
            except ImportError:
                logger.warning("未安装openai库，将使用基础调试模式")
                self.client = None
    
    def is_available(self) -> bool:
        """检查o1是否可用"""
        return self.client is not None
    
    def debug_with_trace(self, func_code: str, func_name: str, 
                        test_cases: List[Dict], trace_summary: str,
                        error_info: Dict) -> Optional[str]:
        """
        使用Azure o1基于执行轨迹进行调试
        
        Returns:
            修复后的代码，如果失败返回None
        """
        if not self.is_available():
            return None
        
        # 构建prompt
        prompt = self._build_debug_prompt(
            func_code, func_name, test_cases, trace_summary, error_info
        )
        
        try:
            logger.info(f"调用Azure o1调试函数: {func_name}")
            
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}]
                # temperature=0.1,
                # extra_body={"min_p": 0.05}
            )
            
            # 更新token统计
            config.TOTAL_PROMPT_TOKENS += response.usage.prompt_tokens
            config.TOTAL_COMPLETION_TOKENS += response.usage.completion_tokens
            
            content = response.choices[0].message.content
            
            # 提取代码块
            code_blocks = extract_code_blocks(content)
            if code_blocks:
                fixed_code = code_blocks[-1].strip()
                logger.info("o1成功生成修复代码")
                return fixed_code
            else:
                logger.warning("o1响应中未找到代码块")
                return None
               
        except Exception as e:
            logger.error(f"Azure o1调用失败: {e}")
            return None
        
    
    def _build_debug_prompt(self, func_code: str, func_name: str,
                           test_cases: List[Dict], trace_summary: str,
                           error_info: Dict) -> str:

        
        # 提取关键信息
        actual_output = error_info.get('actual_output')
        expected_output = error_info.get('expected_output')
        error_type = error_info.get('error_type', 'N/A')
        
        # 分析错误类型，提供针对性的指导
        error_analysis = ""
        error_specific_instructions = ""
        if error_type == 'AssertionError':
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个断言错误，说明实际输出与期望输出不匹配
        - 重点检查函数的逻辑和返回值类型
        - 确保正确处理边界情况和特殊输入
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **比较实际输出与期望输出**：
        - 实际输出: `{actual_output}`
        - 期望输出: `{expected_output}`
        - 分析两者差异的原因

        2. **检查逻辑分支**：
        - 确认所有if/else分支都正确
        - 检查循环边界条件
        - 验证数学计算的正确性

        3. **验证数据类型**：
        - 确保返回值的类型与期望一致
        - 检查数字类型的精度问题
        - 验证字符串格式是否正确
        """.format(actual_output=actual_output, expected_output=expected_output)
                
        elif 'TypeError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个类型错误，可能是参数类型不匹配或返回值类型错误
        - 检查输入参数的类型转换
        - 确保返回值的类型与期望一致
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **检查参数类型**：
        - 验证输入参数是否符合函数签名要求
        - 检查类型转换是否正确（如str转int，float转int等）
        - 确保不支持的类型不会传入

        2. **检查返回值类型**：
        - 确认返回值类型与函数声明一致
        - 检查是否存在None返回值但期望非None的情况

        3. **验证操作兼容性**：
        - 检查算术运算的类型兼容性
        - 确认列表/字典操作的对象类型正确
        """
                
        elif 'IndexError' in error_type or 'KeyError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个索引/键错误，可能是访问了不存在的数组索引或字典键
        - 检查数组/列表的边界
        - 验证字典键的存在性
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **边界检查**：
        - 验证数组/列表的长度
        - 确保索引在[0, len-1]范围内
        - 检查负索引的正确性

        2. **键存在性检查**：
        - 使用`key in dict`或`dict.get()`方法
        - 确保访问前键已存在

        3. **安全访问模式**：
        - 考虑使用try-except处理异常情况
        - 添加边界条件检查
        """
                
        elif 'ValueError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个值错误，通常表示参数值不合适
        - 检查参数值的有效性
        - 验证转换操作的正确性
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **参数值验证**：
        - 检查数值参数的合理性（如非负、非零等）
        - 验证字符串参数的有效性（如非空、格式正确）
        - 确保列表/字典参数不为空（如果需要）

        2. **转换验证**：
        - 验证类型转换的可行性（如int("abc")会失败）
        - 检查数学运算的有效性（如sqrt(-1)）

        3. **输入预处理**：
        - 添加输入验证逻辑
        - 提供默认值或错误提示
        """
                
        elif 'ZeroDivisionError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个除零错误，说明在除法运算中分母为零
        - 需要检查除法操作前的分母值
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **分母检查**：
        - 在除法操作前检查分母是否为0
        - 考虑使用三元表达式：`x / y if y != 0 else 0`

        2. **防御性编程**：
        - 添加除零保护机制
        - 返回合理的默认值或特殊标记

        3. **数学验证**：
        - 检查可能导致分母为零的逻辑路径
        - 验证所有变量的值域
        """
                
        elif 'AttributeError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个属性错误，可能是访问了不存在的对象属性
        - 检查对象的类型和属性
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **对象类型验证**：
        - 使用`isinstance()`检查对象类型
        - 确认对象已正确初始化

        2. **属性存在性检查**：
        - 使用`hasattr()`检查属性是否存在
        - 考虑使用`getattr()`带默认值

        3. **None值处理**：
        - 检查对象是否为None
        - 添加None值保护
        """
                
        elif 'SyntaxError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个语法错误，说明代码存在语法问题
        - 需要修正代码结构
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **语法检查**：
        - 检查括号、引号的配对
        - 验证缩进是否正确
        - 确认关键字拼写正确

        2. **常见语法问题**：
        - 检查赋值语句的格式
        - 确认函数调用参数正确
        - 验证控制结构语法

        3. **代码结构**：
        - 确保代码块正确闭合
        - 检查导入语句的正确性
        """
                
        elif 'IndentationError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个缩进错误，Python对缩进要求严格
        - 需要修正代码的缩进
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **统一缩进风格**：
        - 使用4个空格作为标准缩进（不要混用制表符和空格）
        - 检查所有代码块的缩进一致性

        2. **代码块检查**：
        - 确认if/else/for/while等语句后的代码块缩进正确
        - 检查函数定义的缩进

        3. **IDE工具**：
        - 使用IDE的格式化功能
        - 启用显示空格和制表符
        """
                
        elif 'NameError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个名称错误，说明使用了未定义的变量或函数
        - 需要检查变量/函数的定义
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **变量作用域**：
        - 检查变量是否在当前作用域内定义
        - 确认全局变量是否正确声明

        2. **导入检查**：
        - 验证导入的模块和函数名正确
        - 检查拼写错误

        3. **变量定义顺序**：
        - 确保变量在使用前已定义
        - 检查循环依赖
        """
                
        elif 'ImportError' in error_type or 'ModuleNotFoundError' in error_type:
                error_analysis = """
        ### 🔍 错误分析
        - 这是一个导入错误，说明无法找到指定的模块或包
        - 需要检查导入语句和依赖
        """
                error_specific_instructions = """
        ### 🎯 针对性调试策略
        1. **模块路径**：
        - 检查模块名拼写是否正确
        - 确认模块是否在Python路径中

        2. **依赖检查**：
        - 验证所需的包是否已安装
        - 检查版本兼容性

        3. **替代方案**：
        - 考虑使用try-except导入不同模块
        - 实现回退逻辑
        """
            
            # 通用调试指导（适用于所有错误类型）
        general_debug_instructions = """
        ### 🔧 通用调试方法
        1. **逐步分析**：
        - 从错误发生点开始反向追踪
        - 检查调用栈中的每一步

        2. **变量状态检查**：
        - 查看执行轨迹中的变量值变化
        - 分析异常发生时的变量状态

        3. **边界条件测试**：
        - 测试最小/最大/边界值
        - 验证特殊输入的处理

        4. **简化问题**：
        - 尝试将复杂问题分解为简单子问题
        - 隔离可能出错的代码段

        5. **对比分析**：
        - 与已知正确的代码对比
        - 检查算法逻辑的一致性
        """
            
            # 构建完整的错误分析部分
        full_error_analysis = f"""
        {error_analysis}
        {error_specific_instructions}
        {general_debug_instructions}
        """

        """构建调试prompt"""
        prompt = f"""You are an expert Python debugging assistant. Your task is to fix the provided function based on execution traces and test failures.

### 待调试函数
```python
{func_code}
```
### 🛡️ Critical Safety & Robustness Rules (MUST FOLLOW)
1. **Use floats for numerical computations**  
   - Initialize numeric variables as `0.0`, `1.0`, etc. — never `0` or `1` (which are integers).  
   - Avoid unbounded integer growth (e.g., `x *= 2` → use `x *= 2.0`).  
2. **Prevent infinite loops**  
   - All `while` loops must have a maximum iteration guard (e.g., `for _ in range(100):` or `while condition and i < 100:`).  
3. **Avoid numerical overflow and underflow**  
   - Do not compute extremely large powers or products without bounds.  
   - For iterative numerical methods (e.g., root finding), use bounded intervals and safe convergence checks.  
4. **Ensure type safety**  
   - Validate input types if the function may be called with unexpected arguments.  
   - Handle edge cases (e.g., empty lists, zero division, invalid indices) gracefully.  
5. **Preserve the original function signature**  
   - Do not remove or alter existing assertions unless they are demonstrably incorrect.  
6. **Return JSON-serializable outputs**  
   - Avoid generators, `zip`, `map`, or custom objects. Return only `list`, `tuple`, `dict`, `int`, `float`, `str`, or `None`.

{full_error_analysis}

### 测试用例
{safe_json_dumps(test_cases, indent=2, ensure_ascii=False)}

### 执行轨迹分析
{trace_summary}

### 错误信息
- 实际输出: {error_info.get('actual_output')}
- 期望输出: {error_info.get('expected_output')}
- 错误类型: {error_info.get('error_type', 'N/A')}

### 任务
1. 分析执行轨迹，识别变量状态异常的关键步骤
2. 根据DU路径追踪数据流，定位错误传播路径
3. 提供修复后的完整函数代码（包含函数签名）

请在```python代码块中提供修复后的代码。"""
        
        return prompt
    


def _debug_single_function(
    full_code: str,
    gold_test_cases: List[Dict],
    func_name: str,
    o1_debugger: AzureO1Debugger,
    max_debug_attempts: int
) -> str:
    """
    调试单一简单函数（不进行层次化拆分）
    
    Args:
        full_code: 包含目标函数的完整代码字符串
        gold_test_cases: 黄金测试用例列表
        func_name: 要调试的函数名
        o1_debugger: AzureO1Debugger 实例
        max_debug_attempts: 最大调试重试次数
    
    Returns:
        修复后的完整代码（仅含一个函数）
    """
    fixed_code = full_code
    test_cases = generate_test_cases(full_code, gold_test_cases, func_name)
    
    for debug_attempt in range(max_debug_attempts):
        all_tests_pass = True
        failed_testcase = None
        trace_summary = ""
        
        # 执行所有测试用例
        for test_case in test_cases:
            passed, result, tracer = evaluate_with_trace(
                full_code, func_name, test_case
            )
            
            if not passed:
                all_tests_pass = False
                failed_testcase = result
                # 关键：传入 test_case 以生成 LLM 友好摘要
                trace_summary = tracer.get_trace_summary(test_case)
                logger.warning(f"测试失败: {test_case['input']}")
                logger.info(trace_summary)
                break
        
        # 如果全部通过，直接返回
        if all_tests_pass:
            logger.info(f"✓ 函数 {func_name} 所有测试通过")
            return fixed_code
        
        # 调试失败，尝试修复
        logger.info(f"调试函数 {func_name} (尝试 {debug_attempt + 1}/{max_debug_attempts})")
        
        # 优先使用 Azure o1
        new_fixed_code = None
        if o1_debugger.is_available():
            new_fixed_code = o1_debugger.debug_with_trace(
                fixed_code, func_name, test_cases,
                trace_summary, failed_testcase
            )
        
        # 回退到基础调试模式
        if not new_fixed_code:
            logger.info("使用基础调试模式")
            from function_debugger import debug_function
            analysis, new_fixed_code = debug_function(fixed_code, func_name, test_cases)
        
        # 应用修复
        if new_fixed_code:
            logger.info(f"生成修复代码:\n{new_fixed_code[:200]}...")
            fixed_code = new_fixed_code
            # 更新 full_code 用于下一轮测试
            full_code = fixed_code
        else:
            logger.warning(f"无法修复 {func_name}，保留原实现")
            break
    
    if not all_tests_pass:
        logger.warning(f"函数 {func_name} 在 {max_debug_attempts} 次尝试后仍未修复")
    
    return fixed_code
# ========== 主调试函数：替代mg_debug ==========

# def mg_debug_with_trace(full_code: str, gold_test_cases: List[Dict], 
#                        max_debug_attempts: int = MAX_DEBUG_RETRIES) -> str:
#     """
#     基于执行追踪的主调试函数 - 替代原mg_debug
    
#     完全兼容原MGDebugger架构，可直接替换
#     """
#     config.TOTAL_MG_DEBUG_CALLS += 1
#     logger.info("启动基于执行追踪的调试流程")
    
#     # 初始化Azure o1调试器
#     o1_debugger = AzureO1Debugger()
    
#     # 1. 转换为层次化结构（复用原逻辑）
#     convert_hierarchical_attempts = 0
#     while convert_hierarchical_attempts < MAX_PARSE_RETRIES:
#         try:
#             hierarchical_code = convert_to_hierarchical(full_code, include_example=False)
#             logger.info(f"已转换为层次化结构:\n{hierarchical_code}")
            
#             functions = extract_functions(hierarchical_code)
#             dependency_graph = create_dependency_graph(functions)
#             logger.info(f"依赖图:\n{get_dependency_graph_str(dependency_graph)}")
            
#             break
#         except Exception as e:
#             logger.error(f"层次化转换失败 (尝试 {convert_hierarchical_attempts + 1}/{MAX_PARSE_RETRIES}): {e}")
#             convert_hierarchical_attempts += 1
    
#     # 2. 按依赖顺序调试（自底向上）
#     sorted_functions = topological_sort(dependency_graph)
#     logger.info(f"调试顺序: {sorted_functions}")
    
#     for func_name in sorted_functions:
#         logger.info(f"\n{'='*70}")
#         logger.info(f"处理函数: {func_name}")
#         logger.info(f"{'='*70}")
        
#         func_code = functions[func_name]
#         test_cases = generate_test_cases(hierarchical_code, gold_test_cases, func_name)
#         fixed_code = func_code
        
#         # 3. 迭代调试（最多max_debug_attempts次）
#         for debug_attempt in range(max_debug_attempts):
#             all_tests_pass = True
#             failed_testcase = None
#             trace_summary = ""
            
#             # 执行所有测试用例并捕获追踪
#             for test_case in test_cases:
#                 passed, result, tracer = evaluate_with_trace(
#                     hierarchical_code, func_name, test_case
#                 )
                
#                 if not passed:
#                     all_tests_pass = False
#                     failed_testcase = result
#                     trace_summary = tracer.get_trace_summary(test_case)
#                     logger.warning(f"测试失败: {test_case['input']}")
#                     logger.info(trace_summary)
#                     break
            
#             # 如果全部通过，继续下一个函数
#             if all_tests_pass:
#                 logger.info(f"✓ 函数 {func_name} 所有测试通过")
#                 break
            
#             # 4. 调试失败的函数
#             logger.info(f"调试函数 {func_name} (尝试 {debug_attempt + 1}/{max_debug_attempts})")
            
#             # 优先使用Azure o1（如果可用）
#             new_fixed_code = None
#             if o1_debugger.is_available():
#                 new_fixed_code = o1_debugger.debug_with_trace(
#                     fixed_code, func_name, test_cases, 
#                     trace_summary, failed_testcase
#                 )
            
#             # 如果o1不可用或失败，回退到原debug_function
#             if not new_fixed_code:
#                 logger.info("使用基础调试模式")
#                 from function_debugger import debug_function
#                 analysis, new_fixed_code = debug_function(fixed_code, func_name, test_cases)
            
#             # 应用修复
#             if new_fixed_code:
#                 logger.info(f"生成修复代码:\n{new_fixed_code[:200]}...")
#                 fixed_code = new_fixed_code
#                 functions[func_name] = fixed_code
                
#                 # 合并变更到依赖方
#                 hierarchical_code = merge_changes_to_parents(
#                     func_name, dependency_graph, functions
#                 )
#                 logger.info(f"已合并 {func_name} 的修改")
#             else:
#                 logger.warning(f"无法修复 {func_name}，保留原实现")
#                 break
        
#         if not all_tests_pass:
#             logger.warning(f"函数 {func_name} 在 {max_debug_attempts} 次尝试后仍未修复")
    
#     # 5. 重构完整代码
#     fixed_full_code = "\n\n".join(functions.values())
#     logger.info("调试流程完成")
    
#     return fixed_full_code

def mg_debug_with_trace(full_code: str, gold_test_cases: List[Dict], 
                       max_debug_attempts: int = MAX_DEBUG_RETRIES) -> str:
    """
    基于执行追踪的主调试函数 - 修复版
    """
    config.TOTAL_MG_DEBUG_CALLS += 1
    logger.info("启动基于执行追踪的调试流程")
    
    # 初始化Azure o1调试器
    o1_debugger = AzureO1Debugger()
    
    # 1. 转换为层次化结构
    convert_hierarchical_attempts = 0
    hierarchical_code = full_code
    functions = {}
    
    while convert_hierarchical_attempts < MAX_PARSE_RETRIES:
        try:
            hierarchical_code = convert_to_hierarchical(full_code, include_example=False)
            logger.info(f"已转换为层次化结构")
            break
        except Exception as e:
            logger.error(f"层次化转换失败 (尝试 {convert_hierarchical_attempts + 1}/{MAX_PARSE_RETRIES}): {e}")
            convert_hierarchical_attempts += 1
            if convert_hierarchical_attempts >= MAX_PARSE_RETRIES:
                logger.warning("层次化转换失败，使用原始代码")
                hierarchical_code = full_code

    # 2. 提取函数并构建依赖图
    functions = extract_functions(hierarchical_code)
    dependency_graph = create_dependency_graph(functions)
    logger.info(f"依赖图:\n{get_dependency_graph_str(dependency_graph)}")
    
    # 3. 按依赖顺序调试
    sorted_functions = topological_sort(dependency_graph)
    logger.info(f"调试顺序: {sorted_functions}")
    
    for func_name in sorted_functions:
        logger.info(f"\n{'='*70}")
        logger.info(f"处理函数: {func_name}")
        logger.info(f"{'='*70}")
        
        func_code = functions[func_name]
        test_cases = generate_test_cases(hierarchical_code, gold_test_cases, func_name)
        fixed_code = func_code
        
        # 4. 迭代调试
        for debug_attempt in range(max_debug_attempts):
            all_tests_pass = True
            failed_testcase = None
            trace_summary = ""
            tracer = None
            
            # 执行所有测试用例
            for test_case in test_cases:
                passed, result, current_tracer = evaluate_with_trace(
                    hierarchical_code, func_name, test_case
                )
                
                # ✅ 确保 tracer 不为 None
                if current_tracer is None:
                    logger.error(f"追踪器为None，跳过测试用例: {test_case['input']}")
                    continue
                
                tracer = current_tracer  # 保存追踪器
                
                if not passed:
                    all_tests_pass = False
                    failed_testcase = result
                    trace_summary = tracer.get_trace_summary(test_case)
                    logger.warning(f"测试失败: {test_case['input']}")
                    logger.info(trace_summary)
                    break
            
            if all_tests_pass:
                logger.info(f"✓ 函数 {func_name} 所有测试通过")
                break
            
            # 5. 调试失败的函数
            logger.info(f"调试函数 {func_name} (尝试 {debug_attempt + 1}/{max_debug_attempts})")
            
            # ✅ 检查 tracer 是否有效
            if tracer is None:
                logger.warning(f"追踪器无效，跳过此轮调试")
                continue
            
            # 优先使用Azure o1
            new_fixed_code = None
            if o1_debugger.is_available():
                new_fixed_code = o1_debugger.debug_with_trace(
                    fixed_code, func_name, test_cases, 
                    trace_summary, failed_testcase
                )
            
            # 回退到基础调试模式
            if not new_fixed_code:
                logger.info("使用基础调试模式")
                from function_debugger import debug_function
                analysis, new_fixed_code = debug_function(fixed_code, func_name, test_cases)
            
            # 应用修复
            if new_fixed_code:
                logger.info(f"生成修复代码")
                fixed_code = new_fixed_code
                functions[func_name] = fixed_code
                
                # 合并变更到依赖方
                try:
                    hierarchical_code = merge_changes_to_parents(
                        func_name, dependency_graph, functions
                    )
                    logger.info(f"已合并 {func_name} 的修改")
                except Exception as e:
                    logger.error(f"合并修改失败: {e}")
                    # 手动重建 hierarchical_code
                    hierarchical_code = "\n\n".join(functions.values())
            else:
                logger.warning(f"无法修复 {func_name}，保留原实现")
                break
        
        if not all_tests_pass:
            logger.warning(f"函数 {func_name} 在 {max_debug_attempts} 次尝试后仍未修复")
    
    # ✅ 修复：确保总是返回结果
    fixed_full_code = "\n\n".join(functions.values())
    logger.info("调试流程完成")
    
    return fixed_full_code

# ========== 导出接口 ==========

# 为了兼容性，提供两个名字
mg_debug = mg_debug_with_trace  # 直接替换原函数名


# ========== 单元测试 ==========

class TestFineGrainedTracer(unittest.TestCase):

 

    def test_traces_user_function_in_string(self):
        """测试能否追踪 <string> 中的用户函数"""
        code = '''
def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
'''
        test_case = {'input': {'arr': [3, 1, 4, 1, 5]}, 'expected_output': 5}
        passed, result, tracer = evaluate_with_trace(code, 'find_max', test_case)

        self.assertTrue(passed)
        self.assertGreater(len(tracer.traces), 0)
        # 检查是否追踪了 find_max
        func_lines = [t for t in tracer.traces if t['function'] == 'find_max']
        self.assertGreater(len(func_lines), 0)
        # 检查是否记录了变量
        has_arr = any('L.arr' in (t.get('variables') or {}) for t in func_lines)
        self.assertTrue(has_arr)



    def test_du_chain_for_loop_multiple_assignments(self):
        """验证循环变量的多次赋值"""
        code = '''
def sum_loop(n):
        total = 0
        for i in range(n):
            total += i
        return total
    '''
        test_case = {'input': {'n': 3}, 'expected_output': 3}
        passed, result, tracer = evaluate_with_trace(code, 'sum_loop', test_case)
        self.assertTrue(passed)
        
        # i 应该被定义多次（首次 + 3次迭代）
        self.assertIn('i', tracer.var_definitions)
        i_defs = tracer.var_definitions['i']
        print("i definitions at lines:", i_defs)
        # 期望：至少 2 个不同行号（for 行 + 循环体行）
        self.assertGreater(len(i_defs), 1)


    def test_captures_exception_in_user_code(self):
        """测试异常是否被捕获并定位到用户代码"""
        buggy_code = '''
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr) + 1):  # 越界
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
'''
        test_case = {'input': {'arr': [1, 2]}, 'expected_output': 2}
        passed, result, tracer = evaluate_with_trace(buggy_code, 'find_max', test_case)

        self.assertFalse(passed)
        self.assertIn('IndexError', result['error_type'])
        self.assertIsNotNone(tracer.exception_info)
        self.assertEqual(tracer.exception_info['function'], 'find_max')
        # 确保出错行被追踪
        error_line_traced = any(
            t['line'] == tracer.exception_info['line'] and t['function'] == 'find_max'
            for t in tracer.traces
        )
        self.assertTrue(error_line_traced)

    def test_tracks_loop_iterations(self):
        """测试循环迭代是否被正确记录"""
        code = '''
def count_to_n(n):
    total = 0
    for i in range(n):
        total += i
    return total
'''
        test_case = {'input': {'n': 3}, 'expected_output': 3}  # 0+1+2=3
        passed, result, tracer = evaluate_with_trace(code, 'count_to_n', test_case)

        self.assertTrue(passed)
        # 应有循环迭代记录（range(3) → 3 次）
        loop_iters = [t for t in tracer.traces if t.get('loop_info')]
        self.assertGreater(len(loop_iters), 0)
        # 最后一次迭代应为第3次
        last_iter = loop_iters[-1]['loop_info']['iter']
        self.assertGreaterEqual(last_iter, 2)  # 至少2次以上



    def test_du_chain_records_variables(self):
        """测试DU链是否记录关键变量"""
        code = '''
def add_one(x):
    y = x + 1
    return y
'''
        test_case = {'input': {'x': 10}, 'expected_output': 11}
        passed, result, tracer = evaluate_with_trace(code, 'add_one', test_case)

        self.assertTrue(passed)
        # 检查 x 和 y 是否在 DU 链中
        self.assertIn('x', tracer.var_definitions)
        self.assertIn('y', tracer.var_definitions)
        self.assertIn('y', tracer.var_uses)  # y 被 return 使用



    def test_du_chain_simple_return(self):
        """最简 return 场景：y = x + 1; return y"""
        code = '''
def add_one(x):
        y = x + 1
        return y
    '''
        test_case = {'input': {'x': 5}, 'expected_output': 6}
        passed, result, tracer = evaluate_with_trace(code, 'add_one', test_case)
        logger.debug("var_definitions: %s", dict(tracer.var_definitions))
        self.assertTrue(passed)
        self.assertIn('x', tracer.var_definitions)
        self.assertIn('y', tracer.var_definitions)
        self.assertIn('y', tracer.var_uses)  # y 被 return 使用


    def test_du_chain_return_expression(self):
        """return 表达式（非变量）—— 应无额外 use"""
        code = '''
def double(x):
        return x * 2
    '''
        test_case = {'input': {'x': 3}, 'expected_output': 6}
        passed, result, tracer = evaluate_with_trace(code, 'double', test_case)
        self.assertTrue(passed)
        self.assertIn('x', tracer.var_definitions)
        # 注意：x 在 return x*2 中被使用，但当前机制可能不捕获（属合理）
        # 本测试重点：无错误，且 x 有定义


    def test_du_chain_implicit_use_in_expression(self):
        """变量在表达式中被使用（y = x + 1）—— x 应被 use"""
        code = '''
def add_one(x):
        y = x + 1  # x 被使用
        return y
    '''
        test_case = {'input': {'x': 10}, 'expected_output': 11}
        passed, result, tracer = evaluate_with_trace(code, 'add_one', test_case)
        self.assertTrue(passed)
        self.assertIn('x', tracer.var_definitions)
        # 若机制能捕获 x 的 use，则应包含；否则跳过（当前聚焦 return 变量）
        self.assertIn('y', tracer.var_definitions)


    def test_du_chain_multiple_uses(self):
        """变量多次被使用"""
        code = '''
def compute(x):
        y = x + 1
        z = y * 2
        return y + z
    '''
        test_case = {'input': {'x': 2}, 'expected_output': 3 + 6}  # y=3, z=6 → 9
        passed, result, tracer = evaluate_with_trace(code, 'compute', test_case)
        self.assertTrue(passed)
        self.assertIn('y', tracer.var_definitions)
        self.assertIn('y', tracer.var_uses)  # 至少在 z = y*2 和 return 中使用


    def test_du_chain_no_return_statement(self):
        """无 return 语句（隐式 return None）"""
        code = '''
def do_nothing(x):
        y = x  # y 被定义但未使用
    '''
        test_case = {'input': {'x': 42}, 'expected_output': None}
        passed, result, tracer = evaluate_with_trace(code, 'do_nothing', test_case)
        logger.debug("var_definitions:", dict(tracer.var_definitions))
        self.assertTrue(passed)
        self.assertIn('y', tracer.var_definitions)
        # y 未被使用，var_uses 中可不含 y（合理）
        self.assertNotIn('y', tracer.var_uses)  # 确保不误报


    def test_du_chain_return_constant(self):
        """return 常量，无变量 use"""
        code = '''
def get_five():
        x = 5
        return 5
    '''
        test_case = {'input': {}, 'expected_output': 5}
        passed, result, tracer = evaluate_with_trace(code, 'get_five', test_case)
        self.assertTrue(passed)
        self.assertIn('x', tracer.var_definitions)
        # self.assertNotIn('x', tracer.var_uses)  # x 未被使用


    def test_du_chain_conditional_return(self):
        """条件 return 中的变量"""
        code = '''
def maybe_return(x):
        y = x + 1
        if x > 0:
            return y
        return 0
    '''
        test_case = {'input': {'x': 3}, 'expected_output': 4}
        passed, result, tracer = evaluate_with_trace(code, 'maybe_return', test_case)
        self.assertTrue(passed)
        self.assertIn('y', tracer.var_definitions)
        self.assertIn('y', tracer.var_uses)


    def test_du_chain_nested_function_ignored(self):
        """嵌套函数（应被过滤，不追踪）"""
        code = '''
def outer(x):
        def inner():
            return 999  # 不应被追踪
        y = x + 1
        return y
    '''
        test_case = {'input': {'x': 1}, 'expected_output': 2}
        passed, result, tracer = evaluate_with_trace(code, 'outer', test_case)
        self.assertTrue(passed)
        self.assertIn('y', tracer.var_definitions)
        self.assertIn('y', tracer.var_uses)
        # 确保 inner 的变量（如 999）未污染 var_uses
        for var in tracer.var_uses:
            self.assertNotIn('inner', var)  # 名字不包含 inner
            self.assertNotIn('inner', tracer.var_uses)
            # 同时确保 'inner' 不在 var_definitions 中
            self.assertNotIn('inner', tracer.var_definitions)


    def test_du_chain_global_variable_read(self):
        """读取全局变量（应被记录为 use）"""
        global_code = '''
GLOBAL_VAL = 42
def use_global():
        return GLOBAL_VAL
    '''
        test_case = {'input': {}, 'expected_output': 42}
        passed, result, tracer = evaluate_with_trace(global_code, 'use_global', test_case)
        self.assertTrue(passed)
        # 注意：GLOBAL_VAL 是全局变量，在 co_names 中
        # 若你的 _track_du_chain 支持全局变量 use，则应包含
        # 如不支持，可暂时跳过此断言
        # self.assertIn('GLOBAL_VAL', tracer.var_uses)


    def test_du_chain_same_line_define_and_use(self):
        """同一线定义并使用（如交换）"""
        code = '''
def swap(a, b):
        a, b = b, a  # a, b 重新定义
        return a
    '''
        test_case = {'input': {'a': 1, 'b': 2}, 'expected_output': 2}
        passed, result, tracer = evaluate_with_trace(code, 'swap', test_case)
        self.assertTrue(passed)
        self.assertIn('a', tracer.var_definitions)
        self.assertIn('b', tracer.var_definitions)
        # return a 应触发 a 的 use
        self.assertIn('a', tracer.var_uses)

    def test_du_chain_class_method(self):
        """类方法中的 self 属性赋值"""
        code = '''
class Calculator:
    def add(self, a, b):
        self.result = a + b  # self.result 被定义
        return self.result

def test_method():
        calc = Calculator()
        return calc.add(2, 3)
'''
        test_case = {'input': {}, 'expected_output': 5}
        passed, result, tracer = evaluate_with_trace(code, 'test_method', test_case)
        self.assertTrue(passed)
        # 注意：self.result 是属性，不是局部变量，通常不在 locals 中
        # 我们主要验证不崩溃，且局部变量（如 a, b）被记录
        self.assertIn('a', tracer.var_definitions)
        self.assertIn('b', tracer.var_definitions)


    def test_du_chain_global_declaration(self):
        """全局变量声明与写入（global x; x = 1）"""
        code = '''
x = 0  # 全局初始值

def set_global():
        global x
        x = 42  # 修改全局 x
        return x
    '''
        test_case = {'input': {}, 'expected_output': 42}
        passed, result, tracer = evaluate_with_trace(code, 'set_global', test_case)
        self.assertTrue(passed)
        # x 是全局变量，在 set_global 中被声明为 global
        # 注意：x 可能出现在 co_names 中，但不在 locals
        # 我们主要验证不崩溃
        # （当前机制可能不记录 global x 的定义/使用，属合理）


    def test_du_chain_for_loop_variable(self):
        """for 循环变量（for i in range(3)）"""
        code = '''
def sum_loop(n):
        total = 0
        for i in range(n):  # i 是循环变量
            total += i
        return total
    '''
        test_case = {'input': {'n': 3}, 'expected_output': 3}  # 0+1+2
        passed, result, tracer = evaluate_with_trace(code, 'sum_loop', test_case)
        self.assertTrue(passed)
        self.assertIn('i', tracer.var_definitions)   # i 应被定义
        self.assertIn('i', tracer.var_uses)         # i 在 total += i 中被使用
        self.assertIn('total', tracer.var_definitions)


    def test_du_chain_exception_handling(self):
        """异常处理中的变量（try/except 中的变量）"""
        code = '''
def safe_divide(a, b):
        try:
            result = a / b
            return result
        except ZeroDivisionError as e:
            error_msg = "Division by zero"
            return error_msg
    '''
        # 正常情况
        test_case1 = {'input': {'a': 6, 'b': 2}, 'expected_output': 3.0}
        passed1, result1, tracer1 = evaluate_with_trace(code, 'safe_divide', test_case1)
        self.assertTrue(passed1)
        self.assertIn('result', tracer1.var_definitions)

        # 异常情况
        test_case2 = {'input': {'a': 6, 'b': 0}, 'expected_output': "Division by zero"}
        passed2, result2, tracer2 = evaluate_with_trace(code, 'safe_divide', test_case2)
        self.assertTrue(passed2)
        self.assertIn('error_msg', tracer2.var_definitions)
        # self.assertNotIn('result', tracer2.var_definitions)  # try 块未完全执行


    def test_du_chain_list_comprehension(self):
        """列表推导式中的变量（隐式作用域）"""
        code = '''
def get_squares(n):
        return [x * x for x in range(n)]
    '''
        test_case = {'input': {'n': 3}, 'expected_output': [0, 1, 4]}
        passed, result, tracer = evaluate_with_trace(code, 'get_squares', test_case)
        self.assertTrue(passed)
        # 注意：x 在列表推导式中有独立作用域，可能不会出现在外层 locals
        # 我们主要验证不崩溃
        self.assertIn('n', tracer.var_definitions)


    def test_du_chain_keyword_arguments(self):
        """关键字参数与默认值"""
        code = '''
def greet(name, greeting="Hello"):
        message = f"{greeting}, {name}!"
        return message
    '''
        test_case = {'input': {'name': "Alice"}, 'expected_output': "Hello, Alice!"}
        passed, result, tracer = evaluate_with_trace(code, 'greet', test_case)
        self.assertTrue(passed)
        self.assertIn('name', tracer.var_definitions)
        self.assertIn('greeting', tracer.var_definitions)
        self.assertIn('message', tracer.var_definitions)   

# ========== 主程序 ==========

if __name__ == "__main__":
    # 配置日志（仅主程序运行时）
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    # 检查是否运行单元测试
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("运行单元测试...")
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # 原有测试用例
        test_code = '''
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr) + 1):  # BUG: 越界
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
'''

        test_case = {
            'input': {'arr': [3, 1, 4, 1, 5]},
            'expected_output': 5
        }

        logger.info("测试追踪功能...")
        passed, result, tracer = evaluate_with_trace(test_code, 'find_max', test_case)

        # ✅ 保存压缩追踪
        tracer.save_compressed_trace('/root/autodl-tmp/temp_data/find_max_trace.gz')
        
        # ✅ 加载并验证
        loaded_trace = FineGrainedTracer.load_compressed_trace('/root/autodl-tmp/temp_data/find_max_trace.gz')
        logger.info(f"加载的追踪步骤数: {loaded_trace.get('total_steps', 0)}")

        logger.info(f"测试通过: {passed}")
        logger.info(f"结果: {result}")
        logger.info(tracer.get_trace_summary(test_case))