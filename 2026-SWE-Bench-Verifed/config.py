# Configuration and hyperparameters
import openai


GITHUB_TOKEN = "g***"  # 替换为你的 GitHub 访问令牌


MODEL="kimi-k2.5"
LLM_BASE_URL="https://api.kimi.com/coding"
LLM_API_KEY = "sk-kimi-***"





# Hyperparameters
MAX_VLLM_RETRIES = 10  # maximum number of retries for the VLLM call
MAX_PARSE_RETRIES = 3  # we sometimes fail to parse the code/test cases from the response, so we retry
MAX_DEBUG_RETRIES = 3  # 3 seems to be slightly better than 1, but the difference is not significant
RETRY_DELAY = 0  # seconds
REPEAT_CONVERT_HIERARCHICAL_NUM = 1  # seems unimportant
REPEAT_TEST_CASE_GENERATION_NUM = 1 # 1 seems to be better than 3  REPEAT_TEST_CASE_GENERATION_NUM -》来进行重复生成和投票，这是保证测试用例质量
MAX_OUTER_RETRY = 10  # maximum number of retries for the entire debugging process
CONTINUOUS_RETRY = False  # whether to retry from the last fixed code, else retry from the original buggy code. it seems better to set this to False
TEMPERATURE = 0.8  # 0.8 better than 1.0 better than 0.2
MINP = 0.05

# Global statistics tracking
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_MG_DEBUG_CALLS = 0
TOTAL_DEBUG_FUNCTION_CALLS = 0
TOTAL_GENERATE_TEST_CASES_CALLS = 0
TOTAL_CONVERT_HIERARCHICAL_CALLS = 0 


# ========== 追踪器配置 ==========
TRACER_MAX_DEPTH = 1000          # 最大追踪步数
TRACER_LOOP_SAMPLING = 5         # 循环采样频率（每 N 次记录 1 次）
TRACER_MAX_VAR_SIZE = 100        # 变量 repr 最大长度
TRACER_ENABLE_DU_CHAIN = True    # 是否追踪定义 - 使用链

# ========== 知识库配置 ==========
KB_SIMILARITY_THRESHOLD = 0.6    # 卡片检索相似度阈值
KB_MAX_CANDIDATES = 10           # 模糊匹配返回的最大候选数
KB_AUTO_MERGE_SIMILARITY = 0.95  # 自动合并相似卡片的阈值

# ========== Solid Card 配置 ==========
# 设置为 False 可完全禁用 Solid Card，使用纯 Tracer 模式（参考旧版 37/38 成功率）
ENABLE_SOLID_CARD = True         # 是否启用 Solid Card 知识库
ENABLE_SOLID_CARD_PROMPT = True  # 是否将 Solid Card 指导加入 Prompt
SOLID_CARD_ONLY_PERFECT_MATCH = False  # 仅使用完美匹配（相似度=1.0）的卡片

# ========== MemGovern 风格 Agentic Experience Search 配置 ==========
ENABLE_AGENTIC_SEARCH = True     # 是否启用 MemGovern 风格的 Agentic Search
AGENTIC_SEARCH_MAX_ROUNDS = 3    # Agentic Search 最大轮数
AGENTIC_SEARCH_THRESHOLD = 0.6   # 相关性阈值

# ========== 代码转换配置 ==========
MAX_FUNC_LINES_FOR_DECOMPOSE = 9999   # [修改] 禁用层次化拆分，设置极高阈值
MIN_SUBFUNCTIONS_FOR_HIERARCHICAL = 2  # 至少生成 N 个子函数才保留层次化
HALSTEAD_WEIGHT = 0.3            # 复杂度评分中 Halstead 的权重

# ========== 性能监控 ==========
SLOW_CALL_THRESHOLD = 5.0        # 慢调用告警阈值（秒）
ENABLE_PERF_LOGGING = True       # 是否记录性能指标


# 知识库存储路径，建议指定一个 json 文件
KB_PATH = "kb.json"


# 安全白名单：允许的内置函数
SAFE_BUILTINS = {
    # 类型
    'bool', 'int', 'float', 'str', 'list', 'tuple', 'dict', 'set', 'frozenset',
    'object', 'type', 'isinstance', 'issubclass', 'callable',
    # 数学
    'abs', 'round', 'min', 'max', 'sum', 'pow', 'divmod',
    # 迭代
    'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
    'iter', 'next',  # ✅ 添加这两个
    # 函数式
    'any', 'all',
    # 字符串
    'ord', 'chr', 'format', 'repr', 'ascii',
    # 对象
    'hasattr', 'getattr', 'setattr', 'delattr', 'vars', 'dir',
    # 类相关
    'staticmethod', 'classmethod', 'property', 'super',
    # 其他
    'print', 'input', 'open', 'help', 'id', 'hash', 'hex', 'oct', 'bin',
    # 异常
    'Exception', 'ValueError', 'TypeError', 'NameError', 'IndexError',
    'KeyError', 'AssertionError', 'StopIteration', 'RuntimeError',
    'AttributeError', 'ZeroDivisionError', 'OverflowError',
}