import time
from openai import OpenAI
from loguru import logger
try:
    from config import MODEL, LLM_BASE_URL, LLM_API_KEY, MAX_VLLM_RETRIES, RETRY_DELAY, TEMPERATURE, MINP
    import config
except ImportError:
    from src.config import MODEL, LLM_BASE_URL, LLM_API_KEY, MAX_VLLM_RETRIES, RETRY_DELAY, TEMPERATURE, MINP
    import src.config as config

# 使用 config.py 中配置的 API 设置

# OpenAI
# client = OpenAI(
#     base_url=LLM_BASE_URL,
#     api_key=LLM_API_KEY,
# )

# anthropic
import anthropic

client = anthropic.Anthropic(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)
#AzureOpenAI
# from openai import AzureOpenAI  # ← 注意是 AzureOpenAI！

# client = AzureOpenAI(
#     azure_endpoint=LLM_BASE_URL,      # 参数名是 azure_endpoint，不是 base_url
#     api_key=LLM_API_KEY,
#     api_version="2025-01-01-preview"  # Azure 强制要求指定 API 版本
# )

#Gemini
#http://cn-bj.api.sutefangzhou.com
# client = OpenAI(
#     api_key="sk-hwqkhtT6aRN2nYo3AnC9Vhl3mheWcfMSENhayqvzx307Tum8",  # <-- 替换成你的 Google API Key
#     base_url="http://cn-bj.api.sutefangzhou.com/v1" # <-- 这是关键！
# )

# client = OpenAI(
#     api_key=LLM_API_KEY,  # <-- 替换成你的 Google API Key
#     base_url=LLM_BASE_URL # <-- 这是关键！
# )



def get_completion_with_retry(messages, model=MODEL, max_retries=MAX_VLLM_RETRIES):
    """Get completion from LLM with retry mechanism"""
    # --- 新增：自动兼容字符串输入 ---
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
        
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting LLM call (attempt {attempt + 1}/{max_retries})")
            logger.info(f"Input messages: {messages[-1]['content']}")
        #     chat_completion = client.chat.completions.create(
        #         messages=messages,
        #         model=model,
        #         temperature=TEMPERATURE,
        #         extra_body={"min_p": MINP}
        #     )
        #    #AzureOpenAI
        #     chat_completion = client.chat.completions.create(
        #         messages=messages,
        #         model=model
    
        #     )
        #     response = chat_completion.choices[0].message.content
        #     logger.info(f"LLM response: {response}")
        #     logger.info("LLM call received")
        #     config.TOTAL_PROMPT_TOKENS += chat_completion.usage.prompt_tokens
        #     config.TOTAL_COMPLETION_TOKENS += chat_completion.usage.completion_tokens
        #     return response

            # Anthropic Messages API 调用
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=messages,
                # Anthropic 支持 temperature，但参数位置不同
                temperature=TEMPERATURE if 'TEMPERATURE' in dir(config) else 0.7
            )
            
            # 提取文本内容（Anthropic 格式）
            response_text = response.content[0].text
            logger.info(f"LLM response: {response_text[:100]}...")
            logger.info("LLM call received")
            
            # 更新 Token 计数（Anthropic 使用 input/output_tokens）
            if response.usage:
                config.TOTAL_PROMPT_TOKENS += response.usage.input_tokens
                config.TOTAL_COMPLETION_TOKENS += response.usage.output_tokens
            
            return response_text  # 返回字符串而非对象
            
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            error_str = str(e).lower()
            is_rate_limit = 'rate_limit' in error_str or '429' in error_str or 'overloaded' in error_str
            if attempt < max_retries - 1:
                if is_rate_limit:
                    # Exponential backoff for rate limit: 30s, 60s, 120s, 240s
                    backoff = min(30 * (2 ** attempt), 300)
                    logger.info(f"Rate limit detected. Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                else:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Giving up.")
                raise 



# ===== 主函数 =====
def main():
    # 构造对话消息（符合 OpenAI 格式）
    messages = [
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请用一句话解释量子计算。"}
    ]

    try:
        response = get_completion_with_retry(messages)
        print("\n✅ 最终回复:")
        print(response)

        print(f"\n📊 总共使用 Tokens - Prompt: {config.TOTAL_PROMPT_TOKENS}, Completion: {config.TOTAL_COMPLETION_TOKENS}")
    except Exception as e:
        print(f"\n❌ 调用失败: {e}")
        exit(1)

# ===== 执行入口 =====
if __name__ == "__main__":
    main()
