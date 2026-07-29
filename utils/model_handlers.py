import re

class BaseModelHandler:
    def prepare_tokenizer(self, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def extract_sql(self, raw_output: str) -> str:
        raise NotImplementedError


class QwenHandler(BaseModelHandler):
    def extract_sql(self, raw_output: str) -> str:
        # Strip Qwen thinking tags
        cleaned = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
        
        # Extract SQL from markdown block
        sql_match = re.search(r'```[a-zA-Z]*\s*\n(.*?)\n```', cleaned, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        # Handle unclosed markdown blocks
        sql_match_unclosed = re.search(r'```[a-zA-Z]*\s*\n(.*)', cleaned, re.DOTALL)
        if sql_match_unclosed:
            return sql_match_unclosed.group(1).strip()
            
        return cleaned


class GemmaHandler(BaseModelHandler):
    def prepare_tokenizer(self, tokenizer):
        # Gemma 4 uses native <pad> tokens, no need to override with EOS
        return tokenizer

    def extract_sql(self, raw_output: str) -> str:
        # Strip Gemma native thought channels if they leak into text, or custom <think> tags
        cleaned = re.sub(r'<\|channel\|>thought\n.*?<\|channel\|>', '', raw_output, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        
        sql_match = re.search(r'```[a-zA-Z]*\s*\n(.*?)\n```', cleaned, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
            
        sql_match_unclosed = re.search(r'```[a-zA-Z]*\s*\n(.*)', cleaned, re.DOTALL)
        if sql_match_unclosed:
            return sql_match_unclosed.group(1).strip()
            
        return cleaned


# Factory to fetch the correct handler at runtime
def get_model_handler(model_name: str) -> BaseModelHandler:
    if "gemma" in model_name.lower():
        return GemmaHandler()
    return QwenHandler()
