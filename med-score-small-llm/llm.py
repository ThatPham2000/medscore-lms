from typing import List, Dict

import backoff
import requests


class LLM(object):
    def __init__(self, model_name: str, seed: int = 42, temperature: float = 0.0, top_p: float = 1.0,
                 max_tokens: int = 256):
        self.model_name = model_name
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_time=60
    )
    async def batch_response(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        raise NotImplementedError

    def normalize_llm_response(self, completions) -> List[str]:
        raise NotImplementedError
