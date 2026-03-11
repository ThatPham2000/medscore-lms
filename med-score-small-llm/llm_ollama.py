import asyncio
from typing import List, Dict

import backoff
import nest_asyncio
import ollama
import requests

from llm import LLM

nest_asyncio.apply()


class LLMOllama(LLM):
    def __init__(self, model_name: str, ollama_async_client: ollama.AsyncClient = None):
        super().__init__(model_name)
        self.ollama_async_client = ollama_async_client

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_time=60
    )
    async def batch_response(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        async_responses = [
            self.ollama_async_client.chat(
                model=self.model_name,
                messages=x,
                # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
                options={
                    "seed": self.seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                }
            ) if self.ollama_async_client
            else ollama.AsyncClient().chat(
                model=self.model_name,
                messages=x,
                options={
                    "seed": self.seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                }
            )
            for x in batch
        ]
        return await asyncio.gather(*async_responses)

    def normalize_llm_response(self, completions) -> List[str]:
        return [c.message.content for c in completions]
