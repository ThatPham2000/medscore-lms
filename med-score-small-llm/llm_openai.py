import asyncio
from functools import partial
from typing import List, Dict

import backoff
import nest_asyncio
import requests
from openai import AsyncOpenAI

from llm import LLM

nest_asyncio.apply()


class LLMOpenAI(LLM):
    def __init__(self, model_name: str, server_path: str, api_key: str):
        super().__init__(model_name)
        self.client = AsyncOpenAI(base_url=server_path,api_key=api_key)

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_time=60
    )
    async def batch_response(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        # https://platform.openai.com/docs/api-reference/chat/create
        agent = partial(
            self.client.chat.completions.create,
            model=self.model_name,
            seed=self.seed,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        async_responses = [
            agent(messages=x)
            for x in batch
        ]
        return await asyncio.gather(*async_responses)

    def normalize_llm_response(self, completions) -> List[str]:
        return [c.choices[0].message.content for c in completions]
