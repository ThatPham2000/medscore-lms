import asyncio
import time
from typing import Optional, List, Dict, Any

import nest_asyncio
from tqdm import tqdm

from llm import LLM
from utils import chunker, process_claim, remove_think_tags

nest_asyncio.apply()


class Decomposer(object):
    def __init__(
            self,
            llm: LLM = None,
            random_state: int = 42,
            batch_size: int = 16, #TODO(THAT): edit
    ):
        self.llm = llm
        self.random_state = random_state
        self.batch_size = batch_size

    def do_decompose(self, decomposition_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        system_prompt = self.get_system_prompt()

        messages = []
        for d in decomposition_input:
            formatted_input = self.format_input(d['context'], d['sentence'])
            if system_prompt:
                messages.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_input}
                ])
            else:
                messages.append([
                    {"role": "user", "content": formatted_input}
                ])

        # Async calls for batch_size items
        all_completions = []
        n_iter = len(messages) // self.batch_size
        for batch in tqdm(chunker(messages, self.batch_size), desc="Decomposer process", total=n_iter, ncols=0):
            completions = asyncio.run(self.llm.batch_response(batch))
            all_completions.extend(completions)
            time.sleep(3) # TODO(THAT): edit

        # Format claims
        decompositions = self.format_completions(decomposition_input, self.llm.normalize_llm_response(all_completions))
        return decompositions

    def format_completions(self, decomp_input: List[Dict[str, Any]], completions: List[str]) -> List[Dict[str, Any]]:
        decompositions = []
        for d_input, completion in zip(decomp_input, completions):
            completion = remove_think_tags(completion)
            claim_list = completion.split("\n")
            claim_list = process_claim(claim_list)
            for idx, claim in enumerate(claim_list):
                decomp = {k: v for k, v in d_input.items() if k != "context"}
                decomp["claim"] = claim
                decomp["claim_id"] = idx
                decomp["model_response"] = completion
                decompositions.append(decomp)
            if not claim_list:
                decomp = {k: v for k, v in d_input.items() if k != "context"}
                decomp["claim"] = None
                decomp["model_response"] = completion
                decompositions.append(decomp)
        return decompositions

    def get_system_prompt(self) -> Optional[str]:
        return None

    def format_input(self, context: str, sentence: str) -> Optional[str]:
        return None
