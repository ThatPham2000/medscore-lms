from decomposer import Decomposer
from llm import LLM
from prompts import FACTSCORE_PROMPT


class DecomposerFactScore(Decomposer):
    def __init__(
            self,
            llm: LLM = None,
    ):
        super().__init__(llm=llm)

        # Override llm to match settings from FActScore
        self.llm.temperature = 0.7
        self.llm.top_p = 1.0
        self.llm.max_tokens = 2048

    def get_system_prompt(self) -> str:
        return FACTSCORE_PROMPT

    def format_input(self, context: str, sentence: str) -> str:
        return f"Please breakdown the following sentence into independent facts: {sentence}"
