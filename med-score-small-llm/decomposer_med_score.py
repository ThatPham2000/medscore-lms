from decomposer import Decomposer
from llm import LLM
from prompts import MEDSCORE_PROMPT


class DecomposerMedScore(Decomposer):
    def __init__(
            self,
            llm: LLM = None,
    ):
        super().__init__(llm=llm)

    def get_system_prompt(self) -> str:
        return MEDSCORE_PROMPT

    def format_input(self, context: str, sentence: str) -> str:
        return f"Context: {context}\nPlease breakdown the following sentence into independent facts: {sentence}\nFacts:\n"
