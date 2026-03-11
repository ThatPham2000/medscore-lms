from typing import Dict, List, Any

from llm import LLM
from verifier import Verifier


class VerifierProvidedEvidence(Verifier):
    def __init__(
            self,
            id_to_evidence: Dict[str, str],
            llm: LLM = None,
    ):
        super().__init__(llm)
        self.id_to_evidence = id_to_evidence

    """Verify claims against a pre-provided `evidence` key"""

    def add_evidence_to_verification_input(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        verification_input = []
        for d in decompositions:
            d["evidence"] = self.id_to_evidence[d['id']]
            verification_input.append(d)
        return verification_input

    def prepare_messages(self, verification_input: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        messages = []
        for d in verification_input:
            formatted_input = f"""Answer the question based on the given context.\n\n{d['evidence']}\n\nInput: {d['claim']} True or False?\nOutput:"""
            messages.append([
                {"role": "user", "content": formatted_input}
            ])
        return messages
