from typing import List, Dict, Any

from prompts import INTERNAL_KNOWLEDGE_PROMPT
from verifier import Verifier


class VerifierInternal(Verifier):
    """Verify claims against internal model knowledge"""

    def add_evidence_to_verification_input(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        verification_input = []
        for d in decompositions:
            d["evidence"] = None
            verification_input.append(d)
        return verification_input

    def prepare_messages(self, verification_input: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        messages = []
        for d in verification_input:
            formatted_input = f"""Using your own knowledge, answer the question.\n\nInput: {d['claim']} True or False?\n\nOutput:"""
            messages.append([
                {"role": "system", "content": INTERNAL_KNOWLEDGE_PROMPT},
                {"role": "user", "content": formatted_input}
            ])
        return messages
