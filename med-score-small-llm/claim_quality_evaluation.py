import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional

import nest_asyncio
from tqdm import tqdm

from llm import LLM
from utils import chunker, remove_think_tags

nest_asyncio.apply()


class ClaimQualityEvaluation(object):
    def __init__(
            self,
            normalized_llm: LLM = None,
            llm: LLM = None,
            is_only_classification: bool = False,
            random_state: int = 42,
            batch_size: int = 16, #TODO(THAT): update it
    ):
        self.llm = llm
        self.is_only_classification = is_only_classification
        self.llm.max_tokens = 32000
        self.random_state = random_state
        self.batch_size = batch_size
        self.normalized_llm = normalized_llm
        self.normalized_llm.max_tokens = 10240

    def do_claim_quality_evaluation(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Complete pipeline for claim quality evaluation.
        
        Pipeline:
        1. Decompose: Extract atomic facts (claims) - already done in input (decompositions param)
        2. Classify: Label each claim according to 7 MedScore taxonomy
        3. Normalize: Fix invalid claims (Incomplete, Context-dependent, Incorrectly-Structured)
        4. Classify again: Re-classify normalized claims
        5. Filter: Keep only Valid claims
        
        Args:
            decompositions: List of atomic claims (already decomposed) with 'id', 'sentence_id', 'sentence', 'claim_id', 'claim' fields.
        
        Returns:
            List of claim dictionaries with quality evaluation information.
        """
        # Step 2: Initial Classification
        classified_claims = self.classify_claims(decompositions)
        print(f"Classified claims: done, total={len(classified_claims)}")

        if self.is_only_classification:
            return classified_claims
        else:
            # save to file for analysis
            with open("classified_claims.jsonl", "w") as f:
                for claim in classified_claims:
                    f.write(json.dumps(claim) + "\n")

        # get classified claims from file for continuing the pipeline
        # classified_claims = []
        # with open("classified_claims.jsonl", "r") as f:
        #     for line in f:
        #         classified_claims.append(json.loads(line.strip()))

        # Group claims by (id, sentence) for processing
        claims_by_sentence = self._group_claims_by_sentence(classified_claims)

        # Process each sentence group
        final_claims = []
        for (response_id, sentence), claims in claims_by_sentence.items():
            if not claims:
                continue

            # Step 3: Normalize invalid claims
            normalized_claims = self.normalize_invalid_claims(claims)
            print(
                f"Normalized claims for response_id={response_id}, sentence_id={claims[0]['sentence_id']}: done, total={len(normalized_claims)}")

            # save to file for analysis
            with open("normalized_claims.jsonl", "a") as f:
                for claim in normalized_claims:
                    f.write(json.dumps(claim) + "\n")

            # Step 4: Re-classify normalized claims
            reclassified_claims = self.classify_claims(normalized_claims)
            print(
                f"Re-classified claims for response_id={response_id}, sentence_id={claims[0]['sentence_id']}: done, total={len(reclassified_claims)}")

            # save to file for analysis
            with open("reclassified_claims.jsonl", "a") as f:
                for claim in reclassified_claims:
                    f.write(json.dumps(claim) + "\n")

            # Step 5: Filter to get only Valid claims
            # valid_claims = [c for c in reclassified_claims if c.get('claim_quality_type') == 'Valid']

            final_claims.extend(reclassified_claims)
            time.sleep(2) # TODO(THAT): update it

        return final_claims

    def classify_claims(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Group decompositions by id (same response)
        decompositions_by_id = {}
        for d in decompositions:
            response_id = d['id']
            if response_id not in decompositions_by_id:
                decompositions_by_id[response_id] = []
            decompositions_by_id[response_id].append(d)

        messages = []
        for d in decompositions:
            # Get all other claims from the same response (same id) and same sentence, excluding current claim
            response_id = d['id']
            sentence = d.get('sentence', '')
            other_claims = [
                other_d['claim']
                for other_d in decompositions_by_id[response_id] if
                other_d['claim_id'] != d['claim_id'] and other_d.get('claim') is not None and other_d.get('sentence',
                                                                                                          '') == sentence
            ]

            formatted_input = self.format_classification_input(d['context'], d['sentence'], d['claim'], other_claims)
            messages.append([
                {"role": "user", "content": formatted_input}
            ])
            d['other_claims'] = other_claims  # Store other claims for output

        all_completions = []
        n_iter = len(messages) // self.batch_size
        for batch in tqdm(chunker(messages, self.batch_size), desc="Classify claims process", total=n_iter):
            completions = asyncio.run(self.llm.batch_response(batch))
            all_completions.extend(completions)
            time.sleep(3) # TODO(THAT): update it

        claim_quality_output = []
        for decomposition, completion in zip(decompositions, self.llm.normalize_llm_response(all_completions)):
            if completion is not None:
                raw_output = completion.strip()
                claim_quality_type = self.parse_classify_claims_output(raw_output)
                output = {k: v for k, v in decomposition.items()}
                output["raw_claim_quality_response"] = raw_output
                output["claim_quality_type"] = claim_quality_type
                claim_quality_output.append(output)
            else:
                print("No claim quality response", decomposition)
                # save in file jsonl for analysis
                with open("no_claim_quality_response.jsonl", "a") as f:
                    f.write(json.dumps(decomposition) + "\n")

        return claim_quality_output

    def parse_classify_claims_output(self, completion_message: str) -> str:
        valid_types = [
            "Valid",
            "Unverifiable",
            "Incorrectly structured",
            "Context-dependent",
            "Hallucinated",
            "Incomplete",
            "Redundant"
        ]

        completion_message = completion_message.strip()

        # Try to find the classification label after "Classification:"
        classification_patterns = [
            "Classification:",
            "**Classification:**"
        ]

        classification_label = None

        # Search for classification label
        for pattern in classification_patterns:
            if pattern in completion_message:
                # Extract text after the pattern
                parts = completion_message.split(pattern, 1)
                if len(parts) > 1:
                    label_text = parts[1].strip()
                    # Extract the first word or phrase (handle multi-word labels)
                    # Remove any trailing punctuation or newlines
                    label_text = label_text.split('\n')[0].split('.')[0].strip()

                    # Try exact match first (case-insensitive)
                    for valid_type in valid_types:
                        if label_text.lower() == valid_type.lower():
                            classification_label = valid_type
                            break

                    # If no exact match, try partial match
                    if classification_label is None:
                        label_lower = label_text.lower()
                        for valid_type in valid_types:
                            if valid_type.lower() in label_lower or label_lower in valid_type.lower():
                                classification_label = valid_type
                                break

                    if classification_label:
                        break

        # Fallback: search for any valid type in the entire message (case-insensitive)
        if classification_label is None:
            message_lower = completion_message.lower()
            for valid_type in valid_types:
                if valid_type.lower() in message_lower:
                    # Check if it appears near the end (more likely to be the classification)
                    last_occurrence = message_lower.rfind(valid_type.lower())
                    if last_occurrence > len(completion_message) * 0.5:  # In the latter half
                        classification_label = valid_type
                        break

        # Default fallback
        if classification_label is None:
            classification_label = "Parse Failed"

        return classification_label

    def format_classification_input(self, context: str, sentence: str, claim: str, other_claims: List[str]) -> str:
        formatted_other_claims = json.dumps(other_claims)
        prompt = f"""You are a Medical Information Auditor. Your task is to evaluate the quality of a specific "Atomic Claim" extracted from a medical sentence.

INPUT DATA:
- "Context": The full response (used for resolving pronouns/references).
- "Original Sentence": The specific source sentence in the Context.
- "Atomic Claim": The specific fact you need to classify.
- "Other Claims": A list of other facts extracted from the same sentence.

---
INSTRUCTIONS:
Classify the "Atomic Claim" into exactly ONE of the following 7 categories.
You must evaluate the claim against the rules below in order (Step 1 to Step 7). The first step that matches determines the label.

The 7 Labels:
1. Unverifiable: Personal narratives or empathy.
2. Incorrectly structured: Questions, commands, or contains reporting frames.
3. Context-dependent: Contains unresolved pronouns or vague terms.
4. Incomplete: Missing critical modifiers (may, severe) or conditions (if, when).
5. Hallucinated: Adds info not in "Atomic Claim" or distorts meaning.
6. Redundant: A duplicate or a composite of other claims.
7. Valid: A perfect, standalone atomic fact.

---
REASONING PROCESS:
Step 1: Check for Unverifiable
Is the claim a personal narrative ("I spoke to..."), a patient-specific experience ("You are feeling..."), or an empathetic "bedside manner" statement ("It is understandable...")? These claims cannot be verified by an external knowledge base.
- IF YES -> STOP. Label: Unverifiable
- IF NO -> Proceed to Step 2.
Step 2: Check for Incorrectly structured
Does the claim:
a) Retain the reporting frame (e.g., "The doctor noted that...", "It is believed that...")?
b) Exist as a Question ("Did you take it?") or Command ("Take this.")?
c) Contain non-factual text (e.g., "Facts:", "Bullet point")?
- IF YES -> STOP. Label: Incorrectly structured
- IF NO -> Proceed to Step 3.
Step 3: Check for Context-dependent
Does the claim contain words that refer to something outside the claim itself?
- Look for Pronouns: he, she, it, they, this, these.
- Look for possessive adjectives (their, your, its, his, her).
- Is the sentence used in the first person? (I, we, my, our)
- Look for Vague References: the medication, the symptoms, the condition, the patient.
- Can a stranger understand exactly what this claim is about without reading the Context?
- IF NO (It needs context) -> STOP. Label: Context-dependent
- IF YES (It is specific) -> Proceed to Step 4.
Step 4: Check for Incomplete
Compare the Claim to the Original Sentence. Did the claim drop a Critical Medical Modifier?
- Critical Modifiers: Probability (may, could, likely), Frequency (daily, rarely), Severity (severe, mild), Conditions (if, when).
- Crucial Exception (Valid Decomposition): Do NOT flag as Incomplete if the claim splits a list "AND/OR" (e.g., "A causes B and C" -> "A causes B"). This is valid. Only flag if the meaning of the individual fact has changed due to a missing word.
- IF YES (Modifier/Condition missing) -> STOP. Label: Incomplete
- IF NO -> Proceed to Step 5.
Step 5: Check for Hallucinated
Does the claim assert something NOT present or implied in the Original Sentence?
- Check 1 (New Info): Does it add details (e.g., "Aspirin" becomes "Aspirin (NSAID)")? -> If the Context explicitly supports this substitution, it is VALID. If not, it is HALLUCINATED.
- Check 2 (Distortion): Does it change the meaning (e.g., "reduces pain" -> "eliminates pain")?
- IF YES -> STOP. Label: Hallucinated
- IF NO -> Proceed to Step 6.
Step 6: Check for Redundant
Look at the "Other Claims" list.
- Condition A (Composite): Is the Atomic Claim just a combination of two facts already in the list? (e.g., Claim="A and B", List=["A", "B"]).
- Condition B (Subset): Is the Atomic Claim a shorter/less detailed version of another claim in the list? (e.g., Claim="Pain", List=["Severe Pain"]).
- IF YES to A or B -> STOP. Label: Redundant
- IF NO -> Proceed to Step 7.
Step 7: Assign Valid
If the claim passed all checks above, Label is Valid.

---
FEW-SHOT EXAMPLES:
Example 1:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease."
Atomic Claim: "I spoke to your doctor."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. The Atomic Claim "I spoke to your doctor" describes a specific personal interaction (event narrative) between the speaker and the doctor. It is not a verifiable general medical fact. STOP. Label: Unverifiable
Classification: Unverifiable

Example 2:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "The doctor noted that these substances may have positive effects on muscle and bone health."
Other Claims: [
"Anabolic steroids may have positive effects on muscle health.",
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects"
]
Reasoning:
Step 1: Check for Unverifiable. The claim contains medical content regarding the effects of substances, so it is not a purely personal narrative or empathetic statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim explicitly retains the reporting frame "The doctor noted that...", which should have been removed to make the claim a standalone fact. This matches the condition for incorrectly structured claims. STOP. Label: Incorrectly structured
Classification: Incorrectly structured

Example 3:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "They also carry significant risks."
Other Claims: [
"Anabolic steroids may have positive effects on muscle health.",
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry potential side effects"
]
Reasoning:
Step 1: Check for Unverifiable. The claim "They also carry significant risks" is a statement regarding medical risk, not a personal narrative or an empathetic statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim is a declarative sentence and does not include the reporting frame "The doctor noted that". Proceed to Step 3.
Step 3: Check for Context-dependent. The claim begins with the pronoun "They". Without reading the Context or Original Sentence, it is impossible to know what "They" refers to (which is "anabolic steroids"). Therefore, the claim relies on outside context. STOP. Label: Context-dependent
Classification: Context-dependent

Example 4:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids have positive effects on muscle health."
Other Claims: [
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects"
]
Reasoning:
Step 1: Check for Unverifiable. The claim is not a personal narrative or empathy statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. It is a declarative sentence and does not contain the reporting frame "The doctor noted". Proceed to Step 3.
Step 3: Check for Context-dependent. The vague term "these substances" has been correctly resolved to "Anabolic steroids". It is standalone. Proceed to Step 4.
Step 4: Check for Incomplete. The Original Sentence states "these substances may have positive effects". The Atomic Claim states "Anabolic steroids have positive effects". The claim drops the critical probability modifier "may", which changes the meaning from a possibility to a certainty. This matches the Incomplete criteria. STOP. Label: Incomplete 
Classification: Incomplete

Example 5:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids may have negative effects on muscle health."
Other Claims: [
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects"
]
Reasoning:
Step 1: Check for Unverifiable. The claim "Anabolic steroids may have negative effects on muscle health" is a general medical statement, not a personal narrative or empathy. Proceed to Step 2.
Step 2: Check for Incorrectly structured. It is a declarative sentence, does not contain a reporting frame ("The doctor noted that..."), and is not a question or command. Proceed to Step 3.
Step 3: Check for Context-dependent. The claim uses "Anabolic steroids" (specific entity) instead of "these substances". It stands alone clearly. Proceed to Step 4.
Step 4: Check for Incomplete. The claim retains the modifier "may". Proceed to Step 5.
Step 5: Check for Hallucinated. Compare to the Original Sentence: "these substances may have positive effects on muscle...". The claim states "may have negative effects on muscle health." This contradicts and distorts the meaning of the original sentence regarding muscle health. This matches Check 2 (Distortion). Stop. Label: Hallucinated
Classification: Hallucinated

Example 6:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids carry significant risks and potential side effects."
Other Claims: [
"Anabolic steroids may have positive effects on muscle health.", 
"Anabolic steroids may have positive effects on bone health.", 
"Anabolic steroids carry significant risks.", 
"Anabolic steroids carry potential side effects."
]
Reasoning:
Step 1: Check for Unverifiable. The claim "Anabolic steroids carry significant risks and potential side effects" is a factual medical statement. It is not a personal narrative ("I spoke..."), patient experience ("You feel..."), or empathy statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim is a declarative sentence. It does not contain a reporting frame (e.g., "The doctor noted that..."). It is not a question or command. Proceed to Step 3.
Step 3: Check for Context-dependent. The claim uses the specific entity "Anabolic steroids" instead of the pronoun "they" or "these substances". It contains no vague terms requiring external context. It is specific and standalone. Proceed to Step 4.
Step 4: Check for Incomplete. Comparing to the original sentence segment "...they also carry significant risks and potential side effects", the claim preserves the modifiers "significant" and "potential". No critical conditions or modifiers are dropped. Proceed to Step 5.
Step 5: Check for Hallucinated. The claim is strictly grounded in the original text. It does not add new information or distort the meaning. Proceed to Step 6.
Step 6: Check for Redundant. Looking at the "Other Claims" list: ["Anabolic steroids may have positive effects on muscle health.", "Anabolic steroids may have positive effects on bone health.", "Anabolic steroids carry significant risks.", "Anabolic steroids carry potential side effects."]
- Condition A (Composite Check): The Atomic Claim "Anabolic steroids carry significant risks and potential side effects" is a direct combination of two facts already present in the list: "Anabolic steroids carry significant risks" AND "Anabolic steroids carry potential side effects". Since the more atomic versions exist in the list, this composite claim is Redundant. STOP. Label: Redundant.
Classification: Redundant

Example 7:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids may have positive effects on muscle health."
Other Claims: [
"Anabolic steroids may have positive effects on muscle and bone health.",
"Anabolic steroids may have positive effects on bone health.", 
"Anabolic steroids carry significant risks.", 
"Anabolic steroids carry potential side effects."
]
Reasoning:
Step 1: Check for Unverifiable. The claim is a factual statement about the effects of a substance, not a personal narrative or empathy. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim is a declarative sentence. It does not contain the reporting frame ("The doctor noted that..."). Proceed to Step 3.
Step 3: Check for Context-dependent. The claim uses "Anabolic steroids" which correctly replaces the pronoun/reference "these substances" found in the Original Sentence based on the Context. It is specific and understandable to a stranger. Proceed to Step 4.
Step 4: Check for Incomplete. The claim retains the critical modifier "may". It separates "muscle health" from "muscle and bone health", which is a Valid Decomposition (Crucial Exception) of an "AND" list. It is a complete atomic fact. Proceed to Step 5.
Step 5: Check for Hallucinated. The substitution of "Anabolic steroids" for "these substances" is explicitly supported by the Context ("...safety of using anabolic steroids..."). It does not add new unsupported info. Proceed to Step 6.
Step 6: Check for Redundant.
Condition A (Composite Check): The claim is atomic, not a combination of others.
Condition B (Subset Check): The claim is "Anabolic steroids may have positive effects on muscle health". The Other Claims list contains the composite "Anabolic steroids may have positive effects on muscle and bone health". However, the atomic claim is a distinct, validly separated concept, not a "less detailed/vague" version of the composite. It is not redundant to the other atomic claim ("bone health"). Proceed to Step 7.
Step 7: Assign Valid. The claim has passed all six checks.
Classification: Valid

Example 8:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids may have positive effects on bone health."
Other Claims: [
"Anabolic steroids may have positive effects on muscle health.", 
"Anabolic steroids carry significant risks.", 
"Anabolic steroids carry potential side effects."
]
Reasoning:
Step 1: Check for Unverifiable. The claim is a medical statement regarding health effects, not a personal narrative or empathetic statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim properly removes the reporting frame ("The doctor noted that...") and is a complete declarative sentence. Proceed to Step 3.
Step 3: Check for Context-dependent. The claim resolves the vague phrase "these substances" from the Original Sentence to the specific entity "Anabolic steroids" based on the Context. It is specific and standalone. Proceed to Step 4.
Step 4: Check for Incomplete. The claim retains the critical probability modifier "may". Although the original sentence mentions "muscle and bone health", extracting "bone health" as a separate fact is a Valid Decomposition (splitting an "AND" list) and is not considered Incomplete. Proceed to Step 5.
Step 5: Check for Hallucinated. The claim is grounded in the text. The substitution of "Anabolic steroids" is explicitly supported by the Context. Proceed to Step 6.
Step 6: Check for Redundant. The claim focuses on "bone health". The Other Claims list focuses on "muscle health", "risks", and "side effects". The claim is distinct and is neither a composite nor a subset of the other claims. Proceed to step 7.
Step 7: Assign Valid. The claim passed all checks.
Classification: Valid

---
OUTPUT FORMAT:
Reasoning:
Step 1: [Step 1 reasoning]
Step 2: [Step 2 reasoning]
...
Classification: (Your single-word classification label)

---
YOUR TASK

Context: "{context}"
Original Sentence: "{sentence}"
Atomic Claim: "{claim}"
Other Claims: {formatted_other_claims}"""
        return prompt

    def format_normalization_input(self, context: str, sentence: str, claim: str, error_label: str) -> str:
        prompt = f"""You are an expert Medical Text Normalization System. Your task is to REPAIR an "Invalid Atomic Claim" based on its "Error Label" and the "Original Source" information.

Your goal is to transform the Invalid Claim into a Valid, Standalone, Declarative, and Complete atomic fact without altering the original medical meaning.

INPUT DATA:
- Context: The full paragraph containing the information.
- Original Sentence: The specific sentence the claim was extracted from.
- Invalid Claim: The claim that needs fixing.
- Error Label: The specific reason the claim was rejected (Context-dependent, Incorrectly structured, or Incomplete).

---
REPAIR STRATEGIES (Follow the strategy corresponding to the Error Label):

STRATEGY A: IF Error Label is "Context-dependent"
- Problem: The claim contains unresolved pronouns (it, they, he, she) or unresolved possessive adjectives (their, your, its, his, her) or vague references (the medication, this condition) or the first person.
- Fix: Locate the specific entity (noun/proper noun) in the "Context" or "Original Sentence" that the pronoun refers to. Replace the pronoun/vague term with the specific entity.
- Example: "It causes nausea" + Context "Metformin..." -> "Metformin causes nausea."

STRATEGY B: IF Error Label is "Incorrectly structured"
- Problem: The claim is an Imperative (command), a Question, or contains a Reporting Frame (e.g., "The doctor said that...").
- Fix (Imperative/Question): Convert the command into a passive advice statement or a declarative fact. Consider the context, use phrases like "Patients should..." or correct ones.
- Fix (Reporting Frame): Remove the reporting frame (e.g., "The study shows that", "The doctor noted") and keep only the core medical fact.
- Example: "Take with food" -> "The medication should be taken with food."
- Example: "The doctor noted that Aspirin reduces pain" -> "Aspirin reduces pain."

STRATEGY C: IF Error Label is "Incomplete"
- Problem: The claim has lost a critical modifier (may, likely), a condition (if, when), or a dependency due to over-decomposition.
- Fix: Retrieve the missing modifier, condition, or clause from the "Original Sentence" and re-attach it to the claim.
- Example: Claim "Stop taking the drug" + Original "If rash occurs, stop taking..." -> "Patients should stop taking the drug if a rash occurs."

---
REASONING PROCESS:
Step 1: Analyze the Error: Identify clearly what makes the claim invalid based on the label.
Step 2: Locate Source Info: Find the missing context, entity, or modifier in the "Context" or "Original Sentence".
Step 3: Draft Repair: Apply the specific Fix Strategy.
Step 4: Final Check: Ensure the new claim is now Valid (Standalone, Declarative, Complete) and does not hallucinate new info.

---
FEW-SHOT EXAMPLES:

Example 1: Repairing "Context-dependent"
Context: "Metformin is the first-line treatment for type 2 diabetes. However, it often causes gastrointestinal issues."
Original Sentence: "However, it often causes gastrointestinal issues."
Invalid Claim: "It often causes gastrointestinal issues."
Error Label: Context-dependent
Reasoning:
Step 1: Analyze the Error: The claim uses "It", which is vague.
Step 2: Locate Source Info: In the context, "It" refers to "Metformin".
Step 3: Draft Repair: Replace "It" with "Metformin".
Step 4: Final Check: "Metformin often causes gastrointestinal issues." is valid.
Normalized Claim: Metformin often causes gastrointestinal issues.

Example 2: Repairing "Context-dependent" (Possessive Adjective + The first person)
Context: "I was prescribed Lisinopril for hypertension. My blood pressure improved significantly."
Original Sentence: "My blood pressure improved significantly."
Invalid Claim: "My blood pressure improved significantly."
Error Label: Context-dependent
Reasoning:
Step 1: Analyze the Error: The claim uses "My", which is first-person possessive.
Step 2: Locate Source Info: "My" refers to the patient's blood pressure.
Step 3: Draft Repair: Replace "My" with "The patient's".
Step 4: Final Check: "The patient's blood pressure improved significantly." is valid.
Normalized Claim: The patient's blood pressure improved significantly.

Example 3: Repairing "Incorrectly structured" (Imperative)
Context: "Managing acid reflux involves lifestyle changes. Avoid eating heavy meals right before bedtime."
Original Sentence: "Avoid eating heavy meals right before bedtime."
Invalid Claim: "Avoid eating heavy meals right before bedtime."
Error Label: Incorrectly structured
Reasoning:
Step 1: Analyze the Error: The claim is an imperative command ("Avoid...").
Step 2: Locate Source Info: The context implies this is advice for managing acid reflux.
Step 3: Draft Repair: Convert to declarative advice. "Patients should avoid eating..."
Step 4: Final Check: The sentence is now a declarative statement.
Normalized Claim: Patients should avoid eating heavy meals right before bedtime.

Example 4: Repairing "Incorrectly structured" (Reporting Frame)
Context: "A recent study verified the effects of the drug. The study shows that Ibuprofen reduces inflammation."
Original Sentence: "The study shows that Ibuprofen reduces inflammation."
Invalid Claim: "The study shows that Ibuprofen reduces inflammation."
Error Label: Incorrectly structured
Reasoning:
Step 1: Analyze the Error: The claim includes the frame "The study shows that".
Step 2: Locate Source Info: Core fact is "Ibuprofen reduces inflammation".
Step 3: Draft Repair: Remove the frame.
Step 4: Final Check: The core fact is preserved.
Normalized Claim: Ibuprofen reduces inflammation.

Example 5: Repairing "Incomplete" (Missing Condition)
Context: "Dosing instructions for Amoxicillin. If you have kidney disease, the dosage requires adjustment."
Original Sentence: "If you have kidney disease, the dosage requires adjustment."
Invalid Claim: "The dosage requires adjustment."
Error Label: Incomplete
Reasoning:
Step 1: Analyze the Error: The claim states dosage adjustment is needed but misses the condition "If you have kidney disease". This makes the advice dangerous/incorrect generally.
Step 2: Locate Source Info: The condition "If you have kidney disease" is in the Original Sentence.
Step 3: Draft Repair: Re-attach the condition.
Step 4: Final Check: The claim now accurately reflects the medical condition.
Normalized Claim: The dosage of Amoxicillin requires adjustment if the patient has kidney disease.

Example 6: Repairing "Incomplete" (Missing Modifier)
Context: "Anabolic steroids may have positive effects on muscle health, but they also carry significant risks."
Original Sentence: "Anabolic steroids may have positive effects on muscle health, but they also carry significant risks."
Invalid Claim: "Anabolic steroids have positive effects on muscle health."
Error Label: Incomplete
Reasoning:
Step 1: Analyze the Error: The claim drops the critical modifier "may", changing the meaning.
Step 2: Locate Source Info: The modifier "may" is present in the Original Sentence.
Step 3: Draft Repair: Re-attach the modifier "may".
Step 4: Final Check: The claim is now complete and accurate.
Normalized Claim: Anabolic steroids may have positive effects on muscle health.

---
OUTPUT FORMAT:
Reasoning:
Step 1: Analyze the Error: [Step 1 reasoning]
Step 2: Locate Source Info: [Step 2 reasoning]
Step 3: Draft Repair: [Step 3 reasoning]
Step 4: Final Check: [Step 4 reasoning]
Normalized Claim: (Your repaired atomic claim)

---
YOUR TASK:

Context: "{context}"
Original Sentence: "{sentence}"
Invalid Claim: "{claim}"
Error Label: {error_label}

Output ONLY the Reasoning and the Final Normalized Claim.
"""
        return prompt

    def _group_claims_by_sentence(self, claims: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
        """Group claims by (response_id, sentence) tuple.
        Output: {(response_id, sentence): [claim_dicts]}
        """
        grouped = {}
        for claim in claims:
            key = (claim.get('id', ''), claim.get('sentence', ''))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(claim)
        return grouped

    def normalize_invalid_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize claims that are Context-dependent, Incorrectly structured, or Incomplete."""
        normalized_claims = []
        normalizable_types = ['Context-dependent', 'Incorrectly structured', 'Incomplete']

        messages = []
        claims_to_normalize = []

        for claim in claims:
            claim_type = claim.get('claim_quality_type', '')
            if claim_type in normalizable_types:
                messages.append([{
                    "role": "user",
                    "content": self.format_normalization_input(
                        claim.get('context', ''),
                        claim.get('sentence', ''),
                        claim.get('claim', ''),
                        claim_type
                    )
                }])
                claims_to_normalize.append(claim)
            else:
                # Keep non-normalizable claims as-is
                normalized_claims.append(claim)

        if not messages:
            return claims

        # Batch normalize
        all_completions = []
        n_iter = len(messages) // self.batch_size + (1 if len(messages) % self.batch_size else 0)
        for batch in tqdm(chunker(messages, self.batch_size), desc="Normalizing claims process", total=n_iter):
            completions = asyncio.run(self.normalized_llm.batch_response(batch))
            all_completions.extend(completions)

        # Parse normalized claims
        for claim, completion in zip(claims_to_normalize, self.normalized_llm.normalize_llm_response(all_completions)):
            completion = remove_think_tags(completion)
            normalized_text = self.parse_normalized_claim(completion)
            if normalized_text:
                new_claim = {k: v for k, v in claim.items()}
                new_claim['claim'] = normalized_text
                new_claim['normalized_from'] = claim.get('claim', '')
                new_claim['normalization_response'] = completion
                normalized_claims.append(new_claim)
            else:
                # If normalization failed, keep original
                normalized_claims.append(claim)

        return normalized_claims

    def parse_normalized_claim(self, completion: str) -> Optional[str]:
        """Extract normalized claim from LLM response."""
        completion = completion.strip()

        # Look for "Normalized Claim:" pattern
        patterns = [
            r"Normalized Claim:\s*(.+?)(?:\n|$)",
            r"Normalized Claim\s*:\s*(.+?)(?:\n|$)",
            r"normalized claim:\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, completion, re.IGNORECASE | re.DOTALL)
            if match:
                claim = match.group(1).strip()
                # Remove any trailing reasoning or explanations
                claim = claim.split('\n')[0].strip()
                if claim and not claim.lower().startswith('reasoning'):
                    return claim

        # Fallback: look for the last line that looks like a claim
        lines = completion.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.lower().startswith('reasoning') and len(line) > 10:
                # Remove bullet points
                line = re.sub(r'^[-*]\s*', '', line)
                if line:
                    return line

        return None
