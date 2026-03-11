import json


def get_prompt(context: str, sentence: str) -> str:
    prompt = f"""You are a meticulous medical expert specializing in information extraction. Your task is to decompose a medical sentence into individual, verifiable facts by following a rigorous reasoning process.

INPUT DATA:
- Context: full medical response or passage (used ONLY to resolve pronouns and references).
- Sentence: the specific sentence that must be decomposed into atomic facts.

---
INSTRUCTIONS:
Decompose the Sentence into a list of independent facts by following these guidelines:
1. Filter Unverifiable Narratives: Ignore event narratives and patient-empathy.
2. No Hallucinations: Only include information explicitly stated in the Sentence.
3. Preserve Modifiers: Never drop "may", "could", "likely", dosages, or frequencies; Do not lose the original dependency condition or modifier.
4. Correct Structure: Convert commands or questions into declarative statements. Remove reporting frames.
5. Resolve References: Replace all pronouns (he, it, they), all possessive adjectives (their, your, its, his, her) with specific ones from the Context.
6. No Redundancies: Each fact must be unique and non-overlapping both semantically and wording.
7. Atomic Concepts: Each fact must contain only ONE medical concept.
8. No Complex Claims: Split compound sentences containing "AND/OR".

---
REASONING CHAIN OF THOUGHT:
Step 1: TRIAGE (Filter Narratives)
- Does it include event narratives ("I spoke with...", "you are experiencing pain") or present for patient-empathy (i.e., "bedside manner" such as "Your pain can be very tiring") or have no medical content to extract?
If YES, STOP here and output "No verifiable claim".
If NO, PROCEED to Step 2.

Step 2: ISOLATE & FORMAT (Structure Check)
- Remove the reporting frame (e.g., "The doctor believes that...", "The study shows that..."). Extract only the core medical content.
- Convert commands/questions into a passive advice statement or a declarative fact. Consider the context, use phrases like "Patients should..." or correct ones.

Step 3: DECONTEXTUALIZE (Resolve References)
- Replace all pronouns (it, they, your) with the specific entities in the Context.
- Replace all possessive adjectives (their, your, its, his, her) with the specific entities in the Context.
- Replace all general terms (the medication, your symptoms) with the specific entities in the Context. For example, "your symptoms" must be replaced with the actual symptoms mentioned, like "irregular periods and extreme pain".

Step 4: DECOMPOSE (Split Atomic Facts)
- Now that the sentence is clean and resolved, break it into the smallest possible pieces of information.
- Action: Split compound sentences containing "AND" / "OR" into separate lines.
- Rule: Each atomic fact must represent a SINGLE, distinct medical concept.

Step 5: Check Incomplete Claims
- Did you lose any modifiers (e.g., "may", "could", "likely")? Put them back.
- Did you lose any frequency (e.g., "twice daily")? Put them back.
- Did you lose any dosages? Put them back.
- Did you lose any conditions or dependencies are retained (e.g., "if", "when", "in case of")? Put them back.

Step 6: Check Hallucinations
- All claims must be grounded in the original Sentence with no additional information.
- If any information is inferred or added that is not explicitly stated in the Sentence, remove it.

Step 7: Check Redundant Claims
- Ensure that no two facts are semantically or wording-wise overlapping.
- Compare extracted claims against each other.
- Rule 1 (Subset Removal): If Claim A is merely a shorter, less specific version of Claim B without adding new information, REMOVE Claim A.
  (Example: "Do not worry" is a subset of "Do not worry about symptom changes" -> DELETE "Do not worry").
- Rule 2 (Split Exception): Do NOT remove claims that resulted from splitting "AND/OR" lists.
  (Example: Sentence: "Patient has cough and fever". Atomic Facts: "Patient has cough" and "Patient has fever" are NOT redundant to each other, nor are they subsets of a hypothetical combined claim. Keep BOTH).
- Goal: Keep the most specific version of a fact, but preserve all distinct items from a list.

Step 8: Coverage Check
- Final Audit: Do the final list of atomic facts capture 100% of the medical info from the original Sentence?

---
FEW-SHOT EXAMPLES:

Example 1:
Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit.
Sentence: I spoke to your doctor and they wanted to address your concerns about tetanus.
Reasoning:
Step 1: TRIAGE (Filter Narratives): The sentence "I spoke to your doctor..." describes a personal interaction. This is an unverifiable narrative. The process stops here.
Facts:
- No verifiable claim

Example 2:
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Sentence: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "The doctor noted that". The core medical content is: "while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Step 3: DECONTEXTUALIZE (Resolve References): Step 3: I am resolving references using the Context. "These substances" and "they" refer to "anabolic steroids". The specific medical content becomes: "while [anabolic steroids] may have positive effects on muscle and bone health, [anabolic steroids] also carry significant risks and potential side effects."
Step 4: DECOMPOSE (Split Atomic Facts): I am decomposing the sentence based on the "AND" conjunctions.
- "Positive effects on muscle AND bone health" splits into: (1) muscle health and (2) bone health.
- "Significant risks AND potential side effects" splits into: (3) significant risks and (4) potential side effects.
Step 5: Check Incomplete Claims: I am checking for missing modifiers.
- For the benefits, I preserved the modifier "may have" and "positive".
- For the harms, I preserved the modifiers "significant" (for risks) and "potential" (for side effects).
Step 6: Check Hallucinations: All claims are grounded in the original text. No information was added.
Step 7: Check Redundant Claims: I am checking for redundancies.
- The facts are derived from splitting lists (muscle/bone and risks/side effects).
- According to Rule 2 (Split Exception), these are distinct concepts sharing a subject/verb, so I will keep all of them. None are subsets of each other.
Step 8: Coverage Check: I have captured all four key concepts: muscle health, bone health, risks, and side effects. Coverage is complete.
Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids carry significant risks.
- Anabolic steroids carry potential side effects.

Example 3:
Context: I spoke to your doctor and they wanted to address your concerns about your irregular periods and extreme pain. They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods.
Sentence: They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods.
Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "They believe that". The core medical content is: "your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- "your symptoms" refers to "irregular periods and extreme pain".
- "your body" refers to "the body".
- "your menstrual cycle" refers to "the menstrual cycle". 
The specific medical content becomes: "[irregular periods and extreme pain] could be related to anovulatory cycles, which means that [the body] is not releasing an egg during [the menstrual cycle], and primary dysmenorrhea, which is a condition that causes painful periods."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- Split the main compound link: "irregular periods and extreme pain" is linked to "anovulatory cycles" AND "primary dysmenorrhea".
- Extract the nested definitions: "anovulatory cycles" are defined in the text, and "primary dysmenorrhea" is defined in the text.
Step 5: Check Incomplete Claims: Preserve the modifier "could be related to" for the diagnostic claims. Preserve the full definitions.
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: Check Redundancy: The claims regarding anovulatory cycles and primary dysmenorrhea are distinct (Split Exception applies). The definitions are distinct. No subsets found.
Step 8: Coverage Check: All concepts (symptoms, two potential causes, two definitions) are captured.
Facts:
- Irregular periods and extreme pain could be related to anovulatory cycles.
- Anovulatory cycles mean that the body is not releasing an egg during the menstrual cycle.
- Irregular periods and extreme pain could be related to primary dysmenorrhea.
- Primary dysmenorrhea is a condition that causes painful periods.

---
OUTPUT FORMAT:
Reasoning:
Step 1: [Step 1 reasoning]
Step 2: [Step 2 reasoning]
Step 3: [Step 3 reasoning]
Step 4: [Step 4 reasoning]
Step 5: [Step 5 reasoning]
Step 6: [Step 6 reasoning]
Step 7: [Step 7 reasoning]
Step 8: [Step 8 reasoning]
Facts:
- [Fact 1]
- [Fact 2]
...

---
YOUR TASK:
Context: {context}

Sentence: {sentence}

Facts:"""
    return prompt


import ollama

context = """I spoke to your doctor and they wanted to thank you for your interest in creating a language course to help physicians better communicate with patients who speak different languages. \n\nThey mentioned that while language barriers can contribute to the \\\"revolving door syndrome,\\\" it's just one of many factors. Other important factors include education, home support, medication noncompliance, and lack of primary care. \n\nIn terms of a language course, your doctor thinks that Duolingo is a good option. However, they noted that it's challenging for doctors to find the time to learn multiple languages, as there are many languages spoken by patients in their area, including Spanish, Hmong, Chinese, and Polish. They also mentioned that many Spanish-speaking patients have some knowledge of English or have family members who are fluent in English.\n\nYour doctor didn't specify a preferred medium for the course, but they seemed to appreciate the idea of a convenient and accessible program. They also didn't provide specific vocabulary recommendations, but it's likely that a course focused on medical terminology and common patient interactions would be most useful."""
sentence = """However, they noted that it's challenging for doctors to find the time to learn multiple languages, as there are many languages spoken by patients in their area, including Spanish, Hmong, Chinese, and Polish."""

print('Context:', context)
print('Sentence:', sentence)

content = get_prompt(context=context, sentence=sentence)
print(content)

result = ollama.chat(
    model='llama3.2-vision:11b',
    messages=[
        {"role": "user", "content": content}
    ],
    options={
        "seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 4096,
    }
)

print(result)
with open('test_decomposition_output.jsonl', 'w') as f:
    f.write(json.dumps({"context": context, "sentence": sentence, "response": result.message.content}) + '\n')
