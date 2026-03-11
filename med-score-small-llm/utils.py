import re
from itertools import islice
from typing import List, Dict, Any, Iterable

import spacy

# nlp = spacy.load('vi_spacy_model')
nlp = spacy.load("en_core_web_sm")


# input: passage = "Hello world. This is an example."
# output: [{'text': 'Hello world.', 'span_start': 0, 'span_end': 12}, {'text': 'This is an example.', 'span_start': 13, 'span_end': 32}]
def parse_sentences(
        passage: str,
) -> List[Dict[str, Any]]:
    doc = nlp(passage)
    sentences = []
    for sent in doc.sents:
        sentences.append({
            "text": sent.text,
            "span_start": sent.start_char,
            "span_end": sent.end_char
        })
    return sentences


# input: iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#        n = 3
# output: [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18), (19, 20)]
def chunker(
        iterable: Iterable,
        n: int
):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk


# input: claims = ["- Claim 1", "- Claim 2", "No verifiable claim"]
# output: ["Claim 1", "Claim 2"]
def process_claim(claims: List[str]) -> List[str]:
    # drop - in front of the claim
    claims = [claim.strip('-').strip() for claim in claims]

    # remove 'no verifiable claim' from the claims
    claims = [claim for claim in claims if 'no verifiable claim' not in claim.lower()]

    return claims


def extract_confidence_score(completion: str) -> float:
    """Extract confidence score from completion text"""
    import re

    # Look for confidence patterns like "Confidence: 0.8" or "[Confidence: 0.8]"
    confidence_patterns = [
        r'confidence:\s*(\d+\.?\d*)',
        r'\[confidence:\s*(\d+\.?\d*)\]',
        r'confidence\s*=\s*(\d+\.?\d*)',
        r'confidence\s*(\d+\.?\d*)',
    ]

    for pattern in confidence_patterns:
        match = re.search(pattern, completion.lower())
        if match:
            try:
                confidence = float(match.group(1))
                # Ensure confidence is between 0.0 and 1.0
                return max(0.0, min(1.0, confidence))
            except ValueError:
                continue

    # Look for percentage patterns like "80%" or "80 percent"
    percentage_patterns = [
        r'(\d+\.?\d*)\s*%',
        r'(\d+\.?\d*)\s*percent',
    ]

    for pattern in percentage_patterns:
        match = re.search(pattern, completion.lower())
        if match:
            try:
                percentage = float(match.group(1))
                return max(0.0, min(1.0, percentage / 100.0))
            except ValueError:
                continue

    # Look for word-based confidence indicators
    completion_lower = completion.lower()
    if any(word in completion_lower for word in ['very confident', 'highly confident', 'extremely confident']):
        return 0.9
    elif any(word in completion_lower for word in ['confident', 'certain', 'sure']):
        return 0.8
    elif any(word in completion_lower for word in ['somewhat confident', 'moderately confident']):
        return 0.6
    elif any(word in completion_lower for word in ['uncertain', 'unsure', 'not sure']):
        return 0.3
    elif any(word in completion_lower for word in ['very uncertain', 'highly uncertain']):
        return 0.1

    # Default confidence based on response clarity
    if any(word in completion_lower for word in ['true', 'false', 'correct', 'incorrect']):
        return 0.7  # Medium confidence for clear responses
    else:
        return 0.4  # Low confidence for unclear responses


def parse_reasoning_response_with_confidence(completion: str) -> tuple:
    """Parse reasoning response to extract True/False, score, and confidence using same methodology as parse_verification_output"""
    import string

    # Extract confidence score from the response first
    confidence = extract_confidence_score(completion)

    # Use the same logic as parse_verification_output for score calculation
    generated_answer = completion.strip().lower()
    is_supported = 0.0
    raw_response = "False"

    if "true" in generated_answer or "false" in generated_answer:
        if "true" in generated_answer and "false" not in generated_answer:
            is_supported = 1.0
            raw_response = "True"
        elif "false" in generated_answer and "true" not in generated_answer:
            is_supported = 0.0
            raw_response = "False"
        else:
            # If the last occurrence of 'true' appears later than 'false' in the output, then think the conclusion is true.
            is_supported = generated_answer.rindex("true") > generated_answer.rindex("false")
            is_supported = 1.0 if is_supported else 0.0
            raw_response = "True" if is_supported else "False"
    else:
        generated_answer = generated_answer.translate(str.maketrans("", "", string.punctuation)).split()
        is_supported = all(
            [keyword not in generated_answer for keyword in ["not", "cannot", "unknown", "information"]])
        is_supported = 1.0 if is_supported else 0.0
        raw_response = "True" if is_supported else "False"

    return raw_response, is_supported, confidence


def remove_think_tags(text):
    pattern = r'<think>.*?</think>'

    # Replace the pattern with an empty string
    clean_text = re.sub(pattern, '', text, flags=re.DOTALL)

    # .strip() removes leading/trailing whitespace left over
    return clean_text.strip()
