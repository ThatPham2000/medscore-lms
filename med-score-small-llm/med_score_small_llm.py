import json
import os
import time
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any

import jsonlines
import ollama

from claim_quality_evaluation import ClaimQualityEvaluation
from decompose_dnd_score import DecomposeDnDScore
from decomposer_fact_score import DecomposerFactScore
from decomposer_med_score import DecomposerMedScore
from decomposer_small_llm import DecomposerSmallLLM
from exceptions import InvalidArgumentException, IllegalArgumentException
from llm_ollama import LLMOllama
from llm_openai import LLMOpenAI
from utils import parse_sentences
from verifier_internal import VerifierInternal
from verifier_provided_evidence import VerifierProvidedEvidence


def initialize_llm(llm_provider: str, model_name: str, server: Optional[str], openai_api_key: Optional[str]):
    llm_provider = llm_provider.lower()
    if llm_provider == "ollama":
        if server is None:
            return LLMOllama(model_name=model_name)
        return LLMOllama(
            model_name=model_name,
            ollama_async_client=ollama.AsyncClient(host=server, verify=False)
        )

    if llm_provider == "openai":
        if server is None:
            raise InvalidArgumentException("Server URL must be provided for OpenAI LLM provider")
        if openai_api_key is None:
            raise InvalidArgumentException("OpenAI API key must be provided for OpenAI LLM provider")
        return LLMOpenAI(model_name=model_name, server_path=server, api_key=openai_api_key)

    raise IllegalArgumentException(f"Unknown LLM provider: {llm_provider}")


def initialize_decomposer(
        decomposition_mode: str,
        decomposition_llm_provider: str,
        decomposition_model_name: str,
        decomposition_server: Optional[str],
        openai_api_key: Optional[str],
):
    mode = decomposition_mode.lower()
    llm = initialize_llm(decomposition_llm_provider, decomposition_model_name, decomposition_server, openai_api_key)

    if mode == "small_llm":
        return DecomposerSmallLLM(llm=llm)
    if mode == "medscore":
        return DecomposerMedScore(llm)
    if mode == "factscore":
        return DecomposerFactScore(llm)
    if mode == "dndscore":
        return DecomposeDnDScore(llm)
    else:
        raise IllegalArgumentException(f"Unknown decomposition mode: {mode}")


def initialize_claim_quality_evaluation(
        decomposition_llm_provider: str,
        decomposition_model_name: str,
        decomposition_server: str,  # used to continue normalizing invalid claims
        claim_quality_evaluation_llm_provider: str,
        claim_quality_evaluation_model_name: str,
        claim_quality_evaluation_server: Optional[str],
        is_only_classification: bool = False,
        openai_api_key: Optional[str] = None,
):
    llm = initialize_llm(claim_quality_evaluation_llm_provider, claim_quality_evaluation_model_name,
                         claim_quality_evaluation_server, openai_api_key)
    normalized_llm = initialize_llm(decomposition_llm_provider, decomposition_model_name, decomposition_server,
                                    openai_api_key)

    return ClaimQualityEvaluation(llm=llm, is_only_classification=is_only_classification,
                                  normalized_llm=normalized_llm)


def initialize_verifier(
        verification_mode: str,
        verification_llm_provider: str,
        verification_model_name: str,
        verification_server: Optional[str],
        provided_evidence: Optional[Dict[str, str]] = None,
        openai_api_key: Optional[str] = None,
):
    """Initialize verifier with multiple modes support"""
    mode = verification_mode.lower()
    llm = initialize_llm(verification_llm_provider, verification_model_name, verification_server, openai_api_key)

    if mode == "internal":
        return VerifierInternal(llm)
    if mode == "provided":
        if provided_evidence is None:
            raise InvalidArgumentException("Provided evidence is required for 'provided' verification mode")
        return VerifierProvidedEvidence(provided_evidence, llm)

    raise IllegalArgumentException(f"Unknown verification mode: {mode}")


class MedScoreSmallLLM(object):
    """
    Enhanced MedScore implementation with multiple modes support.
    
    This class supports both traditional MedScore modes
    and enhanced small language model modes with:
    
    1. Chain-of-thought prompting
    2. Multi-step reasoning processes
    3. Enhanced decomposition with reasoning steps
    4. Improved verification with confidence scoring
    5. Context-aware fact extraction
    """

    def __init__(
            self,
            decomposition_mode: str = "small_llm",
            decomposition_llm_provider: str = "ollama",
            decomposition_model_name: str = "llama3.2:3b",
            decomposition_server: Optional[str] = None,
            decomposition_prompt_path: Optional[str] = None,
            verification_mode: str = "small_llm",
            verification_llm_provider: str = "ollama",
            verification_model_name: str = "llama3.2:3b",
            verification_server: Optional[str] = None,
            provided_evidence: Optional[Dict[str, str]] = None,
            claim_quality_evaluation_llm_provider: str = "ollama",
            claim_quality_evaluation_model_name: str = "llama3.2:3b",
            claim_quality_evaluation_server: Optional[str] = None,
            is_only_classification: bool = False,
            openai_api_key: Optional[str] = None,
    ):
        self.decomposer = initialize_decomposer(
            decomposition_mode,
            decomposition_llm_provider,
            decomposition_model_name,
            decomposition_server,
            openai_api_key,
        )

        self.verifier = initialize_verifier(
            verification_mode,
            verification_llm_provider,
            verification_model_name,
            verification_server,
            provided_evidence,
            openai_api_key,
        )

        self.claim_quality_evaluation = initialize_claim_quality_evaluation(
            decomposition_llm_provider,
            decomposition_model_name,
            decomposition_server,
            claim_quality_evaluation_llm_provider,
            claim_quality_evaluation_model_name,
            claim_quality_evaluation_server,
            is_only_classification,
            openai_api_key,
        )

    def decompose(
            self,
            dataset: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enhanced decomposition with reasoning for small LLMs"""
        # Split each response
        decomposer_input = []
        for item in dataset:
            sentences = parse_sentences(item['response'])
            for idx, sentence in enumerate(sentences):
                decomposer_input.append({
                    "id": item["id"],
                    "sentence_id": idx,
                    "context": item['response'],
                    "sentence": sentence['text'].strip(),
                })

        decompositions = self.decomposer.do_decompose(decomposer_input)
        return decompositions

    def evaluate_claim_quality(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        non_empty_decompositions = [d for d in decompositions if d["claim"] is not None]
        claim_quality_evaluation_output = self.claim_quality_evaluation.do_claim_quality_evaluation(
            non_empty_decompositions)
        return claim_quality_evaluation_output

    def verify(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced verification with reasoning for small LLMs"""
        non_empty_decompositions = [d for d in decompositions if d["claim"] is not None]
        verifier_output = self.verifier.do_verify(non_empty_decompositions)
        return verifier_output


def parse_args():
    parser = ArgumentParser(description="MedScore Implementation with Multiple Modes Support")

    # Small LLM: gemma3:27b, gpt-oss:20b, gemma3:12b, llama3.1:8b, llama3.2:3b

    # General
    parser.add_argument("--input_file", required=True, type=str, help="Input JSONL file with medical responses")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for results")
    parser.add_argument("--decompose_only", action="store_true", help="Only run decomposition")
    parser.add_argument("--verify_only", action="store_true", help="Only run verification")
    parser.add_argument("--openai_api_key", type=str, help="Input OPENAI API key")

    # Decomposition
    parser.add_argument("--decomposition_mode", type=str,
                        choices=["small_llm", "medscore", "factscore"],
                        default="small_llm", help="Decomposition mode")
    parser.add_argument("--decomposition_llm_provider", type=str, choices=["ollama", "openai"],
                        default="ollama", help="LLM provider for decomposition")
    parser.add_argument("--decomposition_model_name", type=str, default="gemma3:12b",
                        help="Model name for decomposition")
    parser.add_argument("--decomposition_server", type=str, default=None,
                        help="Server URL for decomposition LLM")
    parser.add_argument("--decomposition_input_file", type=str, default=None,
                        help="Path to decomposition input file (required for verify_only mode)")
    parser.add_argument("--decomposition_prompt_path", type=str, default=None,
                        help="Path to custom decomposition prompt")

    # Claim quality evaluation
    parser.add_argument("--evaluate_claim_quality_only", action="store_true")
    parser.add_argument("--is_only_classification", action="store_true")
    parser.add_argument("--valid_decomposition_input_file", type=str, default=None,
                        help="Path to valid decomposition input file")
    parser.add_argument("--claim_quality_evaluation_llm_provider", type=str, choices=["ollama", "openai"],
                        default="ollama", help="LLM provider for claim quality evaluation")
    parser.add_argument("--claim_quality_evaluation_model_name", type=str, default="gemma3:12b",
                        help="Model name for claim quality evaluation")
    parser.add_argument("--claim_quality_evaluation_server", type=str, default=None,
                        help="Server URL for claim quality evaluation LLM")

    # Verification
    parser.add_argument("--verification_mode", type=str,
                        choices=["internal", "provided"],
                        default="internal_small_llm", help="Verification mode")
    parser.add_argument("--verification_llm_provider", type=str, choices=["ollama", "openai"],
                        default="ollama", help="LLM provider for verification")
    parser.add_argument("--verification_model_name", type=str, default="gemma3:12b",
                        help="Model name for verification")
    parser.add_argument("--verification_server", type=str, default=None,
                        help="Server URL for verification LLM")
    parser.add_argument("--provided_evidence_path", type=str, default=None,
                        help="Path to provided evidence file (required for 'provided' mode)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with jsonlines.open(args.input_file) as reader:
        dataset = [item for item in reader.iter()]

    # use 100 record from dataset[100:200]
    dataset = dataset[100:200]
    print(f"len dataset: {len(dataset)}")

    # Handle provided evidence for 'provided' verification mode
    provided_evidence = None
    if args.verification_mode == "provided":
        if args.provided_evidence_path is not None:
            with open(args.provided_evidence_path, "r") as f:
                provided_evidence = json.load(f)
        else:
            raise InvalidArgumentException("Provided evidence path is required when verification_mode is 'provided'")

    # Set output file names based on modes
    mode_prefix = f"{args.decomposition_mode}_{args.verification_mode}"
    decomposition_output_file = os.path.join(args.output_dir, f"{mode_prefix}_decompositions.jsonl")
    verification_output_file = os.path.join(args.output_dir, f"{mode_prefix}_verifications.jsonl")
    output_file = os.path.join(args.output_dir, f"{mode_prefix}_med_score_output.jsonl")

    scorer = MedScoreSmallLLM(
        decomposition_mode=args.decomposition_mode,
        decomposition_llm_provider=args.decomposition_llm_provider,
        decomposition_model_name=args.decomposition_model_name,
        decomposition_server=args.decomposition_server,
        decomposition_prompt_path=args.decomposition_prompt_path,
        verification_mode=args.verification_mode,
        verification_llm_provider=args.verification_llm_provider,
        verification_model_name=args.verification_model_name,
        verification_server=args.verification_server,
        provided_evidence=provided_evidence,
        claim_quality_evaluation_llm_provider=args.claim_quality_evaluation_llm_provider,
        claim_quality_evaluation_model_name=args.claim_quality_evaluation_model_name,
        claim_quality_evaluation_server=args.claim_quality_evaluation_server,
        is_only_classification=args.is_only_classification,
        openai_api_key=args.openai_api_key,
    )

    decompositions = []
    if args.decompose_only:
        decompose_start_time = time.time()
        print(f"Running decomposition with {args.decomposition_mode} mode...")
        decompositions = scorer.decompose(dataset)
        with jsonlines.open(decomposition_output_file, 'w') as writer:
            writer.write_all(decompositions)
        decompose_end_time = time.time()
        print(f"Decomposition time: {decompose_end_time - decompose_start_time:.2f} seconds")
        print(f"Decomposition completed. Results saved to {decomposition_output_file}")
        exit(0)

    if args.evaluate_claim_quality_only:
        # Load existing decompositions
        with jsonlines.open(args.decomposition_input_file, 'r') as reader:
            decompositions = [item for item in reader.iter()]

        # Evaluate claim quality
        time_claim_quality_start = time.time()
        # Add context to each decomposition
        decompositions_with_context = []
        for item in decompositions:
            for data_item in dataset:
                if data_item["id"] == item["id"]:
                    item["context"] = data_item["response"]
                    break
            decompositions_with_context.append(item)
        claim_quality_decompositions = scorer.evaluate_claim_quality(decompositions_with_context)
        claim_quality_output_file = os.path.join(args.output_dir, f"{mode_prefix}_claim_quality_evaluations.jsonl")
        with jsonlines.open(claim_quality_output_file, 'w') as writer:
            writer.write_all(claim_quality_decompositions)
        time_claim_quality_end = time.time()
        print(f"Claim quality evaluation time: {time_claim_quality_end - time_claim_quality_start:.2f} seconds")
        print(f"Claim quality evaluation completed. Results saved to {claim_quality_output_file}")
        exit(0)

    if args.verify_only:
        # Load existing decompositions
        with jsonlines.open(args.valid_decomposition_input_file, 'r') as reader:
            decompositions = [item for item in reader.iter()]
        print(f"len decompositions: {len(decompositions)}")

        verification_start_time = time.time()
        # Process verification
        print(f"Running verification with {args.verification_mode} mode...")
        print(f"len decompositions: {len(decompositions)}")
        verifications = scorer.verify(decompositions)
        with jsonlines.open(verification_output_file, 'w') as writer:
            writer.write_all(verifications)
        verification_end_time = time.time()
        print(f"Verification time: {verification_end_time - verification_start_time:.2f} seconds")

        # Combine results
        combined_output = {
            d["id"]: {
                "id": d["id"],
                "claims": []
            } for d in decompositions
        }
        for verification in verifications:
            claim_info = {
                k: v for k, v in verification.items() if k not in {"id", "sentence_id", "claim_id"}
            }
            combined_output[verification['id']]['claims'].append(claim_info)

        # Aggregate scores
        for idx in combined_output:
            claim_scores = [claim['score'] for claim in combined_output[idx]['claims']]
            if len(claim_scores) == 0:
                combined_output[idx]["score"] = None
            else:
                combined_output[idx]["score"] = sum(claim_scores) / len(claim_scores)

        combined_output = [v for k, v in combined_output.items()]
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(combined_output)

        # Calculate and display final metrics
        scores = [item['score'] for item in combined_output if item['score'] is not None]
        final_score = sum(scores) / len(scores) if scores else None

        print(f"\n=== MedScore Results ({args.decomposition_mode} + {args.verification_mode}) ===")
        print(f"Total responses evaluated: {len(combined_output)}")
        print(f"Responses with valid scores: {len(scores)}")
        print(f"Final MedScore: {final_score:.4f}")
        print(f"Results saved to: {output_file}")

        # Additional metrics for comparison
        if scores:
            import statistics

            print(f"Score statistics:")
            print(f"  Mean: {statistics.mean(scores):.4f}")
            print(f"  Median: {statistics.median(scores):.4f}")
            if len(scores) > 1:
                print(f"  Std Dev: {statistics.stdev(scores):.4f}")
            print(f"  Min: {min(scores):.4f}")
            print(f"  Max: {max(scores):.4f}")
        exit(0)

    # Full flow: decompose -> filter valid -> verify
    # ==========Decomposition==========
    decompose_start_time = time.time()
    print(f"Running decomposition with {args.decomposition_mode} mode...")
    decompositions = scorer.decompose(dataset)
    with jsonlines.open(decomposition_output_file, 'w') as writer:
        writer.write_all(decompositions)
    decompose_end_time = time.time()
    print(f"Decomposition time: {decompose_end_time - decompose_start_time:.2f} seconds")
    print(f"Decomposition completed. Results saved to {decomposition_output_file}")

    # ==========Claim Quality Evaluation==========
    # Evaluate claim quality
    time_claim_quality_start = time.time()
    # Add context to each decomposition
    decompositions_with_context = []
    for item in decompositions:
        for data_item in dataset:
            if data_item["id"] == item["id"]:
                item["context"] = data_item["response"]
                break
        decompositions_with_context.append(item)
    claim_quality_decompositions = scorer.evaluate_claim_quality(decompositions_with_context)
    claim_quality_output_file = os.path.join(args.output_dir, f"{mode_prefix}_claim_quality_evaluations.jsonl")
    with jsonlines.open(claim_quality_output_file, 'w') as writer:
        writer.write_all(claim_quality_decompositions)
    time_claim_quality_end = time.time()
    print(f"Claim quality evaluation time: {time_claim_quality_end - time_claim_quality_start:.2f} seconds")
    print(f"Claim quality evaluation completed. Results saved to {claim_quality_output_file}")

    # ===========Verification==========
    decompositions = [d for d in claim_quality_decompositions]
    print(f"len decompositions: {len(decompositions)}")

    verification_start_time = time.time()
    # Process verification
    print(f"Running verification with {args.verification_mode} mode...")
    print(f"len decompositions: {len(decompositions)}")
    verifications = scorer.verify(decompositions)
    with jsonlines.open(verification_output_file, 'w') as writer:
        writer.write_all(verifications)
    verification_end_time = time.time()
    print(f"Verification time: {verification_end_time - verification_start_time:.2f} seconds")

    # Combine results
    combined_output = {
        d["id"]: {
            "id": d["id"],
            "claims": []
        } for d in decompositions
    }
    for verification in verifications:
        claim_info = {
            k: v for k, v in verification.items() if k not in {"id", "sentence_id", "claim_id"}
        }
        combined_output[verification['id']]['claims'].append(claim_info)

    # Aggregate scores
    for idx in combined_output:
        claim_scores = [claim['score'] for claim in combined_output[idx]['claims']]
        if len(claim_scores) == 0:
            combined_output[idx]["score"] = None
        else:
            combined_output[idx]["score"] = sum(claim_scores) / len(claim_scores)

    combined_output = [v for k, v in combined_output.items()]
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(combined_output)

    # Calculate and display final metrics
    scores = [item['score'] for item in combined_output if item['score'] is not None]
    final_score = sum(scores) / len(scores) if scores else None

    print(f"\n=== MedScore Results ({args.decomposition_mode} + {args.verification_mode}) ===")
    print(f"Total responses evaluated: {len(combined_output)}")
    print(f"Responses with valid scores: {len(scores)}")
    print(f"Final MedScore: {final_score:.4f}")
    print(f"Results saved to: {output_file}")

    # Additional metrics for comparison
    if scores:
        import statistics

        print(f"Score statistics:")
        print(f"  Mean: {statistics.mean(scores):.4f}")
        print(f"  Median: {statistics.median(scores):.4f}")
        if len(scores) > 1:
            print(f"  Std Dev: {statistics.stdev(scores):.4f}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
    exit(0)
