#!/usr/bin/env python3
"""
CLI tool to analyze decomposition JSONL files and calculate metrics:
- Average number of claims per response
- Standard deviation of #claims/response
- Average number of claims per sentence
- Standard deviation of #claims/sentence
- Average 0-claim rate per response
"""

import json
import argparse
import statistics
from collections import defaultdict
from typing import Dict, Set, List, Tuple


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def calculate_metrics(data: List[Dict]) -> Dict[str, float]:
    """
    Calculate metrics from decomposition data.
    
    Args:
        data: List of decomposition records
        
    Returns:
        Dictionary with calculated metrics
    """
    # Group data by response ID
    response_claims: Dict[str, List[Dict]] = defaultdict(list)
    
    # Track unique sentences (id + sentence_id combination)
    unique_sentences: Set[Tuple[str, int]] = set()
    
    # Track total claims
    total_claims = 0
    
    # Process each record
    for record in data:
        response_id = record['id']
        sentence_id = record.get('sentence_id', 0)
        claim = record.get('claim')
        
        # Track unique sentences
        unique_sentences.add((response_id, sentence_id))
        
        # Count non-null claims
        if claim is not None:
            total_claims += 1
        
        # Store record for response-level analysis
        response_claims[response_id].append(record)
    
    # Calculate average claims per response
    claims_per_response = []
    zero_claim_responses = 0
    
    for response_id, records in response_claims.items():
        # Count non-null claims for this response
        num_claims = sum(1 for r in records if r.get('claim') is not None)
        claims_per_response.append(num_claims)
        
        if num_claims == 0:
            zero_claim_responses += 1
    
    avg_claims_per_response = (
        sum(claims_per_response) / len(claims_per_response) 
        if claims_per_response else 0.0
    )
    
    # Calculate standard deviation of claims per response
    std_claims_per_response = (
        statistics.stdev(claims_per_response) 
        if len(claims_per_response) > 1 else 0.0
    )
    
    # Calculate average claims per sentence
    # Group records by sentence to calculate claims per sentence
    sentence_claims: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for record in data:
        response_id = record['id']
        sentence_id = record.get('sentence_id', 0)
        sentence_claims[(response_id, sentence_id)].append(record)
    
    claims_per_sentence = []
    for sentence_key, records in sentence_claims.items():
        # Count non-null claims for this sentence
        num_claims = sum(1 for r in records if r.get('claim') is not None)
        claims_per_sentence.append(num_claims)
    
    num_sentences = len(unique_sentences)
    avg_claims_per_sentence = (
        total_claims / num_sentences 
        if num_sentences > 0 else 0.0
    )
    
    # Calculate standard deviation of claims per sentence
    std_claims_per_sentence = (
        statistics.stdev(claims_per_sentence) 
        if len(claims_per_sentence) > 1 else 0.0
    )
    
    # Calculate 0-claim rate
    num_responses = len(response_claims)
    zero_claim_rate = (
        (zero_claim_responses / num_responses * 100.0) 
        if num_responses > 0 else 0.0
    )
    
    return {
        'avg_claims_per_response': avg_claims_per_response,
        'std_claims_per_response': std_claims_per_response,
        'avg_claims_per_sentence': avg_claims_per_sentence,
        'std_claims_per_sentence': std_claims_per_sentence,
        'zero_claim_rate': zero_claim_rate,
        'total_responses': num_responses,
        'total_sentences': num_sentences,
        'total_claims': total_claims,
        'zero_claim_responses': zero_claim_responses,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze decomposition JSONL files and calculate metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_decompositions.py decompositions.jsonl
  python analyze_decompositions.py -v decompositions.jsonl
        """
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        help='Path to input JSONL file with decompositions'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output with additional statistics'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} records")
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Display results
    print("\n" + "="*60)
    print("DECOMPOSITION ANALYSIS RESULTS")
    print("="*60)
    print(f"\nAverage number of claims per response: {metrics['avg_claims_per_response']:.4f}")
    print(f"Standard deviation (#claims/response):  {metrics['std_claims_per_response']:.4f}")
    print(f"Average number of claims per sentence:  {metrics['avg_claims_per_sentence']:.4f}")
    print(f"Standard deviation (#claims/sentence): {metrics['std_claims_per_sentence']:.4f}")
    print(f"Average 0-claim rate per response:      {metrics['zero_claim_rate']:.2f}%")
    
    if args.verbose:
        print("\n" + "-"*60)
        print("Additional Statistics:")
        print("-"*60)
        print(f"Total responses:              {metrics['total_responses']}")
        print(f"Total sentences:              {metrics['total_sentences']}")
        print(f"Total claims:                 {metrics['total_claims']}")
        print(f"Responses with 0 claims:      {metrics['zero_claim_responses']}")
        print(f"Responses with claims:        {metrics['total_responses'] - metrics['zero_claim_responses']}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

