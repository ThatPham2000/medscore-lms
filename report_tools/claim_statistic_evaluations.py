#!/usr/bin/env python3
"""
CLI tool to analyze manual evaluation JSONL files and calculate claim statistics:
- Total claims
- Valid claims
- Unverifiable claims
- Hallucinated claims
- Incomplete claims
- Wrong-structure claims
- Context-dependent claims
- Redundant claims
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def calculate_statistics(data: List[Dict]) -> Dict[str, int]:
    """
    Calculate statistics from manual evaluation data.
    
    Args:
        data: List of evaluation records with claims
        
    Returns:
        Dictionary with claim statistics
    """
    stats = defaultdict(int)
    
    for record in data:
        claims = record.get('claims', [])
        for claim in claims:
            manual_evaluation = claim.get('manual', '').lower()
            
            # Map manual evaluation types
            if manual_evaluation == 'valid':
                stats['valid'] += 1
            elif manual_evaluation == 'unverifiable':
                stats['unverifiable'] += 1
            elif manual_evaluation == 'hallucinated':
                stats['hallucinated'] += 1
            elif manual_evaluation == 'incomplete':
                stats['incomplete'] += 1
            elif manual_evaluation == 'incorrect-structure':
                stats['wrong_structure'] += 1
            elif manual_evaluation == 'context-dependent':
                stats['context_dependent'] += 1
            elif manual_evaluation == 'redundant':
                stats['redundant'] += 1
            else:
                # Count unknown types as well for total
                stats['unknown'] += 1
            
            # Always count total
            stats['total'] += 1
    
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze manual evaluation JSONL files and calculate claim statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python claim_statistic_evaluations.py medscore_manual_evaluations.jsonl
  python claim_statistic_evaluations.py --percentage medscore_manual_evaluations.jsonl
        """
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        help='Path to input JSONL file with manual evaluations'
    )
    
    parser.add_argument(
        '-p', '--percentage',
        action='store_true',
        help='Show percentages in addition to counts'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output with additional statistics'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    try:
        data = load_jsonl(args.input_file)
        print(f"Loaded {len(data)} records\n")
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{args.input_file}': {e}")
        return 1
    
    # Calculate statistics
    stats = calculate_statistics(data)
    
    # Display results
    print("="*70)
    print("CLAIM STATISTICS")
    print("="*70)
    
    total = stats.get('total', 0)
    
    if total == 0:
        print("\nNo claims found in the data.")
        return 0
    
    # Define claim types in order
    claim_types = [
        ('total', 'Total claims'),
        ('valid', 'Valid claims'),
        ('unverifiable', 'Unverifiable claims'),
        ('hallucinated', 'Hallucinated claims'),
        ('incomplete', 'Incomplete claims'),
        ('wrong_structure', 'Wrong-structure claims'),
        ('context_dependent', 'Context-dependent claims'),
        ('redundant', 'Redundant claims'),
    ]
    
    print(f"\n{'Claim Type':<30} {'Count':<15}", end='')
    if args.percentage:
        print('Percentage')
    else:
        print()
    print("-"*70)
    
    for key, label in claim_types:
        count = stats.get(key, 0)
        if args.percentage and total > 0:
            percentage = (count / total) * 100
            print(f"{label:<30} {count:<15} {percentage:>6.2f}%")
        else:
            print(f"{label:<30} {count:<15}")
    
    # Show unknown types if any
    if stats.get('unknown', 0) > 0:
        print(f"\n{'Unknown claim types':<30} {stats['unknown']:<15}")
        if args.verbose:
            print("\nWarning: Some claims have unrecognized manual evaluation types.")
    
    if args.verbose:
        print("\n" + "-"*70)
        print("Additional Information:")
        print("-"*70)
        print(f"Total records (responses):      {len(data)}")
        avg_claims_per_record = total / len(data) if len(data) > 0 else 0
        print(f"Average claims per record:       {avg_claims_per_record:.2f}")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())

