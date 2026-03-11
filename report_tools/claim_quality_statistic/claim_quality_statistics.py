#!/usr/bin/env python3
"""
CLI tool to analyze claim quality JSONL files produced by `ClaimQualityEvaluation`
and calculate separate statistics for:

1. Model evaluation (`claim_quality_type`)
2. Human evaluation (`manual_claim_quality_type`)

For each of the above, it reports:
- Total claims
- Unverifiable claims
- Hallucinated claims
- Incomplete claims
- Incorrectly structured claims
- Context-dependent claims
- Redundant claims
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _normalize_label(label: str) -> str:
    """Normalize raw label string to one of the known buckets."""
    if not label:
        return ""

    l = label.strip().lower()

    # Canonical mapping based on `ClaimQualityEvaluation.valid_types`
    if l == "valid":
        return "valid"
    if l == "unverifiable":
        return "unverifiable"
    if l == "incorrectly structured":
        return "incorrectly_structured"
    if l in {"context-dependent", "context dependent"}:
        return "context_dependent"
    if l == "hallucinated":
        return "hallucinated"
    if l == "incomplete":
        return "incomplete"
    if l == "redundant":
        return "redundant"

    return l  # unknown, but keep something non-empty


def calculate_statistics(
    data: List[Dict], label_key: str
) -> Dict[str, int]:
    """
    Calculate statistics from claim quality data for a given label field.

    Args:
        data: List of claim records (one claim per JSONL line).
        label_key: Field name to read labels from
                   (e.g. 'claim_quality_type' or 'manual_claim_quality_type').

    Returns:
        Dictionary with claim statistics.
    """
    stats: Dict[str, int] = defaultdict(int)

    for record in data:
        raw_label = record.get(label_key, "") or ""
        norm_label = _normalize_label(raw_label)

        # Always count total
        stats["total"] += 1

        if not norm_label:
            stats["unknown"] += 1
            continue

        if norm_label == "valid":
            stats["valid"] += 1
        elif norm_label == "unverifiable":
            stats["unverifiable"] += 1
        elif norm_label == "hallucinated":
            stats["hallucinated"] += 1
        elif norm_label == "incomplete":
            stats["incomplete"] += 1
        elif norm_label == "incorrectly_structured":
            stats["incorrectly_structured"] += 1
        elif norm_label == "context_dependent":
            stats["context_dependent"] += 1
        elif norm_label == "redundant":
            stats["redundant"] += 1
        else:
            stats["unknown"] += 1

    return dict(stats)


def _print_section(
    title: str,
    stats: Dict[str, int],
    show_percentage: bool,
) -> None:
    """Pretty-print one statistics section in a compact format."""
    total = stats.get("total", 0)

    print("=" * 60)
    print(title)
    print("=" * 60)

    if total == 0:
        print("No claims found.\n")
        return

    # Order and labels requested by the user (now includes Valid)
    rows: List[Tuple[str, str]] = [
        ("total", "Total claims"),
        ("valid", "Valid claims"),
        ("unverifiable", "Unverifiable claims"),
        ("hallucinated", "Hallucinated claims"),
        ("incomplete", "Incomplete claims"),
        ("incorrectly_structured", "Incorrectly structured claims"),
        ("context_dependent", "Context-dependent claims"),
        ("redundant", "Redundant claims"),
    ]

    print(f"\n{'Type':<28} {'Count':>7}", end="")
    if show_percentage:
        print("Percentage")
    else:
        print()
    print("-" * 60)

    for key, label in rows:
        count = stats.get(key, 0)
        if show_percentage and total > 0:
            pct = (count / total) * 100
            print(f"{label:<28} {count:>7}  {pct:6.2f}%")
        else:
            print(f"{label:<28} {count:>7}")

    if stats.get("unknown", 0) > 0:
        print(f"\n{'Unknown / other labels':<28} {stats['unknown']:>7}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze claim quality JSONL and report statistics for both "
            "model predictions (`claim_quality_type`) and human labels "
            "(`manual_claim_quality_type`)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python claim_quality_statistics.py \\
      --input_file med-score-small-llm/med-score-small-llm/output_medscore_claim_quality_llama3_2_vision_11b/baseline_manual_claim_quality_evaluations.jsonl

  python claim_quality_statistics.py -p \\
      --input_file med-score-small-llm/med-score-small-llm/output_medscore_claim_quality_llama3_2_vision_11b/baseline_manual_claim_quality_evaluations.jsonl
        """,
    )

    default_path = (
        "med-score-small-llm/med-score-small-llm/"
        "output_medscore_claim_quality_llama3_2_vision_11b/"
        "baseline_manual_claim_quality_evaluations.jsonl"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=default_path,
        help=f"Path to input claim-quality JSONL file (default: {default_path})",
    )
    parser.add_argument(
        "-p",
        "--percentage",
        action="store_true",
        help="Show percentages in addition to counts",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_file}...")
    try:
        data = load_jsonl(args.input_file)
        print(f"Loaded {len(data)} claim records\n")
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{args.input_file}': {e}")
        return 1

    # Model evaluation statistics
    model_stats = calculate_statistics(data, "claim_quality_type")

    # Human evaluation statistics
    human_stats = calculate_statistics(data, "manual_claim_quality_type")

    # Print sections
    _print_section("MODEL EVALUATION (claim_quality_type)", model_stats, args.percentage)
    _print_section(
        "HUMAN EVALUATION (manual_claim_quality_type)", human_stats, args.percentage
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


