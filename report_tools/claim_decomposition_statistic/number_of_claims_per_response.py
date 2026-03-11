#!/usr/bin/env python3

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def count_claims_per_response(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            response_id = obj.get("id")
            if response_id is None:
                continue
            # Count only entries where claim is not null
            if obj.get("claim", None) is not None:
                counts[str(response_id)] += 1
    return dict(counts)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate number of claims per response from a JSONL decompositions file and emit JSONL (id,num_claims).",
    )
    parser.add_argument(
        "--input",
        nargs="?",
        type=Path,
        default=Path(
            "/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/output_small_llm_llama3_2_vision_11b/small_llm_provided_decompositions.jsonl"
        ),
        help="Path to JSONL decomposition file. Defaults to the gemma3_12b output.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    counts = count_claims_per_response(input_path)
    i = 1
    count = 0
    for response_id, num in counts.items():
        print(json.dumps({"id": response_id, "num_claims": num}, ensure_ascii=False))

        count += num
        if i % 10 == 0:
            print(f"{i} Number of claims: {count}")
            count = 0
        i += 1

    all_claims = sum(counts.values())
    total_responses = len(counts)
    print(f"Total claims: {all_claims}")
    print(f"Total responses: {total_responses}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
