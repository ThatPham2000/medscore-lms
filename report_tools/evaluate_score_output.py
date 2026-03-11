"""
=== MedScore Results: /Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/output_pipeline_gemma3_12b/small_llm_provided_med_score_output.jsonl ===
Total responses evaluated: 20
Responses with valid scores: 20
Final MedScore: 0.8507
Number of generated claims: 235
Score statistics:
  Mean: 0.8507
  Median: 0.8944
  Std Dev: 0.1906
  Min: 0.2222
  Max: 1.0000

=== MedScore Results: /Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/output_pure_gemma3_12b/medscore_provided_med_score_output.jsonl ===
Total responses evaluated: 20
Responses with valid scores: 20
Final MedScore: 0.8914
Number of generated claims: 251
Score statistics:
  Mean: 0.8914
  Median: 0.9282
  Std Dev: 0.1306
  Min: 0.5714
  Max: 1.0000

=== MedScore Results: /Users/that.phamvan/my_ws/master/med-score-small-llm/demo/Provided_medscore_output.jsonl ===
Total responses evaluated: 20
Responses with valid scores: 20
Final MedScore: 0.9600
Number of generated claims: 213
Score statistics:
  Mean: 0.9600
  Median: 1.0000
  Std Dev: 0.0767
  Min: 0.7500
  Max: 1.0000
"""

from jsonlines import jsonlines


def evaluate(file: str):
    with jsonlines.open(file) as reader:
        dataset = [item for item in reader.iter()]

    combined_output = dataset

    # Calculate and display final metrics
    scores = [item['score'] for item in combined_output if item['score'] is not None]
    final_score = sum(scores) / len(scores) if scores else None

    num_of_claims = 0
    for item in combined_output:
        num_of_claims += len(item['claims'])

    print(f"\n=== MedScore Results: {file} ===")
    print(f"Total responses evaluated: {len(combined_output)}")
    print(f"Responses with valid scores: {len(scores)}")
    print(f"Final MedScore: {final_score:.4f}")
    print(f"Number of generated claims: {num_of_claims}")

    # Additional metrics for comparison
    if scores:
        import statistics

        print(f"Score statistics:")
        print(f"  Mean: {statistics.mean(scores):.4f}")
        print(f"  Median: {statistics.median(scores):.4f}")
        print(f"  Std Dev: {statistics.stdev(scores):.4f}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")


if __name__ == '__main__':
    file = '/Users/that.phamvan/my_ws/master/med-score-small-llm/run_result_with_full_data/output_small_llm_gemma3_12b/small_llm_provided_gpt_oss_med_score_output.jsonl'

    evaluate(file)
