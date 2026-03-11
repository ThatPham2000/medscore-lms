import json
import os

def remove_manual_claim_quality_type():
    input_file = "/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/output_medscore_claim_quality_llama3_2_vision_11b/medscore_small_llm_manual_claim_quality_evaluations.jsonl"
    temp_file = input_file + ".tmp"

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(temp_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            if 'manual_claim_quality_type' in data:
                del data['manual_claim_quality_type']

            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    os.replace(temp_file, input_file)
    print("Done!")

if __name__ == '__main__':
    remove_manual_claim_quality_type()

