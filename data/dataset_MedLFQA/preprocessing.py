import json
import uuid


def process_medical_jsonl(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
            open(output_filepath, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            must_have = item.get("Must_have", [])
            nice_to_have = item.get("Nice_to_have", [])

            combined_list = must_have + nice_to_have
            doctor_response = " ".join(combined_list)

            new_item = {
                "id": str(uuid.uuid4()),
                "question": item.get("Question", ""),
                "doctor_response": doctor_response,
                "response": item.get("Free_form_answer", "")
            }

            outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')

    print(f"✅ Process finished, the output file was saved at: {output_filepath}")


if __name__ == "__main__":
    INPUT_FILE = "/Users/that.phamvan/my_ws/master/med-score-small-llm/data/dataset_MedLFQA/live_qa_test_MedLFQA.jsonl"
    OUTPUT_FILE = "/Users/that.phamvan/my_ws/master/med-score-small-llm/data/dataset_MedLFQA/live_qa_test_MedLFQA_medscore.jsonl"

    # Chạy hàm xử lý
    process_medical_jsonl(INPUT_FILE, OUTPUT_FILE)