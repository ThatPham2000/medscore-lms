import json
import uuid


def process_medical_jsonl(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
                open(output_file_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                processed_data = {
                    "id": str(uuid.uuid4()),
                    "question": data.get("input", ""),
                    "doctor_response": data.get("answer_icliniq", ""),
                    "response": data.get("answer_chatdoctor", "")
                }

                json_line = json.dumps(processed_data, ensure_ascii=False)
                outfile.write(json_line + "\n")

        print(f"Saved at: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found {input_file_path}")
    except json.JSONDecodeError:
        print("Error: Not a valid JSONL.")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == '__main__':
    input_file = '/Users/that.phamvan/my_ws/master/med-score-small-llm/data/dataset_ChatDoctor-iCliniq/raw_data.jsonl'
    output_file = '/Users/that.phamvan/my_ws/master/med-score-small-llm/data/dataset_ChatDoctor-iCliniq/chat_doctor_icliniq_medscore.jsonl'

    process_medical_jsonl(input_file, output_file)
