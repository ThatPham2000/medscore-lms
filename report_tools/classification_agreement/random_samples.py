import json
import random


def print_random_unique_keys(file_path, num_samples=30):
    """
    Đọc file JSONL, chọn ngẫu nhiên các dòng và in ra unique_key.
    unique_key = f"{id}_{sentence_id}_{claim_id}"
    """
    try:
        # Mở và đọc tất cả các dòng hợp lệ vào một danh sách
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Xử lý trường hợp file có ít dòng hơn số lượng cần lấy
        actual_samples = min(num_samples, len(lines))
        if actual_samples == 0:
            print("File trống hoặc không có dữ liệu hợp lệ.")
            return

        # Chọn ngẫu nhiên không lặp lại
        sampled_lines = random.sample(lines, actual_samples)

        print(f"--- Đã chọn ngẫu nhiên {actual_samples} unique_keys ---")

        # Phân tích JSON và in ra kết quả
        for line in sampled_lines:
            try:
                data = json.loads(line)

                doc_id = data.get('id', 'unknown_id')
                sent_id = data.get('sentence_id', 'unknown_sent')
                claim_id = data.get('claim_id', 'unknown_claim')

                unique_key = f"{doc_id}_{sent_id}_{claim_id}"
                print(unique_key)

            except json.JSONDecodeError:
                print("⚠️ Bỏ qua: Dòng không đúng định dạng JSON.")

    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{file_path}'")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi: {e}")

if __name__ == '__main__':
    input_file="/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/small_llm/ministral3_14b/small_llm_provided_claim_quality_evaluations.jsonl"
    print_random_unique_keys(input_file, 30)