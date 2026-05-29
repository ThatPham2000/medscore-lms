import json
import sys


def filter_claims_by_list(input_file, output_file, target_ids):
    """
    Lọc các dòng trong file JSONL dựa trên danh sách ID.
    Unique key được tạo thành từ id, sentence_id và claim_id.
    """
    # Chuyển list thành set để tốc độ tìm kiếm O(1) và loại bỏ các id trùng lặp
    target_set = set(target_ids)

    filtered_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                # Trích xuất các trường ID và nối lại thành chuỗi khóa (key)
                doc_id = data.get('id')
                sent_id = data.get('sentence_id')
                claim_id = data.get('claim_id')

                unique_key = f"{doc_id}_{sent_id}_{claim_id}"

                # Nếu key tồn tại trong tập hợp cần lọc, ghi nguyên dòng text vào file mới
                if unique_key in target_set:
                    outfile.write(line)
                    filtered_count += 1
            except json.JSONDecodeError:
                print("Lỗi đọc JSON ở dòng:", line)
                continue

    print(f"Hoàn tất! Đã lọc và lưu thành công {filtered_count} claims vào '{output_file}'.")


# list gẹnerated by random_samples.py
target_list = [
    "y8s41w_20241101_3_0",
    "ygdixm_20241101_2_2",
    "6j6ep0_20241101_4_0",
    "zcdwtc_20241101_1_2",
    "7kmdch_20241101_6_2",
    "5ngxon_20241101_1_1",
    "zkwmhm_20241101_6_0",
    "xnrkm3_20241101_2_1",
    "6gxcdk_20241101_1_2",
    "z3qsoe_20241101_5_0",
    "zwyr7i_20241101_2_0",
    "zbtvfs_20241101_1_1",
    "5ngxon_20241101_3_0",
    "6uq8x7_20241101_2_0",
    "4ij3dv_20241101_1_0",
    "57g8mz_20241101_8_3",
    "xh8xez_20241101_10_2",
    "78m0ev_20241101_2_3",
    "566kfy_20241101_6_2",
    "xo7dky_20241101_1_1",
    "77yenh_20241101_3_1",
    "4rkwx9_20241101_3_0",
    "x4giq8_20241101_4_2",
    "4z65ux_20241101_3_1",
    "61vc6g_20241101_3_1",
    "yk55uh_20241101_4_0",
    "4e0v2x_20241101_3_1",
    "566kfy_20241101_5_3",
    "6jttwf_20241101_6_1",
    "46ij7f_20241101_2_2"
]

if __name__ == '__main__':
    input_file = "small_llm_ministral3_askdocai_ministral3_8b_classification/small_llm_provided_claim_quality_evaluations.jsonl"
    output_file = "small_llm_ministral3_askdocai_ministral3_8b_classification/manual_samples_ministral3_8b.jsonl"
    filter_claims_by_list(input_file, output_file, target_list)
