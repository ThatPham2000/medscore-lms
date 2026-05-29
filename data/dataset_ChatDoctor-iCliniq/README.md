---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: input
    dtype: string
  - name: answer_icliniq
    dtype: string
  - name: answer_chatgpt
    dtype: string
  - name: answer_chatdoctor
    dtype: string
  splits:
  - name: train
    num_bytes: 16962106
    num_examples: 7321
  download_size: 9373080
  dataset_size: 16962106
---
# Dataset Card for "ChatDoctor-iCliniq"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)