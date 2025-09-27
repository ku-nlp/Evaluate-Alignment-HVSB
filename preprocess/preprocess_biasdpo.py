import json
import copy
import os
from datasets import load_dataset

from preprocess_utils import (
    get_biasdpo_chosen_save_path,
    get_biasdpo_rejected_save_path
)
biasdpo_path = "ahmedallam/BiasDPO"
biasdpo_dataset = load_dataset(biasdpo_path)["train"]

chosen_list = []
rejected_list = []
for index,item in enumerate(biasdpo_dataset):
    print(item)
    prompt = item["prompt"]
    chosen = item["chosen"]
    rejected = item["rejected"]
    item['scenario'] = prompt + ' ' + rejected
    item['gold_label'] = 'biased'
    item['id'] = index
    tmp_re = copy.deepcopy(item)
    rejected_list.append(tmp_re)

    item['scenario'] = prompt + ' ' + chosen
    item['gold_label'] = 'unbiased'
    item['id'] = index
    tmp_ch = copy.deepcopy(item)

    chosen_list.append(tmp_ch)

def save_jsonl(data_list, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json.dump(item, f)
            f.write('\n')

# biasdpo_chosen_save_path = get_biasdpo_chosen_save_path()
biasdpo_rejected_save_path = get_biasdpo_rejected_save_path()

# save_jsonl(chosen_list, biasdpo_chosen_save_path)
save_jsonl(rejected_list, biasdpo_rejected_save_path)

