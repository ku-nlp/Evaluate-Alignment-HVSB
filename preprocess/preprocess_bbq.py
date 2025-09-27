import json
import os

from preprocess_utils import (
    category_mapping_dict,
    get_bbq_dataset_path,
    get_bbq_neg_stereotype_path,
    get_bbq_neg_nonstereotype_path,
    get_bbq_pos_stereotype_path,
    get_bbq_pos_nonstereotype_path
)

def save_jsonl(data_list, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json.dump(item, f)
            f.write('\n')

def get_answer(data, category):
    group_map = category_mapping_dict[category]
    stereotyped_groups = data['additional_metadata']['stereotyped_groups']
    stereo_answer = ""
    nonstereo_answer = ""
    unknown_answer = ""
    for k,v in data['answer_info'].items():
        if 'disabled' == v[-1]:
            pass
        true_group_name = group_map[v[-1]]
        if type(true_group_name) != list:
            true_group_name = [true_group_name]
        intersection = list(set(true_group_name) & set(stereotyped_groups))

        if len(intersection)>0:
            stereo_answer = f"The {v[0]}"
        elif list(set(true_group_name) & set(['unknown'])):
            unknown_answer = v[0]
        else:
            nonstereo_answer = f"The {v[0]}"
    return stereo_answer, nonstereo_answer, unknown_answer

def append_data(data_list, data, answer, gold_label):
    tmp_data = {k:v for k,v in data.items()}
    tmp_data['scenario'] = f"Context: {data['context']}\n" \
                                f"Question: {data['question']}\n" \
                                f"Answer: {answer}"
    tmp_data['gold_label'] = gold_label
    data_list.append(tmp_data)

def save_file(category):
    data_list = []
    bbq_dataset_path = get_bbq_dataset_path()
    with open(bbq_dataset_path.format(category), 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))

    neg_stereotype_list = []
    neg_nonstereotype_list = []
    neg_unknown_list = []
    pos_stereotype_list = []
    pos_nonstereotype_list = []
    pos_unknown_list = []

    for data in data_list:
        if "example_id" in data:
            data["id"] = data.pop("example_id")
        if data['context_condition'] == 'ambig':            
            stereo_answer, nonstereo_answer , unknown_answer = get_answer(data,category)
            if data['question_polarity'] == 'neg':
                append_data(neg_stereotype_list, data, stereo_answer, "biased")
                append_data(neg_nonstereotype_list, data, nonstereo_answer, "biased")
                append_data(neg_unknown_list, data, unknown_answer, "unbiased")
            else:
                append_data(pos_stereotype_list, data, stereo_answer, "biased")
                append_data(pos_nonstereotype_list, data, nonstereo_answer, "biased")
                append_data(pos_unknown_list, data, unknown_answer, "unbiased")

    assert len(neg_nonstereotype_list) == len(neg_nonstereotype_list) == len(neg_unknown_list) == len(pos_nonstereotype_list) == len(pos_nonstereotype_list) == len(pos_unknown_list)
    save_jsonl(neg_stereotype_list, get_bbq_neg_stereotype_path(category))
    save_jsonl(neg_nonstereotype_list, get_bbq_neg_nonstereotype_path(category))
    save_jsonl(pos_stereotype_list, get_bbq_pos_stereotype_path(category))
    save_jsonl(pos_nonstereotype_list, get_bbq_pos_nonstereotype_path(category))


for category in category_mapping_dict.keys():
    save_file(category)
    