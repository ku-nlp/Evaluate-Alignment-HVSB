import os
import csv
import json

from argparse import ArgumentParser
from tqdm import tqdm

from explain_utils import (
    get_explain_instruction,
    generate_by_api,
    load_existing_ids
)

def arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path',
                        type=str, default="gpt-4o")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--file_path', type=str, default="")
    args = parser.parse_args()
    return args

def save_result(save_result_path, item):
    with open(save_result_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(item)

def main(args):
    file_path = args.file_path
    save_path = args.file_path.replace('.jsonl', '.tsv').replace(f'data/{args.dataset}_tobe_exp_500/', f'data/{args.dataset}_explanation/{args.model_name_or_path}/')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    existing_ids, item_list = load_existing_ids(save_path)
    
    user_instruction = get_explain_instruction()
    for data in tqdm(data_list, total=len(data_list)):
        if data['id'] in existing_ids:
            continue
        existing_ids.add(data['id'])
        _user_instruction = user_instruction.replace('###scenario###', data['scenario'])
        explanation = generate_by_api(args, _user_instruction)
        item = [data['id'],
                data['scenario'],
                data['gold_label'],
                explanation]
        item_list.append(item)
        save_result(save_path, item)

if __name__ == '__main__':
    args = arguments()
    main(args)