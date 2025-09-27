import os
import csv

import pandas as pd
from argparse import ArgumentParser

from tqdm import tqdm

from agree_utils import (
    format_response,
    load_existing_ids,
    get_agree_instruction,
    generate_by_api
)

def arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path',
                        type=str, default="gpt-4o")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--file_path', type=str)
    args = parser.parse_args()
    return args

def save_result(save_result_path, item):
    with open(save_result_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(item)

def main(args):
    file_path = args.file_path
    save_path = args.file_path.replace(f'data/{args.dataset}_explanation/', f'data/{args.dataset}_agreeability/{args.model_name_or_path}/data_')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.read_csv(file_path, sep='\t')
    existing_ids, item_list = load_existing_ids(save_path)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['id'] in existing_ids:
            continue
        existing_ids.add(row['id'])
        scenario = row['scenario']
        explanation = format_response(row['response'])

        user_instruction = get_agree_instruction(args.dataset, scenario, explanation)
        response = generate_by_api(args, user_instruction)
        item = [row['id'],
                scenario,
                explanation,
                response]
        item_list.append(item)
        save_result(save_path, item)
        
if __name__ == '__main__':
    args = arguments()
    main(args)