import json
import os
from tqdm import tqdm
import csv

from openai import OpenAI

from preprocess_utils import (
    ss_bias_type_list, 
    get_ss_dataset_path, 
    get_ss_save_pro_path,
    get_ss_save_anti_path,
    get_emulator_prompt,
    api_key,
    process_response
)
def get_biased_scenario(statement, example_list=[]):
    user_prompt = get_emulator_prompt(statement, example_list)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{user_prompt}"
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=2000,
				response_format = {"type": "json_object"}
            )
    result = json.loads(response.choices[0].message.content)
    return result

def load_existing_ids(file_path):
    existing_ids = set()

    item_list = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                existing_ids.add(row['id'])
                item_list.append([row['id'],
                                  row['sentence'],
                                  row['bias_type'],
                                  row['scenario'],
                                  row['gold_label']])
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            header = ['id',
                      'sentence',
                      'bias_type',
                      'scenario',
                      'gold_label']
            writer.writerow(header)
    return existing_ids, item_list


def save_result(save_result_path, item):
    with open(save_result_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(item)

def get_data_by_bias_type(current_bias_type):
    pro_list = []
    anti_list = []
    ss_path = get_ss_dataset_path()
    with open(ss_path) as f:
            input = csv.DictReader(f, delimiter='\t')
            for row in input:
                id = row['id']
                sentence = row['context']
                bias_word = row['bias_word']
                bias_type = row['bias_type']
                gold_label = row['gold_label']
                if current_bias_type != bias_type:
                    continue
                
                sentence = sentence.replace('BLANK', bias_word)
                
                if gold_label == 'anti-stereotype':
                    item = {
                        'id': id,
                        'sentence': sentence,
                        'bias_type': bias_type,
                    }
                    anti_list.append(item)
                elif gold_label == 'stereotype':
                    item = {
                        'id': id,
                        'sentence': sentence,
                        'bias_type': bias_type,
                    }
                    pro_list.append(item)
    return pro_list[:200], anti_list[:200] # todo 


def processing(data_list, data_path):
    existing_ids, item_list = load_existing_ids(data_path)
    for row in tqdm(data_list, total=len(data_list)):
        id = row['id']
        if id in existing_ids:
            continue
        existing_ids.add(id)
        sentence = row['sentence']
        bias_type = row['bias_type']
        response = get_biased_scenario(sentence)
        biased_scenario = process_response(response)
        item = [id, sentence, bias_type, biased_scenario, 'biased']
        item_list.append(item)
        save_result(data_path, item)


for biastype in ss_bias_type_list:
    pro_list, anti_list = get_data_by_bias_type(biastype)

    pro_path = get_ss_save_pro_path(biastype)
    anti_path = get_ss_save_anti_path(biastype)

    processing(pro_list,pro_path)
    processing(anti_list,anti_path)




