import json
import os
from tqdm import tqdm
import csv

from openai import OpenAI

from preprocess_utils import (
    cp_bias_type_list, 
    get_cp_dataset_path, 
    get_cp_save_path,
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
                existing_ids.add(int(row['id']))
                item_list.append([int(row['id']),
                                  row['sentence'],
                                  row['bias_type'],
                                  row['scenario'],
                                  row['gold_label']])
    else:
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

def progress_by_stereotype(current_bias_type, stereotype='pro'):
    save_path = get_cp_save_path(stereotype, current_bias_type)
    existing_ids, _ = load_existing_ids(save_path)
    print(current_bias_type)
    cp_path = get_cp_dataset_path()    
    with open(cp_path) as f:
        reader = csv.DictReader(f)
        reader = list(reader)

        for row in tqdm(reader, total=len(reader)):
            id = int(row[''])
            if id in existing_ids:
                continue
            
            bias_type = row['bias_type']
             
            if current_bias_type != bias_type:
                    continue
            if len(existing_ids) >= 200:
                continue
            existing_ids.add(id)
            direction = row['stereo_antistereo']
            if direction == 'stereo':
                pro_stereo_sentence = row['sent_more']
                anti_stereo_sentence = row['sent_less']
                bias_type = bias_type
            else:
                pro_stereo_sentence = row['sent_less']
                anti_stereo_sentence = row['sent_more']
                bias_type = bias_type

            if stereotype == 'pro':
                response = get_biased_scenario(pro_stereo_sentence)
                biased_scenario = process_response(response)
                item = [id, pro_stereo_sentence, bias_type, biased_scenario, 'biased']
            else:
                response = get_biased_scenario(anti_stereo_sentence)
                biased_scenario = process_response(response)
                item = [id, anti_stereo_sentence, bias_type, biased_scenario, 'biased']
            save_result(save_path, item)

def get_data_by_bias_type(current_bias_type):
    progress_by_stereotype(current_bias_type,"pro")
    progress_by_stereotype(current_bias_type,"anti")

for bias_type in cp_bias_type_list:
    get_data_by_bias_type(bias_type)