import os
import json
import random


def gather_sample_bbq():
    folder_path = "data/bbq_tobe_test/"
    file_list = os.listdir(folder_path)
    data_list = []
    for file_path in file_list:
        file_path = folder_path + file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
    
    sampled_list = random.sample(data_list, 500)
    save_path = "data/bbq_tobe_exp_500/bbq.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in sampled_list:
            json.dump(item, f)
            f.write('\n')

def gather_sample_biasdpo():
    folder_path = "data/biasdpo_tobe_test/"
    file_list = os.listdir(folder_path)
    data_list = []
    for file_path in file_list:
        file_path = folder_path + file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
    
    sampled_list = random.sample(data_list, 500)
    save_path = "data/biasdpo_tobe_exp_500/biasdpo.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in sampled_list:
            json.dump(item, f)
            f.write('\n')

def gather_sample_ss():
    folder_path = "data/ss_tobe_test/"
    file_list = os.listdir(folder_path)
    data_list = []
    id = 1
    for file_path in file_list:
        file_path = folder_path + file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tmp = json.loads(line)
                tmp['id'] = id
                id += 1
                data_list.append(tmp)
    
    sampled_list = random.sample(data_list, 500)
    save_path = "data/ss_tobe_exp_500/ss.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in sampled_list:
            json.dump(item, f)
            f.write('\n')

def gather_sample_cp():
    folder_path = "data/cp_tobe_test/"
    file_list = os.listdir(folder_path)
    data_list = []
    id = 1
    for file_path in file_list:
        file_path = folder_path + file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tmp = json.loads(line)
                tmp['id'] = id
                id += 1
                data_list.append(tmp)
    
    sampled_list = random.sample(data_list, 500)
    save_path = "data/cp_tobe_exp_500/cp.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in sampled_list:
            json.dump(item, f)
            f.write('\n')
            
gather_sample_biasdpo()
gather_sample_bbq()
gather_sample_cp()
gather_sample_ss()