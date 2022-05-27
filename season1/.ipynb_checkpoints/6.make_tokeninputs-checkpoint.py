import csv
import sys
import os
import torch
from tqdm import tqdm

sys.path.append("..")
from simcse.models import BertForCL
from transformers import AutoTokenizer
import json
device = "cuda:1"
tokenizer = AutoTokenizer.from_pretrained("./result/MLM_external_plus_bart25/checkpoint-58000/")
batch_size = 100
use_pinyin = False

if os.path.exists('../ranker/data/train.query.json'):
    os.remove('../ranker/data/train.query.json')
if os.path.exists('../ranker/data/corpus.json'):
    os.remove('../ranker/data/corpus.json')

def write_to_json(ips, file_path):
    with open(file_path, 'a+', encoding='utf-8') as f:
        json.dump(ips, f, ensure_ascii=False)
        f.write('\n')


def encode_fun(texts, model):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=108)
    inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return inputs['input_ids'].cpu().numpy(), embeddings


if __name__ == '__main__':
    '''
    用于生成复赛 train.query.json corpus.json,在train阶段，根据data.py所写，不需要加[cls] [sep]
    '''
    model = BertForCL.from_pretrained("./result/MLM_external_plus_bart25/checkpoint-58000")
    model.to(device)
    corpus = [line[1] for line in csv.reader(open("./data/corpus.tsv"), delimiter='\t')]
    query = [line[1] for line in csv.reader(open("./data/train.query.txt"), delimiter='\t')]

    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        ids, temp_embedding = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            query_dict = {}
            ids_str = ids[j].tolist()
            ids_str = [int(t) for t in ids_str]
            ids_str = ids_str[1:] + [0]  # 去掉cls
            ids_str = [0 if t==102 else t for t in ids_str]  # 去掉seq
            query_dict['qid'] = str(i+j+1)
            query_dict['input_ids'] = ids_str + [0] * (128-len(ids_str))
            assert len(query_dict['input_ids'])==128
            write_to_json(query_dict, '../ranker/data/train.query.json')
            
    
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
        ids, temp_embedding = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            doc_dict = {}
            ids_str = ids[j].tolist()
            ids_str = [int(t) for t in ids_str]
            ids_str = ids_str[1:] + [0]  # 去掉cls
            ids_str = [0 if t==102 else t for t in ids_str]  # 去掉seq
            doc_dict['qid'] = str(i+j+1)
            doc_dict['input_ids'] = ids_str + [0] * (128-len(ids_str))
            assert len(doc_dict['input_ids'])==128
            write_to_json(doc_dict, '../ranker/data/corpus.json')


