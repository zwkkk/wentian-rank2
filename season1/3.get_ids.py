import csv
import sys

import torch
from tqdm import tqdm

sys.path.append("..")
from simcse.models import BertForCL
from transformers import AutoTokenizer
import os
device = "cuda:0"
paths = os.listdir('./result/macbert_result')
last_one = 0
for path in paths:
    if 'checkpoint' in path:
        idx = int(path.split('-')[-1])
        if idx > last_one:
            last_one = idx
final_path = os.path.join('./result/macbert_result', 'checkpoint-'+str(last_one))
print("load checkpoints from {}".format(final_path))
tokenizer = AutoTokenizer.from_pretrained(final_path+'/')
batch_size = 100
use_pinyin = False


def encode_fun(texts, model):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=110)
    inputs.to(device)
    return inputs['input_ids'].cpu().numpy()


if __name__ == '__main__':
    model = BertForCL.from_pretrained(final_path)
    model.to(device)
    corpus = [line[1] for line in csv.reader(open("./data/raw_data/corpus.tsv"), delimiter='\t')]
    query = [line[1] for line in csv.reader(open("./data/raw_data/dev.query.txt"), delimiter='\t')]

    query_embedding_file = csv.writer(open('./multi_run_macbert/final/query_ids', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        ids = encode_fun(batch_text, model)
        for j in range(len(ids)):
           
            ids_str = ids[j].tolist()
            ids_str = [str(t) for t in ids_str]
            ids_str = ids_str + ['0'] * (128-len(ids_str))
            assert len(ids_str)==128
            ids_str = ','.join(ids_str)
            query_embedding_file.writerow([i + j + 200001, ids_str])

    doc_embedding_file = csv.writer(open('./multi_run_macbert/final/doc_idx', 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
        ids = encode_fun(batch_text, model)
        for j in range(len(ids)):
            ids_str = ids[j].tolist()[1:] + [0] # 不需要【101】
            ids_str = [str(t) for t in ids_str]
            ids_str = ids_str + ['0'] * (128-len(ids_str))
            assert len(ids_str)==128
            ids_str = ','.join(ids_str)
            doc_embedding_file.writerow([i + j + 1, ids_str])
