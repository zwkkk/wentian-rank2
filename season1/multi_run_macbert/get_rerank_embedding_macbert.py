import csv
import sys
import os
import torch
from tqdm import tqdm
import numpy as np

sys.path.append("..")
from simcse.models import BertForCL
from transformers import AutoTokenizer
import os
device = "cuda:0"
paths = os.listdir('../result/macbert_result')
batch_size = 100
use_pinyin = False
last_one = 0
idxs = []

def read_corpus(file_path1, file_path2):
    reader = csv.reader(open(file_path1), delimiter='\t')
    total_dict = dict()
    for line in reader:
        corpus_id = int(line[0]) # 1开始-100000
        corpus = line[1]
        total_dict[corpus_id] = corpus
        
    reader = csv.reader(open(file_path2), delimiter='\t')
    for line in reader:
        corpus_id = int(line[0])+100000 # 100001开始-200000
        corpus = line[1]
        total_dict[corpus_id] = corpus
    return total_dict




def encode_fun(texts, model):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=110)
    inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return embeddings


for path in paths:
    if '.ipynb_checkpoints' in path:
        continue
    if 'checkpoint' in path:
        idx = int(path.split('-')[-1])
        idxs.append(idx)
        if idx > last_one:
            last_one = idx
            
idxs = np.sort(idxs)  # checkpoint 从小到大排序


if __name__ == '__main__':
    final_path = os.path.join('../result/macbert_result', 'checkpoint-'+str(idxs[-1]))
    print("load checkpoints from {}".format(final_path))
    tokenizer = AutoTokenizer.from_pretrained(final_path+'/')
    model = BertForCL.from_pretrained(final_path)
    model.to(device)
    
    q_dict = read_corpus('../data/raw_data/train.query.txt','../data/CPR_data/data/ecom/train.query.txt')
    with open('../data/raw_data/qrels.train.tsv', encoding="utf-8") as f:
        lines1 = f.readlines()
    with open('../data/CPR_data/data/ecom/qrels.train.tsv', encoding="utf-8") as f:
        lines2 = f.readlines()

    query = []
    for line in tqdm(lines1):
        q_id = int(line.split('\t')[0])
        q = q_dict[q_id]
        query.append(q)

    for line in tqdm(lines2):
        q_id = int(line.split('\t')[0])+100000
        q = q_dict[q_id]
        query.append(q)

    query_embedding_file = csv.writer(open('rerank_query_embedding', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            try:
                writer_str = temp_embedding[j].tolist()
                writer_str = [format(s, '.8f') for s in writer_str]
                writer_str = ','.join(writer_str)
                query_embedding_file.writerow([i + j + 1, writer_str])
            except:
                print(temp_embedding[j].tolist())

