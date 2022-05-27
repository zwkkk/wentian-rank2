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
paths = os.listdir('./result/macbert_result')
batch_size = 100
use_pinyin = False
last_one = 0
idxs = []


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

checks = [idxs[-1]]  #取最后的checkpoints

if __name__ == '__main__':
    for index, n in enumerate(checks):  
        final_path = os.path.join('./result/macbert_result', 'checkpoint-'+str(n))
        print("load checkpoints from {}".format(final_path))
        tokenizer = AutoTokenizer.from_pretrained(final_path+'/')
        model = BertForCL.from_pretrained(final_path)
        model.to(device)
        corpus = [line[1] for line in csv.reader(open("./data/raw_data/corpus.tsv"), delimiter='\t')]
        query = [line[1] for line in csv.reader(open("./data/raw_data/dev.query.txt"), delimiter='\t')]
        
        query_embedding_file = csv.writer(open('multi_run_macbert/final/query_embedding', 'w'), delimiter='\t')
        #query_embedding_file = csv.writer(open('query_embedding'+str(index), 'w'), delimiter='\t')

        for i in tqdm(range(0, len(query), batch_size)):
            batch_text = query[i:i + batch_size]
            temp_embedding = encode_fun(batch_text, model)
            for j in range(len(temp_embedding)):
                writer_str = temp_embedding[j].tolist()
                writer_str = [format(s, '.8f') for s in writer_str]
                writer_str = ','.join(writer_str)
                query_embedding_file.writerow([i + j + 200001, writer_str])

        doc_embedding_file = csv.writer(open('multi_run_macbert/final/doc_embedding', 'w'), delimiter='\t')
        #doc_embedding_file = csv.writer(open('doc_embedding'+str(index), 'w'), delimiter='\t')
        for i in tqdm(range(0, len(corpus), batch_size)):
            batch_text = corpus[i:i + batch_size]
            temp_embedding = encode_fun(batch_text, model)
            for j in range(len(temp_embedding)):
                writer_str = temp_embedding[j].tolist()
                writer_str = [format(s, '.8f') for s in writer_str]
                writer_str = ','.join(writer_str)
                doc_embedding_file.writerow([i + j + 1, writer_str])
