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
paths = os.listdir('../result/macbert_result10')
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


if __name__ == '__main__':
    final_path = os.path.join('../result/macbert_result10', 'checkpoint-'+str(idxs[-1]))
    print("load checkpoints from {}".format(final_path))
    tokenizer = AutoTokenizer.from_pretrained(final_path+'/')
    model = BertForCL.from_pretrained(final_path)
    model.to(device)
    query = [line[0] for line in csv.reader(open("../data/corpus_summar_total_10_train.csv"), delimiter=',')]

    query_embedding_file = csv.writer(open('train_query_embedding', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            query_embedding_file.writerow([i + j + 1, writer_str])

