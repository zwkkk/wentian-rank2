import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import random
import csv
# load data
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

q_dict = read_corpus('./data/raw_data/train.query.txt','data/CPR_data/data/ecom/train.query.txt')
with open('./data/raw_data/qrels.train.tsv', encoding="utf-8") as f:
    lines1 = f.readlines()
with open('./data/CPR_data/data/ecom/qrels.train.tsv', encoding="utf-8") as f:
    lines2 = f.readlines()
    
q_txt = []
for line in tqdm(lines1):
    q_id = int(line.split('\t')[0])
    #print(line)
    #print(q_id)
    q = q_dict[q_id]
    q_txt.append(q)
    
for line in tqdm(lines2):
    q_id = int(line.split('\t')[0])+100000
    q = q_dict[q_id]
    q_txt.append(q)
    
# doc 需要总的corpus库
with open('data/raw_data/corpus.tsv', encoding="utf-8") as f:
    lines = f.readlines()

doc_id = {}
doc_id = {index:text.strip().split('\t')[-1] for index, text in tqdm(enumerate(lines))}
query_id = {}
query_id = {index:text for index, text in tqdm(enumerate(q_txt))}

# load embedding
query = pd.read_csv('multi_run_macbert/rerank_query_embedding', sep='\t', header=None, names=['index', 'embeddings'])
tmp = query['embeddings'].apply(lambda x: np.array([float(x) for x in x.split(',')]))
query_embedding = np.zeros((len(query), 128))
for i in tqdm(range(len(query_embedding))):
    query_embedding[i] = tmp[i]
query_embedding = query_embedding.astype(np.float32)

# corpus库对应的embedding
doc = pd.read_csv('multi_run_macbert/doc_embedding0', sep='\t', header=None, names=['index', 'embeddings'])
tmp = doc['embeddings'].apply(lambda x: np.array([float(x) for x in x.split(',')]))
doc_embedding = np.zeros((len(doc), 128))
for i in tqdm(range(len(doc_embedding))):
    doc_embedding[i] = tmp[i]
doc_embedding = doc_embedding.astype(np.float32)

def get_knn(reference_embeddings, test_embeddings, k,
            embeddings_come_from_same_source=False):
    """
    Finds the k elements in reference_embeddings that are closest to each
    element of test_embeddings.
    Args:
        reference_embeddings: numpy array of size (num_samples, dimensionality).
        test_embeddings: numpy array of size (num_samples2, dimensionality).
        k: int, number of nearest neighbors to find
        embeddings_come_from_same_source: if True, then the nearest neighbor of
                                         each element (which is actually itself)
                                         will be ignored.
    """
    d = reference_embeddings.shape[1]
    print("[INFO]running k-nn with k=%d"%k)
    print("[INFO]embedding dimensionality is %d"%d)
    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0: #GPU
        print("[INFO]faiss gpu: {}".format(faiss.get_num_gpus()))
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings) #
    _, indices = index.search(test_embeddings, k + 1)
    if embeddings_come_from_same_source:
        return indices[:, 1:]
    return indices[:, :k]
  

#
#query, reference, query_labels, reference_labels = feats, feats, labels, labels
embeddings_come_from_same_source = False
#
querys = query_embedding
references = doc_embedding
query_labels = range(len(query))
reference_labels = range(len(references))
# 取top 100
num_k = 100
knn_indices = get_knn(references, querys, num_k, 
                      embeddings_come_from_same_source)


for k in range(num_k):
    labels = []
    for index in knn_indices:
        label = reference_labels[index[k]]
        labels.append(label)
    
    query['knn_{}'.format(k)] = labels
    #query['negative_{}'.format(k)] = query['knn_{}'.format(k)].apply(lambda x: doc_id[x])

query['query'] = q_txt
query.to_csv('data/hard_negative.csv', index=False)
#query[['query', 'doc', 'negative']].to_csv('save.csv', index=False)
print("[INFO]Done.")
