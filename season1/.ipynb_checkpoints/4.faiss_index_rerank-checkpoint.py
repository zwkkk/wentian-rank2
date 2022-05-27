import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import random

# load data
all_text = pd.read_csv('./data/corpus_summar_total_10_train.csv')
doc_id = {}
doc_id = {index:text for index, text in tqdm(enumerate(all_text['doc'].values))}
query_id = {}
query_id = {index:text for index, text in tqdm(enumerate(all_text['query'].values))}

# load embedding
query = pd.read_csv('multi_run_macbert/train_query_embedding', sep='\t', header=None, names=['index', 'embeddings'])
tmp = query['embeddings'].apply(lambda x: np.array([float(x) for x in x.split(',')]))
query_embedding = np.zeros((len(query), 128))
for i in tqdm(range(len(query_embedding))):
    query_embedding[i] = tmp[i]
query_embedding = query_embedding.astype(np.float32)

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
num_k = 40
knn_indices = get_knn(references, querys, num_k, 
                      embeddings_come_from_same_source)


for k in range(num_k):
    if k!=39:
        continue
    labels = []
    for index in knn_indices:
        label = reference_labels[index[k]]
        labels.append(label)
    
    query['knn_{}'.format(k)] = labels
    query['negative_{}'.format(k)] = query['knn_{}'.format(k)].apply(lambda x: doc_id[x])

query['query'] =all_text['query']
#query['negative'] = query['knn'].apply(lambda x: doc_id[x])
query['doc'] = all_text['doc']

query.to_csv('data/hard_negative.csv', index=False)
df = query[['query', 'doc', 'negative_39']]
df.columns = ['query', 'doc', 'negative']
#train = df.sample(int(len(df)*0.9), replace=False, random_state=1)
#test_index = [i for i in range(len(df)) if i not in train.index.tolist() ]
#test = df.loc[test_index]
#train.to_csv('data/train_hard_negative.csv', index=False)
#test.to_csv('data/test_hard_negative.csv', index=False)
#print("train nums: {}, test nums: {}".format(len(train), len(test)))

df[:-5000].to_csv('data/train_hard_negative.csv', index=False)
df[-5000:].to_csv('data/test_hard_negative.csv', index=False)
#query[['query', 'doc', 'negative']].to_csv('save.csv', index=False)
print("[INFO]Done.")
