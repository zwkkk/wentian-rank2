import os
from tqdm import tqdm

path = './data/CPR_data/data/ecom'
path2 = './data/CPR_data_process/MLM.txt'
name = ['train.query.txt', 'dev.query.txt', 'corpus.tsv']

with open(path2, 'w+', encoding='utf-8' ) as F:
    for n in tqdm(name):
        with open(os.path.join(path, n), encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            idx, line = line.split('\t')
            F.write(line)