import pandas as pd
import random
from tqdm import tqdm
import numpy as np
import gc 
gc.collect()
random.seed(2022)
# 所有有监督样本
with open('./data/query_doc_all.csv', encoding='utf-8') as f:
    old = f.readlines()

# 所有根据无监督样本生成的数据
tmp = []
with open('./data/bart_10.csv', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        doc, querys = line.split('\t')
    
        qs = [''.join(i[2:-1].strip().split(' ')) for i in querys.strip()[1:-1].split(',')]
        for q in qs:
            tmp.append([q, doc])
            
random.shuffle(tmp)

# 最终的训练集
success = 0
with open('./data/bart_10_train.csv', 'w', encoding='utf-8') as f:
    f.write('query,doc\n')
    for line in tqdm(tmp):
        if line[0]=='':
            continue
        else:
            q = line[0].replace(',', '')
            d = line[1].replace(',', '')
            f.write(q+','+d+'\n')
            success += 1
            
    for line in tqdm(old[1:]):
        f.write(line)
print(success)