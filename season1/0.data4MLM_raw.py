# -*- encoding: utf-8 -*-

import time

from tqdm import tqdm
import pandas as pd
import random
from sklearn.model_selection import train_test_split


def timer(func):
    """ time-consuming decorator 
    """
    def wrapper(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()
        print(f"function: `{func.__name__}` running time: {te - ts:.4f} secs")
        return res
    return wrapper


qrels = pd.read_csv("data/raw_data/qrels.train.tsv", sep='\t', header=None, names=['query_id', 'doc_id'])
corpus = pd.read_csv("data/raw_data/corpus.tsv", sep='\t', header=None, names=['doc_id', 'title'])

with open('data/raw_data/train.query.txt', encoding="utf-8") as f:
    train_querys = f.readlines()

    
# query_id query
train_query = []
for line in train_querys:
    line = line.strip().split('\t')
    train_query.append(line[1])

corpus = corpus['title'].values


@timer
def doc_preprocess(src_path: str, dst_path:str) -> None:
    """处理数据
    Args:
        src_path (str): 原始文件地址
        dst_path (str): 输出文件地址
    """
    # 写文件
    with open(dst_path, 'w') as writer:
        for line in tqdm(train_query):
            writer.write(line)
            writer.write('\n')
        for line in tqdm(corpus):
            writer.write(line)
            writer.write('\n')
            
    with open('./data/raw_data_process/MLM_eval.txt', 'w') as writer:
        num = 0
        for line in tqdm(train_query):
            if num % 20 ==0:
                num += 1
                writer.write(line)
                writer.write('\n')
        num = 0
        for line in tqdm(corpus):
            if num % 20 ==0:
                num += 1
                writer.write(line)
                writer.write('\n')
        
            
if __name__ == '__main__':
    d_src, d_dst = '_', './data/raw_data_process/MLM.txt'
    doc_preprocess(d_src, d_dst)
 