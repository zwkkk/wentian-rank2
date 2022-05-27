import pandas as pd
from tqdm import tqdm
import random
import os
random.seed(2022)

# corpus汇总
with open('./data/raw_data/corpus.tsv', encoding="utf-8") as f:
    lines1 = f.readlines()

with open('./data/CPR_data/data/ecom/corpus.tsv', encoding="utf-8") as f:
    lines2 = f.readlines()

idx = 1
# 总的corpus raw data 编号从1开始到1001500 CPR data 编号从1001501开始到2004322
with open('./data/season2_process/corpus.tsv', 'w', encoding="utf-8") as f:
    for line in tqdm(lines1+lines2):
        _, txt = line.split('\t')
        f.write(str(idx)+'\t'+txt)
        idx += 1
        
# query汇总
with open('./data/raw_data/train.query.txt', encoding="utf-8") as f:
    lines1 = f.readlines()

with open('./data/CPR_data/data/ecom/train.query.txt', encoding="utf-8") as f:
    lines2 = f.readlines()
    
'''bad case 
403\t内六角\td-0.9单支平头公制\t日本eight百利\n
403	内六角	d-0.9单支平头公制	日本eight百利

1145	佛前供水碗白铜吉祥花高	 脚雕花观音圣水杯佛堂手工净水碗大号

36875	标准量块	哈量83块（0级）

66145	人民邮电出版社	isbn: 	9787115331410

68383	杨红樱	《漂亮老师和坏小子》作家出版社
'''

idx = 1
# 总的corpus raw data 编号从1开始到100000 CPR data 编号从100001开始到200000
with open('./data/season2_process/train.query.txt', 'w', encoding="utf-8") as f:
    for line in tqdm(lines1+lines2):
        _, txt = line.split('\t',1) # 只切一次
        f.write(str(idx)+'\t'+txt.replace('\t', ' '))
        idx += 1
        
# 放到 trainset

if os.path.exists('../season2/data/trainset/x0.out'):
    os.remove('../season2/data/trainset/x0.out')
if os.path.exists('../season2/data/trainset/x1.out'):
    os.remove('../season2/data/trainset/x1.out')
if os.path.exists('../season2/data/trainset/x2.out'):
    os.remove('../season2/data/trainset/x2.out')


# hard nagetive
df = pd.read_csv('data/hard_negative.csv')
columns = df.columns
neg_index = [i for i in columns if 'knn' in i]
hards = df.loc[:,neg_index]


with open('./data/raw_data/qrels.train.tsv', encoding="utf-8") as f:
    lines1 = f.readlines()
with open('./data/CPR_data/data/ecom/qrels.train.tsv', encoding="utf-8") as f:
    lines2 = f.readlines()

epochs = 3
for epoch in range(epochs):
    num = 0
    with open('../season2/data/trainset/x{}.out'.format(epoch), 'w', encoding="utf-8") as f:
        for line in tqdm(lines1):
            line = line.strip().split('\t')
            qid = line[0]
            did = line[1]
            candidate = [str(int(i+1)) for i in list(hards.loc[num].values)]
            num += 1
            if epoch==0:
                can = candidate[40:46]
                if did in can:
                    can.remove(did)
                    can.append(candidate[47])
                row = qid + '\t' + did + '\t' + "#".join(can) + '\n'
            elif epoch==1:
                can = candidate[20:26]
                if did in can:
                    can.remove(did)
                    can.append(candidate[27])
                row = qid + '\t' + did + '\t' + "#".join(can) + '\n'
            elif epoch==2:
                can = candidate[5:11]
                if did in can:
                    can.remove(did)
                    can.append(candidate[12])
                row = qid + '\t' + did + '\t' + "#".join(can) + '\n'
            f.write(row)

        for line in tqdm(lines2):
            line = line.strip().split('\t')
            qid = str(int(line[0]) + 100000)
            did = str(int(line[2]) + 1001500)
            candidate = [str(int(i+1)) for i in list(hards.loc[num].values)]
            num += 1
            if epoch==0:
                can = candidate[40:46]
                if did in can:
                    can.remove(did)
                    can.append(candidate[47])
                row = qid + '\t' + did + '\t' + "#".join(can) + '\n'
            elif epoch==1:
                can = candidate[20:26]
                if did in can:
                    can.remove(did)
                    can.append(candidate[27])
                row = qid + '\t' + did + '\t' + "#".join(can) + '\n'
            elif epoch==2:
                can = candidate[5:11]
                if did in can:
                    can.remove(did)
                    can.append(candidate[12])
                row = qid + '\t' + did + '\t' + "#".join(can) + '\n'
            f.write(row)


    