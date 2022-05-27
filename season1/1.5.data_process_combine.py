# 将external准备的数据 合并到本比赛数据中
#external准备的数据：data/external_data/data/query_doc.csv query_doc_test.csv
#到本比赛数据:data/query_doc.csv query_doc_test.csv

import csv
from tqdm import tqdm

reader_train1 = csv.reader(open('data/raw_data_process/query_doc.csv'), delimiter='\t') 
reader_train2 = csv.reader(open('data/CPR_data_process/query_doc.csv'), delimiter='\t') 
reader_test1 = csv.reader(open('data/raw_data_process/query_doc_test.csv'), delimiter='\t') 
reader_test2 = csv.reader(open('data/CPR_data_process/query_doc_test.csv'), delimiter='\t') 

reader_train1 = [line for line in reader_train1][1:]
reader_train2 = [line for line in reader_train2][1:]
reader_test1 = [line for line in reader_test1][1:]
reader_test2 = [line for line in reader_test2][1:]
print(reader_train1[0])
writer = csv.writer(open('data/query_doc_all.csv', 'w'))
test_writer = csv.writer(open('data/query_doc_test_all.csv', 'w'))

writer.writerow(['query', 'doc'])
test_writer.writerow(['query', 'doc'])    

for line in tqdm(reader_train1):
    line = line[0].split(',')
    q, v = line[0], line[1]
    writer.writerow([q,v])

for line in tqdm(reader_train2):
    line = line[0].split(',')
    q, v = line[0], line[1]
    writer.writerow([q,v])

for line in tqdm(reader_test1):
    line = line[0].split(',')
    q, v = line[0], line[1]
    test_writer.writerow([q,v])

for line in tqdm(reader_test2):
    line = line[0].split(',')
    q, v = line[0], line[1]
    test_writer.writerow([q,v])
