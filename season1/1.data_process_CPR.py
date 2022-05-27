import csv
from random import shuffle
from tqdm import tqdm


def read_corpus(file_path='./data/CPR_data/data/ecom/corpus.tsv'):
    reader = csv.reader(open(file_path), delimiter='\t')
    total_dict = dict()
    for line in reader:
        corpus_id = int(line[0])
        corpus = line[1]
        total_dict[corpus_id] = corpus
    return total_dict


def make_qrels(query_dict, corpus_dict,
               qrels_file='./data/CPR_data/data/ecom/qrels.train.tsv',
               writer_file='./data/CPR_data_process/query_doc.csv',
               test_file='./data/CPR_data_process/query_doc_test.csv',
               test_num=500,
               ):
    reader = csv.reader(open(qrels_file), delimiter='\t')
    writer = csv.writer(open(writer_file, 'w'))
    test_writer = csv.writer(open(test_file, 'w'))
    reader = [line for line in reader]
    shuffle(reader)
    train_lines = reader[:-test_num]
    test_lines = reader[-test_num:]
    print(len(train_lines),len(test_lines))
    max_len = 0

    writer.writerow(['query', 'doc'])
    test_writer.writerow(['query', 'doc'])

    for line in tqdm(train_lines):
        q_id = int(line[0])
        v_id = int(line[2])
        q = query_dict[q_id]
        v = corpus_dict[v_id]
        if v== '"':
            continue
        writer.writerow([q, v])
        max_len = max(len(q), max_len)
        max_len = max(len(v), max_len)

    for line in tqdm(test_lines):
        q_id = int(line[0])
        v_id = int(line[2])
        q = query_dict[q_id]
        v = corpus_dict[v_id]
        if v== '"':
            continue
        test_writer.writerow([q, v])

        max_len = max(len(q), max_len)
        max_len = max(len(v), max_len)

    print(max_len)


if __name__ == '__main__':
    corpus_dict = read_corpus() # id: txt
    query_dict = read_corpus('./data/CPR_data/data/ecom/train.query.txt') # id:txt
    make_qrels(query_dict, corpus_dict,test_num=1000)
