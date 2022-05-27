# query
with open('./embedding/final/query_embedding', encoding="utf-8") as f:
    q_emb = f.readlines()
    
with open('./embedding/final/query_ids', encoding="utf-8") as f:
    q_ids = f.readlines()

with open('./embedding/final/season2/query_embedding', 'w', encoding="utf-8") as f:
    num_q = len(q_ids)
    for i in range(num_q):
        loc1 = q_emb[i].strip()
        loc2 = q_ids[i].split('\t')[-1]
        f.write(loc1+'\t'+loc2)
# doc
with open('./embedding/final/doc_embedding', encoding="utf-8") as f:
    d_emb = f.readlines()
    
with open('./embedding/final/doc_idx', encoding="utf-8") as f:
    d_ids = f.readlines()

with open('./embedding/final/season2/doc_embedding', 'w', encoding="utf-8") as f:
    num_d = len(d_ids)
    for i in range(num_d):
        loc1 = d_emb[i].strip()
        loc2 = d_ids[i].split('\t')[-1]
        f.write(loc1+'\t'+loc2)