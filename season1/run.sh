export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 将数据转换，用于MLM训练
cd data
mkdir raw_data_process
mkdir CPR_data_process
cd ..
python3 0.convert4MLM_CPR.py
python3 0.data4MLM_raw.py
# MLM训练
sh 0.MLM.sh # train 12*2=24h

# 将数据转换，用于主模型训练
python3 1.data_process_CPR.py
python3 1.data_process_raw.py
python3 1.5.data_process_combine.py

# 数据复制到bart文件夹
cd bart
mkdir data
cd ..
cd data
cp query_doc_all.csv query_doc_test_all.csv ../bart/data  # 复制一份到 bart文件夹 用于训练bart
cd ..

# bart 数据生成
cd bart
mkdir result
sh run.sh  # train: 41 min，生成 17h
cd ..
# bart生成的数据集处理+合并有监督数据，生成最终的训练集
python3 1.6.data_process_combine_final.py

# 模型训练
python3 2.preprosess.py  #训练前的数据处理
sh train.sh  # macbert-based model训练 约28h

# 生成embedding
sh predict.sh
#cd multi_run_macbert
#mkdir final
#cd ..
#python3 3.get_embedding_macbert.py # 生成初赛格式embedding，存于multi_run_macbert/final文件夹下

# 生成复赛embedding
mkdir multi_run_macbert/final/season2
python3 3.get_ids.py # 生成复赛的input token
cd multi_run_macbert
python3 season2_embedding.py # 生成最终复赛embedding，存于multi_run_macbert/final/season2文件夹下

# 建立精排模型 hard negative
python3 get_rerank_embedding_macbert.py  # 建立训练集query的embedding，形成文件 rerank_query_embedding
cd ..
python3 4.faiss_index_rerank.py # 生成season2 训练集query的 topk个hard negative

# 生成复赛训练集 数据集
cd ..
cd season2
mkdir submit
cd submit
mkdir bert_submit
cd ..
mkdir data
cd data
mkdir trainset
cd ..
cd ..
cd season1
mkdir data/season2_process
python3 5.make_data_process.py
python3 6.make_tokeninputs.py

# 复赛要提交的embedding复制到season2/submit/bert_submit文件夹下
cp multi_run_macbert/final/season2/doc_embedding ../season2/submit/bert_submit
cp multi_run_macbert/final/season2/query_embedding ../season2/submit/bert_submit



