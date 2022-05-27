## 写在前面  
环境配置和运行过程有任何问题，联系：  
手机：18628026406（微信同号）  
邮箱：396635841@qq.com  

队伍名：诗人藏夜里.  
复赛排名：2  
复赛成绩：0.3890  
备注：1. 召回侧训练步骤较多，换机子运行对最终结果会存在0.00几的波动。  2.提交的代码除了原始数据放置在season1/data文件下意外，其他预训练模型，中间过程数据/文件均已删除。  3.最终代码提交版对过程做了稍许优化调整，在小数据集测试通过，若存在运行问题，请及时联系我，谢谢🙏。  
  
代码讲解视频：链接: https://pan.baidu.com/s/1vf1XM9MqiIbIgJrPJgYpJA 提取码: 3kdo  
    
方案快速讲解视频：链接: https://pan.baidu.com/s/14Id8WQYiDx6GDAvhC5AxtA 提取码: 2nah  
  
过程中生成的文件可参考：链接: https://pan.baidu.com/s/1uqpB7jdYeTw7BWloZ5O1bA 提取码: dhvf ，主要为season1/pretrained_model/MLM_macbert_external,season1/bart/result,season1/result/macbert_result下的模型文件。   
  
外部数据地址：https://github.com/Alibaba-NLP/Multi-CPR  

## 环境配置与运行指南  
### season1  
0 运行要求  
GPU：A100 * 8 （必须A100）
CUDA: 11.4  
CUDNN: 8  
整体运行时长：召回侧约70h  

cd season1  
1 安装虚拟环境  
conda create -n season1 python=3.7  
conda activate season1 # 进入虚拟环境
2 安装pytorch  
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  
3 apex 安装  
git clone https://www.github.com/nvidia/apex # 若网不好，可以先下载https://www.github.com/nvidia/apex到本地  
cd apex  
python3 setup.py install  
cd ..  
4 sentval 安装  
git clone https://github.com/cqulun123/SentEval.git  # 若网不好 可以先下载到本地  
cd sentval  
python3 setup.py install  
cd ..  
5 其他库安装  
python3 -m pip install -r requirements.txt  
6 预训练模型下载  
chinese-macbert-large： https://huggingface.co/hfl/chinese-macbert-large/tree/main，下载config.json,pytorch_model.bin,vocab.txt放置到pretrained_model/chinese-macbert-large下  
bart-large-chinese： https://huggingface.co/fnlp/bart-large-chinese/tree/main，下载config.json,pytorch_model.bin,vocab.txt放置到bart/bart-large-chinese下  
7 season1代码运行  
sh run.sh  

### season2  
0 运行要求  
GPU：P40 * 1  
CUDA: 9.0  
CUDNN: 7  
整体运行时长：精排侧约18h  
cd season2  
1 安装虚拟环境  
conda create -n season2 python=3.6    
conda activate season2 # 进入虚拟环境  
2 ubuntu20.04安装cuda9环境可参考  
https://zwk.notion.site/tensorflow1-12-21efa645c6ec4a7d82460368de06172f  
3 库安装  
python3 -m pip install -r requirements.txt  
4 预训练模型下载  
chinese_bert_wwm_L-12_H-768_A-12： https://github.com/ymcui/Chinese-BERT-wwm下载【BERT-wwm-ext, Chinese】对应的tensorflow文件，下载网址：https://pan.baidu.com/s/1x-jIw1X2yNYHGak2yiq4RQ?pwd=wgnt  
下载完放置到pretrained_model文件夹  
5 模型训练  
sh run.sh

## 目录结构
-- season1  
---- pretrained_model  预训练模型地址  
-------- chinese-macbert-large    
---- simcse simcse模型文件  
---- data  数据信息  
-------- raw_data 存放tianchi官网原始数据  
-------- CPR_data 存放Multi-CPR数据  
-------- raw_data_process 存放raw_data处理后的数据  
-------- CPR_data_process 存放CPR_data处理后的数据  
-------- season2_process 为复赛的数据准备    
---- 0.data4MLM_raw.py raw_data生成用于MLM的训练集/测试集  生成文件'./data/raw_data_process/MLM.txt'，'./data/raw_data_process/MLM_eval.txt'分别为MLM的训练集，测试集  
---- 0.convert4MLM_CPR.py CPR_data生成用于MLM的训练集 生成文件'./data/CPR_data_process/MLM.txt'  
---- 0.run_language_model_roberta.py  MLM训练  
---- 0.MLM.sh MLM训练入口  
---- 1.data_process_raw.py 利用raw_data生成query-doc对，输入训练模型  生成文件'./data/raw_data_process/query_doc.csv'，'./data/raw_data_process/query_doc_test.csv'分别为训练集，测试集    
---- 1.data_process_CPR.py 利用CPR_data生成query-doc对，输入训练模型  生成文件'./data/CPR_data_process/query_doc.csv'，'./data/CPR_data_process/query_doc_test.csv'分别为训练集，测试集  
---- 1.5.data_process_combine.py 将raw_data和CPR_data生成的训练集，测试集合并，生成文件'./data/query_doc_all.csv'，'./data/query_doc_test_all.csv'  
---- bart  用bart进行数据扩增    
-------- bart-large-chinese bart预训练模型    
-------- data bart训练数据  
-------- train.py bart训练代码  
-------- predict.py bart生成代码  
-------- result bart训练后的模型 用于数据生成  
-------- run.sh 开启bart训练,并根据corpus生成数据, 生成数据地址'./data/bart_10.csv'  
---- 1.6.data_process_combine_final.py bart生成数据处理+合并有监督数据集和生成数据集，形成最终用于训练的数据集，生成数据地址'./data/bart_10_train.csv'  
---- 2.preprosess.py 数据输入模型前的最后清洗  
---- 2.train.py 模型训练    
---- train.sh macbert模型训练入口  
---- predict.sh 用于推理并生成初赛提交格式的embedding     
---- multi_run_macbert embedding处理文件夹  
-------- get_rerank_embedding_macbert.py 得到训练集query的embedding，用于faiss取hard negative  
-------- season2_embedding.py 用于得到复赛提交的embedding  
---- 3.get_embedding_macbert.py  生成初赛格式embedding，存于multi_run_macbert/final文件夹下   
---- 3.get_ids.py 生成复赛query/doc的token  
---- 4.faiss_index_rerank.py 建立复赛训练集query的 hard negative  
---- 5.make_data_process.py 为复赛数据作准备  
---- 6.make_tokeninputs.py 生成复赛训练集 train.query.json corpus.json  
---- run.sh 总流程入口  
---- requirements.txt 依赖安装  

-- season2  
---- code 总代码文件夹  
-------- bert bert建模的文件  
-------- pretrained_model  
------------ chinese_bert_wwm_L-12_H-768_A-12 bert_base 预训练模型  
-------- result 模型训练结果文件夹   
-------- data.py 数据处理文件  
-------- rank_model.py 精排模型  
-------- trainer.py 模型训练  
-------- train.sh 模型训练代码入口  
-------- wrapper.py 模型打包  
-------- predict.sh 生成复赛模型  存放到submit/bert_submit文件夹下  
-------- run.sh 总程序入口  
---- data 总数据文件夹  
-------- trainset 总的训练集文件夹  
-------- corpus.json  corpus的token格式  
-------- train.query.json query的token格式  
---- submit 复赛提交的文件夹  
---- requirements.txt 依赖安装  
   






 

