#!/bin/bash
rm -rf query_embedding doc_embedding foo.tar.gz
# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID  2.train.py \
    --model_name_or_path pretrained_model/MLM_macbert_external \
    --train_file "./data/corpus_summar_total_25_train.csv" \
    --validation_file "./data/query_doc_test.csv" \
    --output_dir ./result/MLM_macbert_external_plus_bart25 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --metric_for_best_model eval_acc \
    --learning_rate 3e-5 \
    --max_seq_length 108 \
    --save_total_limit 20 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --eval_steps 50 \
    --save_steps 250 \
    --logging_steps 10 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_fgm True \
    "$@"
