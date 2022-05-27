export CUDA_VISIBLE_DEVICES=1,2,3
NUM_GPU=3
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path "fnlp/bart-large-chinese" \
    --train_file "../data/query_doc_total.csv" \
    --validation_file "../data/query_doc_total_test.csv" \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy steps \
    --num_train_epochs 3 \
    --source_lang zh_CN \
    --target_lang zh_CN \
    --output_dir './result/fnlp/' \
    --save_total_limit 3 \
    --per_device_train_batch_size 80 \
    --per_device_eval_batch_size 80 \
    --cache_dir '/data1/miaopu/bowen_distance/cache/' \
    --overwrite_output_dir \
    --predict_with_generate \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 100 \
    --max_source_length 156 \
    --max_target_length 156 \
    --generation_max_length 156 \
    --preprocessing_num_workers 12 \
    --fp16 \
    "$@"
