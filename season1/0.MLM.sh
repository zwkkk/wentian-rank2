# MLM macbert in data_raw
python3 0.run_language_model_roberta.py \
        --output_dir=pretrained_model/MLM_macbert    --model_type=bert     --model_name_or_path=pretrained_model/chinese-macbert-large \
        --do_train     --train_data_file=data/raw_data_process/MLM.txt \
        --mlm --per_device_train_batch_size=64  --per_device_train_batch_size=64 --num_train_epochs=25 --save_steps=5000 
# MLM macbert in data_CPR
python3 0.run_language_model_roberta.py \
        --output_dir=pretrained_model/MLM_macbert_external    --model_type=bert     --model_name_or_path=pretrained_model/MLM_macbert \
        --do_train     --train_data_file=data/CPR_data_process/MLM.txt \
        --mlm --per_device_train_batch_size=64  --per_device_train_batch_size=64 --num_train_epochs=25 --save_steps=5000 
