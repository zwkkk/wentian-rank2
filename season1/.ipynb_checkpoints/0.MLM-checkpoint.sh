export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#cd pretrained_model
#rm -rf MLM
#cd ..
#python3 0.run_language_model_roberta.py \
#        --output_dir=pretrained_model/MLM_nezha-large-wwm     --model_type=bert     --model_name_or_path=pretrained_model/nezha-large-wwm \
#        --do_train     --train_data_file=data/MLM.txt     --do_eval     --eval_data_file=data/MLM.txt \
#        --mlm --per_device_train_batch_size=64  --per_device_train_batch_size=64 --num_train_epochs=10 --save_steps=5000

python3 0.run_language_model_roberta.py \
        --output_dir=pretrained_model/MLM_external_plusJD    --model_type=bert     --model_name_or_path=pretrained_model/MLM_external \
        --do_train     --train_data_file=data/external_data/for_pretrain/JD.txt \
        --mlm --per_device_train_batch_size=64 --num_train_epochs=10 --save_steps=5000
        

#python3 0.run_language_model_roberta.py \
#        --output_dir=pretrained_model/MLM_macbert_external     --model_type=bert     --#model_name_or_path=pretrained_model/MLM_macbert \
#        --do_train     --train_data_file=data/external_data/for_pretrain/MLM.txt --do_eval     --#eval_data_file=data/external_data/for_pretrain/MLM.txt \
#        --mlm --per_device_train_batch_size=64  --per_device_train_batch_size=64 --num_train_epochs=10 --#save_steps=5000

#python3 0.run_language_model_roberta.py \
#        --output_dir=pretrained_model/MLM_25_bert   --model_type=bert     --model_name_or_path=pretrained_model/bert_large_chinese \
#        --do_train     --train_data_file=data/MLM_all.txt --mlm --per_device_train_batch_size=64  --num_train_epochs=50 --#save_steps=10000
