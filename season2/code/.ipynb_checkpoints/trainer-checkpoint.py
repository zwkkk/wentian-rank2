import os
import json
import argparse
import tensorflow as tf
import time 
import random
from datetime import datetime

from data import DataProcessor
from bert import modeling
from rank_model import Ranker

seed_value = 2022
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)
     
class Trainer(object):
    def __init__(self, args):
         self.args = args
         self.bert_config_path = args.bert_config_path
         self.batch_size = args.batch_size
         self.num_epochs = args.num_epochs
         self.warmup_proportion = args.warmup_proportion
         self.max_seq_length = args.max_seq_length
         self.learning_rate = args.learning_rate
         self.bert_model_path = args.bert_model_path
         self.processor = DataProcessor(args.corpus_ids_file_path, args.bert_vocab_path)

    def create_model(self):
        model = Ranker(self.bert_config_path, is_training=self.args.is_training, num_train_steps=self.num_train_steps, num_warmup_steps=self.num_warmup_steps, learning_rate=self.learning_rate)
        return model

    def train(self):
        train_examples = self.processor.get_train_examples(self.args.query_ids_file_path, self.args.trainset_dir, 0)
        self.num_train_steps = int(len(train_examples) / self.batch_size * self.num_epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion) 
    
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        tf.logging.info("***** Running training *****")
        tf.logging.info(" Num examples = %d", len(train_examples))
        tf.logging.info(" Batch size = %d", self.batch_size)
        tf.logging.info(" Num steps = %d", self.num_train_steps)
        
        num_batches = len(train_examples) // self.batch_size
        self.model = self.create_model()

        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.bert_model_path)
            tf.train.init_from_checkpoint(self.bert_model_path, assignment_map) 
            tf.logging.info("***** Trainable Variables *****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                #tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)
            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0
            starttime = time.time()
            for i in range(self.num_epochs):
                print("***** epoch-{} *******".format(i))
                if i > 0:
                    tf.logging.info("**** get train_examples ****")
                    train_examples = self.processor.get_train_examples(self.args.query_ids_file_path, self.args.trainset_dir, i)
                    random.shuffle(train_examples) 
                features = self.processor.get_features(train_examples, max_seq_length=self.max_seq_length)
                input_ids_lst, input_mask_lst, token_type_ids_lst, label_ids_lst = self.processor.get_inputs(features)
                for j in range(num_batches):
                    start = j * self.batch_size
                    end = start + self.batch_size
                    batch_features = {"input_ids": input_ids_lst[start: end], "input_mask": input_mask_lst[start: end], "token_type_ids": token_type_ids_lst[start: end], "label_ids": label_ids_lst[start: end]}
                    loss, logits, prob = self.model.train(sess, batch_features)
                    if current_step % self.args.eval_steps == 0:
                        tf.logging.info("*****【%s】, training_step: %d 【%s】, loss: %f", datetime.now().strftime("%H:%M:%S"), current_step, str(100*current_step/self.num_train_steps)+"%", loss)
                    current_step += 1
                    if current_step % self.args.save_steps == 0:
                        tf.logging.info("***** saving model to %s ****", self.args.output_dir)
                        ckpt_name_prefix = "models"
                        save_path = os.path.join(self.args.output_dir, ckpt_name_prefix)
                        self.model.saver.save(sess, save_path, global_step=current_step)
            tf.logging.info("total training time: %f", time.time()-starttime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_path", type=str, default='./pretrained_model/chinese_bert_wwm_L-12_H-768_A-12/bert_model.ckpt',help="The path of bert_model")
    parser.add_argument("--bert_config_path", type=str, default='./pretrained_model/chinese_bert_wwm_L-12_H-768_A-12/bert_config.json', help="The path of bert_config file")
    parser.add_argument("--bert_vocab_path", type=str, default='./pretrained_model/chinese_bert_wwm_L-12_H-768_A-12/vocab.txt', help="The path of vocab.")
    parser.add_argument("--output_dir", type=str, default='./result/bert_base_hard', help="The path of checkpoint you want to save")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--warmup_proportion", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=110)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--query_ids_file_path", type=str, default="../data/train.query.json", help="It's a json file, the result of converting text to ids, and the format is {'qid': xx, 'input_ids': xx}")
    parser.add_argument("--trainset_dir", type=str, default="../data/trainset", help="The directory of trainset, which contains queryid, golden_id, a series of docids.")
    parser.add_argument("--corpus_ids_file_path", type=str, default="../data/corpus.json", help="Its format is the same as query_ids_file_path.") 
    parser.add_argument("--is_training", type=bool, default=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    tf.logging.set_verbosity(tf.logging.INFO)
    trainer = Trainer(args)
    trainer.train()
