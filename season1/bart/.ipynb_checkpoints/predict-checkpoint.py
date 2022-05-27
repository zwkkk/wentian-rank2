import csv
import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, BartForConditionalGeneration


class summarization:
    def __init__(self, model_path="fnlp/bart-base-chinese", cache_dir='/data1/miaopu/bowen_distance/cache/',
                 device="cuda:2"):
        print(model_path)
        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-large-chinese")
        if model_path == "fnlp/bart-base-chinese" or model_path == "fnlp/bart-large-chinese":
            self.model = BartForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model = self.model.to(device).eval()
        self.device = device

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", return_token_type_ids=False)
        inputs.to(self.device)
        zh_outputs = self.model.generate(**inputs, num_beams=4, num_return_sequences=4)
        zh_text = self.tokenizer.batch_decode(zh_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        zh_text = list(set(zh_text))
        return zh_text

    def batch_generate(self, text,num_return_sequences=5):
        batch_size = len(text)
        inputs = self.tokenizer.batch_encode_plus(text, return_tensors="pt", padding=True, truncation=True,
                                                  max_length=200, return_token_type_ids=False)
        inputs.to(self.device)
        zh_outputs = self.model.generate(**inputs, num_beams=num_return_sequences, num_return_sequences=num_return_sequences)
        zh_text = self.tokenizer.batch_decode(zh_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print(zh_text)
        # exit()
        '''
        for t in range(len(zh_text)):
            zh_text[t] = zh_text[t].replace(" ","")
            # print(zh_text[t])
        '''
        num_each = int(len(zh_text) // batch_size)
        return_list = []
        for i in range(batch_size):
            return_list.append(zh_text[i * num_each:(i + 1) * num_each])

        return return_list


if __name__ == '__main__':
    file_path = '../data/corpus_total.tsv'
    writer_file = '../data/corpus_summar_total.csv'
    model_path = '/data1/miaopu/EmbeddingTest/query_doc/summarization/result/fnlp/'
    print('loading')
    summarizationer = summarization(model_path=model_path, cache_dir='/data1/miaopu/bowen_distance/cache/',
                                    device="cuda:1")
    print('load over')
    reader = csv.reader(open(file_path), delimiter='\t')
    writer = csv.writer(open(writer_file, 'w'), delimiter='\t')
    lines = [line[1] for line in reader]
    batch_size = 150
    tqbar = tqdm.tqdm(total=len(lines))
    for i in range(0, len(lines), batch_size):
        temp_line = lines[i:i + batch_size]
        result = summarizationer.batch_generate(temp_line)
        for j in range(len(temp_line)):
            writer.writerow([temp_line[j], result[j]])
        tqbar.update(batch_size)
