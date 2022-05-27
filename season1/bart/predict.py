import csv
import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, BartForConditionalGeneration

def generate_summary(lines, device):
    tokenizer = BertTokenizer.from_pretrained("result/")
    model = BartForConditionalGeneration.from_pretrained("result/").to(device).eval()
    writer = csv.writer(open(writer_file, 'w'), delimiter='\t')
    tqbar = tqdm.tqdm(total=len(lines))
    for i in range(0, len(lines), batch_size):
        text = lines[i:i + batch_size]
        
        inputs = tokenizer.batch_encode_plus(text, return_tensors="pt", padding=True, truncation=True,
                                                  max_length=108, return_token_type_ids=False)
        inputs.to(device)
        outputs = model.generate(**inputs, num_beams=10, num_return_sequences=10)
        text_generate = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        num_each = int(len(text_generate) // batch_size)
        return_list = []
        for k in range(batch_size):
            return_list.append(text_generate[k * num_each:(k + 1) * num_each])
            
        for j in range(len(text)):
            writer.writerow([text[j], return_list[j]])
        tqbar.update(batch_size)



if __name__ == '__main__':
    file_path_CPR = '../data/CPR_data/data/ecom/corpus.tsv'
    file_path_raw = '../data/raw_data/corpus.tsv'
    writer_file = '../data/bart_10.csv'
    model_path = 'result/'

    reader_CPR = csv.reader(open(file_path_CPR), delimiter='\t')
    reader_raw = csv.reader(open(file_path_raw), delimiter='\t')
    
    lines  = [line[1] for line in reader_CPR] + [line[1] for line in reader_raw]
    batch_size = 100
    generate_summary(lines, "cuda:0")