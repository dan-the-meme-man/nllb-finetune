import os

from transformers import AutoTokenizer

from make_tokenizer import lang_code_to_lang_token as c2t

dev_dir = os.path.join('proj_data_final', 'dev')

longest = 0

for file in os.listdir(dev_dir):

    with open(os.path.join(dev_dir, file), 'r', encoding='utf-8') as f:
        
        lang_code = file.split('.')[0]
        
        es = []
        other = []
        
        lines = f.readlines()
        for line in lines:
            es_line, other_line = line.split('\t')
            es.append(es_line.strip())
            other.append(other_line.strip())

        tokenizer = AutoTokenizer.from_pretrained(
            'facebook/nllb-200-distilled-600M',
            src_lang='spa_Latn',
            tgt_lang=c2t[lang_code],
            use_fast=True,
            return_tensors='pt',
            padding=False,
            truncation=False
        )
        
        for i in range(len(es)):
            tokenized = tokenizer(
                text = es[i],
                text_target = other[i],
                return_tensors = 'pt',
                padding=False,
                truncation=False
            )
            input_ids = tokenized['input_ids'][0]
            labels = tokenized['labels'][0]
            longer = max(len(tokenized['input_ids'][0]), len(tokenized['labels'][0]))
            if longer > longest:
                longest = longer

print(longest)