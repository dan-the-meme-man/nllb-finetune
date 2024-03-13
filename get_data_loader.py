import os

import random

from torch import manual_seed
from torch.utils.data import DataLoader, Dataset

from make_tokenizer import make_tokenizer, lang_code_to_lang_token as c2t, lang_token_to_id as t2i

random.seed(42)
manual_seed(42)

tokenizers = {lang: make_tokenizer(tgt_lang=lang) for lang in c2t.values()}
    
class ParallelDataset(Dataset):
    def __init__(self, files, batch_size, num_batches):
        
        super().__init__()
        
        # list of tokenized batches, each batch is the same target language
        # but different batches may differ in target language
        # since they are pre-tokenized, they are ready to be used in training without worrying
        # about which tokenizer to use
        self.examples = []
        
        # lists to be sampled from later to build self.examples
        temp = dict.fromkeys(c2t.values())
        temp.pop('spa_Latn', None)
        for lang_token in temp:
            temp[lang_token] = []
        
        # process each file (language) and add list of examples to temp
        for file in files:

            # tokenizer from spanish to this language
            lang_token = c2t[os.path.basename(file).split('.')[0]]
            tokenizer = tokenizers[lang_token]
            assert tokenizer._src_lang == 'spa_Latn'
            assert tokenizer.tgt_lang == lang_token
            
            # lists of spanish and other language sentences
            es_batch = []
            other_batch = []
            
            lines = open(file, 'r', encoding='utf-8').readlines()
            for i in range(len(lines)):
                
                strip_line = lines[i].strip()
                if not strip_line:
                    continue
                split = strip_line.split('\t')
                if len(split) != 2:
                    continue
                
                # accumulate a batch of es sentences and other language sentences
                es_batch.append(split[0].strip())
                other_batch.append(split[1].strip())
                
                if len(es_batch) == batch_size or i == len(lines) - 1:
                    
                    assert len(es_batch) == len(other_batch)
                    
                    # tokenize a batch and append to temp dict
                    tokenized = tokenizers[lang_token](
                        text = es_batch,
                        text_target = other_batch,
                        return_tensors = 'pt',
                        padding = 'max_length',
                        truncation = True,
                        max_length = 1024
                    )
                    temp[lang_token].append(tokenized)
                    
                    # clear batch accumulation
                    es_batch = []
                    other_batch = []
                    
                    # no way we could need this much data
                    if i >= num_batches * batch_size:
                        break
                    
                    if i % 1000 == 999:
                        print(f'Loaded {len(temp[lang_token]) * batch_size}/{len(lines)} lines of {lang_token}.')

        # build pmf using exponential reweighting
        probs = dict.fromkeys(c2t.values())
        probs.pop('spa_Latn', None)
        for lang_token in probs:
            probs[lang_token] = 0
            try:
                probs[lang_token] = len(temp[lang_token])**(-0.4)
            except ZeroDivisionError:
                pass
        total = sum(probs.values())
        probs = {lang_token: probs[lang_token] / total for lang_token in probs}
        
        # sample from pmf until num_batches examples are in self.examples
        lang_token_list = list(c2t.values())
        lang_token_list.remove('spa_Latn')
        weights = [probs[lang_token] for lang_token in lang_token_list]
        for i in range(num_batches):
            
            # randomly select a language based on pmf
            lang_token = random.choices(lang_token_list, weights=weights)[0]
            
            # randomly select an example of that language
            self.examples.append(random.choice(temp[lang_token]))
            
            if i % 1000 == 999:
                print(f'Loaded {i+1}/{num_batches} batches of {split}.')

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)
    
class DevSet(Dataset):
    def __init__(self, lang_code, num_examples=None):
        
        super().__init__()
        
        self.examples = []        
        self.lang_token = c2t[lang_code]
        
        tokenizer = tokenizers[self.lang_token]
        assert tokenizer._src_lang == 'spa_Latn'
        assert tokenizer.tgt_lang == self.lang_token
        
        file_path = os.path.join('proj_data_final', 'dev', lang_code+'.tsv')
        lines = open(file_path, 'r', encoding='utf-8').readlines()
        for i in range(len(lines)):
            
            strip_line = lines[i].strip()
            if not strip_line:
                continue
            split = strip_line.split('\t')
            if len(split) != 2:
                continue
            
            tokenized = tokenizer(
                text = split[0].strip(),
                text_target = split[1].strip(),
                return_tensors = 'pt',
                padding = 'max_length',
                truncation = True,
                max_length = 1024
            )
            self.examples.append(tokenized)
            if i % 1000 == 999:
                print(f'Loaded {i+1}/{len(lines)} lines for {self.lang_token}.')
            if num_examples is not None and i >= num_examples:
                break
                
    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)
    
def collate_fn(batch):
    return batch[0]

def get_data_loader(split, batch_size, num_batches, shuffle, num_workers, lang=None):
    if split != 'dev':
        split_loc = os.path.join('proj_data_final', split)
        files = [os.path.join(split_loc, f) for f in os.listdir(split_loc) if f.endswith('.tsv')]
        if lang is not None:
            files = [f for f in files if lang in f]
        return DataLoader(
            ParallelDataset(files, batch_size=batch_size, num_batches=num_batches),
            batch_size=1, # batches are already created, so just load one at a time
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn # remove the extra dimension that DataLoader adds
        )
    else:
        if lang is not None:
            return [DataLoader(
                DevSet(lang, num_examples=num_batches),
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn # remove the extra dimension that DataLoader adds
            )]
        else:
            return [DataLoader(
                DevSet(lang),
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn # remove the extra dimension that DataLoader adds
            ) for lang in c2t.keys() if lang != 'es']