import os

from torch import manual_seed
from torch.utils.data import DataLoader, Dataset

from make_tokenizer import make_tokenizer, lang_code_to_lang_token as c2t

manual_seed(42)

tokenizers = {}
for lang in c2t.values():
    tokenizers[lang] = make_tokenizer(lang)

class MonoDataset(Dataset):
    def __init__(self, file, num_examples):
        self.examples = [line.strip() for line in open(file, 'r', encoding='utf-8').readlines()]
        if num_examples:
            self.examples = self.examples[:num_examples]
        self.lang_code = os.path.basename(file).split('.')[0].split('_')[0]
        self.lang_token = c2t[self.lang_code]
        self.tokenizer = tokenizers[self.lang_token]

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)
    
class ParallelDataset(Dataset):
    def __init__(self, file, num_examples):
        self.examples = [line.strip().split('\t') for line in open(file, 'r', encoding='utf-8').readlines()]
        if num_examples:
            self.examples = self.examples[:num_examples]
        self.lang_code = os.path.basename(file).split('.')[0].split('_')[1]
        self.lang_token = c2t[self.lang_code]
        self.tokenizer = tokenizers[self.lang_token]

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

def get_data_loader(file, batch_size, shuffle, num_workers, mono, num_examples=None):
    if mono:
        return DataLoader(
            MonoDataset(file, num_examples=num_examples),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    else:
        return DataLoader(
            ParallelDataset(file, num_examples=num_examples),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )