import os
from gc import collect

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache, memory_allocated
from torch import no_grad, load, manual_seed

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from get_data_loader import get_data_loader
from make_tokenizer import make_tokenizer, c2t, t2i

def main():
    
    freeze_support()
    
    #device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42)

    model_name = 'facebook/nllb-200-distilled-600M'
    
    print('Loading model...')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True
    ).to(device)
    tokenizer = make_tokenizer('ayr_Latn', 'spa_Latn', 384)
    model.resize_token_embeddings(len(tokenizer))
    print('Model loaded.\n')
    
    overfit           = True
    num_workers       = 1
    batch_size        = 8    if not overfit else 1
    dev_num_batches   = None if not overfit else 10 # None for full dev set
    max_length        = 384
    lang_code         = None if not overfit else 'aym' # None for all languages
    
    print('Loading dev data...')
    dev_loaders = get_data_loader(
        split='dev',
        batch_size=batch_size,
        num_batches=dev_num_batches,
        max_length=max_length,
        lang_code=lang_code,
        shuffle=False, # ignored
        num_workers=num_workers,
        use_tgts=False, # needed to do decoding
        get_tokenized=False
    )
    print('Dev data loaded.\n')
    
    tr_dir = os.path.join('outputs', 'translations')
    if not os.path.exists(tr_dir):
        os.mkdir(tr_dir)
        
    model_tr_dir = os.path.join(tr_dir, 'untrained')
    if not os.path.exists(model_tr_dir):
        os.mkdir(model_tr_dir)
    
    with no_grad():
        model.eval()
        for dev_loader in dev_loaders:
            
            lang_code = dev_loader.dataset.lang_code
            lang_token = dev_loader.dataset.lang_token
            #tokenizer = dev_loader.dataset.tokenizer
            
            translations = []
            
            # tokenizer = AutoTokenizer.from_pretrained(
            #     'facebook/nllb-200-distilled-600M',
            #     src_lang='spa_Latn',
            #     tgt_lang='ayr_Latn',
            #     use_fast=True,
            #     return_tensors='pt',
            #     padding='longest',
            #     max_length=max_length,
            #     truncation=True
            # )
            
            for i, batch in enumerate(dev_loader):
                
                #print(tokenizer._src_lang, tokenizer.tgt_lang, lang_code, lang_token)
                #tokenizer = make_tokenizer('eng_Latn', 'spa_Latn', max_length)
                assert tokenizer._src_lang == 'spa_Latn'
                assert tokenizer.tgt_lang == 'ayr_Latn'
                
                print(batch)
                
                tokenized_batch = tokenizer(
                    batch,
                    return_tensors='pt',
                    padding='longest',
                    max_length=max_length,
                    truncation=True
                )
                
                outputs = model.generate(
                    **tokenized_batch.to(device),
                    forced_bos_token_id=t2i[lang_token],
                    max_length=max_length,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
                
                if i % 100 == 0:
                    print(f'{i} batches decoded for {lang_code}.')
                
            loc = os.path.join(model_tr_dir, lang_code + '.txt')
                
            with open(loc, 'w+', encoding='utf-8') as f:
                for t in translations:
                    f.write(t + '\n')
                        
if __name__ == '__main__':
    main()