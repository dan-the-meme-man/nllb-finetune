import os
from gc import collect

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache
from torch import no_grad, load, manual_seed

from transformers import AutoModelForSeq2SeqLM

from get_data_loader import get_data_loader
from make_tokenizer import make_tokenizer, c2t, t2i

def free():
    collect()
    empty_cache()

def main():
    
    freeze_support()
    
    #device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42)
    
    overfit           = False
    num_workers       = 1
    batch_size        = 4    if not overfit else 1
    test_num_batches  = None if not overfit else 5 # None for full test set
    max_length        = 384
    lang_code         = None if not overfit else 'aym' # None for all languages
    
    print('\nLoading model...')
    free()
    tokenizers = dict.fromkeys(c2t.values())
    for lang_token in tokenizers: # load tokenizers for each language
        tokenizers[lang_token] = make_tokenizer(lang_token, 'spa_Latn', max_length)
        assert tokenizers[lang_token]._src_lang == 'spa_Latn'
        assert tokenizers[lang_token].tgt_lang == lang_token
        assert len(tokenizers[lang_token]) == 256212
    model_name = 'facebook/nllb-200-distilled-600M'
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True
    ).to(device)
    model.resize_token_embeddings(len(tokenizers['ayr_Latn'])) # resize embeddings
    free()
    print('Model loaded.\n')
    
    print('Loading dev data...')
    free()
    test_loaders = get_data_loader(
        split='test',
        batch_size=batch_size,
        num_batches=test_num_batches,
        max_length=max_length,
        lang_code=lang_code,
        shuffle=False, # ignored
        num_workers=num_workers,
        use_tgts=False, # needed to do decoding
        get_tokenized=True
    )
    free()
    print('Test data loaded.\n')
    
    tr_dir = os.path.join('outputs', 'translations_test')
    if not os.path.exists(tr_dir):
        os.mkdir(tr_dir)

    for ckpt in os.listdir(os.path.join('outputs', 'ckpts')):
        
        print(f'Loading checkpoint {ckpt}...')
        free()
        file_path = os.path.join('outputs', 'ckpts', ckpt)
        checkpoint = load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        free()
        print(f'Checkpoint {ckpt} loaded.\n')
        
        model_tr_dir = os.path.join(tr_dir, ckpt[:-4])
        if not os.path.exists(model_tr_dir):
            os.mkdir(model_tr_dir)
        
        with no_grad():
            model.eval()
            for test_loader in test_loaders:
                
                lang_code = test_loader.dataset.lang_code
                lang_token = test_loader.dataset.lang_token
                tokenizer = test_loader.dataset.tokenizer
                
                translations = []
                
                for i, batch in enumerate(test_loader):
                    
                    outputs = model.generate(
                        **batch.to(device),
                        forced_bos_token_id=t2i[lang_token],
                        max_length=max_length,
                        num_beams=4,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
                    translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
                    
                    try:
                        assert len(translations) == (i+1)*batch_size
                    except:
                        print(f'Batch {i} failed for {lang_code}.')
                        print(f'Batch size: {batch_size}, translations length: {len(translations)}.')
                    
                    if i % 100 == 0:
                        print(f'{i} batches decoded for {lang_code}.')
                        
                    del outputs
                    free()
                
                loc = os.path.join(model_tr_dir, lang_code + '.txt')
                    
                with open(loc, 'w+', encoding='utf-8') as f:
                    for t in translations:
                        f.write(t + '\n')
                free()
                        
if __name__ == '__main__':
    main()