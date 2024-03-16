from transformers import AutoTokenizer

t2c = {
    'spa_Latn': 'es', # should be in there - note difference
    'ayr_Latn': 'aym', # should be in there - note difference
    'quy_Latn': 'quy', # should be in there
    'grn_Latn': 'gn', # should be in there - note difference
    'bzd_Latn': 'bzd',
    'cni_Latn': 'cni',
    'ctp_Latn': 'ctp',
    'hch_Latn': 'hch',
    'nah_Latn': 'nah',
    'oto_Latn': 'oto',
    'shp_Latn': 'shp',
    'tar_Latn': 'tar'
}

c2t = {v: k for k, v in t2c.items()}

t2i = {}

def make_tokenizer(tgt_lang, src_lang, max_length):
    
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/nllb-200-distilled-600M',
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        use_fast=True,
        return_tensors='pt',
        padding='max_length',
        max_length=max_length,
        truncation=True
    )
    
    assert tokenizer._src_lang == 'spa_Latn'
    assert tokenizer.tgt_lang == tgt_lang

    new_special_tokens = tokenizer.additional_special_tokens

    for lang_token in t2c:
        if lang_token in new_special_tokens:
            continue
        else:
            new_special_tokens.append(lang_token)

    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

    for lang_token in t2c:
        assert lang_token in tokenizer.additional_special_tokens
        t2i[lang_token] = tokenizer.convert_tokens_to_ids(lang_token)
        
    return tokenizer