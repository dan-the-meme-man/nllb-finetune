from transformers import AutoTokenizer

lang_token_to_lang_code = {
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

lang_code_to_lang_token = {v: k for k, v in lang_token_to_lang_code.items()}

def make_tokenizer(tgt_lang, src_lang=None):
    
    if src_lang is None:
        src_lang = 'spa_Latn'
    
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/nllb-200-distilled-600M',
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        use_fast=True,
        return_tensors='pt',
        padding='max_length',
        max_length=1024,
        truncation=True
    )

    new_special_tokens = tokenizer.additional_special_tokens

    for lang_token in lang_token_to_lang_code:
        if lang_token in new_special_tokens:
            continue
        else:
            new_special_tokens.append(lang_token)

    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

    for lang_token in lang_token_to_lang_code:
        assert lang_token in tokenizer.additional_special_tokens
        
    return tokenizer