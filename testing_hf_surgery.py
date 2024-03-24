import os

from torch.cuda import is_available
from torch import no_grad, load, manual_seed

from transformers import AutoConfig, AutoModelForSeq2SeqLM

from make_tokenizer import make_tokenizer, c2t, t2i

# from get_data_loader import get_data_loader

# dev_loaders = get_data_loader(
#     split='dev',
#     batch_size=1,
#     num_batches=None,
#     max_length=384,
#     lang_code=None,
#     shuffle=False, # ignored
#     num_workers=1,
#     use_tgts=True, # for dev loss
#     get_tokenized=True # ignored
# )
# exit()
    
device = 'cuda' if is_available() else 'cpu'
manual_seed(42)

model_name = 'facebook/nllb-200-distilled-600M'

config = AutoConfig.from_pretrained(model_name)
config.vocab_size += 8 # 8 new special tokens for languages

print('Loading model...')
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
).to(device)
print('Model loaded.\n')

max_length        = 384
lang_code         = 'bzd'
lang_token        = c2t[lang_code]

ckpt = 'checkpoint1_train_4_1_25000_3_25000_10_75000_1e-05_0.01.pth'

tokenizer = make_tokenizer(tgt_lang = lang_token, src_lang = 'spa_Latn', max_length = max_length)

print(f'Loading checkpoint {ckpt}...')
file_path = os.path.join('outputs', 'ckpts', ckpt)
checkpoint = load(file_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Checkpoint {ckpt} loaded.\n')

batch = tokenizer(
    ['Solo dura una semana.'],
    return_tensors='pt',
    padding='max_length',
    max_length=max_length,
    truncation=True
)

with no_grad():
    model.eval()
    outputs = model.generate(
        **batch.to(device),
        forced_bos_token_id=t2i[lang_token],
        max_length=max_length,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))