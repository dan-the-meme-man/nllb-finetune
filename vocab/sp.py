import sentencepiece as spm

input_for_training = 'vocab.txt'
model_name = 'bpe' # unigram or bpe
#train = True
train = False # training is done already

"""TRAINING A SENTENCEPIECE TOKENIZER BELOW"""

if train: # fold up if statement to see tests/usage below

    params = ''
    params = params + ' --input=' + input_for_training
    params = params + ' --model_prefix=' + model_name

    params = params + ' --vocab_size=32000'

    params = params + ' --character_coverage=1.0'

    params = params + ' --normalization_rule_name=nfkc'

    #params = params + ' --model_type=unigram'
    params = params + ' --model_type=bpe'

    params = params + ' --control_symbols=' + ','.join(
    [
        '<aym>',
        '<bzd>',
        '<cni>',
        '<ctp>',
        '<gn>',
        '<hch>',
        '<nah>',
        '<oto>',
        '<quy>',
        '<shp>',
        '<tar>'
    ]
    )

    params = params + ' --shrinking_factor=0.95' # limit
    
    spm.SentencePieceTrainer.Train(params)
 
"""TEST A MODEL BELOW AFTER IT IS TRAINED"""

sp = spm.SentencePieceProcessor()
sp.Load(model_name + '.model')

print(sp.__dict__)
print(sp.this)

print(sp.EncodeAsPieces('Hola mundo.'))
print(sp.EncodeAsIds('Hola mundo.'))

print()
print(sp.EncodeAsPieces('hoy tenemos que entrar a la ciudad porque hay un evento.'))
print(sp.EncodeAsIds('hoy tenemos que entrar a la ciudad porque hay un evento.'))

print()
print(sp.DecodeIds([10, 30, 60, 100, 1000, 2000, 4000]))