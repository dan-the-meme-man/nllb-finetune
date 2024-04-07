import os

translations_test = os.path.join('outputs', 'translations_test')

ckpts = [os.path.join(translations_test, ckpt) for ckpt in os.listdir(translations_test)]

for ckpt in ckpts:
    
    files = [os.path.join(ckpt, f) for f in os.listdir(ckpt)]
    
    for file in files:
        
        lines = open(file, 'r', encoding='utf-8').readlines()
        
        if file.endswith('ctp.txt'):
            assert len(lines) == 1000
            continue
        
        assert len(lines) == 1003