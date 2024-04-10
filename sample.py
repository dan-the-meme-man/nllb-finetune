# implement sampling within each subcorpus (bad_supp, dev, good_supp, train) independently
# use strategy from nllb paper: using parameter alpha=0.3
# for each language, p(language) = file_size(language) / total_file_size in subcorpus
# for each language, p(language) = p(language) ** alpha
# then renormalize p(language) = p(language) / sum(p(language))

import os

data_dir = 'proj_data_final'

bad_supp = os.path.join(data_dir, 'bad_supp')
good_supp = os.path.join(data_dir, 'good_supp')
train = os.path.join(data_dir, 'train')
dev = os.path.join(data_dir, 'dev')

bad_supp_files = [os.path.join(bad_supp, f) for f in os.listdir(bad_supp)]
good_supp_files = [os.path.join(good_supp, f) for f in os.listdir(good_supp)]
train_files = [os.path.join(train, f) for f in os.listdir(train)]
dev_files = [os.path.join(dev, f) for f in os.listdir(dev)]

langs = [
    'aym',
    'quy',
    'gn',
    'bzd',
    'cni',
    'ctp',
    'hch',
    'nah',
    'oto',
    'shp',
    'tar'
]

p = dict.fromkeys(langs)

for lang in p:
    p[lang] = 0

for lang in langs:
    # for file in bad_supp_files:
    #     if lang in file:
    #         p[lang] += os.path.getsize(file)
    # for file in good_supp_files:
    #     if lang in file:
    #         p[lang] += os.path.getsize(file)
    for file in train_files:
        if lang in file:
            p[lang] += os.path.getsize(file)

total = sum(p.values())

for lang in p:
    p[lang] = p[lang] / total
    try:
        p[lang] = p[lang] ** (-0.4)
    except ZeroDivisionError:
        p[lang] = 0
    
total = sum(p.values())

for lang in p:
    p[lang] = p[lang] / total
    
for lang in sorted(p.keys()):
    print(f'{lang}: {p[lang]}')

print(sum(p.values()))