import os

reports_dir = os.path.join('outputs', 'reports')

ckpts = [os.path.join(reports_dir, ckpt) for ckpt in os.listdir(reports_dir)]

for ckpt in ckpts:
    
    good = True
    
    langs = []
    scores = []
    
    for file in os.listdir(ckpt):        
        file_path = os.path.join(ckpt, file)
        
        lang = os.path.basename(file_path)[:-4]
        
        lines = open(file_path, 'r', encoding='utf-8').readlines()
        
        try:
            if 'assert' in lines[3]:
                print('unusable', os.path.basename(ckpt))
                good = False
                break
        except:
            print(file_path)
            exit()
            
        score = float(lines[3].split()[-1])
        
        langs.append(lang)
        scores.append(score)
        
    if len(langs) != 11 or len(scores) != 11:
        good = False
    
    for score in scores:
        if score < 12.3: # threshold score
            good = False
            break
        
    if good:
        print()
        print('good', os.path.basename(ckpt))
        for i in range(11):
            print(f'{langs[i]}: {scores[i]}')
        print()