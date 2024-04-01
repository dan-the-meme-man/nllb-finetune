import os

dev_dir = os.path.join('proj_data_final', 'dev')
ref_dir = os.path.join('proj_data_final', 'ref')

for file in os.listdir(dev_dir):
    in_path = os.path.join(dev_dir, file)
    out_path = os.path.join(ref_dir, file[:-4]+'.txt')
    with open(in_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with open(out_path, 'w+', encoding='utf-8') as g:
            for line in lines:
                g.write(line.split('\t')[1].strip() + '\n')