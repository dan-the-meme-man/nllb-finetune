import os
import shutil

submission_path = os.path.join('outputs', 'submission')
if not os.path.exists(submission_path):
    os.mkdir(submission_path)

ckpts = [
    #'checkpoint10_train_4_0_2500_3_2500_10_2500_1e-05_0.0001',
    'checkpoint3_good_supp_False_4_0_2500_3_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001',
    'checkpoint3_train_False_4_0_2500_3_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001',
    'checkpoint4_train_False_4_0_2500_3_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001',
    'checkpoint10_train_False_4_0_2500_0_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001',
    'checkpoint8_train_False_4_0_2500_3_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001',
    'checkpoint6_train_False_4_0_2500_0_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001',
    'checkpoint4_train_False_4_0_2500_3_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001',
    'checkpoint3_train_False_4_0_2500_3_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001',
    'checkpoint10_train_False_4_0_2500_0_None_10_None_1e-05_0.0001_ckpt_checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001'
]

unique_ckpts = sorted(list(set(ckpts)))

for i, ckpt in enumerate(unique_ckpts):
    
    path_to_results = os.path.join('outputs', 'translations_test', ckpt)
    
    files = [os.path.join(path_to_results, f) for f in os.listdir(path_to_results)]
    
    assert len(files) == 11
    
    print(i+1, ckpt)
    for file in files:
        code = os.path.basename(file)[:-4]
        name = code + '.results.' + str(i+1)
        #print(file, name)
        shutil.copy(file, os.path.join(submission_path, name))