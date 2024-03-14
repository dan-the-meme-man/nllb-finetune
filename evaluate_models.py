import os
import subprocess

def execute_command(command, output_file):
    with open(output_file, 'w+', encoding='utf-8') as outfile:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            outfile.write(line)
        process.wait()

tr_dir = os.path.join('outputs', 'translations')
ckpts = [os.path.join(tr_dir, ckpt) for ckpt in os.listdir(tr_dir) if os.path.isdir(os.path.join(tr_dir, ckpt))]

reports_dir = os.path.join('outputs', 'reports')
if not os.path.exists(reports_dir):
    os.mkdir(reports_dir)

for ckpt in ckpts:
    
    if not os.path.exists(os.path.join(reports_dir, ckpt)):
        os.mkdir(os.path.join(reports_dir, ckpt))
    
    for tr_file in os.listdir(ckpt):
        tr_file_path = os.path.join(ckpt, tr_file)
        lang_code = tr_file.split('.')[0]
        es_file_path = os.path.join('proj_data_final', 'ref', lang_code+'.txt')
        
        command = [
            'python',
            'evaluate.py',
            '--sys',
            tr_file_path,
            '--ref',
            es_file_path,
            '--detailed_output'
        ]
        
        output_file = os.path.join(reports_dir, ckpt, lang_code+'.txt')