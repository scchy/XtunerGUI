
import re
import os 
from os.path import getsize as p_getsize
from os.path import join as p_join


def get_py_config_files(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_tt = p_join(root, file)
            if os.path.isfile(file_tt) and '.py' in file and '__init__' not in file:
                all_files.append(file_tt)
    return all_files


def read_and_find_path(file):
    with open(file, 'r') as f:
        res = f.readlines()[:50]
    return re.findall(r"path\s+=\s+'(.*?)'\n", ''.join(res))


father_p = '/home/scc/sccWork/openProject/xtuner019/xtuner/xtuner/configs' 
py_files = get_py_config_files(father_p)
py_need_files = [i.rsplit('/', 1)[1] for i in py_files]
files_need = set([i.split('lora_')[-1].replace('.py', '') for i in py_need_files])
path_ = [read_and_find_path(i) for i in py_files]
path_final = []
for p in path_:
    path_final.extend(p)

model_str = ['Llama', 'Qwen', 'Baichuan', 'chatglm', '01-ai', 'internlm', 'llama']
path_final = [i for i in sorted(set(path_final)) if all(j not in i for j in model_str)]




