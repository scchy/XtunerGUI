# python3
# Create Date: 2024-01-24
# Author: Scc_hy
# Func: 模型拉取到本地
# ===========================================================================================
import os
import re
import inspect
import ctypes
import requests
import inspect
import ctypes
import requests
from tqdm.auto import tqdm
from os.path import join as p_join
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub.hf_api import HfApi

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
TOKEN = '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'

def get_hf_cache_files(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_tt = p_join(root, file)
            if os.path.isfile(file_tt) and '.incomplete' in file:
                all_files.append(file_tt)
    return all_files


def get_final_out_files(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_tt = p_join(root, file)
            if os.path.isfile(file_tt) and 'cache' not in file_tt:
                all_files.append(file_tt)
    return all_files



def _split_repo(model_repo) -> (str, str):
    """
    Split a full repository name into two separate strings: the username and the repository name.
    """
    # username/repository format check
    pattern = r'.+/.+'
    if not re.match(pattern, model_repo):
        raise ValueError("The input string must be in the format 'username/model_repo'")

    values = model_repo.split('/')
    return values[0], values[1]
 

def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    try:
        _async_raise(thread.ident, SystemExit)
    except Exception as e:
        print(e)


def detect_data_file_bytes(data_name, data_file):
    try:
        txt = requests.get(f'https://hf-mirror.com/datasets/{data_name}/blob/main/{data_file}', timeout=3).text
    except Exception as e:
        print(e)
        return 0.0
    find_out = re.findall(r'Size of remote file:(.*?)B', txt, flags=re.S)
    find_info = find_out[0] if len(find_out) else '0 '
    info_num = float(re.findall(r'\d+.\d+|\d+', find_info)[0])
    info_cat = find_info[-1]+'B'
    # huggingface 直接1000 
    info_map = {
        'kB': 1000,
        'KB': 1000,
        'MB': 1000 ** 2,
        'mB': 1000 ** 2,
        'GB': 1000 ** 3,
        'gB': 1000 ** 3,
        ' B': 1,
    }
    return info_num * info_map.get(info_cat, 1.0)


def get_data_info(data_name):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    api_ = HfApi(token=TOKEN)
    try:
        df_info = api_.dataset_info(repo_id=data_name, token=TOKEN, timeout=3)
    except Exception as e:
        print(e)
        return 0, 0
    df_files = [i.rfilename for i in df_info.siblings]
    exec_ = ThreadPoolExecutor(max_workers=2)
    tasks = [exec_.submit(detect_data_file_bytes, data_name=data_name, data_file=i) for i in df_files]
    res = []
    for t in tqdm(tasks):
        res.append(t.result())
    
    total_MB = sum(res) / 1024 ** 2
    total_file_nums = len(df_files)
    return total_MB, total_file_nums


def detect_model_file_bytes(model_name, data_file):
    try:
        txt = requests.get(f'https://hf-mirror.com/{model_name}/blob/main/{data_file}', timeout=3).text
    except Exception as e:
        print(e)
        return 0.0
    find_out = re.findall(r'Size of remote file:(.*?)B', txt, flags=re.S)
    find_info = find_out[0] if len(find_out) else '0 '
    info_num = float(re.findall(r'\d+.\d+|\d+', find_info)[0])
    info_cat = find_info[-1]+'B'
    # huggingface 直接1000 
    info_map = {
        'kB': 1000,
        'KB': 1000,
        'MB': 1000 ** 2,
        'mB': 1000 ** 2,
        'GB': 1000 ** 3,
        'gB': 1000 ** 3,
        ' B': 1,
    }
    return info_num * info_map.get(info_cat, 1.0)


def get_model_info(model_name):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    api_ = HfApi(token=TOKEN)
    try:
        df_info = api_.model_info(repo_id=model_name, token=TOKEN, timeout=3)
    except Exception as e:
        print(e)
        return 0, 0
    df_files = [i.rfilename for i in df_info.siblings]
    exec_ = ThreadPoolExecutor(max_workers=2)
    tasks = [exec_.submit(detect_model_file_bytes, model_name=model_name, data_file=i) for i in df_files]
    res = []
    for t in tqdm(tasks):
        res.append(t.result())

    total_MB = sum(res) / 1024 ** 2
    total_file_nums = len(df_files)
    return total_MB, total_file_nums


def test_get_data_info():
    print('>>>>>>>> Start test')
    data_name = 'shibing624/medical'
    total_MB, total_file_nums = get_data_info(data_name)
    print(f'[ data_name={data_name} ] total_file_nums={total_file_nums} | total_MB={total_MB:.3f}MiB')

def test_down_load_data():
    print('>>>>>>>> Start test')
    model_name = 'internlm/internlm-chat-7b'
    total_MB, total_file_nums = get_model_info(model_name)
    print(f'[ data_name={model_name} ] total_file_nums={total_file_nums} | total_MB={total_MB:.3f}MiB')



if __name__ == '__main__':
    test_get_data_info()
    test_down_load_data()
