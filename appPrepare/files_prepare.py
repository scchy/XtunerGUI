import os


CUR_PATH = os.path.dirname(__file__)

def dir_create(_dir):
    if not os.path.exists(_dir):
        os.system(f'mkdir -p {_dir}')
    return _dir


MODEL_DOWNLOAD_DIR = dir_create(f"{CUR_PATH}/tmp/model_download")
DATA_DOWNLOAD_DIR = dir_create(f"{CUR_PATH}/tmp/data_download")



