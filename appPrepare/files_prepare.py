import os

def dir_create(_dir):
    if not os.path.exists(_dir):
        os.system(f'mkdir -p {_dir}')
    return _dir


MODEL_DOWNLOAD_DIR = dir_create("/home/xlab-app-center/model_download")
DATA_DOWNLOAD_DIR = dir_create("/home/xlab-app-center/model_download")



