import os


CUR_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.dirname(CUR_PATH)

def dir_create(_dir):
    if not os.path.exists(_dir):
        os.system(f'mkdir -p {_dir}')
    return _dir

DEFAULT_DOWNLOAD_DIR = dir_create(f"{DATA_PATH}/download_cache")
MODEL_DOWNLOAD_DIR = dir_create(f"{DEFAULT_DOWNLOAD_DIR}/model_download")
DATA_DOWNLOAD_DIR = dir_create(f"{DEFAULT_DOWNLOAD_DIR}/data_download")
WORK_DIR = dir_create(f"{CUR_PATH}/work_dir")





