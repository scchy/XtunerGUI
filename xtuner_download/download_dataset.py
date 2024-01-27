# python3
# Create Date: 2024-01-25
# Author: Scc_hy
# Func: 模型拉取到本地
# ===========================================================================================
import os
import gradio as gr
from tqdm.auto import tqdm
from os.path import getsize as p_getsize
from os.path import join as p_join
import threading
import time
from .download_utils import stop_thread, _split_repo, get_hf_cache_files, get_data_info, get_final_out_files, TOKEN
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
CUR_DIR = os.path.dirname(__file__)


class xtunerDataDownload():
    def __init__(self, data_name, out_path, tqdm_class=tqdm, progress_sleep=1, retry_times=0) -> None:
        self.progress_sleep = progress_sleep
        self.run_times = retry_times + 1
        self.tqdm_class = tqdm_class
        self.username, self.repository = _split_repo(data_name)
        self.data_name = data_name
        self.out_path = out_path
        self.final_out_path = p_join(out_path, f'dataset_{self.username}_{self.repository}')
        self.mid_download_dir = self.final_out_path
        self._t_handle_dl = None
        self._t_handle_pg = None
        self.remove_and_create()
        self.get_download_info()

    def get_download_info(self):
        self.total_MB, self.total_file_nums = get_data_info(self.data_name)

    def __check_create_dir(self):
        if not os.path.exists(self.out_path):
            os.system(f'mkdir -p {self.out_path}')
        if not os.path.exists(self.final_out_path):
            os.system(f'mkdir -p {self.final_out_path}')

    def __remove_mid_files(self):
        """中断时删除所有文件"""
        os.system(f'rm -rf {self.mid_download_dir}')
        # cd rm 
        rm_dir = './' + self.mid_download_dir.replace(self.out_path, '.')[2:].split('/')[0]
        os.system(f'cd {self.out_path} && rm -rf  {rm_dir} && rm -rf temp')
        # 删除 hf 的cache
        os.system(f'rm -rf {self.final_out_path}/cache')

    def __remove_final_files(self):
        os.system(f'rm -rf {self.final_out_path}')
        os.system(f'cd {self.out_path} && rm -rf dataset_{self.username}_{self.repository}')
    
    def remove_and_create(self):
        self.__remove_mid_files()
        self.__remove_final_files()
        self.__check_create_dir()
    
    def auto_download(self, progress=gr.Progress(track_tqdm=True)):
        self._t_download(self.safe_download)
        # self._t_start()
        self.progress(progress=progress)
        return self.final_out_path

    def safe_download(self):
        for _ in range(self.run_times):
            self.hf_download()
            time.sleep(0.5)
            # 执行完检验
            if self._finished_check():
                print('finished download all model files')
                break
            self.remove_and_create()
        return

    def hf_download(self):
        print('>>>>>>> Start hf_download')
        # 1- mid download local dir
        self.mid_download_dir = self.final_out_path   
        # 2- download 
        os.system(f"""
        export HF_ENDPOINT=https://hf-mirror.com && \
        huggingface-cli download --resume-download {self.data_name} --local-dir-use-symlinks False \
        --repo-type dataset \
        --local-dir {self.final_out_path} \
        --cache-dir {self.final_out_path}/cache \
        --token {TOKEN}
        """)
        os.system(f'rm -rf {self.final_out_path}/cache')
        return self.final_out_path 

    def _finished_check(self):
        """检查是否下载完整数据
        """
        no_flag = (self.total_file_nums is not None) or (self.total_file_nums <= 0.01)
        if no_flag and os.path.exists(self.final_out_path):
            downloaded_files, download_bytes = self._get_final_out_bytes()
            file_same = len(downloaded_files) == self.total_file_nums
            size_same = download_bytes / 1024**2 / (self.total_MB + 1e-5) >= 0.99
            return size_same &  file_same
        return True

    def _t_start(self):
        self._t_handle_pg = threading.Thread(target=self.progress, name='X-dataset-progress', daemon=True)
        self._t_handle_pg.start()
        
    def _t_download(self, d_func):
        self._t_handle_dl = threading.Thread(target=d_func, name='X-dataset-download', daemon=True)
        self._t_handle_dl.start()

    def _get_final_out_bytes(self):
        # data存在多层的情况
        downloaded_files = get_final_out_files(self.final_out_path)
        cached_mb1 = sum([p_getsize(f) for f in downloaded_files])
        return downloaded_files, cached_mb1

    def progress(self, progress=None):
        hf_cache_dir = p_join(self.final_out_path, 'cache')
        self.bar_ = self.tqdm_class(total=round(self.total_MB*1024**2, 3), unit='iB', unit_scale=True)
        self.bar_.set_description('TotalProgress')
        bf = 0
        while True:
            if self._t_handle_dl is None:
                break
            if not self._t_handle_dl.is_alive():
                break
            hf_cache_files = get_hf_cache_files(hf_cache_dir) if os.path.exists(hf_cache_dir) else []
            _, cached_mb1 = self._get_final_out_bytes()
            cached_mb2 = sum([p_getsize(f) for f in hf_cache_files])
            cached_mb = (cached_mb1 + cached_mb2)
            
            self.bar_.update(round(cached_mb - bf, 3))
            bf = cached_mb
            time.sleep(self.progress_sleep)
        
        # 数据统计可能不准确
        finished_rate =  cached_mb / (self.total_MB + 1e-5) / 1024**2
        if self._t_handle_dl is None and finished_rate <= 0.99:
            left = self.total_MB * 1024**2 - bf
            self.bar_.update(round(left, 3))

        self.bar_.close()
        return 

    def break_download(self):
        # 然后杀死该线程
        # 删除文件
        if self._t_handle_dl is not None:
            print('>>>>>>>>>>>>>>>>> break_download')
            stop_thread(self._t_handle_dl)
            self._t_handle_dl = None
            os.system(f'sh {CUR_DIR}/kill_hf.sh dataset')
            print('>>>>>>>>>>>>>>>>> stop_thread(self._t_handle_dl)')

        if self._t_handle_pg is not None:
            stop_thread(self._t_handle_pg)
            print('>>>>>>>>>>>>>>>>> stop_thread(self._t_handle_pg)')
            self._t_handle_pg = None
        self.remove_and_create()
        try:
            self.bar_.close()
        except Exception as e:
            pass
        return "Done! Break Download"


if __name__ == '__main__':
    print(os.getcwd())
    download_ = xtunerDataDownload(
        'shibing624/medical', 
        out_path='/home/scc/sccWork/myGitHub/My_Learn/tmp/download')
    #     out_path='/root/tmp/download'
    # )
    # download_.hf_download()  # checked 
    download_.auto_download() # checked-download & progress    
    time.sleep(10)
    download_.break_download() # checked-download & progress & break
    print('Yes')
    # chech finished 
    f_ = download_._finished_check() 
    print(f'_finished_check={f_}')

