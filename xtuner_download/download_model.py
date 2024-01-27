# python3
# Create Date: 2024-01-23
# Author: Scc_hy
# Func: 模型拉取到本地
# ===========================================================================================
import os
import gradio as gr
from tqdm.auto import tqdm
from openxlab.model import download as ox_download
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi, ModelScopeConfig
from os.path import getsize as p_getsize
from os.path import join as p_join
import threading
import time
from .download_utils import stop_thread, _split_repo, get_hf_cache_files, get_model_info, TOKEN
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
CUR_DIR = os.path.dirname(__file__)


class xtunerModelDownload():
    def __init__(self, model_name, out_path, tqdm_class=tqdm, progress_sleep=1) -> None:
        self.progress_sleep = progress_sleep
        self.tqdm_class = tqdm_class
        self.username, self.repository = _split_repo(model_name)
        self.model_name = model_name
        self.out_path = out_path
        self.final_out_path = p_join(out_path, f'{self.username}_{self.repository}')
        self.mid_download_dir = self.final_out_path
        self._t_handle_dl = None
        self._t_handle_pg = None
        self.remove_and_create()
        self.get_download_info()

    def _username_map(self, tp):
        """username 映射
        """
        modelscope_map_dict = {
            'internlm': 'Shanghai_AI_Laboratory',
            'meta-llama': 'shakechen', # Llma-2
            'huggyllama': 'skyline2006', # Llma
            'THUDM': 'ZhipuAI',
            '01-ai': '01ai',
            'Qwen': 'qwen'
        }
        hf_map_dict = {}
        openxlab_map_dict = {
            'internlm': 'OpenLMLab',
            'meta-llama': 'shakechen', # Llma-2
            'huggyllama': 'skyline2006', # Llma
            'THUDM': 'ZhipuAI',
            '01-ai': '01ai'
        }
        sp_model_name = '{u_name}/{rep}'.format(
            u_name=eval(f"{tp}_map_dict.get('{self.username}', '{self.username}')"),
            rep=self.repository 
        )
        return sp_model_name

    def get_download_info(self):
        # 优先modelscope查看
        try:
            self.total_MB, self.total_file_nums = self._get_download_info()
        except Exception as e:
            self.total_MB, self.total_file_nums = get_model_info(self.model_name)
    
    def _get_download_info(self):
        _api = HubApi()
        headers = {'user-agent': ModelScopeConfig.get_user_agent(user_agent=None, )}
        snapshot_header = headers if 'CI_TEST' in os.environ else {
                    **headers,
                    **{
                        'Snapshot': 'True'
                    }
                }
        model_id = self._username_map('modelscope')
        model_files = _api.get_model_files(
            model_id=model_id,
            revision=None,
            recursive=True,
            use_cookies=False,
            headers=snapshot_header,
        )
        total_MB = sum([i['Size']/1024**2 for i in model_files])
        total_file_nums = len(model_files)
        return total_MB, total_file_nums
        
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
        os.system(f'cd {self.out_path} && rm -rf {self.username}_{self.repository}')
    
    def remove_and_create(self):
        self.__remove_mid_files()
        self.__remove_final_files()
        self.__check_create_dir()
        self.mid_download_dir = self.final_out_path
    
    def auto_download(self, progress=gr.Progress(track_tqdm=True), tp=None):
        self._t_download(self.loop_download, tp)
        # self._t_start(progress)
        # progress not use thread
        self.progress(progress=progress)
        return self.final_out_path

    def loop_download(self, tp=None):
        if 'internlm' in self.model_name.lower():
            loop_list = [self.openxlab_download, self.modelscope_download, self.hf_download]
        elif tp == 'speed':
            loop_list = [self.modelscope_download, self.hf_download, self.openxlab_download]
        else:
            loop_list = [self.hf_download, self.modelscope_download, self.openxlab_download]

        for download_func in loop_list:
            try:
                download_func()
                time.sleep(1)
            except Exception as e:
                pass
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
        huggingface-cli download --resume-download {self.model_name} --local-dir-use-symlinks False \
        --repo-type model \
        --local-dir {self.final_out_path} \
        --cache-dir {self.final_out_path}/cache \
        --token {TOKEN}
        """)
        os.system(f'rm -rf {self.final_out_path}/cache')
        return self.final_out_path 

    def modelscope_download(self):
        print('>>>>>>> Start modelscope_download')
        # 1- fix-name
        model_name = self._username_map('modelscope')
        # 2- mid download local dir
        self.mid_download_dir = mid_download_dir = p_join(self.out_path, model_name)
        # 3- download 
        snapshot_download(model_id=model_name, cache_dir=self.out_path)
        # 保证目录一致  out_path/sccHyFuture/LLM_medQA_adapter  --> final_out_path
        os.system(f'mv {mid_download_dir}/*  {self.final_out_path}')
        self.__remove_mid_files()
        return self.final_out_path

    def openxlab_download(self):
        print('>>>>>>> Start openxlab_download')
        # 1- fix-name
        model_name = self._username_map('openxlab')
        # 2- mid download local dir
        self.mid_download_dir = self.final_out_path
        # 3- download 
        ox_download(model_repo=model_name, output=self.final_out_path, cache=False)
        return self.final_out_path 

    def _finished_check(self):
        """检查是否下载完整数据
        """
        no_flag = (self.total_file_nums is not None) or (self.total_file_nums <= 0.01)
        if no_flag and os.path.exists(self.final_out_path):
            file_same = len([i for i in os.listdir(self.final_out_path) if os.path.isfile(i) ]) == self.total_file_nums
            size_same = sum([p_getsize(p_join(self.final_out_path, i))/ 1024**2
                         for i in os.listdir(self.final_out_path)])/(self.total_MB + 1e-5) >= 0.9999
            return size_same &  file_same
        return True
    
    def _t_start(self, pg=None):
        self._t_handle_pg = threading.Thread(target=self.progress, args=(pg,), name='X-model-progress', daemon=True)
        self._t_handle_pg.start()
        
    def _t_download(self, d_func, tp):
        self._t_handle_dl = threading.Thread(target=d_func, args=(tp,) ,name='X-model-download', daemon=True)
        self._t_handle_dl.start()

    def progress(self, progress=None):        
        model_scope_cache_dir = p_join(self.out_path, 'temp')
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
            if self.mid_download_dir == self.final_out_path:
                cached_mb1 = sum([p_getsize(p_join(self.final_out_path, i))#/1024**2
                for i in os.listdir(self.final_out_path)])
                cached_mb4 = sum([p_getsize(f) #/1024**2
                for f in hf_cache_files])
                cached_mb = cached_mb1 + cached_mb4
            else:
                # 获取最近创建的temp文件
                model_scope_cache_dir_tmp = sorted([
                    [p_join(model_scope_cache_dir, i), os.stat(p_join(model_scope_cache_dir, i)).st_atime] for i in os.listdir(model_scope_cache_dir)
                ], key=lambda c:c[1])[-1][0]
                cached_mb1 = sum([p_getsize(p_join(self.mid_download_dir, i))#/1024**2
                for i in os.listdir(self.mid_download_dir)])
                cached_mb2 = sum([p_getsize(p_join(self.final_out_path, i))#/1024**2
                for i in os.listdir(self.final_out_path)])
                cached_mb3 = sum([p_getsize(p_join(model_scope_cache_dir_tmp, i))#/1024**2
                for i in os.listdir(model_scope_cache_dir_tmp)])
                cached_mb4 = sum([p_getsize(f) #/1024**2
                for f in hf_cache_files])
                cached_mb = (cached_mb1 + cached_mb2 + cached_mb3 + cached_mb4)
            
            self.bar_.update(round(cached_mb - bf, 3))
            bf = cached_mb
            if cached_mb / (self.total_MB + 1e-5) / 1024**2 > 99.99:
                break
            time.sleep(self.progress_sleep)
        self.bar_.close()
        return 

    def break_download(self):
        # 然后杀死该线程
        # 删除文件
        if self._t_handle_dl is not None:
            print('>>>>>>>>>>>>>>>>> break_download')
            stop_thread(self._t_handle_dl)
            self._t_handle_dl = None
            os.system(f'sh {CUR_DIR}/kill_hf.sh model')
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
    download_ = xtunerModelDownload(
        'internlm/InternLM-chat-7b', 
        out_path='/home/scc/sccWork/myGitHub/My_Learn/tmp/download')
        #'/root/tmp/download')
    #'/home/scc/sccWork/myGitHub/My_Learn/tmp/download')
    # download_.hf_download() # checked-download & progress
    # download_.openxlab_download() # checked-download & progress
    # download_.modelscope_download() # checked-download & progress
    download_.auto_download() # checked-download & progress
    time.sleep(10)
    download_.break_download() # checked-download & progress & break
    print('Yes')
    # chech finished 
    f_ = download_._finished_check() 
    print(f'_finished_check={f_}')
