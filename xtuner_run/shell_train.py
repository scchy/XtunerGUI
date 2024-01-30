# python3
# Create Date: 2024-01-30
# Author: Scc_hy
# Func: 用shell 启动xtuner 
# ===========================================================================================

from .train_utils import prepareConfig, prepareUtil, stop_thread
import threading
import os
import gradio as gr
CUR_DIR = os.path.dirname(__file__)


class quickTrain:
    def __init__(self, 
                 work_dir,
                 config_py_path,
                 deepspeed_seed=None,
                 resume_from_checkpoint=None,
                 run_type='mmengine'):
        self.work_dir = work_dir
        self.config_py_path = config_py_path
        self.resume_from_checkpoint = resume_from_checkpoint
        self.run_type = run_type
        self.deepspeed_seed = deepspeed_seed
        self._t_handle_tr = None
        self.log_file = os.path.join(CUR_DIR, '__xtuner_tr.log')
        self.remove_log_file()
        print(f'config_py_path={config_py_path}')
    
    def reset_deepspeed(self, deepspeed):
        self.deepspeed_seed = deepspeed
    
    def reset_work_dir(self, local_path):
        self.work_dir = os.path.join(local_path, 'work_dir')

    def reset_cfg_py(self, cfg_py):
        self.config_py_path = cfg_py

    def remove_log_file(self):
        if os.path.exists(self.log_file):
            os.system(f'rm -rf {self.log_file}')
    
    def _quick_train(self, progress=gr.Progress(track_tqdm=True)):
        self.remove_log_file()
        add_ = ''
        if str(self.deepspeed_seed).lower() != 'none':
            add_ = f'--deepspeed deepspeed_{self.deepspeed_seed} '
        
        exec_ = f'xtuner train {self.config_py_path} --work_dir {self.work_dir} {add_} > {self.log_file}'
        os.system(exec_)
    
    def _t_start(self):
        self._t_handle_tr = threading.Thread(target=self._quick_train, name=f'X-train-{self.run_type}', daemon=True)
        self._t_handle_tr.start()

    def quick_train(self, progress=gr.Progress(track_tqdm=True)):
        self._break_flag = False
        self._t_start()
        self._t_handle_tr.join()
        if self._break_flag:
            return "Done! Xtuner had interrupted!"
        return self.work_dir
    
    def read_log(self):
        if self._t_handle_tr is None:
            return ""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                res_ = f.readlines()
            return ''.join(res_)

    def break_train(self):
        # 然后杀死该线程
        # 删除文件
        if self._t_handle_tr is not None:
            print('>>>>>>>>>>>>>>>>> break_download')
            stop_thread(self._t_handle_tr)
            os.system(f'sh {CUR_DIR}/kill_xtuner.sh')
            self._t_handle_tr = None
     
        self._break_flag = True
        return "Done! Xtuner had interrupted!"
