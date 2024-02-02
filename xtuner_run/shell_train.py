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
        self.log_file = os.path.join(self.work_dir, '__xtuner_tr.log')
        self.remove_log_file()
        print(f'config_py_path={config_py_path}')
    
    def reset_resume_from_checkpoint(self, ckpt):
        self.resume_from_checkpoint = f'{self.work_dir}/{ckpt}'
        print(f"reset_resume_from_checkpoint({self.resume_from_checkpoint})")
    
    def reset_deepspeed(self, deepspeed):
        print(f"reset_deepspeed({deepspeed})")
        self.deepspeed_seed = deepspeed
    
    def reset_work_dir(self, local_path):
        print(f"reset_work_dir({local_path})")
        self.work_dir = os.path.join(local_path, 'work_dir')
        if not os.path.exists(self.work_dir):
            os.system(f'mkdir -p {self.work_dir}')
        self.log_file = os.path.join(self.work_dir, '__xtuner_tr.log')
        self.remove_log_file()

    def reset_cfg_py(self, cfg_py):
        print(f"reset_cfg_py({cfg_py})")
        self.config_py_path = cfg_py

    def remove_log_file(self):
        if os.path.exists(self.log_file):
            os.system(f'rm -rf {self.log_file}')
    
    def _quick_train(self, resume, progress=gr.Progress(track_tqdm=True)):
        self.remove_log_file()
        add_ = resume_ = ''
        if str(self.deepspeed_seed).lower() not in ['none', 'dropdown']:
            add_ = f'--deepspeed deepspeed_{self.deepspeed_seed} '
        
        if self.resume_from_checkpoint is not None:
            resume_ = f'--resume {self.resume_from_checkpoint}'
        
        if resume:
            exec_ = f'xtuner train {self.config_py_path} --work-dir {self.work_dir} {add_} {resume_} > {self.log_file} 2>&1'
        else:
            exec_ = f'xtuner train {self.config_py_path} --work-dir {self.work_dir} {add_} > {self.log_file} 2>&1'
            
        print(f'exec={exec_}')
        os.system(exec_)

    def _t_start(self, resume=0):
        self._t_handle_tr = threading.Thread(target=self._quick_train, args=(resume,), name=f'X-train-{self.run_type}', daemon=True)
        self._t_handle_tr.start()

    def quick_train(self, progress=gr.Progress(track_tqdm=True)):
        self._break_flag = False
        self._t_start(0)
        self._t_handle_tr.join()
        if self._break_flag:
            return f"Done! Xtuner had interrupted!\nwork_dir={self.work_dir}", self.work_dir
        return "Success", self.work_dir

    def resume_train(self, progress=gr.Progress(track_tqdm=True)):
        self._break_flag = False
        self._t_start(1)
        self._t_handle_tr.join()
        if self._break_flag:
            return f"Done! Xtuner had interrupted!\nRESUME work_dir={self.work_dir}"
        return self.work_dir
    
    def _tail(self, n=100):
        line_list = []
        with open(self.log_file, "rb") as f:
            f.seek(0, 2)
            while 1:
                if f.read(1) == b"\n":
                    now_index = f.tell()
                    line_list.append(f.readline())
                    f.seek(now_index, 0)
                if len(line_list) >= n:
                    return line_list[::-1]
                if f.tell() <= 1:
                    f.seek(0, 0)
                    line_list.append(f.readline())
                    return line_list[::-1]
                f.seek(-2, 1)
    
    def read_log(self):
        if self._t_handle_tr is None:
            return ""
        if os.path.exists(self.log_file):
            # with open(self.log_file, 'r') as f:
            #     res_ = f.readlines()
            # return ''.join(res_)
            line_list = self._tail(5)
            return b"".join(line_list).decode()

    def break_train(self):
        # 然后杀死该线程
        # 删除文件
        if self._t_handle_tr is not None:
            print('>>>>>>>>>>>>>>>>> break_download')
            stop_thread(self._t_handle_tr)
            os.system(f'sh {CUR_DIR}/kill_xtuner.sh')
            self._t_handle_tr = None
     
        self._break_flag = True
        return f"Done! Xtuner had interrupted!\nwork_dir={self.work_dir}", self.work_dir

