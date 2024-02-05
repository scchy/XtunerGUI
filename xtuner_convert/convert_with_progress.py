
import os
from .convert_and_merge import build_convert_and_merged_path, convert_to_hf, merge
import re
from tqdm.auto import tqdm
import threading
import time 
import gradio as gr


class ConvertMerged:
    def __init__(self):
        self.save_hf_dir = None 
        self.save_merged_dir = None
        self.merged_flag = True
        self.model_path = None
        self.out_dir = None
        self.info = None

    def convert_and_merged(self, root_dir, config_file, epoch_pth, model_path, model_personal_path, ft_method):
        if len(model_personal_path) >= 3:
            model_path = model_personal_path
            
        self.model_path = model_path
        work_dir, save_hf_dir, save_merged_dir = build_convert_and_merged_path(root_dir, epoch_pth)
        self.save_hf_dir = save_hf_dir
        self.save_merged_dir = save_merged_dir
        pth_model = os.path.join(work_dir, epoch_pth)
        print(
            f'config_file = {config_file}'
            ,f'\npth_model = {pth_model}' 
            ,f'\nsave_hf_dir = {save_hf_dir}' 
            ,f'\nmodel_path ={model_path}' 
            ,f'\nsave_merged_dir ={save_merged_dir}' 
        )
        self.merged_flag = ft_method.lower() != 'full'
        try:
            convert_to_hf(config_file, pth_model, save_hf_dir)
            self.out_dir = save_hf_dir
            if self.merged_flag:
                merge(model_path, save_hf_dir, save_merged_dir)
                self.out_dir = save_merged_dir
            
            self.info = 'Successfully converted model ! '
        except Exception as e:
            self.info = e
            pass
        return self.info, self.out_dir
    
    def auto_convert_merge(
        self, 
        root_dir, 
        config_file, 
        epoch_pth, 
        model_path, 
        model_personal_path, 
        ft_method,
        progress=gr.Progress(track_tqdm=True)
        ):
        self._t_convert(root_dir, config_file, epoch_pth, model_path, model_personal_path, ft_method)
        time.sleep(2)
        print(
            f'self.model_path={self.model_path}',
            f'self.save_hf_dir={self.save_hf_dir}',
            f'self.save_merged_dir={self.save_merged_dir}'
        )
        self.progress()
        return self.info, self.out_dir

    def _t_convert(self, root_dir, config_file, epoch_pth, model_path, model_personal_path, ft_method):
        self._t_handle_convert = threading.Thread(
            target=self.convert_and_merged, args=(root_dir, config_file, epoch_pth, model_path, model_personal_path, ft_method) ,
            name='X-model-convert-merge', daemon=True)
        self._t_handle_convert.start()
    
    def find_max_sub(self, _dir):
        total_ = [i for i in os.listdir(_dir) if len(re.findall(r'[0-9]+-of-[0-9]+', i))]
        if len(total_):
            info = [int(re.findall(r'(\d+)-of', i)[0]) for i in total_ if len(re.findall(r'(\d+)-of', i))]
            if len(info):
                return max(info)
        return 0
        
    def progress(self, progress=None):  
        big_step = 100
        total = 0
        hf_total = 1
        # /root/share/model_repos/internlm-chat-7b/pytorch_model-00001-of-00008.bin
        base_model_parts = [i for i in os.listdir(self.model_path) if len(re.findall(r'[0-9]+-of-[0-9]+', i))]
        base_max = 0
        if len(base_model_parts):
            fd = re.findall(r'-of-(\d+)', base_model_parts[0])
            print(f'progress fd => {fd}')
            base_max = int(fd[0]) if len(fd) else 0
        if self.merged_flag:
            total = hf_total + base_max 
        else:
            total = hf_total = base_max 
        
        hf_total *= big_step
        total *= big_step
        tq_bar = tqdm(total=total)
        big_step_hf_now = 0
        big_step_mg_now = 0
        hf_now = 0
        mg_now = 0
        while True:
            if self._t_handle_convert is None:
                break
            if not self._t_handle_convert.is_alive():
                break
            
            up_hf = 0
            if os.path.exists(self.save_hf_dir) and not self.merged_flag:
                max_hf_b = self.find_max_sub(self.save_hf_dir) * big_step
                # 在一个的时候
                if big_step_hf_now == max_hf_b and  (big_step_hf_now + big_step) > hf_now and hf_now < hf_total:
                    up_hf = 1
                    hf_now += 1
                elif max_hf_b > hf_now: 
                    up_hf = max_hf_b - hf_now
                    hf_now = max_hf_b
                else:
                    up_hf = 0

                big_step_hf_now = max_hf_b
            elif self.merged_flag and not os.path.exists(self.save_hf_dir):
                if big_step >= hf_now:
                    up_hf = 1
                    hf_now += 1
            else:
                max_hf_b = big_step
                up_hf = max_hf_b - hf_now
                hf_now = max_hf_b

            up_mg = 0
            if self.merged_flag:
                if not os.path.exists(self.save_merged_dir):
                    if big_step > mg_now:
                        up_mg = 1
                        mg_now += 1
                else:
                    max_mg_b = self.find_max_sub(self.save_merged_dir) * big_step
                    # 在一个的时候
                    if big_step_mg_now == max_mg_b and (big_step_mg_now + big_step) > mg_now and (mg_now + hf_now) < total:
                        up_mg = 1
                        mg_now += 1
                    elif max_mg_b > mg_now: 
                        up_mg = max_mg_b - mg_now
                        mg_now = max_mg_b
                    else:
                        up_mg = 0

                    big_step_mg_now = max_mg_b

            tq_bar.update(up_mg + up_hf)
            time.sleep(1)

