
# python3
# Create Date: 2024-01-30
# Author: 爱科研的瞌睡虫

import os
from .merge import merge
from .pth_to_hf import convert_to_hf

def _convert_and_merged(config_file, pth_model, save_hf_dir, model_path, save_merged_dir):
    convert_to_hf(config_file, pth_model, save_hf_dir)
    merge(model_path, save_hf_dir, save_merged_dir)


def build_convert_and_merged_path(root_dir, epoch_pth):
    epoch = os.path.basename(epoch_pth).split('.')[0]
    work_dir = os.path.join(root_dir, 'work_dir')
    if not os.path.exists(work_dir):
        os.system(f'mkdir -p {work_dir}')
    hf = os.path.join(work_dir, f'xtuner_{epoch}_hf')
    mg = os.path.join(work_dir, f'xtuner_{epoch}_merge')
    # clear
    if os.path.exists(hf):
        os.system(f'rm -rf {hf}')
    if os.path.exists(mg):
        os.system(f'rm -rf {mg}')
    return work_dir, hf, mg


def convert_and_merged(root_dir, config_file, epoch_pth, model_path, model_personal_path, merged_flag=False):
    if len(model_personal_path) >= 3:
        model_path = model_personal_path
    work_dir, save_hf_dir, save_merged_dir = build_convert_and_merged_path(root_dir, epoch_pth)
    pth_model = os.path.join(work_dir, epoch_pth)
    print(
        f'config_file = {config_file}'
        ,f'\npth_model = {pth_model}' 
        ,f'\nsave_hf_dir = {save_hf_dir}' 
        ,f'\nmodel_path ={model_path}' 
        ,f'\nsave_merged_dir ={save_merged_dir}' 
    )
    try:
        convert_to_hf(config_file, pth_model, save_hf_dir)
        out_dir = save_hf_dir
        if merged_flag:
            merge(model_path, save_hf_dir, save_merged_dir)
            out_dir = save_merged_dir
        
        info = 'Successfully converted model ! '
    except Exception as e:
        info = e
        pass
    return info, out_dir


if __name__ == '__main__':

    config_file = '/root/ft-oasst1/internlm_chat_7b_qlora_oasst1_e3_copy.py'
    pth_model = '/root/ft-oasst1/work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth'
    save_hf_dir = '/root/ft-oasst1/hf5'
    model_path = '/root/ft-oasst1/internlm-chat-7b'
    save_merged_dir = '/root/ft-oasst1/merged5'

    _convert_and_merged(config_file, pth_model, save_hf_dir, model_path, save_merged_dir)
