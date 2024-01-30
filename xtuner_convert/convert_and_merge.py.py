
# python3
# Create Date: 2024-01-30
# Author: 爱科研的瞌睡虫


from merge import merge
from pth_to_hf import convert_to_hf

def convert_and_merged(config_file, pth_model, save_hf_dir, model_path, save_merged_dir):
    convert_to_hf(config_file, pth_model, save_hf_dir)
    merge(model_path, save_hf_dir, save_merged_dir)


if __name__ == '__main__':

    config_file = '/root/ft-oasst1/internlm_chat_7b_qlora_oasst1_e3_copy.py'
    pth_model = '/root/ft-oasst1/work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth'
    save_hf_dir = '/root/ft-oasst1/hf5'
    model_path = '/root/ft-oasst1/internlm-chat-7b'
    save_merged_dir = '/root/ft-oasst1/merged5'

    convert_and_merged(config_file, pth_model, save_hf_dir, model_path, save_merged_dir)