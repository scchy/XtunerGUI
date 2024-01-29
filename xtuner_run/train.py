# python3
# Create Date: 2024-01-29
# Author: Scc_hy
# Func: 基于指定参数进行模型训练
# nohup xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py > __xtuner.log &
# pip install -U xtuner
# ===========================================================================================

from .train_utils import prepareConfig, prepareUtil, stop_thread
import threading
import os
import gradio as gr
from transformers import Trainer
from xtuner.dataset.collate_fns import default_collate_fn
from functools import partial
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
from xtuner.dataset.map_fns import template_map_fn_factory
from transformers import TrainingArguments
import torch
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from xtuner.model import SupervisedFinetune
from xtuner.apis.datasets import alpaca_data_collator, alpaca_dataset
from mmengine.runner import Runner


def safe_load_data(file):
    tp = file.split('.')[-1]
    if tp == 'csv':
        return load_dataset('csv', data_files=dict(train=file))
    if 'json' in tp:
        return load_dataset('json', data_files=dict(train=file))
    
    # py
    return load_dataset(file, split='train')


def saft_build_model(model_name_or_path,
                            quantization_config=None,
                            lora_config=None,
                            return_tokenizer=True,
                            qlora_flag=True):
    if quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    if lora_config is None:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM')
    
    llm = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quantization_config if qlora_flag else None)

    try:
        model = SupervisedFinetune(llm, lora=lora_config)
    except Exception as e:
        model = SupervisedFinetune(llm, lora=lora_config, use_activation_checkpointing=False)

    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            encode_special_tokens=True)
        return model.llm, tokenizer
    else:
        return model.llm



def mm_run(
    model_name_or_path,
    dataset_name_or_path,
    work_dir,
    xtuner_type='qlora',
    progress=gr.Progress(track_tqdm=True)
):
    cfg_org = prepareConfig(
        model_name_or_path=model_name_or_path,
        dataset_name_or_path=dataset_name_or_path
    )
    pp = prepareUtil(cfg_org, work_dir=work_dir, lora_type=xtuner_type)
    cfg = pp.auto_prepare()
    runner = Runner.from_cfg(cfg)
    # runner = Runner.from_cfg(org_cfg)
    runner.train()
    # runner.test()
    return pp.work_dir


def hf_run(
    model_name_or_path,
    dataset_name_or_path,
    work_dir,
    xtuner_type='qlora',
    progress=gr.Progress(track_tqdm=True)
):
    cfg = prepareConfig(
            model_name_or_path=model_name_or_path,
            dataset_name_or_path=dataset_name_or_path
    )
    cfg.output_dir = work_dir
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(cfg.local_rank)

    cfg_dict = cfg.to_tr_dict()
    tr_args = TrainingArguments(**cfg_dict)
    print('=='*35)
    print('tr_args=', tr_args)
    print('=='*35)
    model, tokenizer = saft_build_model(
        model_name_or_path=cfg.model_name_or_path,
        return_tokenizer=True,
        qlora_flag=xtuner_type == 'qlora'
    )
    train_dataset = process_hf_dataset(
        dataset=safe_load_data(cfg.dataset_name_or_path),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        dataset_map_fn=cfg.dataset_map_fn,
        template_map_fn=template_map_fn_factory(template=cfg.task_prompt_template),
        pack_to_max_length=cfg.pack_to_max_length
    )
    # build trainer
    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_dataset,
        data_collator=partial(default_collate_fn, return_hf_format=True)
    )
    # training 
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=tr_args.output_dir)
    return tr_args.output_dir



class quickTrain:
    def __init__(self, 
                 model_name_or_path, 
                 dataset_name_or_path, 
                 work_dir,
                 xtuner_type='qlora', 
                 run_type='mmengine'):
        self.model_name_or_path = model_name_or_path
        self.dataset_name_or_path = dataset_name_or_path
        self.xtuner_type = xtuner_type
        self.run_type = run_type
        self.work_dir = work_dir
        self._break_flag = False
        self._t_handle_tr = None
    
    def set_model_path(self, model_path):
        self.model_name_or_path = model_path
    
    def set_data_path(self, data_path):
        self.dataset_name_or_path = data_path
    
    def set_xtuner_type(self, xtuner_type):
        self.xtuner_type = xtuner_type
        
    def set_work_dir(self, work_dir):
        self.work_dir = work_dir
        
    def _t_start(self):
        self._t_handle_tr = threading.Thread(target=self.quick_train, name=f'X-train-{self.run_type}', daemon=True)
        self._t_handle_tr.start()
        self._t_handle_tr.join()

    def _quick_train(self, progress=gr.Progress(track_tqdm=True)):
        if self.run_type == 'mmengine':
            return mm_run(self.model_name_or_path, self.dataset_name_or_path, self.work_dir, self.xtuner_type)
        return hf_run(self.model_name_or_path, self.dataset_name_or_path, self.work_dir, self.xtuner_type)
    
    def quick_train(self, progress=gr.Progress(track_tqdm=True)):
        self._break_flag = False
        self._t_start()
        if self._break_flag:
            return "Done! Xtuner had interrupted!"
        return self.final_out_path

    def break_train(self):
        # 然后杀死该线程
        # 删除文件
        if self._t_handle_tr is not None:
            print('>>>>>>>>>>>>>>>>> break_download')
            stop_thread(self._t_handle_tr)
            self._t_handle_tr = None
     
        self._break_flag = True
        return "Done! Xtuner had interrupted!"



def main_test():
    model_ = '/root/share/model_repos/internlm-chat-7b'
    model_2 = '/root/share/model_repos/internlm2-chat-7b'
    TR_ = quickTrain(
        model_name_or_path=model_,
        dataset_name_or_path='/root/ft-medqa/MedQA2019-structured-train.jsonl',
        work_dir='./work_dir',
        xtuner_type='qlora',
        run_type='mmengine'
    )
    TR_.quick_train()


if __name__ == '__main__':
    main_test()




