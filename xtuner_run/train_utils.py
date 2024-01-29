# python3
# Create Date: 2024-01-29
# Author: Scc_hy
# Func: 参数准备
# ===========================================================================================

import os
import transformers
from dataclasses import dataclass, field
from typing import List, Dict, ClassVar, Tuple, AnyStr, Callable
import warnings 
import os
import re
import inspect
import ctypes
import inspect
import ctypes
warnings.filterwarnings(action='ignore')


def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    try:
        _async_raise(thread.ident, SystemExit)
    except Exception as e:
        print(e)


@dataclass
class prepareConfig:
    model_name_or_path: AnyStr
    dataset_name_or_path: AnyStr
    do_train: bool = True
    save_strategy: AnyStr = 'epoch'
    lr_scheduler_type: AnyStr = 'cosine'
    logging_steps: int = 5
    framework: AnyStr = 'huggingface'
    output_dir: AnyStr = './work_dir'
    deepspeed: bool = None
    local_rank: int = -1
    seed: int = 42
    max_length: int =  2048
    pack_to_max_length: bool = True
    batch_size: int = 1
    per_device_train_batch_size: int = 1 # per_device
    per_device_eval_batch_size: int = 1 # per_device
    # accumulative_counts: int = 16
    gradient_accumulation_steps: int = 16
    dataloader_num_workers: int = 0
    max_epochs: int = 3
    num_train_epochs: float = 3.0 # TrainingArguments
    optim: AnyStr = "adamw_torch"
    # optim_args
    optim_type: AnyStr = 'bitsandbytes.optim.PagedAdamW32bit'
    learning_rate: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0
    max_norm: float = 1
    max_grad_norm: float = 1
    warmup_ratio: float = 0.03
    task_prompt_template: AnyStr = 'xtuner.utils.PROMPT_TEMPLATE.internlm_chat'
    # Save
    save_steps: int = 500
    save_total_limit: int = 2  # Maximum checkpoints to keep (-1 means unlimited)
    # Evaluate the generation performance during the training
    evaluation_freq: int = 500
    system: AnyStr = "" # SYSTEM_TEMPLATE.coder
    # prompt trick
    evaluation_inputs: List = field(default_factory=lambda: [
        '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
    ])
    # set visualizer
    visualizer = None
    # set log level
    log_level: AnyStr = 'info'
    # load from which checkpoint
    resume_from_checkpoint: AnyStr = None
    # whether to resume training from the loaded checkpoint
    resume: bool = False
    # Defaults to use random seed and disable `deterministic`
    randomness: Dict = field(default_factory=lambda: dict(seed=None, deterministic=False))
    trust_remote_code: bool = True
    env_cfg: Dict = field(default_factory=lambda: dict(
        # whether to enable cudnn benchmark
        cudnn_benchmark=False,
        # set multi process parameters
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        # set distributed parameters
        dist_cfg=dict(backend='nccl'),
    ))
    # Callable path
    dataset_map_fn:AnyStr = None
    
    def get_device_map(self):
        if self.deepspeed:
            self.device_map = None
        else:
            self.device_map = {
                '': int(os.environ.get('LOCAL_RANK', self.local_rank))
            }

    def to_tr_dict(self):
        self_dict = self.__dict__
        focus_key = [
            'output_dir',
            'deepspeed',
            'local_rank',
            'seed',
            'per_device_train_batch_size',
            'per_device_eval_batch_size',
            'dataloader_num_workers',
            'gradient_accumulation_steps',
            'num_train_epochs',
            'optim',
            'weight_decay',
            'adam_beta1',
            'adam_beta2',
            'max_grad_norm',
            'warmup_ratio',
            'save_steps',
            'save_total_limit',
            'log_level',
            'resume_from_checkpoint',
            'deepspeed',
            'do_train',
            'save_strategy' ,
            'lr_scheduler_type',
            'logging_steps',
        ]
        res = {}
        for k in focus_key:
            res[k] = self_dict[k]
        return res


class prepareUtil:
    def __init__(self, cfg: prepareConfig, work_dir: AnyStr, lora_type: str='qlora'):
        self.cfg = cfg
        self.cfg.output_dir = work_dir
        self.work_dir = work_dir
        self.qlora_flag = lora_type == 'qlora'
        self._model, self._tokenizer = self.prepare_model_tokenizer()

    def auto_prepare(self):
        train_dataset, train_dataloader = self.prepare_data()
        optim_wrapper, param_scheduler = self.prepare_scheduler_optimizer()
        custom_hooks, default_hooks = self.prepare_hook()
        return dict(
            model=self._model,
            work_dir=self.cfg.output_dir,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            test_dataloader=None,
            train_cfg=dict(by_epoch=True, max_epochs=self.cfg.max_epochs, val_interval=1),
            val_cfg=None,
            test_cfg=None,
            optim_wrapper=optim_wrapper,
            param_scheduler=param_scheduler,
            val_evaluator=None,
            test_evaluator=None,
            custom_hooks=custom_hooks,
            default_hooks=default_hooks,
            resume=self.cfg.resume_from_checkpoint is not None,
            env_cfg=self.cfg.env_cfg,
            visualizer=self.cfg.visualizer,
            log_level=self.cfg.log_level.upper(),
            randomness=self.cfg.randomness,
            launcher='none'
        )

    def prepare_scheduler_optimizer(self):
        optim_wrapper = dict(
            type='mmengine.optim.AmpOptimWrapper',
            optimizer=dict(
                type=self.cfg.optim_type, lr=self.cfg.learning_rate, betas=self.cfg.betas, weight_decay=self.cfg.weight_decay),
            clip_grad=dict(max_norm=self.cfg.max_norm, error_if_nonfinite=False),
            accumulative_counts=self.cfg.gradient_accumulation_steps,
            loss_scale='dynamic',
            dtype='float16'
        )
        param_scheduler = dict(
                type='mmengine.optim.CosineAnnealingLR',
                eta_min=0.0,
                by_epoch=True,
                begin=self.cfg.warmup_ratio * self.cfg.max_epochs,
                T_max=self.cfg.max_epochs,
                convert_to_iter_based=True
                )
        return optim_wrapper, param_scheduler
    
    def prepare_model_tokenizer(self):
        tokenizer = dict(
            type='transformers.AutoTokenizer.from_pretrained',
            pretrained_model_name_or_path=self.cfg.model_name_or_path,
            trust_remote_code=True,
            padding_side='right'
        )
        model = dict(
            type='xtuner.model.SupervisedFinetune',
            llm=dict(
                type='transformers.AutoModelForCausalLM.from_pretrained',
                pretrained_model_name_or_path=self.cfg.model_name_or_path,
                trust_remote_code=True,
                torch_dtype='torch.float16',
                quantization_config=dict(
                    type='transformers.BitsAndBytesConfig',
                    load_in_4bit=True,
                    load_in_8bit=False,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype='torch.float16',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4') if self.qlora_flag  else None
                ),
            lora=dict(
                type='peft.LoraConfig',
                r=64,
                lora_alpha=16,
                lora_dropout=0.1,
                bias='none',
                task_type='CAUSAL_LM'))
        return model, tokenizer

    def prepare_hook(self):
        custom_hooks = [
            dict(type='xtuner.engine.DatasetInfoHook', tokenizer=self._tokenizer),
            dict(
                type='xtuner.engine.EvaluateChatHook',
                tokenizer=self._tokenizer ,
                every_n_iters=self.cfg.evaluation_freq,
                evaluation_inputs=self.cfg.evaluation_inputs,
                system=self.cfg.system,
                prompt_template=self.cfg.task_prompt_template)
        ]
        # configure default hooks
        default_hooks = dict(
            checkpoint=dict(interval=1, type='mmengine.hooks.CheckpointHook'),
            logger=dict(interval=10, type='mmengine.hooks.LoggerHook'),
            param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
            sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
            timer=dict(type='mmengine.hooks.IterTimerHook')
        )
        return custom_hooks, default_hooks

    def prepare_data(self):
        train_dataset = dict(
            type='xtuner.dataset.process_hf_dataset',
            dataset=self.safe_load_dataset(self.cfg.dataset_name_or_path),
            tokenizer=self._tokenizer,
            max_length=self.cfg.max_length,
            dataset_map_fn=self.cfg.dataset_map_fn,
            template_map_fn=dict(
                type='xtuner.dataset.map_fns.template_map_fn_factory', 
                template=self.cfg.task_prompt_template),
            remove_unused_columns=True,
            shuffle_before_pack=True,
            pack_to_max_length=self.cfg.pack_to_max_length)

        train_dataloader = dict(
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.dataloader_num_workers,
            dataset=train_dataset,
            sampler=dict(type='mmengine.dataset.DefaultSampler', shuffle=True),
            collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'))
        return train_dataset, train_dataloader
    
    def safe_load_dataset(self, file):
        load_tp = 'datasets.load_dataset'
        tp = file.split('.')[-1]
        if tp == 'csv':
            return dict(type=load_tp, path='csv', data_files=dict(train=file))
        if 'json' in tp:
            return dict(type=load_tp, path='json', data_files=dict(train=file))
        # py
        return dict(type=load_tp, path=file)


def main_test():
    import transformers
    cfg = prepareConfig(
        model_name_or_path='/root/opencompass/InternLM/Shanghai_AI_Laboratory/internlm2-chat-7b', 
        dataset_name_or_path='/root/ft-medqa/MedQA2019-structured-train.jsonl'
    )
    print(f'type(cfg.evaluation_inputs)={type(cfg.evaluation_inputs)}')
    print(cfg.evaluation_inputs)
    pp = prepareUtil(cfg)
    pp_res = pp.auto_prepare()
    # pp_res = cfg.to_tr_dict()
    print('--'*25)
    print(pp_res.custom_hooks)
    print(pp_res.custom_hooks[0]['type'])
    # print('=='*35)


if __name__ == '__main__':
    main_test() # checked


