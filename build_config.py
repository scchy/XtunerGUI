from mmengine import Config, ConfigDict
from mmengine.config.lazy import LazyObject
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE
import torch
import os
CUR_DIR = os.path.dirname(__file__)

DATA2MAPFN = {
    'tatsu-lab/alpaca': 'alpaca_map_fn',
    'silk-road/alpaca-data-gpt4-chinese': 'alpaca_zh_map_fn',
    'garage-bAInd/Open-Platypus': 'alpaca_map_fn',
    'HuggingFaceH4/CodeAlpaca_20K': 'code_alpaca_map_fn',
    'burkelibbey/colors': 'colors_map_fn',
    'shibing624/medical': 'medical_map_fn',
    'damo/MSAgent-Bench': 'msagent_react_map_fn',
    'timdettmers/openassistant-guanaco': 'oasst1_map_fn',
    'Open-Orca/OpenOrca': 'openorca_map_fn',
    'Skywork/SkyPile-150B': 'pretrain_map_fn',
    'mistralai/Mistral-7B-v0.1': 'pretrain_map_fn',
    'b-mc2/sql-create-context': 'sql_map_fn',
    'ArmelR/stack-exchange-instruction': 'stack_exchange_map_fn',
    'nampdn-ai/tiny-codes': 'tiny_codes_map_fn',
    'WizardLM/WizardLM_evol_instruct_V2_196k': 'wizardlm_map_fn',
}

def data_path_map_fn(file):
    if file in DATA2MAPFN:
        return DATA2MAPFN[file]
    for k, v in DATA2MAPFN.items():
        k_list = k.split('/')
        k_fix = '_'.join(k_list)
        if k_fix in file:
            return v
    return None


"""
save_checkpoint_ratio -> save_checkpoint_interval
accumulative_counts -> accumulative_counts
新增 save_total_limit
'bigcode/starcoder' 不是DATA_LIST 
"""
def traverse_keys(cfg_dict, target_keys, new_value):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if key in target_keys:
                cfg_dict[key] = new_value
            else:
                traverse_keys(value, target_keys, new_value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            traverse_keys(value, target_keys, new_value)

def traverse_value(cfg_dict, target_value, new_value):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if value == target_value:
                cfg_dict[key] = new_value
            else:
                traverse_value(value, target_value, new_value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            traverse_value(value, target_value, new_value)


def set_model_related(cfg, model_path):
    traverse_keys(cfg._cfg_dict, ('pretrained_model_name_or_path', ), model_path)


def set_data_related(cfg, dataset, is_custom_dataset, prompt_template, max_length, pack_to_max_length):
    if is_custom_dataset:
        dataset = ConfigDict(path='json', data_files=dataset)
        cfg.alpaca_en.dataset.update(dataset)
        cfg.train_dataloader.dataset.dataset.update(dataset)

        traverse_keys(cfg._cfg_dict, ('dataset_map_fn', ), LazyObject('xtuner.dataset.map_fns', 'openai_map_fn'))
    else:
        traverse_value(cfg._cfg_dict, 'tatsu-lab/alpaca', dataset)

        traverse_keys(cfg._cfg_dict, ('dataset_map_fn', ), LazyObject('xtuner.dataset.map_fns', data_path_map_fn(dataset)))

    assert prompt_template in PROMPT_TEMPLATE, \
        f'Expect prompt_template to be one of {PROMPT_TEMPLATE.keys()}, but got {prompt_template}.'
    prompt_template = PROMPT_TEMPLATE[prompt_template]
    traverse_keys(cfg._cfg_dict, ('template', 'prompt_template'), prompt_template)

    traverse_keys(cfg._cfg_dict, ('max_length', ), max_length)

    traverse_keys(cfg._cfg_dict, ('pack_to_max_length', ), pack_to_max_length)


def set_scheduler_optimizer_related(
        cfg, batch_size_per_device, accumulative_counts, dataloader_num_workers,
        max_epochs, optim_type, lr, beta1, beta2, weight_decay, max_norm, warmup_ratio):
    traverse_keys(cfg._cfg_dict, ('batch_size', ), batch_size_per_device)
    traverse_keys(cfg._cfg_dict, ('accumulative_counts', ), accumulative_counts)
    traverse_keys(cfg._cfg_dict, ('dataloader_num_workers', 'num_workers'), dataloader_num_workers)

    traverse_keys(cfg._cfg_dict, ('max_epochs', 'T_max'), max_epochs)
    cfg.param_scheduler[0].end = warmup_ratio * max_epochs
    cfg.param_scheduler[1].begin = warmup_ratio * max_epochs
    cfg.warmup_ratio = warmup_ratio

    assert hasattr(torch.optim, optim_type)
    cfg.optim_type = LazyObject('torch.optim', optim_type)
    cfg.optim_wrapper.optimizer.type = LazyObject('torch.optim', optim_type)

    cfg.lr = lr
    cfg.optim_wrapper.optimizer.lr = lr

    if optim_type == 'AdamW':
        traverse_keys(cfg._cfg_dict, ('betas', ), (beta1, beta2))
    
    traverse_keys(cfg._cfg_dict, ('weight_decay', ), weight_decay)
    traverse_keys(cfg._cfg_dict, ('max_norm', ), max_norm)


def set_checkpoint_related(cfg, save_checkpoint_interval, save_total_limit):
    cfg.save_steps = save_checkpoint_interval
    cfg.default_hooks.checkpoint.interval = save_checkpoint_interval

    cfg.save_total_limit = save_total_limit
    cfg.default_hooks.checkpoint.max_keep_ckpts = save_total_limit


def set_evaluate_related(cfg, evaluation_freq, evaluation_system_prompt, evaluation_input1, evaluation_input2):
    traverse_keys(cfg._cfg_dict, ('evaluation_freq', 'every_n_iters'), evaluation_freq)

    system_prompt = SYSTEM_TEMPLATE[evaluation_system_prompt] if evaluation_system_prompt else ''
    traverse_keys(cfg._cfg_dict, ('SYSTEM', 'system'), system_prompt)

    evaluation_inputs = [evaluation_input1, evaluation_input2]
    traverse_keys(cfg._cfg_dict, ('evaluation_inputs', ), evaluation_inputs)


def build_config(
        ft_method, model_path, dataset, is_custom_dataset, deepspeed, lr, warmup_ratio, batch_size_per_device,
        accumulative_counts, num_GPU, max_length, pack_to_max_length, max_epochs, save_checkpoint_interval, save_total_limit,
        evaluation_freq, evaluation_system_prompt, evaluation_input1, evaluation_input2,
        optim_type, weight_decay, max_norm, dataloader_num_workers, beta1, beta2, 
        prompt_template):
    if ft_method == 'full':
        cfg = Config.fromfile(f'{CUR_DIR}/template_configs/full_finetune.py')
    elif ft_method == 'lora':
        cfg = Config.fromfile(f'{CUR_DIR}/template_configs/lora.py')
    elif ft_method == 'qlora':
        cfg = Config.fromfile(f'{CUR_DIR}/template_configs/qlora.py')
    else:
        raise NotImplementedError(f'Expect ft_method to be one of (full, lora, qlora), but got {ft_method}.')

    set_model_related(cfg, model_path)
    set_data_related(cfg, dataset, is_custom_dataset, prompt_template, max_length, pack_to_max_length)
    set_scheduler_optimizer_related(cfg, batch_size_per_device, accumulative_counts, dataloader_num_workers,
        max_epochs, optim_type, lr, beta1, beta2, weight_decay, max_norm, warmup_ratio)
    set_checkpoint_related(cfg, save_checkpoint_interval, save_total_limit)
    set_evaluate_related(cfg, evaluation_freq, evaluation_system_prompt, evaluation_input1, evaluation_input2)

    return cfg


kwargs = dict(
    ft_method='full', 
    model_path='/mnt/petrelfs/share_data/caoweihan/official_Ampere_7B_1_0_0', 
    dataset='timdettmers/openassistant-guanaco',
    is_custom_dataset=False, 
    deepspeed=None,  # 与生成config无关
    lr=2e-5, 
    warmup_ratio=0.03, 
    batch_size_per_device=1, 
    accumulative_counts=2, 
    num_GPU=None,  # 与生成config无关
    max_length=2048,
    pack_to_max_length=True,
    max_epochs=2,
    save_checkpoint_interval=1000,
    save_total_limit=2,
    evaluation_freq=100,
    evaluation_system_prompt='',
    evaluation_input1='请给我介绍五个上海的景点',
    evaluation_input2='Please tell me five scenic spots in Shanghai',
    optim_type='AdamW',
    weight_decay=0,
    max_norm=1,
    dataloader_num_workers=0,
    beta1=0.9,
    beta2=0.999,
    prompt_template='internlm2_chat'
    )

int_args = [
    'batch_size_per_device',
    'accumulative_counts',
    'num_GPU',
    'max_length',
    'pack_to_max_length',
    'max_epochs',
    'save_checkpoint_interval',
    'save_total_limit',
    'evaluation_freq',
    'dataloader_num_workers',
]
default_args_key = [
    'ft_method',
    'model_path',
    'dataset',
    'deepspeed',
    'lr',
    'warmup_ratio',
    'batch_size_per_device',
    'accumulative_counts',
    'num_GPU',
    'max_length',
    'pack_to_max_length',
    'max_epochs',
    'save_checkpoint_interval',
    'save_total_limit',
    'evaluation_freq',
    'evaluation_system_prompt',
    'evaluation_input1',
    'evaluation_input2',
    'optim_type',
    'weight_decay',
    'max_norm',
    'dataloader_num_workers',
    'beta1',
    'beta2',
    'prompt_template',
]

def build_config_path(root_dir):
    work_dir = os.path.join(root_dir, 'work_dir')
    if not os.path.exists(work_dir):
        os.system(f'mkdir -p {work_dir}')
    return os.path.join(work_dir, 'xtuner_config.py')


def build_and_save_config(dataset_personal_path, root_dir, *args, **kwargs):
    kwargs.update(
        dict(zip(default_args_key, list(args)))
    )
    print(f'dataset_personal_path={dataset_personal_path}||')
    kwargs['is_custom_dataset'] = False
    if dataset_personal_path is not None and len(dataset_personal_path) >= 3:
        kwargs['is_custom_dataset'] = True 
        kwargs['dataset'] = dataset_personal_path
    for k in int_args:
        kwargs[k] = int(kwargs[k])
    print(f'kwargs={kwargs}')
    cfg = build_config(**kwargs)
    cfg_py = build_config_path(root_dir)
    cfg.dump(cfg_py)
    print('cfg_py=', cfg_py)
    return cfg_py


if __name__ == '__main__':
    build_and_save_config('.', **kwargs)
