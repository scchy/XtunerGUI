custom_hooks = [
    dict(
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.DatasetInfoHook'),
    dict(
        evaluation_inputs=[
            '请给我介绍五个上海的景点',
            'Please tell me five scenic spots in Shanghai',
        ],
        every_n_iters=500,
        prompt_template='xtuner.utils.PROMPT_TEMPLATE.internlm_chat',
        system='',
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.EvaluateChatHook'),
]
default_hooks = dict(
    checkpoint=dict(interval=1, type='mmengine.hooks.CheckpointHook'),
    logger=dict(interval=10, type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
log_level = 'INFO'
model = dict(
    llm=dict(
        pretrained_model_name_or_path='',
        quantization_config=dict(
            bnb_4bit_compute_dtype='torch.float16',
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_has_fp16_weight=False,
            llm_int8_threshold=6.0,
            load_in_4bit=True,
            load_in_8bit=False,
            type='transformers.BitsAndBytesConfig'),
        torch_dtype='torch.float16',
        trust_remote_code=True,
        type='transformers.AutoModelForCausalLM.from_pretrained'),
    lora=dict(
        bias='none',
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        task_type='CAUSAL_LM',
        type='peft.LoraConfig'),
    type='xtuner.model.SupervisedFinetune')
optim_wrapper = dict(
    accumulative_counts=16,
    clip_grad=dict(error_if_nonfinite=False, max_norm=1),
    dtype='float16',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.0002,
        type='bitsandbytes.optim.PagedAdamW32bit',
        weight_decay=0),
    type='mmengine.optim.AmpOptimWrapper')
param_scheduler = dict(
    T_max=3,
    begin=0.09,
    by_epoch=True,
    convert_to_iter_based=True,
    eta_min=0.0,
    type='mmengine.optim.CosineAnnealingLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        dataset=dict(path='', type='datasets.load_dataset'),
        dataset_map_fn=None,
        max_length=2048,
        pack_to_max_length=True,
        remove_unused_columns=True,
        shuffle_before_pack=True,
        template_map_fn=dict(
            template='xtuner.utils.PROMPT_TEMPLATE.internlm_chat',
            type='xtuner.dataset.map_fns.template_map_fn_factory'),
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.dataset.process_hf_dataset'),
    num_workers=0,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
val_cfg = None
val_dataloader = None
val_evaluator = None
visualizer = None
work_dir = '/home/scc/sccWork/myGitHub/XtunerGUI/appPrepare'
