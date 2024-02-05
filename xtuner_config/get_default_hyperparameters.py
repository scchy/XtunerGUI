import copy
from collections import OrderedDict


DEFAULT_HYPERPARAMETERS = OrderedDict(
    warmup_ratio=0.03,
    batch_size_per_device=1, 
    accumulative_counts=2, 
    num_GPU=8,
    max_length=2048,
    pack_to_max_length=True,
    evaluation_freq=500,
    optim_type='AdamW',
    weight_decay=0,
    max_norm=1,
    dataloader_num_workers=0,
    beta1=0.9,
    beta2=0.999,
    lr=2e-5,
    save_checkpoint_interval=500,
    save_total_limit=2
)


def get_default_hyperparameters(ft_method):
    out_dict = copy.deepcopy(DEFAULT_HYPERPARAMETERS)
    if ft_method.lower() == 'full':
        out_dict.update(dict(
            lr=2e-5,
            save_checkpoint_interval=500,
            save_total_limit=2))
    else:
        out_dict.update(dict(
            lr=2e-4,
            save_checkpoint_interval=200,
            save_total_limit=5))
    out_list = []
    for i in DEFAULT_HYPERPARAMETERS.keys():
        out_list.append(out_dict[i])
    return out_list


if __name__ == '__main__':
    print(DEFAULT_HYPERPARAMETERS.keys())
    print(
        get_default_hyperparameters('full')
    )
    print(
        get_default_hyperparameters('lora')
    )
    
