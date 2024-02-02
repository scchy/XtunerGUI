import matplotlib.pyplot as plt
from datasets import load_dataset


def draw(y_axis_name, log_path, save_path):
    """
    Args:
        y_axis_name: One of ('lr', 'loss')
    """
    ds = load_dataset('json', data_files=log_path)
    ds = ds['train']
    x = ds['iter']
    y = ds[y_axis_name]
    plt.figure(figsize=(10,5))

    plt.plot(x, y, marker='.')

    plt.title(f'Training Iterations vs {y_axis_name}')
    plt.xlabel('Training iterations')
    plt.ylabel(y_axis_name)

    plt.savefig(save_path)

draw('loss', './dummy_log.json', 'd1.png')
