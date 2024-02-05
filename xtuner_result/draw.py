import matplotlib.pyplot as plt
from datasets import load_dataset
import altair as alt
import pandas as pd
import os
import re
import gradio as gr


class resPlot:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.log_file = None
        self.iter_dir_list = []
        self.get_log_path()
        self.find_iter_pth()
        print(f"resPlot(self.log_file={self.log_file})")

    def get_log_path(self):
        try:
            list_ =  sorted([i for i in os.listdir(self.work_dir) if '.' not in i and re.match(r'\d+_\d+', i)])
            dir_name = list_[-1]
            self.log_file = os.path.join(self.work_dir, dir_name, 'vis_data' , f'{dir_name}.json')
        except Exception as e:
            print(e)
            pass
    
    def find_iter_pth(self):
        try:
            self.iter_dir_list = sorted([i for i in os.listdir(self.work_dir) if '.pth' in i])
        except Exception as e:
            print(e)
            pass
    
    def get_eval_test(self, ep_pth):
        ep_str = ep_pth.split('.')[0]
        ep = ep_str.split('_')[0] + '_' + str(int(ep_str.split('_')[1]) - 1)
        list_ =  sorted([i for i in os.listdir(self.work_dir) if '.' not in i and re.match(r'\d+_\d+', i)])
        dir_name = list_[-1]
        eval_file = os.path.join(self.work_dir, dir_name, 'vis_data' , f'eval_outputs_{ep}.txt')
        try:
            return open(eval_file, 'r').read()
        except Exception as e:
            return f'eval_file={eval_file}\nERROR: {e} '
    
    def dynamic_eval_drop_down(self):
        list_ =  sorted([i for i in os.listdir(self.work_dir) if '.' not in i and re.match(r'\d+_\d+', i)])
        dir_name = list_[-1]
        # /root/xtunerUITest/test/appPrepare/work_dir/20240204_204337/vis_data/eval_outputs_iter_49.txt
        eval_file = [i for i in os.listdir(os.path.join(self.work_dir, dir_name, 'vis_data')) if '.txt' in i]
        final_list = []
        if len(eval_file):
            final_list = ["iter_{}".format(int(i.split('_')[-1].split('.')[0])+1) for i in eval_file]

        return gr.Dropdown(choices=final_list, interactive=True)
    
    def dynamic_drop_down(self):
        self.iter_dir_list = sorted([i for i in os.listdir(self.work_dir) if '.pth' in i])
        return gr.Dropdown(choices=self.iter_dir_list, interactive=True)

    def reset_work_dir(self, root_dir):
        self.work_dir = f'{root_dir}/work_dir'
        self.get_log_path()
        self.find_iter_pth()
        print(f"resPlot -> self.work_dir={self.work_dir}\nself.log_file={self.log_file}\nself.iter_dir_list={self.iter_dir_list}")

    def lr_plot(self):
        self.get_log_path()
        y_axis_name = 'lr'
        return self.gr_line_plot(y_axis_name, self.log_file)

    def loss_plot(self):
        self.get_log_path()
        y_axis_name = 'loss'
        return self.gr_line_plot(y_axis_name, self.log_file)
    
    @staticmethod
    def make_plot(y_axis_name, log_path):
        ds = load_dataset('json', data_files=log_path)
        ds = ds['train'].to_pandas()
        ds = ds.rename(columns={'iter': 'iter_num'})
        # ['lr', 'data_time', 'loss', 'time', 'grad_norm', 'iter', 'memory', 'step']
        source = pd.DataFrame({
            'iter_num': ds['iter_num'].map(int).tolist(),
            y_axis_name: ds[y_axis_name].map(float).tolist(),
        })
        base = alt.Chart(source).mark_line(
            point=alt.OverlayMarkDef(filled=False, fill="white")
            ).encode(x='iter_num',y=y_axis_name) 
        return base
    
    @staticmethod
    def gr_line_plot(y_axis_name, log_path):
        ds = load_dataset('json', data_files=log_path)
        ds = ds['train'].to_pandas()
        ds = ds.rename(columns={'iter': 'iter_num'})
        source = pd.DataFrame({
            'iter_num': ds['iter_num'].map(int).tolist(),
            y_axis_name: ds[y_axis_name].map(float).tolist(),
        })
        return gr.LinePlot(
            source,
            x="iter_num",
            x_title='iter_num',
            y=y_axis_name,
            y_title=y_axis_name,
            overlay_point=True,
            tooltip=["iter_num", y_axis_name],
            title=y_axis_name,
            height=300,
            width=500,
        )

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


if __name__ == '__main__':
    draw('loss', './dummy_log.json', 'd1.png')
