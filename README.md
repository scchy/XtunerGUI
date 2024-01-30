# XtunerGUI
Xtuner Factory 

[Disign Doc: XtunerGUI](https://aab2vs0do9o.feishu.cn/docx/JWkbdoDiboVKBAxUyQvcg9MQnbb?from=from_copylink)



- 下载`xtuner_donwload`
  - 模型下载
    - 入参: `model: gr.Dropdown`
    - 出参: `model_path: gr.Textbox`
      - 路径： 当前版本路径 `XtunerGUI/appPrepare/download_cache/model_download/{username}_{repository}`
  - 数据下载
    - 入参: `dataset: gr.Dropdown`
    - 出参: `data_path: gr.Textbox`   --cache_dir
      - 路径：当前版本路径 `XtunerGUI/appPrepare/download_cache/data_download/dataset_{username}_{repository}`
    - todo: load_dataset 下载数

- config `xtuner_config`
  - 入参: 非常多
  - 下载的数据是否 load_dataset -> cache  `os.envison[hf-cache-dataset]`
    - load_dataset  环境变量调整

- fintune `xtuner_run`
  - 指定 环境变量
  - shell_train: 直接执行shell `xtuner train {config_py_path} {add_} --work_dir {work_dir} > {log_file}` 
    - add_: `--deepspeed deepspeed_{self.deepspeed_seed} `
    - 入参: `config_py_path`
    - 出参: `config_py_path.work_dir` : 一般情况下的路径？
  - 日志显示: ->  日志显示慢的问题 

- 转换合并 `xtuner_convert`
  - `convert_and_merge.py`
  - 入参: 
    - todo： 选择epoch
    - `config_file`: `xtuner_config` 生成
    - `pth_model`: work_dir 目录下模型问题 
    - `save_hf_dir`: 指定生成目录 {work_dir}/hf
    - `model_path`: 模型路径 `xtuner_donwload` 产出 `model_path`
    - `save_merged_dir`: 指定生成目录 {work_dir}/merge_epoch{n}


todo: 
    1. todo: load_dataset 下载数
    2. 测试问题 是否可以添加,  直接输入list 
    3. 自定义模型 路径校验（template）
       1. template 映射
    4. 路径最终确定： hf merge 

