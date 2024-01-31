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
    - 出参: `data_path: gr.Textbox`
      - 路径：当前版本路径 `XtunerGUI/appPrepare/download_cache/data_download/dataset_{username}_{repository}`

- config `xtuner_config`
  - 入参: 非常多
  - 出参：`cfg_py_box`

- fintune `xtuner_run`
  - 指定 环境变量
  - shell_train: 直接执行shell `xtuner train xxxx` 
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
  - [X] todo: load_dataset 下载数 
  - [ ] 测试问题 是否可以添加,  直接输入list 
  - [ ] 自定义模型 路径校验（template）
     1. template 映射
  - [X] 路径最终确定

```text
customer_path  
|-- download_cache
|   |-- data_download
|   `-- model_download
`-- work_dir
    |-- 20240131_145313
    |-- 20240131_152320
    |-- __xtuner_tr.log
    |-- iter_250.pth
    |-- iter_300.pth
    |-- last_checkpoint
    |-- xtuner_config.py
    |-- xtuner_hf
    |-- xtuner_merge
    `-- zero_to_fp32.py

```


## Test
- /root/sccTest2
- model: /root/share/model_repos/internlm-chat-7b
- data: tatsu-lab/alpaca -> downloading
- config
- xtuner
- convert
  - /root/sccTest2/work_dir/iter_250.pth

