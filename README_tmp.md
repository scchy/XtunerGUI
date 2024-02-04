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
  - [ ] 自定义模型 
     1. template 映射 -> 
     2. 路径校验（template）
  - [X] 路径最终确定
  - [ ] prompt_template 位置改动？
  - [ ] 自定义数据集 只支持openAI 数据集格式


```text
customer_path  
|-- download_cache
|   |-- data_download
|   |   `-- tatsu-lab_alpaca
|   `-- model_download
|       `-- internlm_internlm-chat-7b
`-- work_dir
    |-- 20240202_153301
    |   |-- 20240202_153301.log
    |   `-- vis_data
    |-- iter_100.pth
    |-- iter_50.pth
    |-- last_checkpoint
    |-- xtuner_config.py
    `-- xtuner_iter_100_hf
        |-- README.md
        |-- adapter_config.json
        |-- adapter_model.safetensors
        `-- xtuner_config.py

```


## Test
- [X] customer-root /root/sccTest3
- [X] customer-data-dir /root/download_cache
- [X] customer model: /root/share/model_repos/internlm-chat-7b
  - [X] check customer model template detect
  - [X] -> detect_prompt_template -> prompt_template_show
- [X] customer dataset: 
  - /root/personal_assistant/data/personal_assistant_openai_final.json 
  - /root/xtunerUITest/ttt.json
- [X] data: tatsu-lab/alpaca -> downloading
- [X] config
  - [X] ft_method -> DEFAULT_HYPERPARAMETERS
  - [X] generate check
- [ ] xtuner
  - [ ] running without pregress ?
- show result
  - [X] plot
  - [X] dynamic select_checkpoint -> 
- convert
  - [X] choose pth
  - [ ] convert progress






