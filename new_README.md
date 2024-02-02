<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />    
                
[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)                

<div align="left">

# 项目背景

在开源大语言模型愈发流行的当下，越来越多的普通人也希望能够通过一块家用的消费级显卡（比如说英伟达2080TI、3090等）能够实现开源大语言模型的个性化定制操作。目前而言，主流的自定义开源大语言模型的方式就是基于外挂数据库技术（如 RAG，即检索增强的生成模型）以及模型微调技术（fine-tuning)。

虽然微调模型和使用外挂数据库技术各有优势，但微调模型的主要优势在于其能够更深入地理解和生成与特定任务高度相关的内容。通过在特定数据集上微调，模型能够学习并模仿该数据集的特定语言模式、术语和知识，使得生成的内容更加贴合特定的应用场景。相比之下，外挂数据库技术虽然能够利用庞大的知识库来增强模型的回答能力，但它依赖于检索机制来找到相关信息，可能不如直接微调模型在生成连贯、准确回答方面那么灵活和精准。此外，微调模型可以在没有外部数据库访问的情况下独立运作，这对于需要快速响应或在资源受限的环境中部署的应用来说是一个重要优势。

XTuner作为InternLM团队开发的一款简单易上手的微调开发工具包。一直以来深受开发者们喜爱。但由于对于基础比较薄弱的小白而言，还是具有一定的技术门槛。因此，借由InternLM官方推出的大模型实战训练营的机会，我们小组成员有幸与XTuner官方技术团队合作，在参考了 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的基础上，根据XTuner的特性进行修改完善，从而完成了基于Gradio的XTuner可视化的界面设计。此界面旨在帮助基础薄弱的小白能够通过简单的点击即可尝试微调模型的操作。并且可以实时查看到训练时的信息，训练后的结果，并且能够进行微调模型与原模型的对比测试。另外，除了官方支持的模型和数据集以外，进阶用户也可以将自己的模型和数据集上传并进行微调操作，并且自定义的模型也有利于帮助小白在基于自己微调完的模型基础上完成进一步的微调工作。


# 整体流程逻辑介绍

![XTuner (1)](https://github.com/scchy/XtunerGUI/assets/108343727/06283b3b-d397-4771-aa45-7e6702a0f63f)

## 第一步：路径设置

首先在最开始，我们需要设置两个路径，一个是文件保存路径（里面存放着模型配置文件，模型训练的过程文件及模型最后转化整合的文件），另外一个文件夹则是模型数据集文件保存路径，里面将会存放在网上下载的模型及数据集文件。

## 第二步：模型、数据集及微调方法设置

在这一步，我们需要选择的有三部分内容：

- **合适的模型微调方法（qlora、lora和全量微调）**：不同的微调方法将会影响第三部分超参数的设置。
- **模型选择**：我们可以选择官方支持的模型并点击按钮进行下载，也可以把本地的模型所在的路径填在自定义模型的位置。选择官方支持的模型将自动更改提示词模板（prompt_template)的部分，假如是通过自定义模型上传的需要点击按钮尝试模型是否符合huggingface要求后，根据规则选择匹配的提示词模板。假如发现无法训练，可能需要自行选择或添加提示词模板。
- **数据集选择**：与模型选择类似，我们也可以选择官方支持的数据集，也可以根据本地的数据上传自定义的数据集。但是需要注意的是，目前我们所支持的自定义数据集格式为OpenAI格式，具体可以看页面内的展示。因此在上传数据集前还需要将格式转化为OpenAI格式。自定义的数据集上传完毕后还可通过检查按钮来确保数据集准确无误。

## 第三步：相关参数设置

在参数设置部分，其实根据第一步中选择的微调方法已经设置了大部分的参数，我们可以不更改参数直接进入模型训练的步骤。但是假如有自己独特的需求要对模型进行测试的话，也可以考虑进行修改。参数的话分为基础参数和进阶参数：

- **基础参数**：包括了一些常见的参数，比如说学习率(lr)，所使用的GPU的数据等等。这些可能需要根据自己要求更改。
- **进阶参数**：这部分就是一些不常用的参数，可以视情况自行更改。

在修改完参数后，我们就可以点击按钮生成模型的配置文件。配置文件的生成是基于前三步的参数集中进行生成的，有了配置文件并准备好模型和数据集后就可以开始训练了。

## 第四步：XTuner！启动该！

![元神启动](https://github.com/scchy/XtunerGUI/assets/108343727/c1ac21fb-7c87-4818-8566-ab199a760a0b)

在完成前三步骤后，我们只需要点击“XTuner！启动！“按钮即可开始模型的训练。训练的过程文件可以从下方的终端显示查看，可以查看到模型训练的过程内容。假如我们对模型效果不满意，我们也可以随时中断模型训练的过程。并且对于中途不小心停下来的模型，我们也可以通过点击模型续训来实现，只需要选定最终的权重文件即可。

## 第五步：训练结果展示

对于训练好的模型，我们可以根据训练的过程的json文件来生成一些训练的结果，包括学习率的变化曲线图、损失值的变化曲线图等等。另外我们还能够查看不同权重文件下设置的默认对话的改变以监控模型是否过拟合。

## 第六步：模型转化及测试

对于训练好的模型，我们还需要将其转为标准的huggingface格式才可以真实的进行使用。对于使用QLora或者Lora技术的，也可以选择将训练好的adapter与原模型整合为一个大模型以供后续进行进一步的使用。转换和合并好的模型就可以通过启动对话服务来实际对比原模型和微调完模型的效果了。




# 任务划分

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

  <br /><br />    

