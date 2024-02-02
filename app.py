# python3
# Create Date: 2024-01-26
# ========================================

from xtuner_download.download_model import xtunerModelDownload
from xtuner_download.download_dataset import xtunerDataDownload
from xtuner_convert.convert_and_merge import convert_and_merged
from xtuner_run.shell_train import quickTrain
from appPrepare.files_prepare import DATA_DOWNLOAD_DIR, MODEL_DOWNLOAD_DIR, CUR_PATH, DEFAULT_DOWNLOAD_DIR
from appPrepare.list_prepare import DATA_LIST, MODEL_LIST, PROMPT_TEMPLATE_LIST
from appPrepare.func_prepare import read_first_ten_lines, get_template_format_by_name, OPENAI_FORMAT
from xtuner_config.build_config import build_and_save_config
from xtuner_config.check_custom_dataset import check_custom_dataset
from xtuner_config.get_prompt_template import app_get_prompt_template
from xtuner_config.get_default_hyperparameters import get_default_hyperparameters
from tqdm import tqdm
from xtuner_result.draw import resPlot
import gradio as gr
import warnings
warnings.filterwarnings(action='ignore')


def combine_message_and_history(message, chat_history):
    # 将聊天历史中的每个元素（假设是元组）转换为字符串
    history_str = "\n".join(f"{sender}: {text}" for sender, text in chat_history)

    # 将新消息和聊天历史结合成一个字符串
    full_message = f"{history_str}\nUser: {message}"
    return full_message

def respond(message, chat_history):
#     message1 = combine_message_and_history(message,chat_history)
#     client = OpenAI()
#     messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": message1}
#   ]
  
#     completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=messages,
#     max_tokens=150,        # 设置生成响应的最大 token 数量
#     seed=12345,            # 设置种子以获得确定性采样（如果可能）
#     temperature=0.7,       # 设置采样温度
#     top_p=0.9              # 设置核心采样的概率质量百分比
#     )
#     bot_message_text =  completion.choices[0].message.content
#     #这里的bot_message_text就是最后输出的文本
#     chat_history.append((message, bot_message_text))
      return "", chat_history

def clear_history(chat_history):
    chat_history.clear()
    return chat_history

def regenerate(chat_history):
    if chat_history:
        # 提取上一条输入消息
        last_message = chat_history[-1][0]
        # 移除最后一条记录
        chat_history.pop()
        # 使用上一条输入消息调用 respond 函数以生成新的回复
        msg,chat_history = respond(last_message, chat_history)
    # 返回更新后的聊天记录
    return msg, chat_history

def undo(chat_history):
        chat_history.pop()
        return chat_history


with gr.Blocks() as demo:
    gr.Markdown(value='''  
<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />    
                
[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)                
                ''')
    
    with gr.Tab("基础训练"):
        with gr.Accordion(label='使用指南', open=False):
            gr.Markdown('## 流程图')
            # process = gr.Image(value='/root/XtunerGUI/pictures/workflow.jpg',label='使用流程图',container=False,show_download_button=False )
            gr.Markdown('## 演示视频')
            # video_customer_introduction = gr.Video(label='Xtuner GUI用法演示',value='/mnt/d/xtuner/demo.mp4',interactive=False)
        gr.Markdown("## 1. 本地路径设置")
        with gr.Row():
            local_path = gr.Textbox(
            label='请上传所有文件保存的文件本地路径', 
            value=CUR_PATH,
            info='将会在选择的路径下创建名为xxx的文件夹并将所有文件保存在此处'
        )
            local_model_path = gr.Textbox(label='请确定数据集和模型下载的本地位置', value=DEFAULT_DOWNLOAD_DIR, info='所有通过下方按钮下载的模型和数据集内容将保存在该路径')
            
        # 这个里面是存放着保存数据集的路径
        local_path_button = gr.Button('确认路径')
        gr.Markdown("## 2. 微调方法、模型、数据集设置")
        
        with gr.Row():
            ft_method = gr.Dropdown(choices=['qlora', 'lora', 'full'], value='qlora',label = '微调方法', info='''请选择微调的方法''',interactive=True)            
            with gr.Column():
                model = gr.Dropdown(choices=MODEL_LIST + ['自定义'], value='internlm/internlm-chat-7b',label = '模型', info='请选择配置文件对应的模型',interactive=True)
                DM_CLS = xtunerModelDownload(
                    model_name=model.value,
                    out_path=MODEL_DOWNLOAD_DIR,
                    tqdm_class=tqdm
                )
                local_path_button.click(DM_CLS.reset_path, inputs=[local_model_path])
                model.change(DM_CLS.reset, inputs=[model])
                with gr.Row():
                    model_download_button = gr.Button('模型下载')
                    model_stop_download = gr.Button('取消下载')
                model_path = gr.Markdown(label='模型下载详情')

                # model_download_information = gr.Markdown(label='模型下载信息')
                # model_download_path = gr.Textbox(visible=False)
                model_download_button.click(DM_CLS.auto_download, outputs=[model_path])
                model_stop_download.click(DM_CLS.break_download, outputs=[model_path])

            with gr.Column():            
                dataset = gr.Dropdown(choices=DATA_LIST + ['自定义'], value='shibing624/medical',label = '数据集', info='请选择需要微调的数据集',interactive=True)
                DT_CLS = xtunerDataDownload(
                    data_name= dataset.value, 
                    out_path=DATA_DOWNLOAD_DIR,
                    tqdm_class=tqdm 
                )
                local_path_button.click(DT_CLS.reset_path, inputs=[local_model_path])
                dataset.change(DT_CLS.reset, inputs=[dataset])
                with gr.Row():
                    dataset_download_button = gr.Button('数据集下载')
                    dataset_stop_download = gr.Button('取消下载')
                data_path = gr.Markdown(label='数据下载详情')
                # dataset_download_information = gr.Markdown(label='数据集下载信息')
                # dataset_download_path = gr.Textbox(visible=False)
                dataset_download_button.click(DT_CLS.auto_download, outputs=[data_path])
                dataset_stop_download.click(DT_CLS.break_download, outputs=[data_path])
        wrong_message1 = gr.Markdown()
        with gr.Row():
            with gr.Accordion(label="自定义模型",open=False):
                model_personal_path = gr.Textbox(label='自定义模型本地路径', info = '请输入模型的本地路径在下方，或将文件压缩上传到下方的位置（建议直接填写本地路径）')
                personal_model = gr.Files(label='请上传自定义模型文件')
                check_personal_model = gr.Button('模型检查及提示词模板自动匹配（请务必点击！）')
                # detect_prompt_template = gr.Markdown() #可用于承接检查后得到的结果
                detect_prompt_template = gr.Textbox(label='检测后的prompt_template', value=' ') #可用于承接检查后得到的结果
                check_personal_model.click(app_get_prompt_template, inputs=[model_personal_path], outputs=[detect_prompt_template])
                
            with gr.Accordion(label="自定义数据集（仅支持OpenAI格式）",open=False):
                with gr.Row():
                    with gr.Column():
                        dataset_type = gr.Dropdown(choices=['OpenAI'],value='OpenAI',label = '支持的数据集格式', interactive=False)
                        dataset_type_preview = gr.TextArea(label='OpenAI数据集格式展示', info= '该数据集的标准格式如下所示，请将自定义的数据集格式转化为该格式。',value=OPENAI_FORMAT)
                        #dataset_type_preview = gr.JSON(label='数据集格式展示')
                    with gr.Column():
                        dataset_personal_path = gr.Textbox(label = '数据集本地路径', info='请填入本地数据集路径或直接在下方上传数据文件')
                        dataset_personal_path_button = gr.Button('请点击上传数据集本地路径')
                        dataset_personal = gr.Files(label='请上传自定义的数据集或在上方填入本地路径',type='filepath')
                check_personal_dataset = gr.Button('检查数据集是否符合要求')
                wrong_message3 = gr.Markdown() #判定数据集格式是否符合要求，符合就在上面显示
                check_personal_dataset.click(check_custom_dataset, inputs=[dataset_personal_path, dataset_personal], outputs=wrong_message3)
                
                with gr.Accordion(label="数据集预览",open=False):
                    dataset_preview = gr.TextArea(label='数据集展示', info = '截取前n行内容，可用于对比原数据集格式。')
                    #dataset_preview = gr.JSON(label='数据集展示')

                dataset_personal_path_button.click(fn=read_first_ten_lines, inputs=dataset_personal_path, outputs=dataset_preview)
                dataset_personal.change(fn=read_first_ten_lines, inputs=dataset_personal_path, outputs=dataset_preview)
                
        with gr.Accordion(label="对应提示词模版展示",open=False):
            with gr.Row():
                # todo map function 
                prompt_template = gr.Dropdown(PROMPT_TEMPLATE_LIST, label='提示词模版', info='请选择合适的提示词模版',interactive=True)
                prompt_template_show = gr.TextArea(label='提示词模版展示')
                
                prompt_template.change(fn=get_template_format_by_name, inputs=prompt_template, outputs=prompt_template_show)
                # detect_prompt_template.change(fn=lambda x: x, inputs=detect_prompt_template, outputs=prompt_template)
                detect_prompt_template.change(fn=get_template_format_by_name, inputs=detect_prompt_template, outputs=prompt_template_show)
                # model.change(fn=get_template_name_by_model, inputs=model, outputs=prompt_template)

        gr.Markdown("## 3. 微调参数设置")
        with gr.Accordion(label="参数调整指南",open=False):
            gr.Markdown('#### 参数调整方式为...')
        with gr.Tab("基础参数"):
            with gr.Row():
                lr = gr.Number(label='学习率(lr)', value=2.0e-5, info= '请选择合适的学习率')
                warmup_ratio = gr.Number(label='预热比', value=0.03, info='预热比例用于在训练初期逐渐增加学习率，以避免训练初期的不稳定性。')
                batch_size_per_device = gr.Number(label='设备的样本个数(batch_size_per_device)', value=1, info='请选择每个设备的样本个数')
                max_length = gr.Number(label='数据集最大长度(max_length)', value=2048, info='请设置训练数据最大长度')
                pack_to_max_length = gr.Dropdown(choices=[True, False], value=True, label='选择合并为最长样本(pack_to_max_length)',info='请选择是否将多条样本打包为一条最长长度的样本')
            with gr.Row():
                deepspeed = gr.Dropdown(choices=['None','zero1','zero2','zero3'], value='None', label='deepspeed算子', info='请选择deepspeed算子类型或关闭deepspeed')
                num_GPU = gr.Number(label='GPU的数量', value=1, info='请设置训练是所用GPU的数量')
                max_epochs = gr.Number(label='训练迭代数(max_epochs)', value=2, info='请选择合适的训练迭代数')
                save_checkpoint_interval = gr.Number(label='保存权重的间隔', value=1000, info='请输入保存checkpoint的间隔')
                save_total_limit = gr.Number(label='最多保存权重文件的个数', value=2, info='控制保存权重文件的个数，以免出现内存不足的情况') 
            with gr.Accordion(label="测试问题模版", open=False):
                evaluation_freq = gr.Number(label='验证对话效果频率(evaluation_freq)', value=100, info='请确定模型每多少轮需要验证一次对话效果')
                evaluation_system_prompt = gr.Textbox(label = '系统提示词', value='', info='请设置在评估模式下的System Prompt')
                evaluation_input1 = gr.Textbox(label= '测试问题1',value='请给我介绍五个上海的景点', info='请输入第一个评估的问题')
                evaluation_input2 = gr.Textbox(label='测试问题2',value='Please tell me five scenic spots in Shanghai', info='请输入第二个评估问题')
        with gr.Tab('进阶参数'):
            with gr.Row():
                optim_type = gr.Dropdown(choices=['AdamW'], value='AdamW', label='优化器', info='请选择合适的优化器（默认为AdamW）',visible=False)
                accumulative_counts = gr.Number(label='梯度累计数', value=16, info='请选择合适的梯度累计数') 
                weight_decay = gr.Number(label='权重衰减', value=0, info = '权重衰减是一种正则化方法，用于防止过拟合，通过在损失函数中添加与权重大小成比例的项')
            with gr.Row(): 
                max_norm = gr.Number(label='梯度剪裁', value=1, info = '梯度裁剪通过限制梯度的最大长度来防止训练过程中的梯度爆炸问题。' )
                dataloader_num_workers = gr.Number(label='加载数据时使用的线程数量', value=0, info='更多的线程可以加快数据加载速度，但也会增加内存和处理器的使用。' ) 
            with gr.Accordion(label="AdamW优化器betas", open=False):
                beta1 = gr.Number(label='beta1', value=0.9, info='这个值通常用于计算梯度的一阶矩估计（即梯度的指数移动平均）。较高的 beta1 值意味着过去梯度的权重更大，从而使得优化器更加关注历史梯度信息。')
                beta2 = gr.Number(label='beta2', value=0.999, info= ' 这个值用于计算梯度的二阶矩估计（即梯度的平方的指数移动平均）。较高的 beta2 值使优化器能够在更长的时间跨度内平滑方差的影响。')

        ft_method.change(
            get_default_hyperparameters, inputs=[ft_method],
            outputs=[
                warmup_ratio,
                batch_size_per_device,
                accumulative_counts,
                num_GPU,
                max_length,
                pack_to_max_length,
                evaluation_freq,
                optim_type,
                weight_decay,
                max_norm,
                dataloader_num_workers,
                beta1,
                beta2,
                lr,
                save_checkpoint_interval,
                save_total_limit
            ]
        )
        change_config_button = gr.Button('点击生成配置文件')
        cfg_py_box = gr.Markdown(value="还未生成配置文件")
        change_config_button.click(
            build_and_save_config, 
            inputs=[
                dataset_personal_path,
                model_personal_path,
                detect_prompt_template,
                local_path,
                ft_method,
                model_path,
                data_path,
                deepspeed,
                lr,
                warmup_ratio,
                batch_size_per_device,
                accumulative_counts,
                num_GPU,
                max_length,
                pack_to_max_length,
                max_epochs,
                save_checkpoint_interval,
                save_total_limit,
                evaluation_freq,
                evaluation_system_prompt,
                evaluation_input1,
                evaluation_input2,
                optim_type,
                weight_decay,
                max_norm,
                dataloader_num_workers,
                beta1,
                beta2,
                prompt_template,
            ],
            outputs=[cfg_py_box]
        )
        wrong_message4 = gr.Markdown()

        gr.Markdown("## 4. 微调模型训练")
        TR_CLS = quickTrain(
            config_py_path=cfg_py_box.value,
            work_dir=f'{local_path.value}/work_dir',
            deepspeed_seed=deepspeed
        )
        change_config_button.click(TR_CLS.reset_cfg_py, inputs=[cfg_py_box])
        cfg_py_box.change(TR_CLS.reset_cfg_py, inputs=[cfg_py_box])
        deepspeed.change(TR_CLS.reset_deepspeed, inputs=[deepspeed])
        local_path_button.click(TR_CLS.reset_work_dir, inputs=[local_path])
        with gr.Row():
            train_model = gr.Button('Xtuner！启动！',size='lg')
            stop_button = gr.Button('训练中断',size='lg')

            work_path = gr.Textbox(label='work dir',visible=False)
            train_model.click(TR_CLS.quick_train, outputs=[work_path])
            stop_button.click(TR_CLS.break_train, outputs=[work_path])
            # stop_button.click(empty_break_fn, outputs=[work_path])

        with gr.Accordion(label='模型续训', open=False):
            retry_path_dropdown = gr.Dropdown(choices=['1.pth','50.pth'],label='请选择需要继续训练的权重文件')
            retry_button = gr.Button('继续训练')
            retry_path_dropdown.change(TR_CLS.reset_resume_from_checkpoint, inputs=[retry_path_dropdown])
            retry_button.click(TR_CLS.resume_train, outputs=[work_path])

        # todo: train_model 或者 retry_button
        with gr.Accordion(label="终端界面",open=False):
            log_file = gr.TextArea(label='日志文件打印', info= '点击可查看模型训练信息')        
            # train_model.click(TR_CLS.start_log, outputs=[log_file])
            # retry_button.click(TR_CLS.start_log, outputs=[log_file])
    
        wrong_message5 = gr.Markdown()
        gr.Markdown("## 5. 微调结果展示")
        PLT = resPlot(
            work_dir = f'{local_path.value}/work_dir',
        )
        local_path_button.click(PLT.reset_work_dir, inputs=[local_path])
        with gr.Tab('训练结果'):
            with gr.Row():
                ft_model_save_path = gr.Textbox(label='模型保存路径',visible=False)
                # detect work_dir find newest 
                iter_num = gr.Number(label='训练轮数', scale=1)
                num_pth = gr.Number(label='权重文件数量', scale=1)
            with gr.Row():                
                # lr_plot = gr.Image(label='学习率变化图',container=False,show_download_button=False,interactive=False)
                # loss_graph = gr.Image(label='损失变化图',container=False,show_download_button=False)
                lr_plot = gr.Plot(label='学习率变化图', container=False, show_label=False)
                loss_graph = gr.Plot(label='损失变化图',container=False, show_label=False)
            with gr.Row():
                num_pth_evaluation = gr.Dropdown(choices=['epoch_1.pth', 'epoch_1.pth'], label='请选择权重文件', info='请选择对应的权重文件进行测试',scale=1)
                evaluation_question = gr.TextArea(label='测试问题结果',scale=3)
            show_evaluation_button = gr.Button('微调结果生成')

            show_evaluation_button.click(PLT.reset_work_dir, inputs=[local_path])
            show_evaluation_button.click(PLT.lr_plot, outputs=[lr_plot])
            show_evaluation_button.click(PLT.loss_plot, outputs=[loss_graph])
            show_evaluation_button.click(PLT.dynamic_drop_down, outputs=num_pth_evaluation)
        # gr.Markdown('## 5. 实际案例')
        # ft_examples = gr.Examples(examples=[['qlora','internlm','Medqa2019'],['qlora','自定义','自定义']],inputs=[ft_method ,model ,dataset],label='例子')

        gr.Markdown("## 6. 微调模型转化及测试")
        
        with gr.Accordion(label="模型转换",open=False):
            # Textbox
            # select_checkpoint =gr.Dropdown(choices=['epoch_1.pth', 'epoch_1.pth'], value='epoch_1.pth', label='微调模型的权重文件', info = '请选择需要进行测试的模型权重文件并进行转化')
            select_checkpoint = gr.Dropdown(choices=['epoch_1.pth'],  label='微调模型的权重文件', info = '请选择需要进行测试的模型权重文件并进行转化',interactive = True)
            show_evaluation_button.click(PLT.dynamic_drop_down, outputs=select_checkpoint)
            
            covert_hf = gr.Button('模型转换',scale=1)
            covert_hf_path = gr.Textbox(label='模型转换后地址', visible=False) # False
            wrong_message6 = gr.Markdown()

            # root_dir, config_file, epoch_pth, model_path, customer_model_path)
            covert_hf.click(convert_and_merged, inputs=[local_path, cfg_py_box, select_checkpoint, model_path, model_personal_path], outputs=[wrong_message6, covert_hf_path]) 
        with gr.Accordion(label='对话测试', open=False):
            with gr.Row():
                with gr.Accordion(label="原模型对话测试", open=True):
                    with gr.Column():
                        with gr.Accordion(label='参数设置',open=False):
                            max_new_tokens = gr.Slider(minimum=0, maximum=4096 ,value=1024, label='模型输出的最长Toekn', info='Token越多，模型能够回复的长度就越长')
                            bits =  gr.Radio(choices=['int4', 'int8', 'None'],value='None', label='量化', info='请选择模型量化程度', interactive=True)
                            
                            temperature = gr.Slider(maximum=1,minimum=0,label='温度值',info='温度值越高，模型输出越随机')
                            top_k=gr.Slider(minimum=0, maximum=100, value=40, label='top-k',info='')
                            top_p = gr.Slider(minimum=0, maximum=2, value=0.75, label='top-p',info='')
                            
                            #还可以添加更多
                            wrong_message9 = gr.Markdown()
                        start_testing_model = gr.Button('模型启动')
                        chatbot = gr.Chatbot(label='微调模型测试')
                        msg = gr.Textbox(label="输入信息")
                        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
                    with gr.Row():
                        clear = gr.Button('记录删除').click(clear_history, inputs=[chatbot], outputs=[chatbot])
                        undo = gr.Button('撤回上一条').click(undo, inputs=[chatbot], outputs=[chatbot])
                        regenerate = gr.Button('重新生成').click(regenerate, inputs=[chatbot], outputs = [msg, chatbot]) 
                with gr.Accordion(label="微调模型对话测试", open=True):
                    with gr.Column():
                        with gr.Accordion(label='参数设置',open=False):
                            max_new_tokens = gr.Slider(minimum=0, maximum=4096 ,value=1024, label='模型输出的最长Toekn', info='Token越多，模型能够回复的长度就越长')
                            bits =  gr.Radio(choices=['int4', 'int8', 'None'],value='None', label='量化', info='请选择模型量化程度', interactive=True)
                            
                            temperature = gr.Slider(maximum=1,minimum=0,label='温度值',info='温度值越高，模型输出越随机')
                            top_k=gr.Slider(minimum=0, maximum=100, value=40, label='top-k',info='')
                            top_p = gr.Slider(minimum=0, maximum=2, value=0.75, label='top-p',info='')
                            
                            #还可以添加更多
                            wrong_message9 = gr.Markdown()
                        start_testing_model = gr.Button('模型启动')
                        chatbot = gr.Chatbot(label='微调模型测试')
                        msg = gr.Textbox(label="输入信息")
                        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
                    with gr.Row():
                        clear = gr.Button('记录删除').click(clear_history, inputs=[chatbot], outputs=[chatbot])
                        undo = gr.Button('撤回上一条').click(undo, inputs=[chatbot], outputs=[chatbot])
                        regenerate = gr.Button('重新生成').click(regenerate, inputs=[chatbot], outputs = [msg, chatbot])  
        # with gr.Accordion(label='模型基础能力评估测试',open=False):
        #     mmlu_test_button = gr.Button('MMLU模型能力评估测试')
        with gr.Accordion(label="其他信息", open=True):
            star = gr.Markdown('### 假如感觉能帮助你，请为Xtuner点个小小的star！ https://github.com/InternLM/xtuner.git ')
            
    #with gr.Tab('微调模型部署（LMDeploy）'):

    #with gr.Tab('微调模型测评（OpenCompass）'):

    demo.load(TR_CLS.read_log, outputs=[log_file], every=1)
    demo.launch(share=True) #, server_name="0.0.0.0", server_port=6006, root_path=f'/proxy/6006/')




