# python3
# Create Date: 2024-01-26
# ========================================

from xtuner_download.download_model import xtunerModelDownload
from xtuner_download.download_dataset import xtunerDataDownload
from xtuner_run.train import quickTrain
from appPrepare.files_prepare import DATA_DOWNLOAD_DIR, MODEL_DOWNLOAD_DIR, CUR_PATH
from appPrepare.list_prepare import DATA_LIST, MODEL_LIST
from tqdm import tqdm
import gradio as gr

progress = gr.Progress(track_tqdm=True)

def combine_message_and_history(message, chat_history):
    # 将聊天历史中的每个元素（假设是元组）转换为字符串
    history_str = "\n".join(f"{sender}: {text}" for sender, text in chat_history)

    # 将新消息和聊天历史结合成一个字符串
    full_message = f"{history_str}\nUser: {message}"
    return full_message

# def respond(message, chat_history):
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
#     return "", chat_history

def clear_history(chat_history):
    chat_history.clear()
    return chat_history

# def regenerate(chat_history):
#     if chat_history:
#         # 提取上一条输入消息
#         last_message = chat_history[-1][0]
#         # 移除最后一条记录
#         chat_history.pop()
#         # 使用上一条输入消息调用 respond 函数以生成新的回复
#         msg,chat_history = respond(last_message, chat_history)
#     # 返回更新后的聊天记录
#     return msg, chat_history

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
            # process = gr.Image(value='/mnt/d/xtuner/1.png',label='使用流程图',container=False,show_download_button=False )
            gr.Markdown('## 演示视频')
            # video_customer_introduction = gr.Video(label='Xtuner GUI用法演示',value='/mnt/d/xtuner/demo.mp4',interactive=False)
        gr.Markdown("## 1. 本地路径设置")
        
        local_path = gr.Textbox(
            label='请上传所有文件保存的文件本地路径', 
            value=CUR_PATH,
            info='将会在选择的路径下创建名为xxx的文件夹并将所有文件保存在此处'
        )
        local_path_button = gr.Button('确认路径')

        gr.Markdown("## 2. 微调方法、模型、数据集设置")
        
        with gr.Row():
            ft_method = gr.Dropdown(choices=['qlora', 'lora', '自定义'], value='qlora',label = '微调方法', info='''请选择微调的方法''',interactive=True)
            with gr.Column():
                model = gr.Dropdown(choices=MODEL_LIST + ['自定义'], value='internlm/internlm-chat-7b',label = '模型', info='请选择配置文件对应的模型',interactive=True)
                DM_CLS = xtunerModelDownload(
                    model_name=model.value,
                    out_path=MODEL_DOWNLOAD_DIR,
                    tqdm_class=tqdm
                )
                model.change(DM_CLS.reset, inputs=[model])
                with gr.Row():
                    model_download_button = gr.Button('模型下载')
                    model_stop_download = gr.Button('取消下载')
                    model_path = gr.Textbox(label='下载详情')

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
                dataset.change(DT_CLS.reset, inputs=[dataset])
                with gr.Row():
                    dataset_download_button = gr.Button('数据集下载')
                    dataset_stop_download = gr.Button('取消下载')
                    data_path = gr.Textbox(label='下载详情')

                # dataset_download_information = gr.Markdown(label='数据集下载信息')
                # dataset_download_path = gr.Textbox(visible=False)
                dataset_download_button.click(DT_CLS.auto_download, outputs=[data_path])
                dataset_stop_download.click(DT_CLS.break_download, outputs=[data_path])

        wrong_message1 = gr.Markdown()
        with gr.Row():
            with gr.Accordion(label="自定义模型",open=False):
                personal_model_path = gr.Textbox(label='自定义模型本地路径', info = '请输入模型的本地路径在下方，或将文件压缩上传到下方的位置（建议直接填写本地路径）')
                personal_model = gr.Files(label='请上传自定义模型文件')
                check_personal_model = gr.Button('检查模型是否符合要求')
                wrong_message2 = gr.Markdown()
            with gr.Accordion(label="自定义数据集",open=False):
                with gr.Row():
                    with gr.Column():
                        dataset_type = gr.Dropdown(choices=['Medqa2019'],value='Medqa2019',label = '支持的数据集格式', info='请选择以下支持的数据集格式并上传数据集',interactive=True)
                        dataset_type_preview = gr.TextArea(label='格式展示', info= '该数据集的标准格式如下所示，请将自定义的数据集格式转化为该格式。')
                        #dataset_type_preview = gr.JSON(label='数据集格式展示')
                    with gr.Column():
                        dataset_personal_path = gr.Textbox(label = '数据集本地路径', info='请填入本地数据集路径或直接在下方上传数据文件')
                        dataset_personal = gr.Files(label='请上传自定义的数据集或在上方填入本地路径')
                check_personal_dataset = gr.Button('检查数据集是否符合要求')
                wrong_message3 = gr.Markdown() #判定数据集格式是否符合要求，符合就在上面显示
                with gr.Accordion(label="数据集预览",open=False):
                    dataset_preview = gr.TextArea(label='数据集展示', info = '截取前n行内容，可用于对比原数据集格式。')
                    #dataset_preview = gr.JSON(label='数据集展示')

        gr.Markdown("## 3. 微调参数设置")
        with gr.Accordion(label="参数调整指南",open=False):
            gr.Markdown('#### 参数调整方式为...')
        with gr.Tab("基础参数"):
            with gr.Row():
                deepspeed = gr.Dropdown(choices=['None','zero1','zero2','zero3'],label='deepspeed算子', info='请选择deepspeed算子类型或关闭deepspeed')
                lr = gr.Number(label='学习率(lr)', info= '请选择合适的学习率')
                warmup_ratio = gr.Number(label='预热比', info='预热比例用于在训练初期逐渐增加学习率，以避免训练初期的不稳定性。')
                batch_size_per_device = gr.Number(label='设备的样本个数(batch_size_per_device)', info='请选择每个设备的样本个数') 
                accumlative_counts = gr.Number(label='梯度累计数', info='请选择合适的梯度累计数') 
            with gr.Row():
                num_GPU = gr.Number(label='GPU的数量',info='请设置训练是所用GPU的数量')
                max_length = gr.Number(label='数据集最大长度(max_length)', info='请设置训练数据最大长度')
                pack_to_max_length = gr.Dropdown(choices=[True, False], label='选择合并为最长样本(pack_to_max_length)',info='请选择是否将多条样本打包为一条最长长度的样本')
                max_epochs = gr.Number(label='训练迭代数(max_epochs)', info='请选择合适的训练迭代数')
                save_checkpoint_ratio = gr.Number(label='保存权重的间隔', info='请输入保存checkpoint的间隔')
            with gr.Accordion(label="测试问题模版", open=False):
                evaluation_freq = gr.Number(label='验证对话效果频率(evaluation_freq)', info='请确定模型每多少轮需要验证一次对话效果')
                evaluation_system_prompt = gr.Textbox(label = '系统提示词', info='请设置在评估模式下的System Prompt')
                evaluation_input1 = gr.Textbox(label= '测试问题1', info='请输入第一个评估的问题')
                evaluation_input2 = gr.Textbox(label='测试问题2', info='请输入第二个评估问题')
        with gr.Tab('进阶参数'):
            with gr.Row():
                optim_type = gr.Dropdown(choices=['AdamW'], label='优化器', info='请选择合适的优化器（默认为AdamW）')
                weight_decay = gr.Number(label='权重衰减', info = '权重衰减是一种正则化方法，用于防止过拟合，通过在损失函数中添加与权重大小成比例的项')
            with gr.Row(): 
                max_norm = gr.Number(label='梯度剪裁', info = '梯度裁剪通过限制梯度的最大长度来防止训练过程中的梯度爆炸问题。' )
                dataloader_num_workers = gr.Number(label='加载数据时使用的线程数量', info='更多的线程可以加快数据加载速度，但也会增加内存和处理器的使用。' ) 
            with gr.Accordion(label="AdamW优化器betas", open=False):
                beta1 = gr.Number(label='beta1', info='这个值通常用于计算梯度的一阶矩估计（即梯度的指数移动平均）。较高的 beta1 值意味着过去梯度的权重更大，从而使得优化器更加关注历史梯度信息。')
                beta2 = gr.Number(label='beta2', info= ' 这个值用于计算梯度的二阶矩估计（即梯度的平方的指数移动平均）。较高的 beta2 值使优化器能够在更长的时间跨度内平滑方差的影响。')
            with gr.Accordion(label="提示词模版修改",open=False):
                with gr.Row():
                    prompt_template = gr.Dropdown(['internlm-chat'], label='提示词模版', info='请选择合适的提示词模版')
                    prompt_template_show = gr.TextArea(label='提示词模版展示')
        change_config_button = gr.Button('点击生成配置文件')
        wrong_message4 = gr.Markdown()

        gr.Markdown("## 4. 微调模型训练")
        TR_CLS = quickTrain(
            model_name_or_path=model_path.value,
            dataset_name_or_path=data_path.value,
            work_dir=local_path.value,
            xtuner_type=ft_method.value
        )
        model.change(TR_CLS.set_model_path, inputs=[model])
        dataset.change(TR_CLS.set_data_path, inputs=[dataset])
        ft_method.change(TR_CLS.set_xtuner_type, inputs=[ft_method])
        local_path_button.click(TR_CLS.set_work_dir, inputs=[local_path])
        with gr.Row():
            train_model = gr.Button('Xtuner！启动！',size='lg')
            stop_button = gr.Button('训练中断',size='lg')

            work_path = gr.Textbox(label='work dir')
            train_model.click(TR_CLS.quick_train, outputs=[work_path])
            stop_button.click(TR_CLS.break_train, outputs=[work_path])
    
        with gr.Accordion(label='模型续训', open=False):
            retry_path = gr.Textbox(label='原配置文件地址', info='请查询原配置文件地址并进行填入')
            retry_button = gr.Button('继续训练')
        with gr.Accordion(label="终端界面",open=False):
            log_file = gr.TextArea(label='日志文件打印', info= '点击可查看模型训练信息')        
        wrong_message5 = gr.Markdown()
        gr.Markdown("## 5. 微调结果展示")
        with gr.Tab('训练结果'):
            with gr.Row():
                ft_model_save_path = gr.Textbox(label='模型保存路径',visible=False)
                iter = gr.Number(label='训练轮数',scale=1)
                num_pth = gr.Number(label='权重文件数量',scale=1)
            with gr.Row():
                
                lr_plot = gr.Plot(label='学习率变化图')
                loss_graph = gr.Plot(label='损失变化图')
            show_evaluation_button = gr.Button('微调结果生成')
            with gr.Row():
                num_pth_evaluation = gr.Dropdown(choices=['1.pth', '2.pth'], label='请选择权重文件', info='请选择对应的权重文件进行测试',scale=1)
                evaluation_question = gr.TextArea(label='测试问题结果',scale=3)

        # gr.Markdown('## 5. 实际案例')
        # ft_examples = gr.Examples(examples=[['qlora','internlm','Medqa2019'],['qlora','自定义','自定义']],inputs=[ft_method ,model ,dataset],label='例子')

        gr.Markdown("## 6. 微调模型转化及测试")
        
        with gr.Accordion(label="模型转换",open=True):
            select_checkpoint =gr.Dropdown(choices=[],label='微调模型的权重文件', info = '请选择需要进行测试的模型权重文件并进行转化')
            covert_hf = gr.Button('模型转换',scale=1)
            covert_hf_path = gr.Textbox(label='模型转换后地址', visible=False)
            wrong_message6 = gr.Markdown()


        # with gr.Accordion(label="模型对话测试", open=True):
        #     with gr.Row():
        #         with gr.Column(scale=1):
        #             max_new_tokens = gr.Slider(minimum=0, maximum=4096 ,value=1024, label='模型输出的最长Toekn', info='Token越多，模型能够回复的长度就越长')
        #             bits =  gr.Radio(choices=['int4', 'int8', 'None'],value='None', label='量化', info='请选择模型量化程度', interactive=True)
                    
        #             temperature = gr.Slider(maximum=1,minimum=0,label='温度值',info='温度值越高，模型输出越随机')
        #             top_k=gr.Slider(minimum=0, maximum=100, value=40, label='top-k',info='')
        #             top_p = gr.Slider(minimum=0, maximum=2, value=0.75, label='top-p',info='')
        #             start_testing_model = gr.Button('模型启动')
        #             #还可以添加更多
        #             wrong_message8 = gr.Markdown()
        #         with gr.Column(scale=3):
        #             chatbot = gr.Chatbot(label='微调模型测试')
        #             msg = gr.Textbox(label="输入信息")
        #             msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        #             with gr.Row():
        #                 clear = gr.Button('记录删除').click(clear_history, inputs=[chatbot], outputs=[chatbot])
        #                 undo = gr.Button('撤回上一条').click(undo, inputs=[chatbot], outputs=[chatbot])
        #                 regenerate = gr.Button('重新生成').click(regenerate, inputs=[chatbot], outputs = [msg, chatbot])  
        with gr.Accordion(label='模型基础能力评估测试',open=False):
            mmlu_test_button = gr.Button('MMLU模型能力评估测试')
        with gr.Accordion(label="其他信息", open=True):
            star = gr.Markdown('### 假如感觉能帮助你，请为Xtuner点个小小的star！ https://github.com/InternLM/xtuner.git ')
            
    #with gr.Tab('微调模型部署（LMDeploy）'):

    #with gr.Tab('微调模型测评（OpenCompass）'):


    demo.launch(share=True)



        
