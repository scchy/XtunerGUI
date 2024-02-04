# python3
# Create Date: 2024-01-26
# ========================================

from xtuner_download.download_model import xtunerModelDownload
from xtuner_download.download_dataset import xtunerDataDownload
from xtuner_convert.convert_and_merge import convert_and_merged
from xtuner_convert.convert_with_progress import ConvertMerged
from xtuner_run.shell_train import quickTrain
from appPrepare.files_prepare import DATA_DOWNLOAD_DIR, MODEL_DOWNLOAD_DIR, CUR_PATH, DEFAULT_DOWNLOAD_DIR
from appPrepare.list_prepare import DATA_LIST, MODEL_LIST, PROMPT_TEMPLATE_LIST
from appPrepare.func_prepare import read_first_ten_lines, get_template_format_by_name, OPENAI_FORMAT
from xtuner_config.build_config import build_and_save_config, model_path_map_fn
from xtuner_config.check_custom_dataset import check_custom_dataset
from xtuner_config.get_prompt_template import app_get_prompt_template
from xtuner_config.get_default_hyperparameters import get_default_hyperparameters
from chat.model_center import ModelCenter
from tqdm import tqdm
from xtuner_result.draw import resPlot
import gradio as gr
import warnings
warnings.filterwarnings(action='ignore')
CHAT_ORG = ModelCenter()
FT_CHAT_ORG = ModelCenter()
CVT_MG = ConvertMerged()

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


def evaluation_question_number_change_wrap(max_textboxes):
    def evaluation_question_number_change(k):
        k = int(k)
        return [gr.Textbox(visible=True)]*k + [gr.Textbox(value='', visible=False)]*(max_textboxes-k)
    return evaluation_question_number_change

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
            info='将会在选择的路径下保存模型的配置文件、训练过程文件及模型转换后的文件'
        )
            local_model_path = gr.Textbox(label='请确定数据集和模型下载的本地位置', value=DEFAULT_DOWNLOAD_DIR, info='将保存所有通过下方按钮下载的模型和数据集内容在该路径')
            
        # 这个里面是存放着保存数据集的路径
        local_path_button = gr.Button('确认路径')
        gr.Markdown("## 2. 微调方法、模型、数据集设置")
        
        with gr.Row():
            ft_method = gr.Dropdown(choices=['qlora', 'lora', 'full'], value='qlora',label = '微调方法', info='''请选择需要的微调方法（全量微调（full）需要大量显存，请谨慎选择）''',interactive=True)            
            with gr.Column():
                model = gr.Dropdown(choices=MODEL_LIST + ['自定义'], value='internlm/internlm-chat-7b',label = '模型', info='请选择你希望微调的模型，选择后可点击下方按钮进行下载',interactive=True)
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
                dataset = gr.Dropdown(choices=DATA_LIST + ['自定义'], value='shibing624/medical',label = '数据集', info='请选择合适的数据集，选择后可点击下方按钮进行下载',interactive=True)
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
            with gr.Column(scale=1):
                with gr.Accordion(label="自定义模型",open=False):
                    model_personal_path = gr.Textbox(label='自定义模型本地路径', info = '请输入模型的本地路径在下方文本框中')
                    personal_model = gr.Files(label='请上传自定义模型文件',visible=False)
                    check_personal_model = gr.Button('模型检查及提示词模板自动匹配（请务必点击！）')
                    detect_prompt_status = gr.Markdown() #可用于承接检查后得到的结果
                    # 上传文件自动显示在 model_personal_path
                    personal_model.change(lambda x: x, inputs=[personal_model], outputs=[model_personal_path])
            with gr.Column(scale=2):
                with gr.Accordion(label="自定义数据集（仅支持OpenAI格式）",open=False):
                    with gr.Row():
                        with gr.Column():
                            dataset_type = gr.Dropdown(choices=['OpenAI'],value='OpenAI',label = '支持的数据集格式', interactive=False)
                            dataset_type_preview = gr.TextArea(label='OpenAI数据集格式展示', info= '该数据集的标准格式如下所示，请将自定义的数据集格式转化为该格式。',value=OPENAI_FORMAT)
                            #dataset_type_preview = gr.JSON(label='数据集格式展示')
                        with gr.Column():
                            dataset_personal_path = gr.Textbox(label = '数据集本地路径', info='请填入本地数据集路径或直接在下方上传数据文件')
                            # dataset_personal_path_upload = gr.Button('请点击上传数据集本地路径')
                            dataset_personal = gr.File(label='请上传自定义的数据集或在上方填入本地路径',type='filepath')
                            check_personal_dataset = gr.Button('检查数据集是否符合要求')
                            wrong_message3 = gr.Markdown() #判定数据集格式是否符合要求，符合就在上面显示
                    check_personal_dataset.click(check_custom_dataset, inputs=[dataset_personal_path, dataset_personal], outputs=wrong_message3)
                    
                    # with gr.Accordion(label="数据集预览",open=False):
                    #     dataset_preview = gr.TextArea(label='数据集展示', info = '截取前n行内容，可用于对比原数据集格式。')
                    #     #dataset_preview = gr.JSON(label='数据集展示')
                    # dataset_personal_path_upload.click(fn=read_first_ten_lines, inputs=dataset_personal_path, outputs=dataset_preview, queue=False)
                    # dataset_personal.change(fn=read_first_ten_lines, inputs=dataset_personal, outputs=dataset_preview, queue=False)
        with gr.Accordion(label="对应提示词模版展示",open=False):
            with gr.Row():
                prompt_template = gr.Dropdown(PROMPT_TEMPLATE_LIST, label='提示词模版', value='default', info='请选择合适的提示词模版（请勿随意进行调整）',interactive=True)
                prompt_template_show = gr.TextArea(label='提示词模版展示')
                
                model.change(model_path_map_fn, inputs=[model], outputs=[prompt_template])
                # 检测完毕后 -> 改变 prompt_template -> prompt_template_show
                check_personal_model.click(app_get_prompt_template, inputs=[model_personal_path, personal_model], outputs=[detect_prompt_status, prompt_template])
                prompt_template.change(fn=get_template_format_by_name, inputs=prompt_template, outputs=prompt_template_show)

        gr.Markdown("## 3. 微调参数设置")
        # with gr.Accordion(label="参数调整指南",open=False):
        #     gr.Markdown('#### 参数调整方式为...')
        with gr.Tab("基础参数"):
            with gr.Row():
                lr = gr.Number(label='学习率(Learning Rate)', value=2.0e-5, info='学习率控制模型权重调整的幅度，在训练过程中对损失函数的优化有直接影响。较小的学习率可能导致学习过程缓慢，而较大的学习率可能导致学习过程中出现不稳定。')
                warmup_ratio = gr.Number(label='预热比(Warmup Ratio)', value=0.03, info='预热比例用于在训练初期逐渐增加学习率，这有助于模型训练初期的稳定性，避免因学习率过高导致的训练不稳定。')
                max_length = gr.Number(label='数据集最大长度(Max Length)', value=2048, info='设置数据在处理前的最大长度，确保模型可以处理的序列长度范围内，有助于控制训练过程的内存使用。')
                pack_to_max_length = gr.Dropdown(choices=[True, False], value=True, label='合并为最长样本(Pack to Max Length)', info='决定是否将多个样本合并成一个最大长度的样本。这可以提高数据处理的效率，但可能影响模型学习到的模式。')
            with gr.Row():
                batch_size_per_device = gr.Number(label='每设备样本个数(Batch Size per Device)', value=1, info='定义每个设备上进行处理的样本数量。较大的批量大小可以提高训练效率，但也会增加内存的使用量。')
                accumulative_counts = gr.Number(label='梯度累计数(Gradient Accumulation Steps)', value=16, info='在进行一次参数更新前累积的梯度步数，可以增大批处理大小的效果而不增加内存消耗。')
                deepspeed = gr.Dropdown(choices=['None','zero1','zero2','zero3'], value='None', label='Deepspeed算子(Deepspeed)', info='选择Deepspeed优化策略来加速训练和降低内存使用。不同的优化级别提供了不同的内存和计算优化。')
                num_GPU = gr.Number(label='GPU数量(Number of GPUs)', value=1, info='设置训练过程中使用的GPU数量。增加GPU数量可以提高训练速度，但需要确保硬件资源充足。')            
            with gr.Row():
                max_epochs = gr.Number(label='训练迭代数(Max Epochs)', value=2, info='设置模型训练过程中数据将被遍历的次数。较多的迭代次数可以提高模型性能，但也会增加训练时间。')
                save_checkpoint_interval = gr.Number(label='保存权重间隔(Save Checkpoint Interval)', value=1000, info='设置自动保存模型权重的间隔（以迭代次数计）。这有助于从训练中途的某个点恢复训练过程。')
                save_total_limit = gr.Number(label='最多保存权重文件数(Save Total Limit)', value=2, info='限制保存的模型权重文件的最大数量，有助于管理存储空间，避免因保存过多的模型文件而耗尽存储。')
                evaluation_freq = gr.Number(label='验证对话效果频率(evaluation_freq)', value=100, info='请确定模型每多少轮需要验证一次对话效果，具体的对话问题及系统提示词可以在下方评估问题处进行设置')

            # todo: 测试问题 多个的问题
            with gr.Accordion(label="评估问题设置", open=True):
                evaluation_system_prompt = gr.Textbox(label = '系统提示词（system_prompt）', value='', info='请设置在评估模式下的系统提示词（默认为无）')
                default_evaluation_question_number = 2
                max_evaluation_question_number = 10
                default_evaluation_question_list = [
                    '请给我介绍五个上海的景点',
                    'Please tell me five scenic spots in Shanghai'
                ]
                evaluation_question_list = []
                with gr.Accordion(label='评估问题数量及内容',open=True):
                    with gr.Row():
                        with gr.Column():
                            evaluation_question_number = gr.Number(label='评估问题数', value=default_evaluation_question_number, minimum=1, maximum=max_evaluation_question_number, info='调整评估问题的数量（最多10个问题）')
                        with gr.Column():
                            for i in range(max_evaluation_question_number):
                                evaluation_question_if_visible = True if i < default_evaluation_question_number else False
                                evaluation_question_value = default_evaluation_question_list[i] if i < default_evaluation_question_number else ''
                                t = gr.Textbox(label=f'评估问题{i + 1}', value=evaluation_question_value, interactive=True, placeholder=f"请输入第{i + 1}个评估的问题", visible=evaluation_question_if_visible)
                                evaluation_question_list.append(t)
                evaluation_question_number.change(evaluation_question_number_change_wrap(max_evaluation_question_number), evaluation_question_number, evaluation_question_list)
        with gr.Tab('进阶参数'):
            with gr.Row():
                optim_type = gr.Dropdown(choices=['AdamW'], value='AdamW', label='优化器(Optimizer)', info='选择优化器用于调整网络权重以减少误差；AdamW是Adam优化器的一种变体，提供权重衰减控制，通常用于更好的泛化。', visible=True)
                
                weight_decay = gr.Number(label='权重衰减(Weight Decay)', value=0, info='权重衰减是一种正则化技术，通过为模型的损失函数添加一个与权重大小成比例的惩罚项来防止模型的过拟合。')

            with gr.Row():
                max_norm = gr.Number(label='梯度剪裁(Gradient Clipping)', value=1, info='通过设置梯度的最大阈值来防止在训练过程中梯度爆炸的问题，有助于稳定模型的训练过程。')
                dataloader_num_workers = gr.Number(label='数据加载线程数(Data Loader Number of Workers)', value=0, info='设置在数据加载时并行工作的线程数，较高的值可以加快数据加载速度，但会增加内存和处理器的负担。')

            with gr.Accordion(label="AdamW优化器betas", open=False):
                beta1 = gr.Number(label='beta1 (一阶矩估计)', value=0.9, info='用于计算梯度的一阶矩估计（即梯度的指数移动平均），决定了过去梯度的权重，高值意味着模型更加关注过去的梯度。')
                beta2 = gr.Number(label='beta2 (二阶矩估计)', value=0.999, info='用于计算梯度的二阶矩估计（即梯度平方的指数移动平均），决定了梯度变化率的平滑程度，高值可以使优化过程在长时间内更加平稳。')

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
                dataset_personal,
                model_personal_path,
                personal_model,
                prompt_template,
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
                optim_type,
                weight_decay,
                max_norm,
                dataloader_num_workers,
                beta1,
                beta2,
                prompt_template,
                *evaluation_question_list
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

        tmp_trian_pg_md = gr.Markdown()
        train_model.click(TR_CLS.quick_train, outputs=[tmp_trian_pg_md, work_path])
        stop_button.click(TR_CLS.break_train, outputs=[tmp_trian_pg_md, work_path], queue=False)
        with gr.Accordion(label='模型续训', open=False):
            retry_path_dropdown = gr.Dropdown(label='请选择需要继续训练的权重文件', info='将从训练中断前的模型权重文件进行搜索',interactive=True)
            retry_button = gr.Button('继续训练')
            retry_path_dropdown.change(TR_CLS.reset_resume_from_checkpoint, inputs=[retry_path_dropdown])
            retry_button.click(TR_CLS.resume_train, outputs=[tmp_trian_pg_md, work_path])

        with gr.Accordion(label="终端界面",open=False):
            log_file = gr.TextArea(label='日志文件打印', info= '点击可查看模型训练信息')        
            # train_model.click(TR_CLS.start_log, outputs=[log_file])
            # retry_button.click(TR_CLS.start_log, outputs=[log_file])
    
        wrong_message5 = gr.Markdown()
        gr.Markdown("## 5. 微调结果展示")
        PLT = resPlot(
            work_dir = f'{local_path.value}/work_dir',
        )
        # 点击停止训练的时候 retry_path_dropdown 进行更新
        stop_button.click(PLT.dynamic_drop_down, outputs=retry_path_dropdown, queue=False)
        work_path.change(PLT.dynamic_drop_down, outputs=retry_path_dropdown, queue=False)
        local_path_button.click(PLT.reset_work_dir, inputs=[local_path])
        work_path.change(PLT.reset_work_dir, inputs=[local_path])
        with gr.Tab('训练结果'):
            # with gr.Row():
                # ft_model_save_path = gr.Textbox(label='模型保存路径',visible=False)
                # detect work_dir find newest 
                # iter_num = gr.Number(label='训练轮数', scale=1)
                # num_pth = gr.Number(label='权重文件数量', scale=1)
            with gr.Row():                
                # lr_plot = gr.Image(label='学习率变化图',container=False,show_download_button=False,interactive=False)
                # loss_graph = gr.Image(label='损失变化图',container=False,show_download_button=False)
                lr_plot = gr.LinePlot(label='学习率变化图')
                loss_graph = gr.LinePlot(label='损失变化图')
            with gr.Row():
                num_pth_evaluation = gr.Dropdown(label='请选择权重文件', info='可获取模型训练过程中评估问题的结果展示')
                evaluation_question = gr.TextArea(label='测试问题结果')
                
            stop_button.click(PLT.dynamic_drop_down, outputs=num_pth_evaluation, queue=False)
            show_evaluation_button = gr.Button('微调结果生成')
            show_evaluation_button.click(PLT.reset_work_dir, inputs=[local_path], queue=False)
            show_evaluation_button.click(PLT.lr_plot, outputs=[lr_plot], queue=False)
            show_evaluation_button.click(PLT.loss_plot, outputs=[loss_graph], queue=False)
            show_evaluation_button.click(PLT.dynamic_drop_down, outputs=num_pth_evaluation, queue=False)
            # 找到 & read eval 
            num_pth_evaluation.change(PLT.get_eval_test, inputs=[num_pth_evaluation], outputs=[evaluation_question])

        gr.Markdown("## 6. 微调模型转化及测试")
        
        with gr.Accordion(label="模型转换",open=True):
            # Textbox
            # select_checkpoint =gr.Dropdown(choices=['epoch_1.pth', 'epoch_1.pth'], value='epoch_1.pth', label='微调模型的权重文件', info = '请选择需要进行测试的模型权重文件并进行转化')
            select_checkpoint = gr.Dropdown(label='微调模型的权重文件', info = '请选择需要进行测试的模型权重文件并进行转化',interactive = True)
            stop_button.click(PLT.dynamic_drop_down, outputs=select_checkpoint, queue=False)
            show_evaluation_button.click(PLT.dynamic_drop_down, outputs=select_checkpoint, queue=False)
            
            covert_hf = gr.Button('模型转换',scale=1)
            covert_hf_path = gr.Textbox(label='模型转换后地址', visible=False) # False
            wrong_message6 = gr.Markdown()

            # root_dir, config_file, epoch_pth, model_path, customer_model_path)
            # todo ft_method full-convert  oth-convert+merge
            covert_hf.click(CVT_MG.auto_convert_merge, inputs=[local_path, cfg_py_box, select_checkpoint, model_path, model_personal_path, ft_method], outputs=[wrong_message6, covert_hf_path]) 
        with gr.Accordion(label='对话测试', open=True):
            with gr.Row():
                with gr.Accordion(label="原模型对话测试", open=True):
                    with gr.Column():
                        with gr.Accordion(label='参数设置',open=False):
                            max_new_tokens = gr.Slider(minimum=0, maximum=4096, value=1024, label='模型输出的最长Token(max_new_tokens)', info='这个参数决定了模型输出的最大token数量。增加这个值允许模型生成更长的文本，而减少这个值会导致生成的文本更短。')
                            temperature = gr.Slider(maximum=2, minimum=0, label='温度值(temperature)',value=1, info='控制生成文本的随机性。较高的温度值会使输出更加多样和不可预测，而较低的值使输出更确定和重复。')
                            top_k = gr.Slider(minimum=0, maximum=100, value=40, label='Top-k Sampling(top-k)', info='限制模型在每一步生成文本时考虑的最可能候选词的数量。较大的k值增加了多样性，但可能降低文本的连贯性；较小的k值则相反。')
                            top_p = gr.Slider(minimum=0, maximum=2, value=0.75, label='Top-p Sampling(top-p)', info='类似于top_k，但通过选择累积概率高于某个阈值p的最小词集，动态调整考虑的候选词数量。较高的p值增加多样性，较低的p值提高连贯性。')
                            num_beams = gr.Slider(minimum=0, maximum=12, value=5, label='Beam Search(num_beams)', info='在beam search中，num_beams指定了搜索宽度。更多的beams可以提高生成文本的质量，但也会增加计算负担。')

                            #还可以添加更多
                            wrong_message9 = gr.Markdown()
                        start_testing_model = gr.Button('模型启动')
                        testig_model_loaded = gr.Markdown()
                        chatbot = gr.Chatbot(label='微调模型测试')
                        msg = gr.Textbox(label="输入信息")
                        msg.submit(CHAT_ORG.qa_answer, inputs=[msg, max_new_tokens, temperature, top_k, top_p, num_beams, chatbot], outputs=[msg, chatbot])
                        # 模型载入
                        start_testing_model.click(CHAT_ORG.load_model, inputs=[model_personal_path, personal_model, model_path], outputs=[testig_model_loaded])
                    send = gr.Button('信息发送') # .click(regenerate, inputs=[chatbot], outputs = [msg, chatbot]) 
                    with gr.Row():
                        clear = gr.Button('记录删除') # .click(clear_history, inputs=[chatbot], outputs=[chatbot])
                        undo = gr.Button('撤回上一条') # .click(undo, inputs=[chatbot], outputs=[chatbot])
                        
                        
                        clear.click(CHAT_ORG.qa_clear, inputs=[chatbot], outputs=[chatbot])
                        undo.click(CHAT_ORG.qa_undo, inputs=[chatbot], outputs=[chatbot])
                        send.click(CHAT_ORG.qa_answer, inputs=[msg, max_new_tokens, temperature, top_k, top_p, num_beams, chatbot], outputs=[msg, chatbot])
                        
                with gr.Accordion(label="微调模型对话测试", open=True):
                    with gr.Column():
                        with gr.Accordion(label='参数设置',open=False):
                            ft_max_new_tokens = gr.Slider(minimum=0, maximum=4096, value=1024, label='模型输出的最长Token(max_new_tokens)', info='这个参数决定了模型输出的最大token数量。增加这个值允许模型生成更长的文本，而减少这个值会导致生成的文本更短。')
                            ft_temperature = gr.Slider(maximum=2, minimum=0,value=1, label='温度值(temperature)', info='控制生成文本的随机性。较高的温度值会使输出更加多样和不可预测，而较低的值使输出更确定和重复。')
                            ft_top_k = gr.Slider(minimum=0, maximum=100, value=40, label='Top-k Sampling(top-k)', info='限制模型在每一步生成文本时考虑的最可能候选词的数量。较大的k值增加了多样性，但可能降低文本的连贯性；较小的k值则相反。')
                            ft_top_p = gr.Slider(minimum=0, maximum=2, value=0.75, label='Top-p Sampling(top-p)', info='类似于top_k，但通过选择累积概率高于某个阈值p的最小词集，动态调整考虑的候选词数量。较高的p值增加多样性，较低的p值提高连贯性。')
                            ft_num_beams = gr.Slider(minimum=0, maximum=12, value=5, label='Beam Search(num_beams)', info='在beam search中，num_beams指定了搜索宽度。更多的beams可以提高生成文本的质量，但也会增加计算负担。')  
                            #还可以添加更多
                            ft_wrong_message9 = gr.Markdown()
                        ft_start_testing_model = gr.Button('模型启动')
                        ft_testig_model_loaded = gr.Markdown()
                        ft_chatbot = gr.Chatbot(label='微调模型测试')
                        ft_msg = gr.Textbox(label="输入信息")

                        ft_msg.submit(FT_CHAT_ORG.qa_answer, inputs=[ft_msg, ft_max_new_tokens, ft_temperature, ft_top_k, ft_top_p, ft_num_beams, ft_chatbot], outputs=[ft_msg, ft_chatbot])
                        # 模型载入
                        ft_start_testing_model.click(FT_CHAT_ORG.load_model, inputs=[covert_hf_path, personal_model, model_path], outputs=[ft_testig_model_loaded])
                        
                        ft_send = gr.Button('信息发送')
                    with gr.Row():
                        ft_clear = gr.Button('记录删除')
                        ft_undo = gr.Button('撤回上一条')
                        
                        
                        ft_clear.click(FT_CHAT_ORG.qa_clear, inputs=[ft_chatbot], outputs=[ft_chatbot])
                        ft_undo.click(FT_CHAT_ORG.qa_undo, inputs=[ft_chatbot], outputs=[ft_chatbot])
                        ft_send.click(FT_CHAT_ORG.qa_answer, inputs=[ft_msg, ft_max_new_tokens, ft_temperature, ft_top_k, ft_top_p, ft_num_beams, ft_chatbot], outputs=[ft_msg, ft_chatbot])
        # with gr.Accordion(label='模型基础能力评估测试',open=False):
        #     mmlu_test_button = gr.Button('MMLU模型能力评估测试')
            with gr.Accordion(label="其他信息", open=True):
                star = gr.Markdown('### 如果您觉得该UI界面能帮助您高效完成微调工作，请为[XTuner](https://github.com/InternLM/xtuner.git)和[XTunerGUI](https://github.com/scchy/XtunerGUI.git)点个星星！非常感谢您的支持！')
                thanks = gr.Markdown('''### 最后，感谢XTUnerGUI团队成员对该项目的贡献：
            - [scchy](https://github.com/scchy) - 整体后端开发
            - [jianfeng777](https://github.com/Jianfeng777) - 整体前端开发
            - [l241025097](https://github.com/l241025097) - 模型训练终端可视化
            - [semple030228](https://github.com/semple030228) - 模型转化

            ### 同时也感谢XTuner团队的大力支持：
            - [HIT-cwh](https://github.com/HIT-cwh) - 配置文件生成及相关检查
            - [pppppM](https://github.com/pppppM) - 提供指导意见

            我们相信XTuner将成为国内有影响力的模型微调工具包！
            ''')

    #with gr.Tab('微调模型部署（LMDeploy）'):

    #with gr.Tab('微调模型测评（OpenCompass）'):

    demo.load(TR_CLS.read_log, outputs=[log_file], every=1)
    demo.launch(share=True) #, server_name="0.0.0.0", server_port=6007, root_path=f'/proxy/6007/') #, server_name="0.0.0.0", server_port=6006, root_path=f'/proxy/6006/')




