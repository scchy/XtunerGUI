from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights


MODEL_TEMPLATE_MAPPING = dict(
    InternLM2ForCausalLM='internlm2_chat',
    InternLMForCausalLM='internlm_chat',
    BaiChuanForCausalLM='baichuan_chat',
    BaichuanForCausalLM='baichuan2_chat',
    DeepseekForCausalLM='deepseek_moe',
    MixtralForCausalLM='mixtral',
    QWenLMHeadModel='qwen_chat',
    GPTBigCodeForCausalLM='default'
)


def get_prompt_template(pretrained_model_name_or_path):
    try:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except:
        return f'Model {pretrained_model_name_or_path} can not be loaded.', None
    model_type = type(model).__name__
    if model_type == 'LlamaForCausalLM':
        vocab_size = config.vocab_size
        if vocab_size == 32256:
            return 'Success', 'deepseek_coder'
        elif vocab_size == 64000:  # yi
            return 'Success', 'default'
        elif vocab_size == 32000:  # llama2
            return 'Success', 'llama2_chat'
    elif model_type == 'ChatGLMForConditionalGeneration':
        seq_length = config.seq_length
        if seq_length == 131072:
            return 'Success', 'chatglm3'
        elif seq_length == 32768:
            return 'Success', 'chatglm2'
        else:
            return 'Fail to match automatically, please enter corresponding prompt template manually', None
    elif model_type == 'MistralForCausalLM':
        # 无法判断
        return 'The prompt template should be one of mistral or zephyr, please enter the correct prompt template manually', None
    elif model_type in MODEL_TEMPLATE_MAPPING:
        return 'Success', MODEL_TEMPLATE_MAPPING[model_type]
    else:
        return 'Fail to match automatically, please enter corresponding prompt template manually', None
    

print(get_prompt_template('/mnt/petrelfs/share_data/caoweihan/official_Ampere_7B_1_0_0'))