from xtuner.utils import PROMPT_TEMPLATE

OPENAI_FORMAT = ''' 
[
    {
        "messages": 
        [
            { "role": "system", "content": "You are an assistant that occasionally misspells words." },
            { "role": "user", "content": "Tell me a story." },
            { "role": "assistant", "content": "One day a student went to schoool." }
        ]
    },
    
    {
        "messages": 
        [
            { "role": "user", "content": "Tell me a story." },
            { "role": "assistant", "content": "One day a student went to schoool." }
        ]
    } 
]
'''

def read_first_ten_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # 读取所有行
            first_ten_lines = lines[:10]  # 获取前十行
            return ''.join(first_ten_lines)  # 将前十行合并为一个字符串并返回
    except Exception as e:
        return f"Error reading file: {str(e)}"


def get_template_format_by_name(template_name):
    template = PROMPT_TEMPLATE.get(template_name, None)
    if template is None:
        return "Template not found"
    return str(template)
