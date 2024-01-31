from datasets import load_dataset
import pprint


DATA_EXAMPLE = """example["messages"] = [
    { "role": "system", "content": "You are an assistant that
        occasionally misspells words." },
    { "role": "user", "content": "Tell me a story." },
    { "role": "assistant", "content": "One day a student
        went to schoool." }]"""

def check_custom_dataset(path):
    try:
        data = load_dataset('json', data_files=path)
    except:
        return f"There's a problem with the JSON file in {path}; it can't be read."
    data = data['train']

    if 'messages' not in data.column_names:
        return ('Expect "messages" as a column in the dataset. Here is an '
                f'example:\n{DATA_EXAMPLE}')
    
    if not isinstance(data['messages'], (list, tuple)):
        return ('Expect the type of example["messages"] to be a list or '
                f'a tuple, but got {type(data["messages"])}.'
                f'Here is an example:\n{DATA_EXAMPLE}')
    
    check_first_n_messages = 100
    for message_idx, message in enumerate(data['messages'][:check_first_n_messages]):
        for conv_idx, single_conversation in enumerate(message):
            if not isinstance(single_conversation, dict):
                return ('Expect each single conversation to be a dict, '
                        f'but got {type(single_conversation)}. '
                        f'Here is an example:\n{DATA_EXAMPLE}')
            if not {'role', 'content'}.issubset(single_conversation.keys()):
                return ('Expect "role" and "content" in each single '
                        f'conversation. The {conv_idx + 1} conversation in the'
                        f' {message_idx} message is {single_conversation}.'
                        f'Here is an example:\n{DATA_EXAMPLE}')
    
    return None

out = check_custom_dataset('/mnt/petrelfs/caoweihan/projects/xtuner/data.json')
if out is None:
    print('Data is OK.')
else:
    pprint.pprint(out)
