# python3
# Create Date: 2024-02-03
# Author: Scc_hy
# Func: chat center
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

user_prompt = "<|User|>:{user}\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"

  
class ModelCenter():
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load_model(self,
        model_personal_path,
        personal_model,
        model_path_in
        ):
        # 构造函数，加载检索问答链
        model_path = self.choice_path(model_personal_path, personal_model, model_path_in)
        print(f'ModelCenter.load_model({model_path})')
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            .to(torch.bfloat16)
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print('>>>>> Loading Model Done !')
        return f'>>> Loaded {model_path}'

    @staticmethod
    def choice_path(model_personal_path, personal_model, model_path_in):
        if len(model_personal_path) >= 3:
            return model_personal_path
        if len(personal_model) >= 3:
            return personal_model
        return model_path_in

    def qa_answer(self, question: str, max_new_tokens, temperature, top_k, top_p, num_beams, chat_history: list = []):
            if question == None or len(question) < 1:
                return "", chat_history
            try:
                question = question.replace(" ", '')
                response, history = self.model.chat(
                    self.tokenizer, 
                    question, 
                    history=chat_history,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=int(num_beams)
                )
                chat_history.append((question, response))
                return "", chat_history
            except Exception as e:
                return e, chat_history

    def qa_undo(self, chat_history: list = []):
        if len(chat_history):
            chat_history.pop()
        return chat_history

    def qa_clear(self, chat_history: list = []):
        return []
