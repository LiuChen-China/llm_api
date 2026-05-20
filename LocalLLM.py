from pydantic import BaseModel,ConfigDict,Field
from typing import List, Optional, Union,Dict,Any
from llama_cpp import Llama
import llama_cpp.llama_cpp as llama_cpp
from jinja2 import Template
import asyncio
import json
from llama_cpp import GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0
from HotReloadConfig import config
from schemas import *
from utils import *
import time
import threading

think_tag_start = config.llm.think_tag_start

def get_type_k(type_k_str:str):
    '''根据字符串获取type_k类型'''
    if type_k_str == "q8_0":
        type_k = GGML_TYPE_Q8_0
    elif type_k_str == "f16":
        type_k = GGML_TYPE_F16
    elif type_k_str == "q4_0":
        type_k = GGML_TYPE_Q4_0
    else:
        raise Exception(f"没见过的type_k类型{config.llm.type_k}")
    return type_k

def get_type_v(type_v_str:str):
    '''根据字符串获取type_v类型'''
    if type_v_str == "q8_0":
        type_v = GGML_TYPE_Q8_0
    elif type_v_str == "f16":
        type_v = GGML_TYPE_F16
    elif type_v_str == "q4_0":
        type_v = GGML_TYPE_Q4_0
    else:
        raise Exception(f"没见过的type_v类型{config.llm.type_v}")
    return type_v

class LocalLLM:
    def __init__(self, model_path, n_ctx, n_gpu_layers,type_k,type_v):
        """基于LLAMA_CPP的本地部署大模型 目前只适配qwen模型，它的对话模板需要手动做些操作才能软切换思考模式"""
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=4,
            n_gpu_layers=n_gpu_layers,
            type_k=get_type_k(type_k),
            type_v=get_type_v(type_v),
            flash_attn=True,  
            verbose=False, 
        )
        self.n_ctx = n_ctx
        self.template = self.llm.metadata["tokenizer.chat_template"]
        
        self._async_loop = None          # 长期运行的事件循环
        self._async_thread = None        # 运行循环的线程
        self._async_ready = threading.Event()
        self._start_async_loop()


    def _start_async_loop(self):
        """启动长期运行的异步线程"""
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._async_loop = loop
            self._async_ready.set()
            loop.run_forever()
            # 执行到这里表示 loop.stop() 被调用，可以清理
            # 取消所有剩余任务
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()
        self._async_thread = threading.Thread(target=run_loop, daemon=True)
        self._async_thread.start()
        self._async_ready.wait()  # 确保循环已就绪

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        tokens = self.llm.tokenize(
            text.encode("utf-8"),
            add_bos=False,  # 对话场景一般不加开始符
            special=True    # 解析特殊token如<|im_start|>
        )
        return len(tokens)

    def build_prompt(self,messages:List[Dict[str, Any]],enable_thinking,tools=[],tool_choice="auto") -> str:
        '''使用gguf对话模板将聊天列表转为提示词'''
        # 渲染提示模板
        prompt = Template(self.template).render(
            messages=messages,
            tools=tools,                
            tool_choice=tool_choice,
            enable_thinking=enable_thinking,
            add_generation_prompt=True
        )
        return prompt

    def trim_prompt(self,prompt:str,max_tokens:int) -> str:
        '''根据模型上下文长度 和 最大生成数 裁剪提示词'''
        token_count = self.count_tokens(prompt)# 计算提示词token数量
        max_available_prompt_tokens = self.n_ctx - max_tokens# 计算最大可用提示词token数量
        if token_count > max_available_prompt_tokens:
            tokens = self.llm.tokenize(prompt.encode("utf-8"),add_bos=False,special=True)
            prompt = self.llm.detokenize(tokens[:max_available_prompt_tokens]).decode("utf-8", errors="ignore")
        return prompt

    def create_completion(
            self,
            messages:List[Dict[str, Any]],
            stream:bool=True,
            enable_thinking:bool=True,
            max_tokens:int=512,
            temperature:float=1.0,
            top_p:float=0.95,
            top_k: int=20,
            min_p:float=0.0,
            presence_penalty:float=1.5,
            repeat_penalty:float=1.0,
            tools:List=[],
            tool_choice:str="auto",
            ):
        '''聊天统一接口，只返回文本内容，没有额外元信息开销'''
        #使用模型对话模板构建提示词
        prompt = self.build_prompt(messages=messages,enable_thinking=enable_thinking,tools=tools,tool_choice=tool_choice)
        # 裁剪提示词
        prompt = self.trim_prompt(prompt,max_tokens)
        if stream:
            return self.stream(
                prompt=prompt,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                )
        else:
            return self.generate(
                prompt=prompt,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                )
        # 空生成器 用来避免pydantic的警告
        return (x for x in ())

    def stream(self, prompt, enable_thinking, max_tokens, temperature, top_p, top_k, min_p, presence_penalty, repeat_penalty):
        """对话内容流式生成"""
        if enable_thinking:
            yield think_tag_start

        # 分词（完全对齐llama_cpp内部逻辑）
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        completion_tokens = []
        prompt_tokens = tokens.copy()
        # 原生generate循环，参数严格对齐内部实现
        for token in self.llm.generate(
            tokens=prompt_tokens,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temp=temperature,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
        ):
            # 结束符判断（对齐内部）
            if self.llm._model.vocab is not None and llama_cpp.llama_vocab_is_eog(self.llm._model.vocab, token):
                break
            if token == self.llm.token_eos():
                break

            completion_tokens.append(token)
            
            # 安全解码（对齐内部utf-8处理）
            token_text = self.llm.detokenize([token]).decode("utf-8", errors="ignore")
            yield token_text

                
            # 最大长度限制
            if len(completion_tokens) >= max_tokens:
                break

    def generate(self, prompt, enable_thinking, max_tokens, temperature, top_p, top_k, min_p, presence_penalty, repeat_penalty):
        """对话内容非流式生成"""
        result = ""
        for token_text in self.stream(
            prompt=prompt,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
        ):
            result += token_text
        return result

    async def create_completion_async(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = True,
        enable_thinking: bool = True,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repeat_penalty: float = 1.0,
        tools: List = [],
        tool_choice: str = "auto",
    ):
        """
        修复版：异步流式接口，只执行一次 create_completion
        """
        def sync_func():
            return self.create_completion(
                messages=messages,
                stream=stream,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                tools=tools,
                tool_choice=tool_choice,
            )

        # 关键：只在线程池里执行 一次，并迭代
        if stream:
            # 正确写法：把整个迭代逻辑丢线程池，不拆开发挥
            loop = asyncio.get_event_loop()
            gen = await loop.run_in_executor(None, sync_func)
            # 这里不会再次执行！
            for chunk in gen:
                yield chunk
        else:
            result = await asyncio.get_event_loop().run_in_executor(None, sync_func)
            yield result

if __name__ == "__main__":
    llm = LocalLLM(
        model_path=config.llm.model_path,
        n_ctx=int(config.llm.ctx_size),
        n_gpu_layers=config.llm.n_gpu_layers,
        type_k=config.llm.type_k,
        type_v=config.llm.type_v,
    )

    # ===================== 测试1：普通对话=====================
    messages = [{"role": "user", "content": "查一下长沙2025年4月1日的天气"}]
    tools = [
        {
            "name": "get_weather",
            "description": "查询指定城市、指定日期的天气，日期格式为YYYY-MM-DD",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、长沙"
                    },
                    "date": {
                        "type": "string",
                        "description": "查询日期，格式：YYYY-MM-DD，如：2025-04-01"
                    }
                },
                "required": ["city", "date"]
            }
        }
    ]
    # 调用流式生成
    stream_result = llm.create_completion(
        messages=messages,
        stream=True,
        enable_thinking=False,
        max_tokens=100,
        temperature=0.7,
        tools=tools,
    )
    text = ''

    # 逐块打印输出
    for token in stream_result:
        print(token, end="", flush=True)
        text += token

    print(f"token 数量: {llm.count_tokens(text)}")

