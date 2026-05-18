from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import time
import uuid
from llama_cpp import Llama
from HotReloadConfig import config



# 初始化模型
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=0,  # 有 GPU 可以改 >0
    verbose=False,
)

# 初始化 FastAPI
app = FastAPI(title="Local OpenAI Compatible API", version="1.0")

# 跨域支持（前端可以直接调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- 兼容 OpenAI 请求/响应 数据结构 -------------------
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

# ------------------- 核心接口：兼容 OpenAI /v1/chat/completions -------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # 构造模型输入格式
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # 调用本地模型
        output = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 包装成 OpenAI 格式返回
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output["choices"][0]["message"]["content"]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "completion_tokens": output["usage"]["completion_tokens"],
                "total_tokens": output["usage"]["total_tokens"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理错误：{str(e)}")

# 健康检查
@app.get("/v1/models")
async def models():
    return {
        "data": [
            {
                "id": "local-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)