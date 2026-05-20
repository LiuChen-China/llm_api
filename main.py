from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,ConfigDict,Field
from typing import List, Optional, Union,Dict,Any
from fastapi.responses import StreamingResponse
import asyncio
import json
import uuid
import time
from HotReloadConfig import config
from schemas import *
from utils import *
from LocalLLM import LocalLLM
import orjson

# 初始化模型
localLLM = LocalLLM(
    model_path=config.llm.model_path,
    n_ctx=int(config.llm.ctx_size),
    n_gpu_layers=config.llm.n_gpu_layers,
    type_k=config.llm.type_k,
    type_v=config.llm.type_v,
    )

# 初始化 FastAPI
app = FastAPI()

# 跨域支持（前端可以直接调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- 核心接口：兼容 OpenAI /v1/chat/completions -------------------
@app.post("/v1/chat/completions")
async def chat_completions(req:ChatCompletionRequest):
    '''openai-api风格聊天接口，但模型固定为已部署模型'''
    # 聚合消息 有些请求消息内部元素还是列表
    messages = normalize_openai_messages(req.messages)
    # 清理消息 中 如思考内容的部分
    messages = clean_messages(messages)
    #模型名
    model_name = config.llm.model_name
    #最大生成数
    max_tokens = req.model_extra.get('max_tokens', req.extra.get('max_tokens', config.llm.max_tokens))
    #思考模式
    enable_thinking = req.model_extra.get('enable_thinking', req.extra.get('enable_thinking', True))
    #根据思考模式匹配推荐参数 或者 直接使用请求传参(优先) 
    mode_args = config.llm.think_mode if enable_thinking else config.llm.non_think_mode
    temperature = req.model_extra.get('temperature', req.extra.get('temperature', mode_args.temperature))
    top_p = req.model_extra.get('top_p', req.extra.get('top_p', mode_args.top_p))
    top_k = req.model_extra.get('top_k', req.extra.get('top_k', mode_args.top_k))
    min_p = req.model_extra.get('min_p', req.extra.get('min_p', mode_args.min_p))
    presence_penalty = req.model_extra.get('presence_penalty', req.extra.get('presence_penalty', mode_args.presence_penalty))
    repeat_penalty = req.model_extra.get('repeat_penalty', req.extra.get('repeat_penalty', mode_args.repeat_penalty))

    async def event_generator():
        """SSE 事件生成器：逐 chunk 推送流式响应"""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        include_usage = bool(req.stream_options and req.stream_options.include_usage)
        stream_events: list[dict[str, Any]] = []

        # ---- SSE 格式化辅助函数 ----
        def _sse(chunk: ChatCompletionChunkResponse) -> str:
            """将 chunk 对象序列化为 SSE data 行"""
            payload = chunk.model_dump(exclude_none=True)
            stream_events.append(payload)
            return "data: " + orjson.dumps(payload).decode() + "\n\n"

        def _delta_sse(
            delta: ChatCompletionDelta,
            finish_reason: str | None = None,
        ) -> str:
            """构建包含单个 delta 的 SSE data 行"""
            return _sse(ChatCompletionChunkResponse(
                id=chunk_id,
                created=created,
                model=req.model,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )],
            ))

        def _thinking_delta_sse(field: str, text: str) -> str:
            """将 ThinkingStreamSplitter 输出转换为 SSE data 行"""
            return _delta_sse(ChatCompletionDelta(
                reasoning_content=text if field == "reasoning_content" else None,
                content=text if field == "content" else None,
            ))

        # ---- OpenAI 规范：首个 chunk 仅包含 role="assistant" ----
        yield _delta_sse(ChatCompletionDelta(role="assistant"))

        # ---- 初始化流式解析器 ----
        full_text = ""
        finish_reason = "stop"
        thinking_splitter = ThinkingStreamSplitter()
        # 仅当 tools 存在且 tool_choice != "none" 时开启 ToolCallStreamSplitter
        needs_tool_detection = bool(req.tools and req.tool_choice != "none")
        tool_splitter = ToolCallStreamSplitter() if needs_tool_detection else None
        parsed_calls = None

        async for text_chunk in localLLM.create_completion_async(
            messages=messages,
            stream=True,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            tools=req.tools,
            tool_choice=req.tool_choice,
        ):
            if not text_chunk:
                continue
            full_text += text_chunk

            if tool_splitter is not None:
                # 工具调用检测模式：标记前正文实时推送，标记后缓冲
                for _field, _text in tool_splitter.feed(text_chunk):
                    for tf, tt in thinking_splitter.feed(_text):
                        yield _thinking_delta_sse(tf, tt)
            else:
                # 无工具：直接实时推送
                for field, text in thinking_splitter.feed(text_chunk):
                    yield _thinking_delta_sse(field, text)


        # ---- 生成结束后的处理 ----

        if needs_tool_detection:
            # 优先尝试从 ToolCallStreamSplitter 缓冲区解析（含有明确工具标记的模型）
            if tool_splitter and tool_splitter.has_tool_call_content():
                parsed_calls = parse_tool_calls_from_text(tool_splitter.get_buffer(), extract_allowed_tool_names(req.tools))

            if not parsed_calls:
                # 回退：尝试对全文解析（无明确标记的通用 JSON 格式）
                parsed_calls = parse_tool_calls_from_text(full_text, extract_allowed_tool_names(req.tools))


            if parsed_calls:
                # 按 OpenAI 规范推送 tool_calls 增量 delta
                for idx, tc in enumerate(parsed_calls):
                    fn = tc["function"]
                    yield _delta_sse(ChatCompletionDelta(
                        tool_calls=[ToolCallChunk(
                            index=idx,
                            id=tc["id"],
                            type="function",
                            function=ToolCallChunkFunction(
                                name=fn["name"],
                                arguments="",
                            ),
                        )],
                    ))
                    yield _delta_sse(ChatCompletionDelta(
                        tool_calls=[ToolCallChunk(
                            index=idx,
                            function=ToolCallChunkFunction(
                                arguments=fn["arguments"],
                            ),
                        )],
                    ))
                finish_reason = "tool_calls"
            else:
                # 无工具调用：将 ToolCallStreamSplitter 尾部窗口内容也推送出去
                if tool_splitter:
                    for _field, _text in tool_splitter.flush_content():
                        for tf, tt in thinking_splitter.feed(_text):
                            yield _thinking_delta_sse(tf, tt)
        # 最后刷新 thinking_splitter 的剩余内容
        for field, text in thinking_splitter.flush():
            yield _thinking_delta_sse(field, text)


        # ---- 结束 chunk（含 finish_reason）----
        yield _delta_sse(ChatCompletionDelta(), finish_reason=finish_reason)

        # OpenAI 规范：流式结束时发送 [DONE] 标记
        stream_events.append({"data": "[DONE]"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10002)