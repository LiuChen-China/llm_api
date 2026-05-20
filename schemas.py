from pydantic import BaseModel,ConfigDict,Field
from typing import List, Optional, Union,Dict,Any
from dataclasses import dataclass, field
from HotReloadConfig import config

_OPENAI_TEXT_BLOCK_TYPES = {"text", "input_text"}
_OPENAI_IMAGE_BLOCK_TYPE = "image_url"
_ANTHROPIC_IMAGE_BLOCK_TYPE = "image"
_ANTHROPIC_TEXT_BLOCK_TYPE = "text"
_ANTHROPIC_TOOL_USE_BLOCK_TYPE = "tool_use"
_ANTHROPIC_TOOL_RESULT_BLOCK_TYPE = "tool_result"
_ANTHROPIC_THINKING_BLOCK_TYPE = "thinking"
_DEVELOPER_ROLE = "developer"
_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"
_TOOL_ROLE = "tool"


# ------------------- 兼容 OpenAI 请求/响应 数据结构 -------------------
class ChatContentBlock(BaseModel):
    """消息内容块（多模态支持）

    用于在 messages.content 中传递多模态信息，如文本、图片、音频等。
    当 content 为列表时，每个元素即为一个 ChatContentBlock。

    属性:
        type: 内容块类型标识。常见值: "text"(文本)、"image_url"(图片)、"input_text"(输入文本)
        text: 文本内容，仅当 type 为 "text" 或 "input_text" 时有效
        image_url: 图片信息，仅当 type 为 "image_url" 时有效，格式: {"url": "..."}
    """
    # extra="allow" 确保不会因为新的内容块类型而报 422
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="内容块类型，如 text/image_url/input_text")
    text: Optional[str] = Field(default=None, description="文本内容（type=text/input_text 时有效）")
    image_url: Optional[Dict[str, Any]] = Field(default=None, description="图片信息，格式: {url: str, detail?: str}")

class ToolCallFunction(BaseModel):
    """工具调用中的函数信息

    属性:
        name: 被调用的函数名称
        arguments: 函数参数，为 JSON 格式的字符串（由模型生成）
    """
    name: str = Field(..., description="被调用的函数名称")
    arguments: str = Field(..., description="函数参数，JSON 格式字符串")

class ToolCall(BaseModel):
    """工具调用对象

    当模型决定调用工具时，assistant 消息的 tool_calls 列表中的元素。

    属性:
        id: 工具调用的唯一标识，tool 角色消息需通过 tool_call_id 引用此 ID
        type: 工具类型，目前 OpenAI 仅支持 "function"
        function: 具体的函数调用信息（名称+参数）
    """
    id: str = Field(..., description="工具调用唯一标识，tool 消息通过 tool_call_id 引用")
    type: str = Field(default="function", description="工具类型，固定为 function")
    function: ToolCallFunction = Field(..., description="函数调用详情（名称+参数）")

class ResponseFormat(BaseModel):
    """响应格式控制

    用于指定模型输出的格式要求。

    属性:
        type: 响应格式类型。
            - "text": 普通文本输出（默认）
            - "json_object": 强制输出合法 JSON
            - "json_schema": 按指定 JSON Schema 输出
        json_schema: 当 type="json_schema" 时，指定输出应遵循的 JSON Schema
    """
    model_config = ConfigDict(extra="allow")

    type: str = Field(default="text", description="响应格式类型: text/json_object/json_schema")
    json_schema: Optional[Dict[str, Any]] = Field(default=None, description="当 type=json_schema 时的 Schema 定义")

class ChatMessage(BaseModel):
    """聊天消息单元（完全兼容 OpenAI 规范）

    支持所有 OpenAI 消息角色和字段：
    - system/user/assistant/tool/function/developer
    - content 允许 null（如 assistant 带 tool_calls 时）
    - 支持多模态内容块
    """
    model_config = ConfigDict(extra="allow")

    role: str = Field(..., description="消息角色: system/user/assistant/tool/function/developer")
    content: Optional[Union[str, List[ChatContentBlock]]] = Field(
        default=None, description="消息内容，可为文本、内容块列表或 null"
    )
    name: Optional[str] = Field(default=None, description="发送者名称（可选）")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="assistant 请求的工具调用列表"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="tool 角色消息对应的工具调用 ID"
    )
    refusal: Optional[str] = Field(
        default=None, description="模型拒绝回答的原因"
    )
    reasoning_content: Optional[str] = Field(
        default=None, description="推理/思考过程内容（深度思考模式时返回）"
    )

class ChatCompletionStreamOptions(BaseModel):
    """流式输出选项配置

    OpenAI 兼容：用于控制流式返回时的额外行为
    """
    model_config = ConfigDict(extra="allow")

    include_usage: bool = Field(
        default=False,
        description="是否在流式输出最后一个 chunk 中包含 usage 统计信息",
    )

class ChatCompletionRequest(BaseModel):
    """聊天补全请求（完全兼容 OpenAI API 规范）

    所有 OpenAI 标准字段均已声明，确保任何兼容 OpenAI 的客户端/插件
    都不会因为多传字段而收到 422。

    参数处理策略：
    - 已实现：model, messages, max_tokens, temperature, top_p, stop, stream,
      stream_options, tools, tool_choice, functions, function_call, enable_thinking
    - 显式拒绝（传入非默认值时返回 400）：n>1, response_format(非text),
      parallel_tool_calls=false
    - 接受但忽略（传入非默认值时记录警告）：presence_penalty, frequency_penalty,
      logit_bias, seed, logprobs, top_logprobs
    - 透传忽略（不影响行为）：user, service_tier, store, metadata
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # ---- 必填字段 ----
    model: str = Field(..., description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")

    # ---- 生成参数 ----
    max_completion_tokens: Optional[int] = Field(
        default=None, description="兼容字段：优先生效时会覆盖 max_tokens"
    )
    n: Optional[int] = Field(default=1, ge=1, description="生成候选数量")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="停止序列")
    logit_bias: Optional[Dict[str, float]] = Field(default=None, description="token 偏置")
    logprobs: Optional[bool] = Field(default=None, description="是否返回 logprobs")
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20, description="返回的 top logprobs 数量")
    seed: Optional[int] = Field(default=None, description="随机种子（用于可复现生成）")

    # ---- 流式控制 ----
    stream: Optional[bool] = Field(default=False, description="是否流式输出")
    stream_options: Optional[ChatCompletionStreamOptions] = Field(
        default=None, description="流式输出选项"
    )

    # ---- 工具与函数调用 ----
    tools: Optional[List[Dict[str, Any]]] = Field(default=[], description="可用工具列表")
    tool_choice: Optional[str] = Field(default="auto", description="工具选择策略")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="是否允许并行工具调用")

    # ---- 输出格式 ----
    response_format: Optional[ResponseFormat] = Field(default=None, description="响应格式")

    # ---- 其他标准字段 ----
    user: Optional[str] = Field(default=None, description="终端用户标识")
    service_tier: Optional[str] = Field(default=None, description="服务层级")
    store: Optional[bool] = Field(default=None, description="是否存储对话")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="请求元数据")

    # ------------------- OpenAI 官方标准扩展字段 -------------------
    extra: Optional[Dict[str, Any]] = Field(
        default={},
        description="""OpenAI 官方扩展参数位：
        - 存放自定义配置：思考开关、模型路由、调试参数等
        - 示例：{"enable_thinking": true}
        - 原生 OpenAI 会自动忽略，完美兼容所有客户端"""
    )



@dataclass(frozen=True)
class NormalizedContentPart:
    """标准化内容块。

    当前阶段主要用于保留协议入口中的结构化内容语义；
    现有 text-only 执行链路仍优先消费 `text` 聚合结果。
    """

    type: str
    text: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedMessage:
    """标准化消息对象。

    说明：
    - `content` 为现阶段下游执行链路使用的文本内容。
    - `content_parts` 为后续多模态保留的结构化内容。
    - `tool_calls` / `tool_call_id` 保留 function calling 所需上下文。
    """

    role: str
    content: Optional[str]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    is_error: Optional[bool] = None
    content_parts: List[NormalizedContentPart] = field(default_factory=list)

    def to_chat_dict(self) -> Dict[str, Any]:
        """转换为当前聊天执行链路可直接消费的字典结构。

        文本路径：content 为聚合后的纯文本，下游代码无需修改。
        视觉路径：若含图片内容块，额外写入 _image_parts 字段；
        视觉引擎（TransformersVisionLLMEngine）在策略层 execute() 中提取此字段
        并调用 build_processor_inputs() 构建 MultiModalData。
        """
        message: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            message["name"] = self.name
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.is_error is not None:
            message["is_error"] = self.is_error
        # 视觉路径：保留图片 payload，统一转换为 {"image_url": {"url": "..."}} 格式
        # 支持 OpenAI (type="image_url") 和 Anthropic (type="image", source.type="base64"|"url")
        image_parts: List[Dict[str, Any]] = []
        for p in self.content_parts:
            if p.type == _OPENAI_IMAGE_BLOCK_TYPE:
                # OpenAI 格式：payload 已是 {"image_url": {"url": "...", ...}}
                image_parts.append(p.payload)
        if image_parts:
            message["_image_parts"] = image_parts
        return message

class ToolCallChunkFunction(BaseModel):
    """流式工具调用中的函数增量

    在流式输出中，函数名称和参数会被拆分为多个增量 chunk 逐步推送。

    属性:
        name: 函数名称（通常仅在首个 chunk 中出现）
        arguments: 函数参数的增量片段（逐步拼接为完整 JSON）
    """
    name: Optional[str] = Field(default=None, description="函数名称（首个 chunk 中出现）")
    arguments: Optional[str] = Field(default=None, description="函数参数增量片段")

class ToolCallChunk(BaseModel):
    """
    流式工具调用增量
    流式输出中 delta.tool_calls 数组的元素，逐步推送工具调用信息。
    """
    index: int = Field(..., description="工具调用在 tool_calls 数组中的序号")
    id: Optional[str] = Field(default=None, description="工具调用唯一标识（首个 chunk）")
    type: Optional[str] = Field(default=None, description="工具类型（首个 chunk，固定为 function）")
    function: Optional[ToolCallChunkFunction] = Field(default=None, description="函数调用增量信息")

class FunctionCall(BaseModel):
    """
    函数调用信息（已废弃，保留向后兼容）
    此模型用于兼容旧版 OpenAI API 的 function_call 字段。
    新代码应使用 ToolCall 模型代替。
    """
    name: str = Field(..., description="被调用的函数名称")
    arguments: str = Field(..., description="函数参数，JSON 格式字符串")

class ChatCompletionDelta(BaseModel):
    """
    流式输出的增量内容
    SSE 流式输出中每个 chunk 的 choices[i].delta 字段。
    首个 chunk 通常只包含 role，后续 chunk 包含 content 增量。
    """
    role: Optional[str] = Field(default=None, description="消息角色（仅首个 chunk 出现）")
    content: Optional[str] = Field(default=None, description="增量文本内容")
    tool_calls: Optional[List[ToolCallChunk]] = Field(
        default=None, description="增量工具调用列表"
    )
    function_call: Optional[FunctionCall] = Field(
        default=None, description="（已废弃）增量函数调用信息"
    )
    refusal: Optional[str] = Field(default=None, description="模型拒绝回答的原因")
    reasoning_content: Optional[str] = Field(
        default=None, description="增量推理/思考内容（深度思考模式的流式输出）"
    )

class ChatCompletionChunkChoice(BaseModel):
    """流式输出中的单个选择项"""
    index: int = Field(default=0, description="结果在 choices 数组中的序号")
    delta: ChatCompletionDelta = Field(..., description="本次 chunk 的增量内容")
    finish_reason: Optional[str] = Field(default=None, description="生成结束原因（仅最后一个内容 chunk）")
    logprobs: Optional[Any] = Field(default=None, description="token 级别的对数概率信息")

class CompletionTokensDetails(BaseModel):
    """生成 token 的细分统计"""
    reasoning_tokens: int = Field(default=0, description="推理过程消耗的 token 数")

class PromptTokensDetails(BaseModel):
    """提示 token 的细分统计"""
    cached_tokens: int = Field(default=0, description="命中 KV Cache 的 token 数")

class ChatCompletionUsage(BaseModel):
    """
    Token 使用量统计（兼容 OpenAI 规范）
    用于记录本次对话的 token 消耗情况，可用于计费和监控。
    """
    prompt_tokens: int = Field(default=0, description="输入提示消耗的 token 数")
    completion_tokens: int = Field(default=0, description="模型生成回复消耗的 token 数")
    total_tokens: int = Field(default=0, description="总 token 数 (prompt + completion)")
    completion_tokens_details: Optional[CompletionTokensDetails] = Field(
        default=None, description="生成 token 细分统计"
    )
    prompt_tokens_details: Optional[PromptTokensDetails] = Field(
        default=None, description="提示 token 细分统计"
    )


class ChatCompletionChunkResponse(BaseModel):
    """
    流式聊天补全响应的单个 chunk（完全兼容 OpenAI 规范）
    SSE 流式输出中每个 "data: {...}" 行的数据结构。
    同一次对话的所有 chunk 共享相同的 id 和 created。
    """
    id: str = Field(..., description="响应唯一标识（同一对话的所有 chunk 共享）")
    object: str = Field(default="chat.completion.chunk", description="对象类型，固定为 chat.completion.chunk")
    created: int = Field(..., description="响应创建的 Unix 时间戳（秒）")
    model: str = Field(..., description="实际执行推理的模型名称")
    choices: List[ChatCompletionChunkChoice] = Field(..., description="生成结果列表（usage chunk 中为空列表）")
    usage: Optional[ChatCompletionUsage] = Field(default=None, description="Token 使用量（仅 include_usage=true 的最终 chunk）")
    system_fingerprint: Optional[str] = Field(
        default=None, description="系统指纹，用于追踪后端配置变化"
    )
    service_tier: Optional[str] = Field(default=None, description="服务层级标识")