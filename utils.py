from typing import Any, Dict, List, Optional, Sequence,Tuple
from fastapi import Request
from HotReloadConfig import config
from schemas import _OPENAI_TEXT_BLOCK_TYPES,_OPENAI_IMAGE_BLOCK_TYPE,_DEVELOPER_ROLE,_SYSTEM_ROLE,_USER_ROLE
from schemas import *
import json
import re
import uuid

def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """一键清理对话历史中的  思考内容"""
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
            
        new_msg = msg.copy()
        content = new_msg.get("content", "")
        
        if isinstance(content, str):
            # 模式1: 完整的思考标签对 <think>...</think>
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            # 模式2: 未闭合的思考标签 <think>...（到文本末尾）
            content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)
            # 清理多余的空白字符
            content = re.sub(r'\n\s*\n', '\n', content)# 多个换行合并为一个
            content = content.strip()
            new_msg["content"] = content
            
        if content and content.strip() or new_msg.get("tool_calls") or new_msg.get("tool_call_id") or new_msg.get("_image_parts"):
            cleaned.append(new_msg)
    return cleaned

def _model_dump_if_possible(value: Any) -> Dict[str, Any]:
    """尽量将 Pydantic 对象转成字典。

    未知对象兜底为浅字典或字符串包装，避免归一化阶段抛出无关异常。
    """
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


def extract_openai_content_parts(content: Any) -> List[NormalizedContentPart]:
    """提取 OpenAI content 的结构化内容块。

    当前保持与现有系统一致：
    - 文本块被保留
    - 图片块显式拒绝
    - 未知块兜底为字符串文本
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [NormalizedContentPart(type="text", text=content)]

    if not isinstance(content, list):
        return [NormalizedContentPart(type="text", text=str(content))]

    parts: List[NormalizedContentPart] = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            text_value = block.get("text")
            payload = dict(block)
        elif hasattr(block, "type"):
            block_type = getattr(block, "type", None)
            text_value = getattr(block, "text", None)
            payload = _model_dump_if_possible(block)
        else:
            parts.append(NormalizedContentPart(type="text", text=str(block)))
            continue

        if block_type in _OPENAI_TEXT_BLOCK_TYPES:
            parts.append(
                NormalizedContentPart(
                    type=str(block_type),
                    text=text_value if isinstance(text_value, str) else "",
                    payload=payload,
                )
            )
            continue

        if block_type == _OPENAI_IMAGE_BLOCK_TYPE:
            # 保留图片内容块到 content_parts：不再主动拒绝，而是保留结构化信息
            # 文本消费路径（text = None）不影响，视觉引擎路径通过 content_parts 检测图片
            image_url = block.get("image_url") if isinstance(block, dict) else getattr(block, "image_url", None)
            parts.append(
                NormalizedContentPart(
                    type=_OPENAI_IMAGE_BLOCK_TYPE,
                    text=None,
                    payload={"image_url": image_url} if image_url else payload,
                )
            )
            continue

        parts.append(
            NormalizedContentPart(
                type=str(block_type or "unknown"),
                text=str(block),
                payload=payload,
            )
        )

    return parts

def flatten_content_parts(parts: Sequence[NormalizedContentPart]) -> str:
    """将内容块聚合为当前执行链路可消费的纯文本。"""
    texts = [part.text for part in parts if isinstance(part.text, str) and part.text]
    return "\n".join(texts)

def normalize_openai_message_role(role: str) -> str:
    """标准化 OpenAI 消息角色。

    - `developer` 统一映射为 `system`
    - 空角色兜底为 `user`
    """
    normalized = (role or "").strip().lower()
    if normalized == _DEVELOPER_ROLE:
        return _SYSTEM_ROLE
    return normalized or _USER_ROLE

def normalize_openai_messages(messages: Sequence[ChatMessage]):
    """标准化 OpenAI Chat 消息列表。"""
    normalized_messages: List[NormalizedMessage] = []
    for message in messages:
        parts = extract_openai_content_parts(getattr(message, "content", None))
        content = flatten_content_parts(parts)
        tool_calls = getattr(message, "tool_calls", None)
        normalized_tool_calls = None
        if tool_calls:
            normalized_tool_calls = [
                tool_call.model_dump(exclude_none=True)
                if hasattr(tool_call, "model_dump") else dict(tool_call)
                for tool_call in tool_calls
            ]
            if not content:
                content = None

        normalized_messages.append(
            NormalizedMessage(
                role=normalize_openai_message_role(getattr(message, "role", _USER_ROLE)),
                content=content,
                name=getattr(message, "name", None),
                tool_calls=normalized_tool_calls,
                tool_call_id=getattr(message, "tool_call_id", None),
                content_parts=parts,
            )
        )
    text_messages = []
    for msg in normalized_messages:
        text_messages.append({"role": msg.role, "content": msg.content})
    return text_messages

# </think> 标签长度，用于缓冲区尾部保留
_THINK_END_TAG = "</think>"
_THINK_END_LEN = len(_THINK_END_TAG)
_THINK_START_TAG = "<think>"
_THINK_START_LEN = len(_THINK_START_TAG)


class ThinkingStreamSplitter:
    """流式 <think> 标签拆分器

    逐 chunk 输入模型生成的文本流，实时将 <think>...</think> 内的内容
    与正文内容分离到不同的输出字段。

    状态转换:
        INIT ──检测到<think>──▶ THINKING ──检测到</think>──▶ CONTENT
          │                                                     │
          │ 无<think>标签                                        │
          ▼                                                     ▼
        CONTENT (全部→content)                         (所有后续→content)

    使用示例:
        splitter = ThinkingStreamSplitter()
        for chunk in model_stream:
            for field, text in splitter.feed(chunk):
                # field: "reasoning_content" 或 "content"
                delta = {field: text}
        # 流结束时刷新缓冲区
        for field, text in splitter.flush():
            delta = {field: text}
    """

    def __init__(self) -> None:
        # 状态: "init" → "thinking" → "content"
        self._state: str = "init"
        # 缓冲区，用于处理跨 chunk 的标签边界
        self._buffer: str = ""
        # 从 thinking 切换到 content 后，首次输出需跳过前导空白行
        self._content_started: bool = False

    @property
    def state(self) -> str:
        """当前解析状态"""
        return self._state

    def feed(self, chunk: str) -> List[Tuple[str, str]]:
        """输入一个文本 chunk，返回解析后的 (字段名, 文本) 列表

        参数:
            chunk: 模型流式输出的一个文本片段

        返回:
            列表，每个元素为 (field_name, text) 元组:
            - field_name: "reasoning_content" 或 "content"
            - text: 对应的文本片段
        """
        if not chunk:
            return []

        results: List[Tuple[str, str]] = []
        self._buffer += chunk

        if self._state == "init":
            results.extend(self._handle_init())

        if self._state == "thinking":
            results.extend(self._handle_thinking())

        if self._state == "content":
            results.extend(self._handle_content())

        return results

    def flush(self) -> List[Tuple[str, str]]:
        """流结束时刷新缓冲区中的剩余内容

        返回:
            剩余内容的 (field_name, text) 列表
        """
        results: List[Tuple[str, str]] = []
        if self._buffer:
            if self._state == "thinking":
                # 未闭合的 <think>，剩余内容作为 reasoning_content
                results.append(("reasoning_content", self._buffer))
            else:
                results.append(("content", self._buffer))
            self._buffer = ""
        return results

    def _handle_init(self) -> List[Tuple[str, str]]:
        """处理 INIT 状态：检测是否以 <think> 开头"""
        results: List[Tuple[str, str]] = []

        # 缓冲区不够长，可能是 <think> 的前缀，继续等待
        if len(self._buffer) < _THINK_START_LEN:
            if _THINK_START_TAG.startswith(self._buffer):
                return results
            # 不是 <think> 前缀，切换到 content 状态
            self._state = "content"
            return results

        # 判断是否以 <think> 开头
        if self._buffer.startswith(_THINK_START_TAG):
            # 去掉 <think> 标签本身，切换到 thinking 状态
            self._buffer = self._buffer[_THINK_START_LEN:]
            self._state = "thinking"
        else:
            # 不含 <think>，全部作为 content（无需跳过前导空白）
            self._state = "content"
            self._content_started = True

        return results

    def _handle_thinking(self) -> List[Tuple[str, str]]:
        """处理 THINKING 状态：检测 </think> 并分离内容"""
        results: List[Tuple[str, str]] = []

        end_idx = self._buffer.find(_THINK_END_TAG)
        if end_idx != -1:
            # 找到 </think>，标签前的内容为 reasoning_content
            thinking_text = self._buffer[:end_idx]
            if thinking_text:
                results.append(("reasoning_content", thinking_text))

            # 标签后的内容为 content（去除前导空白行）
            after = self._buffer[end_idx + _THINK_END_LEN:]
            self._buffer = after.lstrip("\n")
            self._state = "content"
        else:
            # 未检测到完整的 </think>
            # 保留尾部缓冲区，防止 </think> 被跨 chunk 切割
            safe_len = len(self._buffer) - _THINK_END_LEN
            if safe_len > 0:
                results.append(("reasoning_content", self._buffer[:safe_len]))
                self._buffer = self._buffer[safe_len:]

        return results

    def _handle_content(self) -> List[Tuple[str, str]]:
        """处理 CONTENT 状态：直接输出为 content"""
        results: List[Tuple[str, str]] = []

        if self._buffer:
            # 从 thinking 切换到 content 后，跳过前导空白行（如 </think> 与正文之间的 \n\n）
            if not self._content_started:
                self._buffer = self._buffer.lstrip("\n")
                if not self._buffer:
                    return results
                self._content_started = True
            results.append(("content", self._buffer))
            self._buffer = ""

        return results

# 工具调用起始标记列表（大小写敏感，按匹配优先级排序）。
# 如需支持新模型的工具调用格式，在此处追加对应起始标记即可，
# 无需修改类的内部逻辑。
_TOOL_CALL_OPEN_MARKERS: List[str] = [
    "<tool_call>",      # Qwen3 / 通用注入模板（最通用，优先匹配）
    "<function=",       # Qwen3-Coder 原生 function tag 格式
    "[TOOL_CALLS]",     # 部分中文模型
]

# 最长标记长度：决定滑动窗口尾部的保留量。
# 保留最后 _MAX_MARKER_LEN - 1 字节，防止起始标记被跨 chunk 切割后漏检。
_MAX_MARKER_LEN: int = max(len(m) for m in _TOOL_CALL_OPEN_MARKERS)

def _find_first_marker(text: str) -> Optional[int]:
    """在文本中找到第一个工具调用起始标记的位置（最早出现的那个）。

    遍历所有已注册的起始标记，返回位置最靠前的那个的索引。

    Args:
        text: 待搜索文本

    Returns:
        起始标记在文本中的起始索引，未找到任何标记则返回 None
    """
    earliest: Optional[int] = None
    for marker in _TOOL_CALL_OPEN_MARKERS:
        idx = text.find(marker)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    return earliest

class ToolCallStreamSplitter:
    """流式工具调用分离器。

    逐 chunk 处理模型流式输出，自动将输出分为：
    1. 工具调用标记前的正文内容 → 实时 yield ("content", text)
    2. 工具调用标记及其后的所有内容 → 缓冲，流结束后通过 get_buffer() 获取

    通过滑动窗口机制处理跨 chunk 的标记边界，保证标记不会因 chunk 切割而漏检。

    使用示例::

        splitter = ToolCallStreamSplitter()
        thinking_splitter = ThinkingStreamSplitter()

        async for chunk in model_stream:
            for field, text in splitter.feed(chunk):
                # field == "content"：工具调用标记之前的正文，实时经过 ThinkingStreamSplitter
                for tf, tt in thinking_splitter.feed(text):
                    yield sse(tf, tt)

        # 流结束后
        if splitter.has_tool_call_content():
            # 有明确工具调用标记的模型路径
            tool_calls = parse_tool_calls(splitter.get_buffer())
        else:
            # 无明确标记的模型：刷新尾部窗口，回退到完整文本解析
            for field, text in splitter.flush_content():
                for tf, tt in thinking_splitter.feed(text):
                    yield sse(tf, tt)
    """

    # 解析状态常量
    _STATE_CONTENT = "content"
    _STATE_BUFFERING = "buffering"

    def __init__(self) -> None:
        # 当前解析状态
        self._state: str = self._STATE_CONTENT
        # 工具调用内容缓冲区（含起始标记及其后所有内容）
        self._buffer: str = ""
        # 滑动窗口尾部：保留最多 _MAX_MARKER_LEN - 1 字节，
        # 防止工具调用起始标记被跨 chunk 切割后漏检。
        # 仅在 CONTENT 状态下使用；进入 BUFFERING 后清空。
        self._tail: str = ""

    @property
    def in_buffering(self) -> bool:
        """是否已进入工具调用缓冲状态（检测到起始标记之后）"""
        return self._state == self._STATE_BUFFERING

    def feed(self, chunk: str) -> List[Tuple[str, str]]:
        """处理一个流式 chunk，返回可实时推送的 (字段名, 文本) 列表。

        字段名固定为 "content"，表示工具调用标记之前的正文内容。
        工具调用标记之后的内容不出现在返回列表中，通过 get_buffer() 获取。

        Args:
            chunk: 模型流式输出的一个文本片段

        Returns:
            [(field_name, text), ...] 列表；
            已进入缓冲模式（in_buffering=True）时始终返回空列表
        """
        if not chunk:
            return []

        # 缓冲模式：所有后续内容直接入缓冲区，不推送给客户端
        if self._state == self._STATE_BUFFERING:
            self._buffer += chunk
            return []

        # CONTENT 模式：将滑动窗口尾部与当前 chunk 合并后检测起始标记
        combined = self._tail + chunk
        marker_pos = _find_first_marker(combined)

        if marker_pos is None:
            # 未检测到标记：输出安全的前部，保留后部作为新的滑动窗口尾部。
            # 安全边界：保留最后 _MAX_MARKER_LEN - 1 字节，
            # 确保跨 chunk 的起始标记不会被截断。
            keep_tail = _MAX_MARKER_LEN - 1
            safe_end = len(combined) - keep_tail
            if safe_end > 0:
                emit_text = combined[:safe_end]
                self._tail = combined[safe_end:]
                return [("content", emit_text)]
            else:
                # 积累量不足，继续等待下一个 chunk
                self._tail = combined
                return []

        # 检测到工具调用起始标记：切换到缓冲状态
        self._state = self._STATE_BUFFERING
        self._tail = ""

        # 标记之前的内容作为正文输出（可能为空）
        content_before = combined[:marker_pos]
        # 标记及其后的所有内容进入工具调用缓冲区
        self._buffer = combined[marker_pos:]

        results: List[Tuple[str, str]] = []
        if content_before:
            results.append(("content", content_before))
        return results

    def flush_content(self) -> List[Tuple[str, str]]:
        """在 CONTENT 状态下刷新滑动窗口尾部的未输出内容。

        应在流正常结束且 in_buffering 为 False 时调用。
        将滑动窗口保留的最后几个字节作为正文内容输出，
        清空内部状态，使对象可重置使用（如需）。

        Returns:
            剩余正文内容列表（可能为空列表）
        """
        if self._state == self._STATE_CONTENT and self._tail:
            text = self._tail
            self._tail = ""
            return [("content", text)]
        return []

    def get_buffer(self) -> str:
        """获取缓冲的工具调用文本（含起始标记）。

        应在流结束后且 in_buffering 为 True 时调用。
        返回的文本可直接传给 CHAT_SERVICE.parse_tool_calls() 解析。

        Returns:
            工具调用文本字符串（从第一个起始标记开始的全部内容）
        """
        return self._buffer

    def has_tool_call_content(self) -> bool:
        """是否已检测到工具调用内容（曾进入过缓冲模式）。

        Returns:
            True 表示检测到工具调用起始标记并缓冲了内容，可调用 get_buffer()
        """
        return bool(self._buffer)

def extract_allowed_tool_names(tools: Optional[List[Dict[str, Any]]]) -> set[str]:
    allowed: set[str] = set()
    if not tools:
        return allowed
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            allowed.add(name)
    return allowed

# 匹配 <tool_call> 标签的起止位置（不依赖内部 JSON 格式）
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*")
_TOOL_CALL_CLOSE = "</tool_call>"

# 匹配 function tag 格式：<function=Name><parameter=key>value</parameter>...</function>
# 用于 Qwen3-Coder 等模型的 chat_template 输出
_FUNCTION_TAG_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)
_PARAMETER_TAG_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)

def _find_json_span(text: str, start: int) -> int | None:
    """从 text[start] 的 '{' 开始，用括号计数找到配对的 '}'。

    正确处理字符串内的转义引号和嵌套大括号。

    Args:
        text: 源文本
        start: '{' 所在的索引位置

    Returns:
        配对 '}' 的索引位置（含），找不到则返回 None
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_str = False
    escape = False
    for j in range(start, len(text)):
        ch = text[j]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return j
    return None

def _normalize_tool_call(raw: dict) -> dict[str, Any]:
    """将模型原始工具调用格式标准化为 OpenAI tool_calls 元素格式

    OpenAI 规范: {"id": "call_xxx", "type": "function",
                  "function": {"name": "...", "arguments": "<json-string>"}}
    """
    name = raw.get("name", "")
    arguments = raw.get("arguments", {})
    # arguments 可能是 dict 或已序列化的 str
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    elif not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }

def _parse_function_tag_calls(
    text: str,
    allowed_tool_names: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """解析 function tag 格式的工具调用

    部分模型（如 Qwen3-Coder）的 chat_template 指示模型输出如下格式：
        <tool_call>
        <function=Bash>
        <parameter=command>ls -la</parameter>
        <parameter=description>List files</parameter>
        </function>
        </tool_call>

    也兼容不带外层 <tool_call> 的情况。

    Returns:
        OpenAI 格式的 tool_calls 列表，空列表表示无工具调用
    """
    tool_calls: list[dict[str, Any]] = []
    allowed_tool_names = allowed_tool_names or set()
    for fn_match in _FUNCTION_TAG_RE.finditer(text):
        func_name = fn_match.group(1)
        if allowed_tool_names and func_name not in allowed_tool_names:
            continue
        inner = fn_match.group(2)
        arguments: dict[str, Any] = {}
        for param_match in _PARAMETER_TAG_RE.finditer(inner):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            # 尝试将值解析为 JSON（支持 dict/list/number/bool）
            try:
                arguments[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                arguments[param_name] = param_value
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    return tool_calls

def _parse_named_json_tool_calls(
    text: str,
    allowed_tool_names: set[str],
) -> list[dict[str, Any]]:
    if not allowed_tool_names:
        return []
    tool_calls: list[dict[str, Any]] = []
    lines = text.splitlines()
    for i, raw_line in enumerate(lines):
        func_name = raw_line.strip()
        if func_name not in allowed_tool_names:
            continue
        remainder = "\n".join(lines[i + 1:]).lstrip()
        if not remainder:
            continue
        if remainder.startswith("```json"):
            remainder = remainder[len("```json"):].lstrip()
        elif remainder.startswith("```"):
            remainder = remainder[len("```"):].lstrip()
        json_start = remainder.find("{")
        if json_start == -1:
            continue
        json_end = _find_json_span(remainder, json_start)
        if json_end is None:
            continue
        try:
            arguments = json.loads(remainder[json_start : json_end + 1])
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(arguments, dict):
            continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    return tool_calls

def _extract_json_objects(text: str) -> list[dict]:
    """从文本中提取所有顶层 JSON 对象（支持嵌套括号）

    使用括号计数而非正则，正确处理任意深度的嵌套 JSON。
    仅返回包含 "name" 和 "arguments" 键的对象（工具调用特征）。
    """
    results: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        end = _find_json_span(text, i)
        if end is None:
            i += 1
            continue
        candidate = text[i : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                results.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
        i = end + 1
    return results

def parse_tool_calls_from_text(
    text: str,
    allowed_tool_names: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """从模型输出文本中提取工具调用

    解析策略（按优先级）：
    1. <tool_call>{...}</tool_call> 标签 + JSON 格式（Qwen3/通用回退模板的标准输出）
    2. <function=Name><parameter=key>value</parameter></function> 格式（Qwen3-Coder 等）
    3. 独立 JSON 对象 {"name": "...", "arguments": {...}}（兜底，支持嵌套参数）

    Returns:
        OpenAI 格式的 tool_calls 列表，空列表表示无工具调用
    """
    tool_calls: list[dict[str, Any]] = []
    allowed_tool_names = allowed_tool_names or set()

    # 策略 1：<tool_call> + JSON 标签（优先，因为边界明确不易误判）
    # 使用 _find_json_span 提取标签内的 JSON，正确处理嵌套参数
    for match in _TOOL_CALL_OPEN_RE.finditer(text):
        json_start = match.end()
        if json_start >= len(text) or text[json_start] != '{':
            continue
        json_end = _find_json_span(text, json_start)
        if json_end is None:
            continue
        # 验证后面紧跟 </tool_call>（允许中间有空白）
        after_json = text[json_end + 1:].lstrip()
        if after_json and not after_json.startswith(_TOOL_CALL_CLOSE):
            continue
        try:
            obj = json.loads(text[json_start : json_end + 1])
            normalized = _normalize_tool_call(obj)
            name = normalized.get("function", {}).get("name", "")
            if allowed_tool_names and name not in allowed_tool_names:
                continue
            tool_calls.append(normalized)
        except (json.JSONDecodeError, KeyError):
            continue

    if tool_calls:
        return tool_calls

    # 策略 2：function tag 格式（Qwen3-Coder 等模型的原生输出）
    tool_calls = _parse_function_tag_calls(text, allowed_tool_names)
    if tool_calls:
        return tool_calls

    tool_calls = _parse_named_json_tool_calls(text, allowed_tool_names)
    if tool_calls:
        return tool_calls

    # 策略 3：从全文提取 JSON 对象（支持任意深度嵌套）
    for obj in _extract_json_objects(text):
        normalized = _normalize_tool_call(obj)
        name = normalized.get("function", {}).get("name", "")
        if allowed_tool_names and name not in allowed_tool_names:
            continue
        tool_calls.append(normalized)

    return tool_calls