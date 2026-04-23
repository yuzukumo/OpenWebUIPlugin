"""
title: OpenAI Responses
author: OVINC CN
git_url: https://github.com/OVINC-CN/OpenWebUIPlugin.git
version: 0.1.7
licence: MIT
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncIterable, Literal, Optional, Tuple

import httpx
from fastapi import Request
from httpx import Response
from open_webui.env import GLOBAL_LOG_LEVEL
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(GLOBAL_LOG_LEVEL)


class APIException(Exception):
    def __init__(
        self,
        status: int,
        content: str,
        response: Response,
        request_id: str = "",
    ):
        self._status = status
        self._content = content
        self._response = response
        self._request_id = request_id

    def __str__(self) -> str:
        message = ""
        try:
            message = json.loads(self._content)["error"]["message"]
        except Exception:
            pass

        if not message:
            try:
                self._response.raise_for_status()
            except Exception as err:
                message = str(err)

        if not message:
            message = "Unknown API error"

        if self._request_id:
            return f"{message} (request_id: {self._request_id})"
        return message


class Pipe:
    class Valves(BaseModel):
        base_url: str = Field(default="https://api.openai.com/v1", title="Base URL")
        api_key: str = Field(default="", title="API Key")
        enable_reasoning: bool = Field(default=True, title="展示思考内容")
        allow_params: Optional[str] = Field(
            default="",
            title="透传参数",
            description="允许透传的参数，英文逗号分隔，例如 temperature,top_p",
        )
        timeout: int = Field(default=600, title="请求超时时间（秒）")
        proxy: Optional[str] = Field(default="", title="代理地址")
        models: str = Field(
            default="gpt-5.1,gpt-5",
            title="模型",
            description="使用英文逗号分隔多个模型",
        )

        enable_web_search: bool = Field(default=True, title="启用 OpenAI Web Search")
        web_search_context_size: Literal["low", "medium", "high"] = Field(
            default="medium",
            title="Web Search 上下文大小",
        )
        web_search_domains: Optional[str] = Field(
            default="",
            title="Web Search 域名白名单",
            description="英文逗号分隔，例如 openai.com,platform.openai.com",
        )
        web_search_country: Optional[str] = Field(
            default="",
            title="搜索国家",
            description="两位国家代码，例如 US、CN",
        )
        web_search_city: Optional[str] = Field(default="", title="搜索城市")
        web_search_region: Optional[str] = Field(default="", title="搜索地区/省州")
        web_search_timezone: Optional[str] = Field(
            default="",
            title="搜索时区",
            description="IANA 时区，例如 Asia/Shanghai",
        )
        append_sources_to_answer: bool = Field(
            default=True,
            title="在答案末尾追加来源链接",
        )

    class UserValves(BaseModel):
        verbosity: Literal["low", "medium", "high"] = Field(
            default="medium",
            title="输出详细程度",
        )
        reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = Field(
            default="low",
            title="思考推理强度",
        )
        summary: Literal["auto", "concise", "detailed"] = Field(
            default="auto",
            title="思考输出摘要程度",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [
            {"id": model.strip(), "name": model.strip()}
            for model in self.valves.models.split(",")
            if model.strip()
        ]

    async def pipe(
        self, body: dict, __user__: dict, __request__: Request
    ) -> StreamingResponse:
        return StreamingResponse(
            self.__stream_pipe(body=body, __user__=__user__, __request__=__request__),
            media_type="text/event-stream",
        )

    async def __stream_pipe(
        self, body: dict, __user__: dict, __request__: Request
    ) -> AsyncIterable[str]:
        user_valves = self._coerce_user_valves((__user__ or {}).get("valves"))
        model, payload = await self._build_payload(body=body, user_valves=user_valves)

        async with httpx.AsyncClient(
            base_url=self.valves.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {self.valves.api_key}"},
            proxy=self.valves.proxy or None,
            trust_env=True,
            timeout=self.valves.timeout,
        ) as client:
            async with client.stream(**payload) as response:
                request_id = response.headers.get("x-request-id", "")

                if response.status_code != 200:
                    text = await self._read_error_text(response)
                    logger.error(
                        "response invalid with %d request_id=%s body=%s",
                        response.status_code,
                        request_id,
                        text,
                    )
                    raise APIException(
                        status=response.status_code,
                        content=text,
                        response=response,
                        request_id=request_id,
                    )

                is_thinking = self.valves.enable_reasoning
                usage = None
                annotations = []
                response_sources = []

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("event:") or not line.startswith("data:"):
                        continue

                    line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("ignore invalid stream line: %s", line)
                        continue

                    event_type = event.get("type", "")

                    if event_type == "response.reasoning_summary_text.delta":
                        if is_thinking and event.get("delta"):
                            yield self._format_stream_data(
                                model=model,
                                reasoning_content=event["delta"],
                            )
                        continue

                    if event_type == "response.output_text.delta":
                        if is_thinking:
                            is_thinking = False
                        if event.get("delta"):
                            yield self._format_stream_data(
                                model=model,
                                content=event["delta"],
                            )
                        continue

                    if event_type == "response.output_text.annotation.added":
                        annotation = event.get("annotation", {})
                        if annotation.get("type") == "url_citation":
                            annotations.append(annotation)
                        continue

                    if event_type in {
                        "response.web_search_call.searching",
                        "response.web_search_call.in_progress",
                    }:
                        yield self._format_status_data(
                            description="web search",
                            done=False,
                        )
                        continue

                    if event_type == "response.web_search_call.completed":
                        yield self._format_status_data(
                            description="web search completed",
                            done=True,
                        )
                        continue

                    if event_type == "response.completed":
                        response_obj = event.get("response", {})
                        usage = response_obj.get("usage")
                        response_sources = self._extract_sources_from_response(
                            response_obj
                        )
                        continue

                    if event_type == "response.incomplete":
                        reason = (
                            event.get("response", {})
                            .get("incomplete_details", {})
                            .get("reason", "unknown")
                        )
                        raise RuntimeError(
                            self._format_upstream_error(
                                f"OpenAI response incomplete: {reason}",
                                request_id=request_id,
                            )
                        )

                    if event_type == "response.failed":
                        err = event.get("response", {}).get("error") or event.get(
                            "error", {}
                        )
                        message = err.get(
                            "message",
                            "An error occurred while processing your request.",
                        )
                        code = err.get("code", "")
                        raise RuntimeError(
                            self._format_upstream_error(
                                message=message,
                                request_id=request_id,
                                code=code,
                            )
                        )

                if self.valves.append_sources_to_answer:
                    source_text = self._render_sources(annotations, response_sources)
                    if source_text:
                        yield self._format_stream_data(model=model, content=source_text)

                yield self._format_stream_data(
                    model=model,
                    usage=usage,
                    if_finished=True,
                )

    async def _build_payload(
        self, body: dict, user_valves: "Pipe.UserValves", stream: bool = True
    ) -> Tuple[str, dict]:
        model = self._extract_model_name(body.get("model", ""))
        messages = self._convert_messages(body.get("messages", []))
        tools = self._build_tools()

        incoming_tools = body.get("tools") or []
        incoming_tool_choice = body.get("tool_choice")

        if incoming_tools:
            logger.info(
                "ignoring incoming OpenWebUI tools, using OpenAI web_search only: %s",
                self._summarize_tools(incoming_tools),
            )

        if incoming_tool_choice not in (None, "", "auto"):
            logger.info(
                "ignoring incoming OpenWebUI tool_choice, forcing auto: %s",
                incoming_tool_choice,
            )

        data: dict[str, Any] = {
            "model": model,
            "input": messages,
            "stream": stream,
            "store": False,
        }

        if self._is_gpt5_family(model):
            data["text"] = {"verbosity": user_valves.verbosity}
            reasoning_effort = self._normalize_reasoning_effort(
                model=model,
                effort=user_valves.reasoning_effort,
                using_web_search=bool(tools),
            )
            data["reasoning"] = {"effort": reasoning_effort}
            if self.valves.enable_reasoning:
                data["reasoning"]["summary"] = user_valves.summary

        if "max_completion_tokens" in body:
            data["max_output_tokens"] = body["max_completion_tokens"]
        elif "max_tokens" in body:
            data["max_output_tokens"] = body["max_tokens"]

        allowed_params = [
            key.strip() for key in self.valves.allow_params.split(",") if key.strip()
        ]
        reserved = {
            "model",
            "input",
            "messages",
            "tools",
            "tool_choice",
            "stream",
            "store",
            "text",
            "reasoning",
            "include",
            "max_completion_tokens",
            "max_tokens",
        }
        for key, val in body.items():
            if key in allowed_params and key not in reserved:
                data[key] = val

        if tools:
            data["tools"] = tools
            data["tool_choice"] = "auto"
            data["include"] = ["web_search_call.action.sources"]

        payload = {"method": "POST", "url": "responses", "json": data}
        logger.debug("responses payload=%s", json.dumps(data, ensure_ascii=False))
        return model, payload

    def _extract_model_name(self, raw_model: str) -> str:
        raw_model = (raw_model or "").strip()
        configured = [
            model.strip() for model in self.valves.models.split(",") if model.strip()
        ]

        if raw_model in configured:
            return raw_model

        for model in sorted(configured, key=len, reverse=True):
            if raw_model.endswith(f".{model}"):
                return model

        return raw_model

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        converted = []

        for message in messages:
            role = (message.get("role") or "user").strip()
            if role not in {"user", "assistant", "system", "developer"}:
                role = "user"

            content_value = message.get("content", "")

            if isinstance(content_value, str):
                converted.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": content_value,
                    }
                )
                continue

            if isinstance(content_value, list):
                if role == "assistant":
                    content = self._convert_assistant_content(content_value)
                else:
                    content = self._convert_input_content(content_value)

                if not content:
                    content = ""

                converted.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": content,
                    }
                )
                continue

            raise TypeError("Invalid message content type %s" % type(content_value))

        return converted

    def _convert_input_content(self, items: list[dict]) -> list[dict]:
        content = []

        for item in items:
            item_type = item.get("type")

            if item_type in {"text", "input_text"}:
                content.append({"type": "input_text", "text": item.get("text", "")})
            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict):
                    image_url = image_url.get("url", "")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    }
                )
            elif item_type == "input_image":
                content.append(item)
            elif item_type == "input_file":
                content.append(item)
            elif item_type == "output_text":
                content.append({"type": "input_text", "text": item.get("text", "")})
            elif item_type == "refusal":
                content.append(
                    {"type": "input_text", "text": item.get("refusal", "")}
                )
            elif item_type == "reasoning_content":
                text = item.get("text") or item.get("reasoning_content") or ""
                if text:
                    content.append({"type": "input_text", "text": text})
            else:
                text = item.get("text")
                if isinstance(text, str):
                    content.append({"type": "input_text", "text": text})
                else:
                    raise TypeError("Invalid message content type %s" % item_type)

        return content

    def _convert_assistant_content(self, items: list[dict]) -> list[dict]:
        content = []

        for item in items:
            item_type = item.get("type")

            if item_type in {"text", "output_text", "input_text"}:
                content.append({"type": "output_text", "text": item.get("text", "")})
            elif item_type == "refusal":
                content.append(
                    {
                        "type": "refusal",
                        "refusal": item.get("refusal", ""),
                    }
                )
            elif item_type == "reasoning_content":
                continue
            else:
                text = item.get("text")
                if isinstance(text, str):
                    content.append({"type": "output_text", "text": text})

        return content

    def _coerce_user_valves(self, raw: Any) -> "Pipe.UserValves":
        if isinstance(raw, self.UserValves):
            return raw

        if raw is None:
            return self.UserValves()

        if isinstance(raw, dict):
            return self.UserValves(**raw)

        if isinstance(raw, BaseModel):
            if hasattr(raw, "model_dump"):
                return self.UserValves(**raw.model_dump())
            return self.UserValves(**raw.dict())

        fields = getattr(self.UserValves, "model_fields", None) or getattr(
            self.UserValves, "__fields__", {}
        )
        data = {name: getattr(raw, name) for name in fields if hasattr(raw, name)}
        return self.UserValves(**data)

    def _build_tools(self) -> list[dict]:
        if not self.valves.enable_web_search:
            return []

        tool = {
            "type": "web_search",
            "search_context_size": self.valves.web_search_context_size,
        }

        domains = self._parse_domains(self.valves.web_search_domains)
        if domains:
            tool["filters"] = {"allowed_domains": domains}

        user_location = self._build_user_location()
        if user_location:
            tool["user_location"] = user_location

        return [tool]

    def _build_user_location(self) -> Optional[dict]:
        values = {
            "type": "approximate",
            "country": (self.valves.web_search_country or "").strip().upper(),
            "city": (self.valves.web_search_city or "").strip(),
            "region": (self.valves.web_search_region or "").strip(),
            "timezone": (self.valves.web_search_timezone or "").strip(),
        }
        if not any(
            [
                values["country"],
                values["city"],
                values["region"],
                values["timezone"],
            ]
        ):
            return None
        return {k: v for k, v in values.items() if v}

    def _parse_domains(self, raw_domains: Optional[str]) -> list[str]:
        domains = []
        for domain in (raw_domains or "").split(","):
            domain = domain.strip().lower()
            if not domain:
                continue
            if domain.startswith("http://"):
                domain = domain[7:]
            elif domain.startswith("https://"):
                domain = domain[8:]
            domain = domain.rstrip("/")
            if domain:
                domains.append(domain)
        return domains

    def _summarize_tools(self, tools: list[dict]) -> list[str]:
        summary = []
        for tool in tools:
            if not isinstance(tool, dict):
                summary.append(str(type(tool)))
                continue

            tool_type = tool.get("type", "")
            if tool_type == "function":
                fn = tool.get("function", {}) or {}
                summary.append(f"function:{fn.get('name', 'unknown')}")
            else:
                summary.append(tool_type or "unknown")
        return summary

    def _is_gpt5_family(self, model: str) -> bool:
        model = (model or "").lower()
        return model.startswith("gpt-5")

    def _normalize_reasoning_effort(
        self, model: str, effort: str, using_web_search: bool
    ) -> str:
        model = (model or "").lower()

        if model.startswith("gpt-5.1") or model.startswith("gpt-5.2"):
            if effort == "xhigh":
                return "high"
            if effort in {"none", "low", "medium", "high"}:
                return effort
            return "none"

        if model.startswith("gpt-5"):
            if effort == "none":
                return "low" if using_web_search else "low"
            if effort == "xhigh":
                return "high"
            if effort in {"low", "medium", "high"}:
                return effort
            return "low"

        return effort

    def _extract_sources_from_response(self, response: dict) -> list[dict]:
        sources = []

        for item in response.get("output", []):
            if item.get("type") == "web_search_call":
                action = item.get("action", {}) or {}
                for source in action.get("sources", []) or []:
                    if source.get("url"):
                        sources.append(
                            {
                                "url": source["url"],
                                "title": source.get("title", ""),
                            }
                        )

            if item.get("type") == "message":
                for content in item.get("content", []):
                    for annotation in content.get("annotations", []) or []:
                        if annotation.get("type") == "url_citation" and annotation.get(
                            "url"
                        ):
                            sources.append(
                                {
                                    "url": annotation["url"],
                                    "title": annotation.get("title", ""),
                                }
                            )

        return sources

    def _render_sources(self, annotations: list[dict], sources: list[dict]) -> str:
        merged = []

        for annotation in annotations:
            if annotation.get("url"):
                merged.append(
                    {
                        "url": annotation["url"],
                        "title": annotation.get("title", ""),
                    }
                )

        merged.extend(sources)

        deduped = []
        seen = set()
        for item in merged:
            url = item.get("url", "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            deduped.append(
                {
                    "url": url,
                    "title": item.get("title", "").strip(),
                }
            )

        if not deduped:
            return ""

        lines = ["", "", "Sources:"]
        for idx, item in enumerate(deduped, start=1):
            title = item["title"] or item["url"]
            lines.append(f"{idx}. {title} - {item['url']}")
        return "\n".join(lines)

    async def _read_error_text(self, response: Response) -> str:
        text = ""
        async for line in response.aiter_lines():
            text += line
        return text

    def _format_upstream_error(
        self, message: str, request_id: str = "", code: str = ""
    ) -> str:
        parts = [message]
        if code:
            parts.append(f"code={code}")
        if request_id:
            parts.append(f"request_id={request_id}")
        return " | ".join(parts)

    def _format_status_data(self, description: str, done: bool) -> str:
        data = {
            "event": {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                },
            }
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _format_stream_data(
        self,
        model: Optional[str] = "",
        content: Optional[str] = "",
        reasoning_content: Optional[str] = "",
        usage: Optional[dict] = None,
        if_finished: bool = False,
    ) -> str:
        delta = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content

        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "finish_reason": "stop" if if_finished else "",
                    "index": 0,
                    "delta": delta,
                }
            ],
            "created": int(time.time()),
            "model": model,
        }

        if usage:
            data["usage"] = usage

        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
