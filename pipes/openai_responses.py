"""
title: OpenAI Responses Minimal
author: OVINC CN
version: 0.1.0
licence: MIT
"""

import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import AsyncIterable, Optional

import httpx
from fastapi import Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse


class Pipe:
    class Valves(BaseModel):
        base_url: str = Field(default="https://api.openai.com/v1", title="Base URL")
        api_key: str = Field(default="", title="API Key")
        models: str = Field(default="gpt-5.5", title="模型")
        timeout: int = Field(default=600, title="请求超时时间（秒）")
        enable_web_search: bool = Field(default=True, title="启用 OpenAI Web Search")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [
            {"id": model.strip(), "name": model.strip()}
            for model in self.valves.models.split(",")
            if model.strip()
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: Optional[list[dict]] = None,
    ) -> StreamingResponse:
        return StreamingResponse(
            self._stream(body, __files__ or []),
            media_type="text/event-stream",
        )

    async def _stream(self, body: dict, injected_files: list[dict]) -> AsyncIterable[str]:
        api_key = (self.valves.api_key or "").strip()
        if not api_key:
            raise RuntimeError("OpenAI API Key is empty. Please set the API Key valve.")

        model = self._extract_model_name(body.get("model", ""))
        body_files = list(body.get("files") or [])
        body_files.extend(injected_files)
        data = {
            "model": model,
            "input": self._convert_messages(
                body.get("messages", []),
                body_files=body_files,
            ),
            "stream": True,
        }

        if model.lower().startswith("gpt-5"):
            data["reasoning"] = {"effort": "high", "summary": "auto"}

        if self.valves.enable_web_search:
            data["tools"] = [{"type": "web_search"}]
            data["tool_choice"] = "auto"

        async with httpx.AsyncClient(
            base_url=self.valves.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.valves.timeout,
        ) as client:
            async with client.stream("POST", "responses", json=data) as response:
                if response.status_code != 200:
                    error = await response.aread()
                    raise RuntimeError(error.decode("utf-8", errors="replace"))

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue

                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue

                    event = json.loads(raw)
                    event_type = event.get("type")

                    if event_type == "response.reasoning_summary_text.delta":
                        delta = event.get("delta")
                        if delta:
                            yield self._chunk(model, {"reasoning_content": delta})

                    elif event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if delta:
                            yield self._chunk(model, {"content": delta})

                    elif event_type in {
                        "response.web_search_call.searching",
                        "response.web_search_call.in_progress",
                    }:
                        yield self._status("web search", done=False)

                    elif event_type == "response.web_search_call.completed":
                        yield self._status("web search completed", done=True)

                    elif event_type == "response.failed":
                        error = event.get("response", {}).get("error") or {}
                        raise RuntimeError(error.get("message", "OpenAI response failed"))

                    elif event_type == "response.incomplete":
                        details = event.get("response", {}).get(
                            "incomplete_details", {}
                        )
                        reason = details.get("reason", "unknown")
                        raise RuntimeError(f"OpenAI response incomplete: {reason}")

        yield self._chunk(model, {}, finished=True)

    def _extract_model_name(self, raw_model: str) -> str:
        raw_model = (raw_model or "").strip()
        models = [m.strip() for m in self.valves.models.split(",") if m.strip()]

        if raw_model in models:
            return raw_model

        for model in sorted(models, key=len, reverse=True):
            if raw_model.endswith(f".{model}"):
                return model

        return raw_model or models[0]

    def _convert_messages(
        self, messages: list[dict], body_files: Optional[list[dict]] = None
    ) -> list[dict]:
        converted = []
        body_files = body_files or []
        last_user_index = self._last_user_message_index(messages)

        for index, message in enumerate(messages):
            role = message.get("role") or "user"
            if role not in {"user", "assistant", "system", "developer"}:
                role = "user"

            files = list(message.get("files") or [])
            if body_files and index == last_user_index:
                files.extend(body_files)

            content = self._convert_content(
                message.get("content", ""),
                role=role,
                files=files,
            )

            converted.append(
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                }
            )

        return converted

    def _last_user_message_index(self, messages: list[dict]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index
        return len(messages) - 1

    def _convert_content(self, content: object, role: str, files: list[dict]) -> object:
        if isinstance(content, str) and not files:
            return content

        parts = []

        if isinstance(content, str):
            if content:
                parts.append(self._text_part(content, role))
        elif isinstance(content, list):
            for item in content:
                part = self._convert_content_part(item, role)
                if part:
                    parts.append(part)
        else:
            raise TypeError("Invalid message content type %s" % type(content))

        for file in files:
            part = self._convert_file_part(file)
            if part:
                parts.append(part)

        return parts or ""

    def _convert_content_part(self, item: object, role: str) -> Optional[dict]:
        if not isinstance(item, dict):
            return None

        item_type = item.get("type")

        if item_type in {"text", "input_text", "output_text"}:
            text = item.get("text", "")
            if isinstance(text, str):
                return self._text_part(text, role)

        if item_type in {"image_url", "input_image"}:
            image_url = self._url_string(item.get("image_url") or item.get("url"))

            part = {"type": "input_image"}
            if image_url:
                part["image_url"] = image_url
            if item.get("file_id"):
                part["file_id"] = item["file_id"]
            if item.get("detail"):
                part["detail"] = item["detail"]
            return part if len(part) > 1 else None

        if item_type in {"file", "input_file"}:
            return self._convert_file_part(item)

        if item_type == "refusal":
            refusal = item.get("refusal", "")
            if isinstance(refusal, str):
                return self._text_part(refusal, role)

        text = item.get("text")
        if isinstance(text, str):
            return self._text_part(text, role)

        return self._convert_file_part(item)

    def _convert_file_part(self, item: dict) -> Optional[dict]:
        file = item.get("file") if isinstance(item.get("file"), dict) else item
        if self._is_image_file(file):
            return self._convert_image_file_part(file)

        part = {"type": "input_file"}

        file_id = self._openai_file_id(file)
        filename = file.get("filename") or file.get("name")
        file_url = self._url_string(file.get("file_url") or file.get("url"))
        file_data = self._data_url(
            file.get("file_data") or file.get("data"),
            filename=filename,
            mime_type=self._mime_type(file, filename),
        )

        if file_id:
            part["file_id"] = file_id
        if file_url:
            part["file_url"] = file_url
        if file_data:
            part["file_data"] = file_data
        if filename:
            part["filename"] = filename

        if file_id or file_url or file_data:
            return part
        return None

    def _convert_image_file_part(self, file: dict) -> Optional[dict]:
        part = {"type": "input_image"}

        file_id = self._openai_file_id(file)
        image_url = self._url_string(file.get("image_url") or file.get("url"))
        image_data = self._data_url(
            file.get("file_data") or file.get("data"),
            filename=file.get("filename") or file.get("name"),
            mime_type=self._mime_type(file, file.get("filename") or file.get("name")),
            default_mime_type="image/png",
        )

        if file_id:
            part["file_id"] = file_id
        if image_url:
            part["image_url"] = image_url
        elif image_data:
            part["image_url"] = image_data

        if file.get("detail"):
            part["detail"] = file["detail"]

        if file_id or image_url or image_data:
            return part
        return None

    def _is_image_file(self, file: dict) -> bool:
        file_type = str(
            file.get("mime_type")
            or file.get("mime")
            or file.get("content_type")
            or file.get("type")
            or ""
        ).lower()
        filename = str(file.get("filename") or file.get("name") or "").lower()
        url = str(self._url_string(file.get("image_url") or file.get("url")) or "").lower()
        data = self._extract_data_string(file.get("file_data") or file.get("data"))
        data_prefix = (data or "")[:32].lower()

        return (
            file_type.startswith("image/")
            or file_type == "image"
            or filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))
            or Path(url).suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
            or data_prefix.startswith("data:image/")
        )

    def _openai_file_id(self, file: dict) -> Optional[str]:
        file_id = file.get("file_id")
        if isinstance(file_id, str) and file_id:
            return file_id

        raw_id = file.get("id")
        if isinstance(raw_id, str) and raw_id.startswith("file-"):
            return raw_id

        return None

    def _data_url(
        self,
        raw: object,
        filename: object = None,
        mime_type: Optional[str] = None,
        default_mime_type: str = "application/octet-stream",
    ) -> Optional[str]:
        data = self._extract_data_string(raw)
        if not data:
            return None

        data = data.strip()
        if data.startswith("data:"):
            return data

        if data.startswith(("http://", "https://")):
            return None

        mime_type = mime_type or self._mime_type({}, filename) or default_mime_type
        return f"data:{mime_type};base64,{data}"

    def _url_string(self, raw: object) -> Optional[str]:
        if isinstance(raw, str):
            return raw

        if isinstance(raw, dict):
            for key in ("url", "href", "file_url", "image_url"):
                value = raw.get(key)
                if isinstance(value, str):
                    return value

        return None

    def _extract_data_string(self, raw: object) -> Optional[str]:
        if isinstance(raw, str):
            return raw

        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")

        if isinstance(raw, dict):
            for key in (
                "data",
                "content",
                "base64",
                "base64_data",
                "file_data",
                "body",
                "value",
            ):
                value = raw.get(key)
                extracted = self._extract_data_string(value)
                if extracted:
                    return extracted

        return None

    def _mime_type(self, file: dict, filename: object = None) -> Optional[str]:
        explicit = (
            file.get("mime_type")
            or file.get("mime")
            or file.get("content_type")
            or file.get("type")
        )
        if isinstance(explicit, str) and "/" in explicit:
            return explicit

        if isinstance(filename, str):
            guessed, _ = mimetypes.guess_type(filename)
            if guessed:
                return guessed

        return None

    def _text_part(self, text: str, role: str) -> dict:
        if role == "assistant":
            return {"type": "output_text", "text": text}
        return {"type": "input_text", "text": text}

    def _chunk(self, model: str, delta: dict, finished: bool = False) -> str:
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop" if finished else None,
                    "delta": delta,
                }
            ],
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _status(self, description: str, done: bool) -> str:
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
