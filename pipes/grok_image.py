"""
title: Grok Image
description: Image generation with Grok
author: OVINC CN, yuzukumo
git_url: https://github.com/OVINC-CN/OpenWebUIPlugin.git
version: 0.1.2
licence: MIT
"""

import base64
import json
import logging
import mimetypes
import time
import uuid
from typing import AsyncIterable, Literal, Optional, Tuple

import httpx
from fastapi import Request
from httpx import Response
from open_webui.env import GLOBAL_LOG_LEVEL
from open_webui.models.users import UserModel
from open_webui.routers.files import get_file_content_by_id
from open_webui.routers.images import upload_image
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(GLOBAL_LOG_LEVEL)


class APIException(Exception):
    def __init__(self, status: int, content: str, response: Response):
        self._status = status
        self._content = content
        self._response = response

    def __str__(self) -> str:
        # error msg
        try:
            return json.loads(self._content)["error"]["message"]
        except Exception:
            pass
        # build in error
        try:
            self._response.raise_for_status()
        except Exception as err:
            return str(err)
        return "Unknown API error"


class Pipe:
    class Valves(BaseModel):
        base_url: str = Field(default="https://api.x.ai/v1", title="Base URL")
        api_key: str = Field(default="", title="API Key")
        num_of_images: int = Field(default=1, title="图片数量", ge=1, le=10)
        timeout: int = Field(default=600, title="请求超时时间 (秒)")
        proxy: Optional[str] = Field(default="", title="代理地址")
        models: str = Field(
            default="grok-imagine-image,grok-imagine-image-pro",
            title="模型",
            description="使用英文逗号分隔多个模型",
        )

    class UserValves(BaseModel):
        resolution: Literal["1k", "2k"] = Field(default="1k", title="图片分辨率")
        aspect_ratio: Literal[
            "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1", "auto"
        ] = Field(default="auto", title="图片比例")

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def pipes(self):
        return [{"id": model, "name": model} for model in self.valves.models.split(",")]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __metadata__: Optional[dict] = None,
    ) -> StreamingResponse:
        return StreamingResponse(
            self._pipe(body=body, __user__=__user__, __request__=__request__, __metadata__=__metadata__)
        )

    async def _pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __metadata__: Optional[dict] = None,
    ) -> AsyncIterable[str]:
        user = self._get_user(__user__)
        model, payload = await self._build_payload(user=user, body=body, user_valves=__user__.get("valves", {}))
        # call client
        async with httpx.AsyncClient(
            base_url=self.valves.base_url,
            headers={"Authorization": f"Bearer {self.valves.api_key}"},
            proxy=self.valves.proxy or None,
            trust_env=True,
            timeout=self.valves.timeout,
        ) as client:
            response = await client.post(**payload)
            if response.status_code != 200:
                raise APIException(status=response.status_code, content=response.text, response=response)
            response = response.json()
            # upload image
            results = []
            for item in response.get("data", []):
                b64_json = item.get("b64_json")
                image_url = item.get("url")
                if b64_json:
                    results.append(
                        await self._upload_image(
                            __request__=__request__,
                            user=user,
                            image_data=b64_json,
                            mime_type=item.get("mime_type", "image/jpeg"),
                            metadata=__metadata__,
                        )
                    )
                elif image_url:
                    results.append(f"![grok-image-remote]({image_url})")

            if not results and response.get("error"):
                raise ValueError(response["error"].get("message", "Unknown API error"))
            if not results:
                raise ValueError("No image returned by xAI image API")
            # format response data
            usage = response.get("usage", None)
            # response
            content = "\n\n".join(results)
            if body.get("stream"):
                yield self._format_data(is_stream=True, model=model, content=content, usage=None)
                yield self._format_data(is_stream=True, model=model, content=None, usage=usage)
            else:
                yield self._format_data(is_stream=False, model=model, content=content, usage=usage)

    async def _upload_image(
        self,
        __request__: Request,
        user: UserModel,
        image_data: str,
        mime_type: str,
        metadata: Optional[dict],
    ) -> str:
        file_item, image_url = await upload_image(
            request=__request__,
            image_data=base64.b64decode(image_data),
            content_type=mime_type,
            metadata={"mime_type": mime_type, **(metadata or {})},
            user=user,
        )
        return f"![grok-image-{file_item.id}]({image_url})"

    async def _get_image_content(self, user: UserModel, markdown_string: str) -> str:
        file_id = markdown_string.split("![grok-image-")[1].split("]")[0]
        file_response = await get_file_content_by_id(id=file_id, user=user)
        mime_type = mimetypes.guess_type(file_response.path)[0] or "image/png"
        with open(file_response.path, "rb") as file_content:
            encoded = base64.b64encode(file_content.read()).decode()
        return f"data:{mime_type};base64,{encoded}"

    async def _build_payload(self, user: UserModel, body: dict, user_valves: UserValves) -> Tuple[str, dict]:
        # payload
        model = body["model"].split(".", 1)[1]
        user_valves = self._normalize_user_valves(user_valves)
        images = []
        prompt_parts = []

        # read messages
        messages = body.get("messages", [])
        if len(messages) >= 2:
            messages = messages[-2:]
        for message in messages:
            # ignore system message
            if message["role"] == "system":
                continue
            # parse content
            message_content = message["content"]
            # str content
            if isinstance(message_content, str):
                for item in message_content.split("\n"):
                    item = item.strip()
                    if not item:
                        continue
                    if item.startswith("![grok-image-"):
                        image_url = await self._get_image_content(user, item)
                        images.append({"type": "image_url", "url": image_url})
                        continue
                    prompt_parts.append(item)
            # list content
            elif isinstance(message_content, list):
                for content in message_content:
                    if content["type"] == "text":
                        prompt_parts.append(content["text"])
                        continue
                    if content["type"] in {"image_url", "input_image"}:
                        image_url = content["image_url"]["url"]
                        images.append({"type": "image_url", "url": image_url})
            else:
                raise TypeError("message content invalid")

        prompt = "\n".join(prompt_parts).strip()
        if not prompt:
            prompt = body.get("prompt", "")

        if len(images) > 5:
            raise ValueError("xAI image editing supports up to 5 input images.")

        # init payload
        payload = {
            "url": "/images/generations",
            "json": {
                "model": model,
                "prompt": prompt,
                "resolution": user_valves.resolution,
                "response_format": "b64_json",
            },
        }

        if not images:
            payload["json"]["n"] = self.valves.num_of_images
            payload["json"]["aspect_ratio"] = user_valves.aspect_ratio
        elif len(images) == 1:
            payload["json"]["image"] = images[0]
            payload["url"] = "/images/edits"
        else:
            payload["json"]["images"] = images
            payload["json"]["aspect_ratio"] = user_valves.aspect_ratio
            payload["url"] = "/images/edits"

        return model, payload

    @staticmethod
    def _normalize_user_valves(user_valves: UserValves | dict) -> UserValves:
        if isinstance(user_valves, Pipe.UserValves):
            return user_valves
        if isinstance(user_valves, BaseModel):
            if hasattr(user_valves, "model_dump"):
                return Pipe.UserValves(**user_valves.model_dump())
            return Pipe.UserValves(**user_valves.dict())
        return Pipe.UserValves(**(user_valves or {}))

    @staticmethod
    def _get_user(user_data: UserModel | BaseModel | dict) -> UserModel:
        if isinstance(user_data, UserModel):
            return user_data
        if isinstance(user_data, BaseModel):
            if hasattr(user_data, "model_dump"):
                user_data = user_data.model_dump()
            else:
                user_data = user_data.dict()
        if isinstance(user_data, dict):
            return UserModel(**{k: v for k, v in user_data.items() if k != "valves"})
        raise ValueError("user not found")

    def _format_data(
        self,
        is_stream: bool,
        model: Optional[str] = "",
        content: Optional[str] = "",
        usage: Optional[dict] = None,
    ) -> str:
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "choices": [],
            "created": int(time.time()),
            "model": model,
        }
        if content:
            data["choices"] = [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "delta" if is_stream else "message": {
                        "content": content,
                    },
                }
            ]
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data)}\n\n"
