import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def load_module():
    open_webui = types.ModuleType("open_webui")
    env = types.ModuleType("open_webui.env")
    env.GLOBAL_LOG_LEVEL = "INFO"
    open_webui.env = env
    sys.modules.setdefault("open_webui", open_webui)
    sys.modules["open_webui.env"] = env

    path = Path(__file__).resolve().parents[1] / "pipes" / "openai_responses.py"
    spec = importlib.util.spec_from_file_location("openai_responses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_module()


def test_extract_model_name_matches_suffix():
    pipe = mod.Pipe()
    pipe.valves.models = "gpt-5.1,gpt-5"
    assert pipe._extract_model_name("openai.gpt-5") == "gpt-5"
    assert pipe._extract_model_name("gpt-5.1") == "gpt-5.1"


def test_convert_messages():
    pipe = mod.Pipe()
    out = pipe._convert_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "https://img.test/a.png"}},
                ],
            }
        ]
    )
    assert out == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hi"},
                {"type": "input_image", "image_url": "https://img.test/a.png"},
            ],
        }
    ]


def test_build_payload_adds_web_search_and_allowed_params():
    pipe = mod.Pipe()
    pipe.valves.allow_params = "temperature"
    pipe.valves.web_search_domains = "https://openai.com/,platform.openai.com"
    pipe.valves.web_search_country = "us"

    model, payload = asyncio.run(
        pipe._build_payload(
            {
                "model": "prefix.gpt-5",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.2,
                "max_tokens": 64,
            },
            pipe.UserValves(
                verbosity="low",
                reasoning_effort="none",
                summary="concise",
            ),
        )
    )

    data = payload["json"]
    assert model == "gpt-5"
    assert data["max_output_tokens"] == 64
    assert data["temperature"] == 0.2
    assert data["reasoning"] == {"effort": "low", "summary": "concise"}
    assert data["tools"][0]["filters"]["allowed_domains"] == [
        "openai.com",
        "platform.openai.com",
    ]
    assert data["tools"][0]["user_location"]["country"] == "US"


def test_render_sources_deduplicates():
    pipe = mod.Pipe()
    text = pipe._render_sources(
        [{"url": "https://a.com", "title": "A"}],
        [
            {"url": "https://a.com", "title": "A again"},
            {"url": "https://b.com", "title": ""},
        ],
    )
    assert "Sources:" in text
    assert text.count("https://a.com") == 1
    assert "https://b.com - https://b.com" in text
