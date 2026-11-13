"""LLM provider adapter.

The LLM extractor sends the parsed blocks of one document and asks for a JSON "labeling"
that references block ids and table rows (see `extract.LABELING_SCHEMA`). Records and
their page/bbox provenance are then rebuilt locally from those references, so a model can
never invent coordinates.

Providers:
- ``offline``: deterministic, no network. It decodes the payload embedded in the prompt and
  answers with the rule-based labeler. Used by tests and CI; it exercises the full prompt ->
  JSON -> validation -> record path.
- ``anthropic``: real Claude call; only usable when ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

PAYLOAD_START = "<<<PAYLOAD"
PAYLOAD_END = "PAYLOAD>>>"


class Provider(Protocol):
    name: str

    def complete_json(self, system: str, user: str) -> dict: ...


def embed_payload(payload: dict) -> str:
    return f"{PAYLOAD_START}\n{json.dumps(payload, ensure_ascii=False)}\n{PAYLOAD_END}"


def decode_payload(text: str) -> dict:
    m = re.search(re.escape(PAYLOAD_START) + r"\n(.*)\n" + re.escape(PAYLOAD_END), text, re.S)
    if not m:
        raise ValueError("prompt does not contain an embedded payload")
    return json.loads(m.group(1))


def parse_json_reply(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    return json.loads(raw)


class OfflineProvider:
    name = "offline"

    def complete_json(self, system: str, user: str) -> dict:
        from .extract import label_payload

        return label_payload(decode_payload(user))


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set; use --provider offline")
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("MANUALTRACE_ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def complete_json(self, system: str, user: str) -> dict:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=32000,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return parse_json_reply(text)


def get_provider(name: str) -> Provider:
    if name == "offline":
        return OfflineProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"unknown provider {name!r}")
