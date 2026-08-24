"""LLMClient protocol, plus the Real and Replay implementations (SDD 7).

TWO RULES, both enforced at construction rather than by convention:

  ReplayLLMClient looks up sha256(prompt) in fixtures/llm/. A MISS RAISES. It
  never falls back to the network, because a fixture-backed test that quietly
  makes a real call is a test that passes for money and cannot be run offline
  or reproducibly.

  RealLLMClient.__init__ RAISES if PYTEST_CURRENT_TEST is set. The suite must
  be incapable of billing anyone, and "we only construct it in production" is a
  property of call sites rather than of the type.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = [
    "LLMClient",
    "LLMResponse",
    "RealLLMClient",
    "ReplayLLMClient",
    "prompt_hash",
]

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "llm"


def prompt_hash(prompt: str) -> str:
    """sha256 of the exact prompt text. The fixture key (SDD 7)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LLMResponse:
    """One model reply, plus what it cost. Cost is reported, never hidden."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(Protocol):
    """The seam. Everything above this layer takes a client, never a vendor SDK."""

    def complete(self, prompt: str) -> LLMResponse: ...


class ReplayMissError(RuntimeError):
    """A prompt had no recorded fixture. Loud, and never a network fallback."""


@dataclass
class ReplayLLMClient:
    """Serves recorded responses by prompt hash. A miss raises."""

    fixture_dir: Path = FIXTURE_DIR
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> LLMResponse:
        digest = prompt_hash(prompt)
        self.calls.append(digest)
        path = self.fixture_dir / f"{digest}.json"
        if not path.exists():
            raise ReplayMissError(
                f"no recorded response for prompt hash {digest}. Record it into "
                f"{path}, or fix the prompt if it changed unintentionally. This "
                "does NOT fall back to the network: a test that silently makes a "
                "real call cannot be run offline and is not reproducible."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(
            text=payload["text"],
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            model=str(payload.get("model", "replay")),
        )


class RealLLMClient:
    """The live client. REFUSES to exist inside a test run."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise RuntimeError(
                "RealLLMClient must not be constructed inside a test run "
                "(PYTEST_CURRENT_TEST is set). Use ReplayLLMClient. The suite has "
                "to be incapable of billing anyone, and that is a property of this "
                "type rather than of who remembered to check."
            )
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def complete(self, prompt: str) -> LLMResponse:  # pragma: no cover - needs network
        from anthropic import Anthropic

        client = Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=self.model,
        )
