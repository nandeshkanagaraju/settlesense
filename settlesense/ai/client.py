"""LLMClient protocol, plus the Real and Replay implementations (SDD 7).

TWO RULES, both enforced at construction rather than by convention:

  ReplayLLMClient looks up sha256(prompt) in fixtures/llm/. A MISS RAISES,
  naming the hash. It never falls back to the network, because a fixture-backed
  test that quietly makes a real call is a test that passes for money and
  cannot be run offline or reproducibly.

  RealLLMClient.__init__ RAISES if PYTEST_CURRENT_TEST is set (D7). The suite
  must be incapable of billing anyone, and "we only construct it in production"
  is a property of call sites rather than of the type.

DETERMINISM SETTINGS ARE NOT OPTIONAL. temperature=0, top_p=1, and a pinned
model string. A sampled model makes the same prompt produce different
hypotheses on two runs, so a golden comparison would fail for reasons nobody
could reproduce - and the whole project rests on same-input-same-output. The
model string is pinned rather than aliased ("latest") for the same reason: an
alias silently repoints and the run stops being reproducible without a single
line changing.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "FIXTURE_DIR",
    "MODEL",
    "FixtureMissError",
    "LLMClient",
    "RealLLMClient",
    "ReplayLLMClient",
    "prompt_hash",
    "record_fixture",
]

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "llm"

MODEL = "claude-sonnet-5"
"""The model string, recorded in every fixture so a run is attributable.

NEVER a "-latest" style alias: an alias silently repoints and the run stops
being reproducible without a line of this repo changing - the same failure as
a floating dependency, in the one place the project can least afford it. If a
dated variant of this model is published, pin that instead; the fixture set
would then need re-recording, which is the point.
"""

TEMPERATURE = 0
TOP_P = 1
MAX_TOKENS = 1024


def prompt_hash(prompt: str) -> str:
    """sha256 of the exact prompt text. The fixture key (SDD 7).

    Of the PROMPT, not of the exception: two exceptions that produce the same
    prompt should share a recording, and a prompt that changed by one character
    must miss rather than serve a stale answer to a question nobody asked.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class FixtureMissError(RuntimeError):
    """No recorded response for this prompt. Loud, and never a network fallback."""


class LLMClient(Protocol):
    """The seam. Everything above this layer takes a client, never a vendor SDK."""

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class ReplayLLMClient:
    """Serves recorded responses by prompt hash. A miss raises.

    `calls` is a list rather than a counter so a test can assert WHICH prompts
    were sent, not merely how many - "zero model calls" and "the right zero
    model calls" are different claims.
    """

    fixture_dir: Path = FIXTURE_DIR
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        del schema  # the recording already conforms; validation is the caller's
        digest = prompt_hash(prompt)
        self.calls.append(digest)
        path = self.fixture_dir / f"{digest}.json"
        if not path.exists():
            raise FixtureMissError(
                f"no recorded response for prompt hash {digest}. Record it into "
                f"{path}, or fix the prompt if it changed unintentionally. This "
                "does NOT fall back to the network: a test that silently makes a "
                "real call cannot be run offline and is not reproducible."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = payload.get("response")
        if not isinstance(response, dict):
            raise FixtureMissError(
                f"fixture {path.name} has no `response` object - it is corrupt, "
                "which is a different problem from being absent and needs a "
                "different fix (re-record, do not re-run)."
            )
        return response


def record_fixture(prompt: str, response: dict[str, Any], fixture_dir: Path = FIXTURE_DIR) -> Path:
    """Write one recording. Used by the fixture-building script, never by tests.

    The prompt is stored ALONGSIDE the response even though the filename is its
    hash. A hash is not readable, and a fixture set nobody can inspect is a
    fixture set nobody will notice has gone stale.
    """
    fixture_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_dir / f"{prompt_hash(prompt)}.json"
    path.write_text(
        json.dumps({"prompt": prompt, "response": response}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


class RealLLMClient:
    """The live client. REFUSES to exist inside a test run (D7)."""

    def __init__(self, model: str = MODEL, api_key: str | None = None) -> None:
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

    def complete(  # pragma: no cover - needs network, never runs in the suite
        self, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        from anthropic import Anthropic

        client = Anthropic(api_key=self._api_key)
        message = client.messages.create(  # type: ignore[call-overload]
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            tools=[
                {
                    "name": "emit_hypotheses",
                    "description": "Return ranked hypotheses for this exception.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "emit_hypotheses"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise RuntimeError("the model returned no tool_use block")
