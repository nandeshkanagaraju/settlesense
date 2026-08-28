"""LLMClient protocol, plus the Real and Replay implementations (SDD 7).

TWO RULES, both enforced at construction rather than by convention:

  ReplayLLMClient looks up sha256(prompt) in fixtures/llm/. A MISS RAISES,
  naming the hash. It never falls back to the network, because a fixture-backed
  test that quietly makes a real call is a test that passes for money and
  cannot be run offline or reproducibly.

  RealLLMClient.__init__ RAISES if PYTEST_CURRENT_TEST is set (D7). The suite
  must be incapable of billing anyone, and "we only construct it in production"
  is a property of call sites rather than of the type.

DETERMINISM COMES FROM THE REPLAY CACHE, NOT FROM THE PROVIDER. This is the
single most important sentence in this module. `temperature=0`, `top_p=1` and
`seed=` reduce variation; NONE of them guarantees it. OpenAI documents `seed`
as BEST EFFORT and pairs it with a `system_fingerprint` that changes when the
backend changes. So the reproducibility this project claims comes from
`fixtures/llm/<sha256(prompt)>.json` - a recorded response replayed byte for
byte - and the provider settings below are there to make a RECORDING SESSION
less noisy, not to make the provider deterministic.

Nobody should later read `seed=` as a reproducibility claim. If the fixtures
are deleted, the guarantee is gone; re-recording produces a new fixture set,
not the old one.

THE MODEL STRING IS PINNED TO A DATED SNAPSHOT. "gpt-4o" is an alias that
moves; "gpt-4o-2024-08-06" does not. An alias silently repoints and a recorded
fixture becomes unreproducible without a line of this repo changing.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "FIXTURE_DIR",
    "MAX_ATTEMPTS",
    "MODEL",
    "RETRYABLE",
    "FixtureMissError",
    "LLMClient",
    "ModelUnavailable",
    "OutageLLMClient",
    "RealLLMClient",
    "ReplayLLMClient",
    "prompt_hash",
    "record_fixture",
    "retry_until_unavailable",
]

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "llm"

MODEL = "gpt-4o-2024-08-06"
"""A DATED SNAPSHOT, never a moving alias.

`gpt-4o` repoints as OpenAI ships new versions; `gpt-4o-2024-08-06` does not.
A fixture recorded against an alias cannot be reproduced later, because the
thing that produced it no longer exists under that name. Changing this string
invalidates the fixture set, and that is the intended consequence.
"""

TEMPERATURE = 0
TOP_P = 1
SEED = 42
"""Passed to the API, and NOT a determinism guarantee.

OpenAI documents `seed` as best-effort and returns a `system_fingerprint` that
changes when the backend does. It reduces variation within a recording session;
it does not make the provider reproducible. The replay cache does that.
"""

MAX_TOKENS = 4096

API_KEY_VARIABLE = "OPENAI_API_KEY"


def prompt_hash(prompt: str) -> str:
    """sha256 of the exact prompt text. The fixture key (SDD 7).

    Of the PROMPT, not of the exception: two exceptions that produce the same
    prompt should share a recording, and a prompt that changed by one character
    must miss rather than serve a stale answer to a question nobody asked.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class FixtureMissError(RuntimeError):
    """No recorded response for this prompt. Loud, and never a network fallback."""


class ModelUnavailable(RuntimeError):
    """The model could not be reached, after retries. M10, SDD 4.9.

    DELIBERATELY NOT A SUBCLASS OF FixtureMissError, and not the reverse. A
    fixture miss is a fact about this repository - a prompt nobody recorded -
    and the honest response is to abstain and say so. An outage is a fact about
    the world, the case is untouched, and the honest response is to put it back
    in the queue for the next run. Collapsing them would let a missing
    recording be reported as a service failure, which is the one thing that
    would make the outage numbers meaningless.

    `attempts` is carried so a caller can say how hard it tried rather than
    asserting a policy it does not own.
    """

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(f"{message} (after {attempts} attempt(s))")
        self.attempts = attempts


MAX_ATTEMPTS = 3
"""One call plus `llm_max_retries` (2) retries. SDD 4.9, config/thresholds.yaml.

THREE ATTEMPTS, NOT THREE RETRIES. "after 2 retries" is three calls, and
reading it as three retries would make the real behaviour four - a silent 33%
more spend and a third longer to fail, in the path that runs when the provider
is already struggling.
"""


RETRYABLE: tuple[type[Exception], ...] = (TimeoutError, json.JSONDecodeError, OSError, RuntimeError)
"""The three SDD 4.9 failures, treated IDENTICALLY. Timeout, HTTP error, bad JSON.

From the orchestrator's side each one means the same thing - no usable answer
arrived - and the response is the same: put the case back. Splitting them into
different handling would invite a caller to catch one and forget another, and
the one forgotten would crash a run instead of degrading it.

`OSError` is the HTTP/connection family; `RuntimeError` covers the empty-message
and wrong-shape cases `_attempt` raises itself.
"""


def retry_until_unavailable(
    call: Callable[[], dict[str, Any]], attempts: int = MAX_ATTEMPTS
) -> dict[str, Any]:
    """Call, retry, then ModelUnavailable. EXTRACTED SO IT CAN BE TESTED.

    It lived inside `RealLLMClient.complete`, which D7 forbids constructing
    inside a test run - so the retry policy was unreachable from the suite and
    the only evidence it worked would have been reading it. A policy nobody can
    exercise is a policy nobody has checked.

    NO ERROR CLASS IS RETRIED DIFFERENTLY. A wrong guess about which failures
    are transient is how a client hammers a provider that is already down.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return call()
        except RETRYABLE as error:
            last = error
    raise ModelUnavailable(f"no usable response: {last}", attempts=attempts) from last


class OutageLLMClient:
    """A client that is always down. For `--simulate-outage` and for tests.

    THIS IS THE ONLY WAY AN OUTAGE IS PRODUCED IN THIS PROJECT. There is no
    flag inside the real client that makes it pretend to fail: a production
    code path that can be told to fail is a production code path that can fail
    by accident. The demo swaps the client instead, which is the same seam the
    replay client already uses.

    `calls` records what was attempted, so "nothing was sent" and "everything
    was sent and everything failed" are distinguishable after the fact.
    """

    def __init__(self, reason: str = "simulated outage") -> None:
        self.reason = reason
        self.calls: list[str] = []

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        del schema
        self.calls.append(prompt_hash(prompt))
        raise ModelUnavailable(self.reason, attempts=MAX_ATTEMPTS)


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
    """The live OpenAI client. REFUSES to exist inside a test run (D7).

    Used ONLY while recording fixtures. Every other path in this project -
    tests, eval, the bench - goes through ReplayLLMClient and never has a
    network. The provider exists during recording and nowhere else.
    """

    def __init__(self, model: str = MODEL, api_key: str | None = None) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise RuntimeError(
                "RealLLMClient must not be constructed inside a test run "
                "(PYTEST_CURRENT_TEST is set). Use ReplayLLMClient. The suite has "
                "to be incapable of billing anyone, and that is a property of this "
                "type rather than of who remembered to check."
            )
        self.model = model
        self._api_key = api_key or os.environ.get(API_KEY_VARIABLE)
        if not self._api_key:
            raise RuntimeError(
                f"{API_KEY_VARIABLE} is not set. Recording fixtures needs a live "
                "OpenAI key; everything else in this project runs from "
                "fixtures/llm/ and needs no key at all."
            )
        self.last_usage: dict[str, int] = {}
        """Token counts from the most recent call. MEASURED, never estimated.

        Populated from the API response so a cost figure is arithmetic on real
        counts rather than an assumption about prompt length.
        """

    def complete(  # pragma: no cover - needs network, never runs in the suite
        self, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """One structured completion, retried, then ModelUnavailable (M10).

        Timeout, HTTP error and invalid JSON are the three failures SDD 4.9
        names, and they are treated IDENTICALLY here on purpose. From the
        orchestrator's side each one means the same thing - no usable answer
        arrived - and the response is the same: put the case back. Splitting
        them into different exception types would invite a caller to handle one
        and forget another, and the one forgotten would crash a run.

        WHAT IS NOT RETRIED: nothing. There is no error class here that is
        retried differently, because a wrong guess about which failures are
        transient is how a client hammers a provider that is already down.
        Three attempts, then give up and say so.
        """
        return retry_until_unavailable(lambda: self._attempt(prompt, schema))

    def _attempt(  # pragma: no cover - needs network, never runs in the suite
        self, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """One call. The schema is enforced PROVIDER-SIDE and locally.

        The provider-side check is a convenience that saves a retry; it is not
        a guarantee, and `hypothesis.parse_hypotheses` is what the pipeline
        relies on. A response that satisfied the provider and not us is
        discarded either way.
        """
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,  # best-effort; see the module docstring
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "hypotheses",
                    "strict": False,
                    "schema": schema,
                },
            },
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage
        self.last_usage = {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("the model returned an empty message")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed
