"""Which model answers the question, and where it runs.

Two backends. Ollama runs a model on this machine: no key, nothing leaves the
laptop, but a small model that needs the deck withheld from it to behave.
Mistral is the hosted API: a key in `.env`, a far larger context window, and a
model big enough to follow the prompt without hand-holding.

Everything provider-specific lives here — default model, default context size,
which knobs the client actually accepts, and what its failures mean — so the
reader itself never has to ask which one it is talking to.
"""

from __future__ import annotations

import os
import sys
from typing import Any

AUTO = "auto"
OLLAMA = "ollama"
MISTRAL = "mistral"

PROVIDERS = (AUTO, OLLAMA, MISTRAL)

DEFAULT_PROVIDER = AUTO

DEFAULT_MODELS = {
    OLLAMA: "llama3.2",
    MISTRAL: "mistral-small-latest",
}

# Ollama's own default context is only 2048 tokens and anything past it is
# silently dropped — fatal for a tool that stuffs whole documents into the
# prompt. The built-in astrology PDF alone is ~1,350 tokens, which left a card
# reading with barely a hundred to spare and got the tail of the system prompt
# cut off: the model would then deal itself cards that were never drawn. Raise
# this further for big source sets, lower it if you run out of RAM.
#
# On Mistral the number buys nothing — the server has its own window, which is
# 128k on every current model — so it is only the budget the usage report is
# drawn against, and is set well below the real limit to stay a useful warning.
DEFAULT_NUM_CTX = {
    OLLAMA: 8192,
    MISTRAL: 32_768,
}

# The variable langchain_mistralai itself reads, plus the spellings this
# project's .env has used. A hyphen is legal in a .env file but not in a shell
# identifier, so MISTRAL-KEY only ever arrives through python-dotenv.
MISTRAL_KEY_VAR = "MISTRAL_API_KEY"
MISTRAL_KEY_ALIASES = (MISTRAL_KEY_VAR, "MISTRAL-KEY", "MISTRAL_KEY")

# The hosted API rejects anything above this outright, while Ollama happily
# takes the wild default this project ships with. Clamped rather than refused:
# a temperature too high for the server is not a reason to abandon the reading.
MISTRAL_MAX_TEMPERATURE = 1.0


def mistral_key() -> str | None:
    """The Mistral key from the environment, under any of its spellings.

    Copies whatever it finds into MISTRAL_API_KEY, which is where the client
    looks, so `.env` may keep spelling it MISTRAL-KEY.
    """
    for name in MISTRAL_KEY_ALIASES:
        key = (os.environ.get(name) or "").strip()
        if key:
            os.environ[MISTRAL_KEY_VAR] = key
            return key
    return None


def resolve_provider(choice: str = DEFAULT_PROVIDER) -> str:
    """Turn `auto` into whichever backend is actually usable right now.

    A key in the environment is taken as the intent to use it; without one
    there is nothing to fall back to but the local model.
    """
    if choice != AUTO:
        return choice
    return MISTRAL if mistral_key() else OLLAMA


def default_model(provider: str) -> str:
    return DEFAULT_MODELS[provider]


def default_num_ctx(provider: str) -> int:
    return DEFAULT_NUM_CTX[provider]


def truncates_silently(provider: str) -> bool:
    """Whether overflowing the window loses text without saying so.

    Only Ollama does; the hosted API answers a prompt over its window with an
    error, which is loud enough on its own.
    """
    return provider == OLLAMA


def build_llm(
    provider: str,
    model: str,
    *,
    num_ctx: int,
    num_predict: int,
    temperature: float,
    base_url: str | None = None,
) -> Any:
    """The chat model for one session, with the knobs each client understands.

    The command line speaks Ollama's vocabulary throughout, so the Mistral
    branch translates: num_predict is max_tokens, num_ctx is the server's
    business, and base_url is only meaningful for a self-hosted Ollama.
    """
    if provider == OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            base_url=base_url,
        )

    if provider == MISTRAL:
        from langchain_mistralai import ChatMistralAI

        key = mistral_key()
        if not key:
            raise ValueError(
                f"No Mistral API key. Put {MISTRAL_KEY_VAR}=... in .env "
                "(or run with --provider ollama for the local model)."
            )
        if temperature > MISTRAL_MAX_TEMPERATURE:
            print(
                f"note: Mistral caps temperature at {MISTRAL_MAX_TEMPERATURE}; "
                f"using that instead of {temperature}.",
                file=sys.stderr,
            )
            temperature = MISTRAL_MAX_TEMPERATURE
        return ChatMistralAI(
            model=model,
            temperature=temperature,
            # -1 means "until the model stops" on Ollama; the hosted API spells
            # that as no limit at all.
            max_tokens=num_predict if num_predict > 0 else None,
            api_key=key,
        )

    raise ValueError(f"Unknown provider: {provider}")


def error_hint(exc: Exception, provider: str, model: str) -> str | None:
    """The usual failure of each backend, turned into something actionable."""
    message = str(exc).lower()
    if provider == OLLAMA:
        if "connect" in message or "refused" in message:
            return "Is the Ollama server up? Start it with: ollama serve"
        if "not found" in message or "no such model" in message:
            return f"Model not pulled yet. Run: ollama pull {model}"
        return None

    if provider == MISTRAL:
        if "401" in message or "unauthorized" in message or "api key" in message:
            return f"Mistral rejected the key. Check {MISTRAL_KEY_VAR} in .env."
        if "429" in message or "rate limit" in message or "capacity" in message:
            return "Mistral is rate-limiting this key. Wait a moment and ask again."
        if "404" in message or "model" in message and "not" in message:
            return (
                f"No such Mistral model: {model}. Try --model "
                f"{DEFAULT_MODELS[MISTRAL]}."
            )
        if "connect" in message or "timed out" in message:
            return "Could not reach the Mistral API — check the network."
    return None
