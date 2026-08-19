"""The reader as a terminal session.

Everything here is about the terminal and nothing else: the banner, the colours,
and the questions asked with `input`. The reading itself belongs to
`FutureReader`, which this only drives — `api.py` drives the same object over
HTTP without repeating a line of it.

Run it with:

    python cli.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Sequence

import models
from FutureReader import (
    DEFAULT_NUM_PREDICT,
    DEFAULT_SOURCES,
    DEFAULT_TEMPERATURE,
    ZODIAC_GLYPHS,
    FutureReader,
    sign_for,
)
from sources import describe
from tools import draw_spread

# Printed once at startup. Kept as plain ASCII/box-drawing so it survives any
# terminal that can show the rest of the session; colour is added separately and
# only when we are actually attached to a TTY.
BANNER = r"""
   ███████╗██╗   ██╗████████╗██╗   ██╗██████╗ ███████╗
   ██╔════╝██║   ██║╚══██╔══╝██║   ██║██╔══██╗██╔════╝
   █████╗  ██║   ██║   ██║   ██║   ██║██████╔╝█████╗
   ██╔══╝  ██║   ██║   ██║   ██║   ██║██╔══██╗██╔══╝
   ██║     ╚██████╔╝   ██║   ╚██████╔╝██║  ██║███████╗
   ╚═╝      ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
      ██████╗ ███████╗ █████╗ ██████╗ ███████╗██████╗
      ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗
      ██████╔╝█████╗  ███████║██║  ██║█████╗  ██████╔╝
      ██╔══██╗██╔══╝  ██╔══██║██║  ██║██╔══╝  ██╔══██╗
      ██║  ██║███████╗██║  ██║██████╔╝███████╗██║  ██║
      ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
"""

TAROT_SPREAD = r"""
          *      .          .         *        .
           .---------.  .---------.  .---------.
           |    *    |  | /\_^_// |  |    |    |
           |  *   *  |  | \|===|  |  |  \ | /  |
           | *  *  * |  |  |o o// |  |-- (o) --|
           |  *   *  |  |  |o o|  |  |  / | \  |
           |    *    |  |  |_n_|  |  |    |    |
           |         |  |         |  |         |
           |THE STAR |  |THE TOWER|  | THE SUN |
           '---------'  '---------'  '---------'
      .           *            .          *        .
"""

def _paint(text: str, code: str, stream=sys.stderr) -> str:
    """Wrap text in an ANSI colour, but only when someone is there to see it.

    Piping the banner into a file or another program should not litter it with
    escape sequences, so anything that is not a TTY gets the plain text back.
    """
    if not stream.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_banner() -> None:
    """Curtain-raiser for the session: title, a spread, and a line of flavour."""
    print(_paint(BANNER, "1;35"), file=sys.stderr)
    print(_paint(TAROT_SPREAD, "36"), file=sys.stderr)
    print(
        _paint("        ~ the mists part · ask, and the future answers ~\n", "2;37"),
        file=sys.stderr,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="FutureReader",
        description=(
            "Ask a local Ollama model or Mistral's API questions about files, "
            "directories, and web pages."
        ),
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help=(
            "extra files, directories, or URLs to load, on top of the built-in "
            f"ones ({', '.join(DEFAULT_SOURCES)})"
        ),
    )
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument(
        "--provider",
        choices=models.PROVIDERS,
        default=models.DEFAULT_PROVIDER,
        help=(
            "which backend answers (default: auto — Mistral when a key is in "
            "the environment, the local Ollama model otherwise)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "default: "
            + ", ".join(
                f"{name} on {provider}"
                for provider, name in models.DEFAULT_MODELS.items()
            )
        ),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help=(
            "context window in tokens (default: "
            + ", ".join(
                f"{size:,} on {provider}"
                for provider, size in models.DEFAULT_NUM_CTX.items()
            )
            + "); on Mistral this only sets what the usage report counts against"
        ),
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help=f"tokens to generate, -1 for unlimited (default: {DEFAULT_NUM_PREDICT})",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"default: {DEFAULT_TEMPERATURE}"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Ollama server, default $OLLAMA_HOST (ignored on Mistral)",
    )
    parser.add_argument("--no-stream", action="store_true", help="wait for the full answer")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="report context-window usage at startup and after every answer",
    )
    return parser.parse_args(argv)


def _report_context(reader: FutureReader, verbose: bool = False) -> None:
    """Print how full the context window is, and warn when it is nearly gone.

    The usage line is opt-in via --verbose; the truncation warning is not,
    since a silently cut prompt is worth knowing about either way.
    """
    estimate = reader.used_tokens()
    left = reader.num_ctx - estimate
    ratio = estimate / reader.num_ctx if reader.num_ctx else 0
    if verbose:
        filled = int(ratio * 20)
        bar = "#" * min(filled, 20) + "." * max(20 - filled, 0)
        print(
            f"context: [{bar}] ~{estimate:,} / {reader.num_ctx:,} tokens "
            f"· {left:,} left ({ratio:.0%} used)",
            file=sys.stderr,
        )
    if ratio > 0.8 and verbose:
        truncates = models.truncates_silently(reader.provider)
        print(
            "warning: the prompt is close to --num-ctx. "
            + (
                "Ollama will silently truncate it — "
                if truncates
                else "The server may refuse it — "
            )
            + "raise --num-ctx, load fewer sources, or restart to drop the history.",
            file=sys.stderr,
        )


def _show_error_hint(exc: Exception, reader: FutureReader) -> None:
    """Turn each backend's usual failure into something actionable."""
    hint = models.error_hint(exc, reader.provider, reader.model)
    if hint:
        print(hint, file=sys.stderr)


def get_horoscope_sign() -> str:
    """Ask for a birth date until one parses, and return its sign."""
    while True:
        birth_date_str = input("Enter your birth date (YYYY-MM-DD): ")
        try:
            birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            continue
        return sign_for(birth_date.date())


def ask_topic() -> str:
    """Ask what the reading is about, before a single card is turned.

    Taken in the querent's own words rather than from a menu: it goes into the
    prompt as-is, and a vague answer makes for a vague prophecy either way.
    """
    while True:
        topic = input("What would you have the future speak of? ").strip()
        if topic:
            return topic
        print("Name the matter you come for.", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    _print_banner()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional; the env var may already be set.

    args = _parse_args(argv)

    try:
        reader = FutureReader(
            provider=args.provider,
            model=args.model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            temperature=args.temperature,
            base_url=args.base_url,
            trace_tools=args.verbose,
        )
    except (ValueError, ImportError) as exc:
        # A missing key or a backend whose package was never installed. Both
        # are fixed before the session starts, not during it.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.verbose:
        print(
            f"Reading with {reader.model} on {reader.provider}\n", file=sys.stderr
        )

    refs = list(DEFAULT_SOURCES)
    refs.extend(args.sources)

    try:
        reader.add(refs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.verbose:
        print(f"Loaded {describe(reader.documents)}\n", file=sys.stderr)

    # The spread is read through the querent's sign and upon the matter they
    # come with, so both are settled before the deck is touched below. Walking
    # out at either prompt is not an error.
    try:
        reader.sign = get_horoscope_sign()
        reader.topic = ask_topic()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    glyph = ZODIAC_GLYPHS.get(reader.sign, "✦")
    print(_paint(f"Your sign: {glyph}  {reader.sign}", "1;33"), file=sys.stderr)
    print(_paint(f"You seek: {reader.topic}\n", "1;33"), file=sys.stderr)

    _report_context(reader, args.verbose)

    def answer(question: str) -> None:
        try:
            if args.no_stream:
                print(reader.ask(question))
            else:
                for chunk in reader.stream(question):
                    print(chunk, end="", flush=True)
                print()
        except Exception as exc:  # noqa: BLE001 - surface anything the server says
            print(f"\nerror: {exc}", file=sys.stderr)
            _show_error_hint(exc, reader)
            return
        _report_context(reader, args.verbose)

    # The reading, given before they say anything themselves. The whole spread
    # is dealt in one call, so no card can repeat, but the cards are turned over
    # one at a time: each is printed, read on its own, and only then added to
    # what the prompt admits is on the table — which is what `reader.turn` does
    # before handing back the question.
    for position, card in draw_spread():
        print(_paint(f"\n  {position} — {card}\n", "1;35"), file=sys.stderr)
        answer(reader.turn(position, card))
    answer(reader.closing_question())

    print("\nAsk a question (Ctrl-D or 'exit' to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            answer(question)


if __name__ == "__main__":
    raise SystemExit(main())
