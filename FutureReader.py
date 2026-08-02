from __future__ import annotations

import argparse
import sys
from typing import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from sources import describe, load_sources

DEFAULT_MODEL = "llama3.2"

# Sources always loaded, whatever the user passes on the command line. Anything
# given as a positional argument is added on top of these.
DEFAULT_SOURCES = ["astrology-sign-meanings.pdf"]

# Ollama's own default context is only 2048 tokens and anything past it is
# silently dropped — fatal for a tool that stuffs whole documents into the
# prompt. Raise --num-ctx for big source sets, lower it if you run out of RAM.
DEFAULT_NUM_CTX = 2048

# Tokens to generate. -1 means "until the model stops".
DEFAULT_NUM_PREDICT = 4_096

# Grounded answers over fixed sources: no reason to sample creatively.
DEFAULT_TEMPERATURE = 0.3

SYSTEM_PROMPT = """You are a psychic old woman that can read the future. You must follow the following rules:
- If you are asked anything not related to the future, you must answer "The future is my concern, thus I cannot answer that question."
- Only answer questions about the future. If you are asked a question about the past or present, you must answer "The future is my concern, thus I cannot answer that question."
- When answering questions about the future, you must always answer in a vague and mysterious way, without giving any specific details.
- You must always answer in a way that is short enough to be cited in an answer, and you must always answer in a way that is short enough to be cited in an answer.

<sources>
{context}
</sources>

The zodiac sign block below is the single most important fact about the person
you are speaking to. It goes last, after the sources, so it stays closest to
the question and is never buried by the documents.

<querent>
zodiac sign: {sign}
</querent>

Every prophecy you give must be read through the traits of that sign, as
described in the sources. Never ask the person for their sign — you already
know it."""


class FutureReader:
    """Holds a set of loaded sources and answers questions about them."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str = SYSTEM_PROMPT,
        base_url: str | None = None,
        **model_kwargs,
    ) -> None:
        # base_url=None lets the ollama client fall back to $OLLAMA_HOST, or
        # http://localhost:11434 if that is unset.
        self.num_ctx = num_ctx
        self.llm = ChatOllama(
            model=model,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            base_url=base_url,
            **model_kwargs,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.documents: list[Document] = []
        self.history: list[BaseMessage] = []
        # Re-rendered into the system prompt on every turn, so it never ages out
        # of the conversation the way a message in `history` would.
        self.sign: str = "unknown"

    def add(self, refs: Sequence[str], quiet: bool = False) -> int:
        """Load files, directories, or URLs. Returns the number of docs added."""
        docs = load_sources(refs, quiet=quiet)
        self.documents.extend(docs)
        return len(docs)

    def clear(self) -> None:
        self.documents.clear()
        self.history.clear()

    @property
    def context(self) -> str:
        """All loaded sources, wrapped so the model can cite them by name."""
        if not self.documents:
            return "(no sources loaded)"
        return "\n\n".join(
            '<document source="{}">\n{}\n</document>'.format(
                doc.metadata.get("source", "unknown"), doc.page_content
            )
            for doc in self.documents
        )

    def ask(self, question: str, *, remember: bool = True) -> str:
        answer = self.chain.invoke(self._inputs(question))
        if remember:
            self._remember(question, answer)
        return answer

    def stream(self, question: str, *, remember: bool = True) -> Iterator[str]:
        chunks: list[str] = []
        for chunk in self.chain.stream(self._inputs(question)):
            chunks.append(chunk)
            yield chunk
        if remember:
            self._remember(question, "".join(chunks))

    def used_tokens(self, question: str = "") -> int:
        """Rough token count of the whole prompt as it would be sent right now.

        Renders the template so the estimate covers the system prompt, the
        sources, and the accumulated history — not just the documents.
        """
        rendered = self.prompt.invoke(self._inputs(question)).to_string()
        return len(rendered) // 4  # ~4 chars per token, good enough here

    def _inputs(self, question: str) -> dict[str, object]:
        return {
            "context": self.context,
            "sign": self.sign,
            "history": self.history,
            "question": question,
        }

    def _remember(self, question: str, answer: str) -> None:
        self.history.extend([HumanMessage(question), AIMessage(answer)])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="FutureReader",
        description="Ask a local Ollama model questions about files, directories, and web pages.",
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
    parser.add_argument( "--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help=f"context window in tokens (default: {DEFAULT_NUM_CTX})",
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
    parser.add_argument("--base-url", default=None, help="Ollama server, default $OLLAMA_HOST")
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
        print(
            "warning: the prompt is close to --num-ctx. Ollama will silently "
            "truncate it — raise --num-ctx, load fewer sources, or restart to "
            "drop the history.",
            file=sys.stderr,
        )


def _hint(exc: Exception, model: str) -> None:
    """Turn the two usual Ollama failures into something actionable."""
    message = str(exc).lower()
    if "connect" in message or "refused" in message:
        print("Is the Ollama server up? Start it with: ollama serve", file=sys.stderr)
    elif "not found" in message or "no such model" in message:
        print(f"Model not pulled yet. Run: ollama pull {model}", file=sys.stderr)

def get_horoscope_sign() -> str:
    """Return the user's horoscope sign based on their birth date."""
    from datetime import datetime

    birth_date_str = input("Enter your birth date (YYYY-MM-DD): ")
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return get_horoscope_sign()

    month = birth_date.month
    day = birth_date.day

    if (month == 3 and day >= 21) or (month == 4 and day <= 19):
        return "Aries"
    elif (month == 4 and day >= 20) or (month == 5 and day <= 20):
        return "Taurus"
    elif (month == 5 and day >= 21) or (month == 6 and day <= 20):
        return "Gemini"
    elif (month == 6 and day >= 21) or (month == 7 and day <= 22):
        return "Cancer"
    elif (month == 7 and day >= 23) or (month == 8 and day <= 22):
        return "Leo"
    elif (month == 8 and day >= 23) or (month == 9 and day <= 22):
        return "Virgo"
    elif (month == 9 and day >= 23) or (month == 10 and day <= 22):
        return "Libra"
    elif (month == 10 and day >= 23) or (month == 11 and day <= 21):
        return "Scorpio"
    elif (month == 11 and day >= 22) or (month == 12 and day <= 21):
        return "Sagittarius"
    elif (month == 12 and day >= 22) or (month == 1 and day <= 19):
        return "Capricorn"
    elif (month == 1 and day >= 20) or (month == 2 and day <= 18):
        return "Aquarius"
    elif (month == 2 and day >= 19) or (month == 3 and day <= 20):
        return "Pisces"
    return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional; the env var may already be set.

    args = _parse_args(argv)

    reader = FutureReader(
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
        base_url=args.base_url,
    )

    refs = list(DEFAULT_SOURCES)
    refs.extend(args.sources)

    try:
        reader.add(refs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {describe(reader.documents)}\n", file=sys.stderr)
    reader.sign = get_horoscope_sign()
    print(f"Your sign: {reader.sign}\n", file=sys.stderr)
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
            _hint(exc, args.model)
            return
        # After _remember(), so this is what the *next* question starts from.
        _report_context(reader, args.verbose)

    if args.question:
        answer(args.question)
        return 0

    print("Ask a question (Ctrl-D or 'exit' to quit).", file=sys.stderr)
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
