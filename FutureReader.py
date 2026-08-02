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

# Ollama's own default context is only 2048 tokens and anything past it is
# silently dropped — fatal for a tool that stuffs whole documents into the
# prompt. Raise --num-ctx for big source sets, lower it if you run out of RAM.
DEFAULT_NUM_CTX = 2048

# Tokens to generate. -1 means "until the model stops".
DEFAULT_NUM_PREDICT = 4_096

# Grounded answers over fixed sources: no reason to sample creatively.
DEFAULT_TEMPERATURE = 10

SYSTEM_PROMPT = """You are a psychic old woman that can read the future. You must follow the following rules:
- If you are asked anything not related to the future, you must answer "The future is my concert, thus I cannot answer that question."
- Only answer questions about the future. If you are asked a question about the past or present, you must answer "The future is my concert, thus I cannot answer that question."
- When answering questions about the future, you must always answer in a vague and mysterious way, without giving any specific details.
- You must always answer in a way that is short enough to be cited in an answer, and you must always answer in a way that is short enough to be cited in an answer.

<sources>
{context}
</sources>"""


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

    # -- sources ----------------------------------------------------------- #

    def add(self, *refs: str, quiet: bool = False) -> int:
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

    # -- questions --------------------------------------------------------- #

    def ask(self, question: str, *, remember: bool = True) -> str:
        answer = self.chain.invoke(
            {"context": self.context, "history": self.history, "question": question}
        )
        if remember:
            self._remember(question, answer)
        return answer

    def stream(self, question: str, *, remember: bool = True) -> Iterator[str]:
        chunks: list[str] = []
        for chunk in self.chain.stream(
            {"context": self.context, "history": self.history, "question": question}
        ):
            chunks.append(chunk)
            yield chunk
        if remember:
            self._remember(question, "".join(chunks))

    def _remember(self, question: str, answer: str) -> None:
        self.history.extend([HumanMessage(question), AIMessage(answer)])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="FutureReader",
        description="Ask a local Ollama model questions about files, directories, and web pages.",
    )
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
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
    return parser.parse_args(argv)


def _warn_if_context_too_small(reader: FutureReader) -> None:
    estimate = len(reader.context) // 4  # ~4 chars per token, good enough here
    print(f"Warning: context is estimated at ~{estimate:,}")
    if estimate > reader.num_ctx * 0.8:
        print(
            f"warning: sources are ~{estimate:,} tokens but --num-ctx is "
            f"{reader.num_ctx:,}. Ollama will silently truncate the prompt — "
            f"raise --num-ctx or load fewer sources.\n",
            file=sys.stderr,
        )


def _hint(exc: Exception, model: str) -> None:
    """Turn the two usual Ollama failures into something actionable."""
    message = str(exc).lower()
    if "connect" in message or "refused" in message:
        print("Is the Ollama server up? Start it with: ollama serve", file=sys.stderr)
    elif "not found" in message or "no such model" in message:
        print(f"Model not pulled yet. Run: ollama pull {model}", file=sys.stderr)


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

    if args.sources:
        try:
            reader.add(*args.sources)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Loaded {describe(reader.documents)}\n", file=sys.stderr)
        _warn_if_context_too_small(reader)
    else:
        print("No sources given — answers will not be grounded in anything.\n", file=sys.stderr)

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
