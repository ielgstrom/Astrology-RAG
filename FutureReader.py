from __future__ import annotations

import argparse
import sys
from typing import Iterator, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain_ollama import ChatOllama

from sources import describe, load_sources
from tools import DEFAULT_TOOLS, asks_for_card, draw_spread

DEFAULT_MODEL = "llama3.2"

# Sources always loaded, whatever the user passes on the command line. Anything
# given as a positional argument is added on top of these.
DEFAULT_SOURCES = ["astrology-sign-meanings.pdf"]

# Ollama's own default context is only 2048 tokens and anything past it is
# silently dropped — fatal for a tool that stuffs whole documents into the
# prompt. The built-in astrology PDF alone is ~1,350 tokens, which left a card
# reading with barely a hundred to spare and got the tail of the system prompt
# cut off: the model would then deal itself cards that were never drawn. Raise
# this further for big source sets, lower it if you run out of RAM.
DEFAULT_NUM_CTX = 8192

# Tokens to generate. -1 means "until the model stops".
DEFAULT_NUM_PREDICT = 4_096

# Grounded answers over fixed sources: no reason to sample creatively.
DEFAULT_TEMPERATURE = 100

# How many times a single question may bounce back for tool calls before we
# force an answer. Small models sometimes loop on the same tool forever.
MAX_TOOL_ROUNDS = 2

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

# Stands in for the matter of the reading when nobody has named one — a phrase
# rather than "unknown", so the prompt still reads as a sentence.
DEFAULT_TOPIC = "whatever the days ahead may hold for them"

SYSTEM_PROMPT = """You are a psychic old woman that can read the future. You must follow the following rules:
- If you are asked anything not related to the future, you must answer "The future is my concern, thus I cannot answer that question."
- Only answer questions about the future. If you are asked a question about the past or present, you must answer "The future is my concern, thus I cannot answer that question."
- When answering questions about the future, you must always answer in a vague and mysterious way, without giving any specific details.
- You must always answer in a way that is short enough to be cited in an answer, and you must always answer in a way that is short enough to be cited in an answer.

<sources>
{context}
</sources>

The block below holds the two most important facts about the person you are
speaking to: their sign, and the matter they came to ask about. It goes last,
after the sources, so it stays closest to the question and is never buried by
the documents.

<querent>
zodiac sign: {sign}
what they came to ask about: {topic}
</querent>

Every prophecy you give must be read through the traits of that sign, as
described in the sources, and must speak to the matter they came to ask about.
Never ask the person for their sign or for what they seek — you already know
both.

{deck}"""

# How the prompt ends when no cards have been dealt: the deck exists, but the
# model has to ask for it through the tool.
DECK_WRAPPED = """You keep a tarot deck hidden, but it stays wrapped in its cloth unless it is asked
for. Use the draw_tarot_card tool only when the person explicitly asks for a
card or a tarot spread, and then build the prophecy around whatever
it gives you. Every other question they ask you, you answer from the sources and their sign
alone, without touching the deck. Never name a card you did not draw."""

# How it ends once cards are on the table. This replaces the paragraph above
# rather than joining it: telling the reader to draw cards while some are
# already face up is a contradiction, and a 3B model resolves it by inventing
# a fresh deck of its own. Only the cards turned so far are listed — they are
# revealed one at a time, and a card the model has not been shown yet is one it
# cannot spoil.
DECK_DEALT = """These cards are face up on the table, in the order they were turned:

<spread>
{cards}
</spread>

Those are the whole reading. Speak of them by name, in their position, and of
nothing else. There is no further card to draw or to name, and you never guess
at one still face down. Every card is read upon the matter the querent came to
ask about, and upon nothing else.

A spread runs from what has passed, through what stands now, to what is coming:
the earlier cards are how the later ones are read. Reading them is telling the
future, so the rule about the past and the present does not apply here, and you
never refuse a question about these cards."""


def _latest_question(messages: Sequence[BaseMessage]) -> str:
    """The question this turn is answering.

    Tool rounds append to the same message list, so the question has to be
    fished back out of it rather than passed down: it is the last thing the
    person actually said.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def _tool_rounds(messages: Sequence[BaseMessage]) -> int:
    """How many times the model has already reached for a tool this turn."""
    rounds = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break  # anything older belongs to an earlier turn
        if isinstance(message, AIMessage) and message.tool_calls:
            rounds += 1
    return rounds


class FutureReader:
    """Holds a set of loaded sources and answers questions about them."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: str | None = None,
        trace_tools: bool = False,
    ) -> None:
        self.num_ctx = num_ctx
        self.llm = ChatOllama(
            model=model,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            base_url=base_url,
        )
        self.tools = list(DEFAULT_TOOLS)
        self.trace_tools = trace_tools
        self.documents: list[Document] = []
        self.history: list[BaseMessage] = []
        self.sign: str = "unknown"
        # What the querent came to ask about, in their own words. Asked for
        # before the deck is touched, so the cards are read upon it.
        self.topic: str = DEFAULT_TOPIC
        # Cards already turned face up, as (position, card) pairs, in the order
        # they were revealed. Grows one card at a time as the reading is given,
        # and stays empty when the querent came for a reading of their sign.
        self.spread: list[tuple[str, str]] = []
        self.agent = create_agent(
            self.llm,
            tools=self.tools,
            middleware=[
                self._prompt_middleware(),
                self._deck_middleware(),
                self._trace_middleware(),
            ],
        )

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

    @property
    def deck_block(self) -> str:
        """How the prompt ends: cards on the table, or a deck still wrapped.

        The dealt cards go in the prompt rather than through a tool result
        because they were drawn before the question that reads them — there is
        no decision left for the model to make about them.
        """
        if not self.spread:
            return DECK_WRAPPED
        return DECK_DEALT.format(cards=self.cards_text)

    @property
    def cards_text(self) -> str:
        """The spread as lines of `position: card`."""
        return "\n".join(f"{position}: {card}" for position, card in self.spread)

    def ask(self, question: str, *, remember: bool = True) -> str:
        """The whole answer at once, in one non-streaming request.

        Same agent and same middleware as `stream`; only the transport differs.
        The turn always ends on a prophecy rather than a tool result — the deck
        middleware sees to that — so the last message is the answer.
        """
        result = self.agent.invoke(self._state(question))
        answer = result["messages"][-1].text
        if remember:
            self._remember(question, answer)
        return answer

    def stream(self, question: str, *, remember: bool = True) -> Iterator[str]:
        """Run one turn through the agent, yielding prose as it arrives.

        stream_mode="messages" gives token-level chunks from every model call in
        the turn, tool rounds included. Rounds that end in a tool call carry no
        text, so nothing reaches the caller until the model finally speaks.
        """
        chunks: list[str] = []
        for chunk, _metadata in self.agent.stream(self._state(question), stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk) and chunk.text:
                chunks.append(chunk.text)
                yield chunk.text
        if remember:
            self._remember(question, "".join(chunks))

    def _state(self, question: str) -> dict[str, object]:
        """The turn handed to the agent: the conversation so far, plus this.

        The sources and the sign are not in here — the prompt middleware puts
        them in the system message on every call, so they never age out.
        """
        return {"messages": [*self.history, HumanMessage(question)]}

    def _prompt_middleware(self) -> AgentMiddleware:
        """Rebuild the system prompt before every model call.

        The sources and the sign are read off `self` each time rather than
        frozen into the agent, so `add` and a late-arriving sign both take
        effect without rebuilding the graph.
        """

        @dynamic_prompt
        def render(request: ModelRequest) -> str:
            return self._render_prompt()

        return render

    def _render_prompt(self) -> str:
        """The system prompt as it stands right now.

        One place, so the middleware and the token estimate can never drift
        apart about what is actually being sent.
        """
        return SYSTEM_PROMPT.format(
            context=self.context,
            sign=self.sign,
            topic=self.topic,
            deck=self.deck_block,
        )

    def _deck_middleware(self) -> AgentMiddleware:
        """Hand over the deck only when the question actually asks for a card.

        A 3B model calls whatever tool it can see, however firmly the prompt
        tells it not to, so the deck is withheld at the API boundary instead:
        no schema in the request, no card in the answer. The same override ends
        a turn that keeps reaching for tools — once MAX_TOOL_ROUNDS cards have
        been drawn the next call goes out empty-handed, so the turn finishes in
        a prophecy instead of another round trip.

        A spread already on the table withholds it outright. Those questions
        name cards by definition, so the word test alone would hand over the
        deck on every one of them, and a fourth card drawn mid-reading is
        exactly what the dealt-spread prompt swears does not exist.
        """

        @wrap_model_call
        def offer_deck(request: ModelRequest, handler):
            if (
                self.tools
                and not self.spread
                and asks_for_card(_latest_question(request.messages))
                and _tool_rounds(request.messages) < MAX_TOOL_ROUNDS
            ):
                return handler(request)
            return handler(request.override(tools=[]))

        return offer_deck

    def _trace_middleware(self) -> AgentMiddleware:
        """Echo each draw to stderr under --verbose."""

        @wrap_tool_call
        def trace(request, handler):
            result = handler(request)
            if self.trace_tools:
                drawn = getattr(result, "text", result)
                print(_paint(f"Card drawn: {drawn} \n", "31"), file=sys.stderr)
            return result

        return trace

    def used_tokens(self, question: str = "") -> int:
        """Rough token count of the whole prompt as it would be sent right now.

        Mirrors what the prompt middleware builds, so the estimate covers the
        system prompt, the sources, and the accumulated history — not just the
        documents.
        """
        rendered = "\n".join(
            [
                self._render_prompt(),
                *(message.text for message in self.history),
                question,
            ]
        )
        return len(rendered) // 4  # ~4 chars per token, good enough here

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


def _show_error_hint(exc: Exception, model: str) -> None:
    """Turn the two usual Ollama failures into something actionable."""
    message = str(exc).lower()
    if "connect" in message or "refused" in message:
        print("Is the Ollama server up? Start it with: ollama serve", file=sys.stderr)
    elif "not found" in message or "no such model" in message:
        print(f"Model not pulled yet. Run: ollama pull {model}", file=sys.stderr)

ZODIAC_GLYPHS = {
    "Aries": "♈",
    "Taurus": "♉",
    "Gemini": "♊",
    "Cancer": "♋",
    "Leo": "♌",
    "Virgo": "♍",
    "Libra": "♎",
    "Scorpio": "♏",
    "Sagittarius": "♐",
    "Capricorn": "♑",
    "Aquarius": "♒",
    "Pisces": "♓",
}


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


READING_CARDS = "cards"
READING_SIGN = "sign"

# What the opening menu accepts for each reading. The number is the documented
# way in; the words are there so someone who types "cartas" or "tarot" instead
# of "1" is not told they answered wrong.
READING_WORDS = {
    READING_CARDS: {
        "1", "cards", "card", "tarot", "spread",
        "cartas", "carta", "tirada", "baraja",
    },
    READING_SIGN: {
        "2", "sign", "horoscope", "zodiac", "stars",
        "signo", "horoscopo", "zodiaco", "astros",
    },
}

# The question the sign reading opens with.
SIGN_QUESTION = (
    "I have come to ask about this: {topic}\n"
    "Read what my sign says of it in the days ahead."
)

# Asked once per card, as it is turned. Like the spread block in the system
# prompt, this repeats the card verbatim: with a whole PDF of sources in
# between, restating it right next to the question is what actually stops a
# small model from reading a card nobody dealt.
CARD_QUESTION = (
    "I have come to ask about this: {topic}\n"
    "You have just turned the card for «{position}»:\n"
    "{card}\n"
    "Read this one card alone, and what it says of what I asked about. Say "
    "nothing of the cards still face down."
)

# Asked after the last card, so the three are finally read as one spread —
# which is the whole point of a three-card layout, and something no single-card
# reading can say on its own.
CLOSING_QUESTION = (
    "I have come to ask about this: {topic}\n"
    "Now the whole spread lies open:\n"
    "{cards}\n"
    "Tie the cards together into one prophecy, and name no other."
)


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


def choose_reading() -> str:
    """Ask which of the two readings the person has come for."""
    print(_paint("  1) A spread of three cards", "36"), file=sys.stderr)
    print(_paint("  2) A reading of your sign\n", "36"), file=sys.stderr)
    while True:
        choice = input("What do you seek? [1/2] ").strip().lower()
        for reading, words in READING_WORDS.items():
            if choice in words:
                return reading
        print("Answer 1 or 2.", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    _print_banner()

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
        trace_tools=args.verbose,
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

    # Both readings are given through the querent's sign, so the birth date is
    # asked for either way, and both are read upon the matter they come with —
    # which is why the topic is asked for before the deck is touched below.
    # Walking out at any of these prompts is not an error.
    try:
        reading = choose_reading()
        reader.sign = get_horoscope_sign()
        reader.topic = ask_topic()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if reader.sign != "unknown":
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
            _show_error_hint(exc, args.model)
            return
        _report_context(reader, args.verbose)

    # The reading the person chose, given before they say anything themselves.
    if reading == READING_CARDS:
        # The whole spread is dealt in one call, so no card can repeat, but the
        # cards are turned over one at a time: each is printed, read on its own,
        # and only then added to what the prompt admits is on the table.
        for position, card in draw_spread():
            print(_paint(f"\n  {position} — {card}\n", "1;35"), file=sys.stderr)
            reader.spread.append((position, card))
            answer(
                CARD_QUESTION.format(
                    position=position, card=card, topic=reader.topic
                )
            )
        answer(CLOSING_QUESTION.format(cards=reader.cards_text, topic=reader.topic))
    else:
        answer(SIGN_QUESTION.format(topic=reader.topic))

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
