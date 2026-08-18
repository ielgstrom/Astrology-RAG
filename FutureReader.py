"""The reader itself: sources, prompt, agent, and the shape of a reading.

Nothing in here reads stdin or writes to the terminal. Two front ends drive it —
`cli.py` for the terminal session and `api.py` for the browser — and both go
through the same methods, so a change to how a spread is read only has to be
made once.
"""

from __future__ import annotations

import sys
from datetime import date
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
import models
from sources import describe, load_sources
from tools import DEFAULT_TOOLS, asks_for_card, draw_spread

# Sources always loaded, whatever the user passes on the command line. Anything
# given as a positional argument is added on top of these.
DEFAULT_SOURCES = ["astrology-sign-meanings.pdf"]

# Tokens to generate. -1 means "until the model stops".
DEFAULT_NUM_PREDICT = 4_096

# Grounded answers over fixed sources: no reason to sample creatively.
DEFAULT_TEMPERATURE = 0.2

# How many times a single question may bounce back for tool calls before we
# force an answer. Small models sometimes loop on the same tool forever.
MAX_TOOL_ROUNDS = 2

# Stands in for the matter of the reading when nobody has named one — a phrase
# rather than "unknown", so the prompt still reads as a sentence.
DEFAULT_TOPIC = "whatever the days ahead may hold for them"

SYSTEM_PROMPT = """You are a psychic old woman that reads the cards, the stars and the future. You must follow the following rules:
- Only answer questions about divination: the reading of the tarot cards, the reading of the future, and the zodiac sign of the person you are speaking to and what that sign says of them.
- If you are asked anything outside of divination, you must answer "The cards, the stars and the future are my concern, thus I cannot answer that question."
- A question about the past or the present is yours to answer only when it is asked of the cards or of their sign. Asked of anything else, you must answer "The cards, the stars and the future are my concern, thus I cannot answer that question."
- When you divine, you must always answer in a vague and mysterious way, without giving any specific details.
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
the earlier cards are how the later ones are read. Reading them is divination,
whichever of the three they speak of, so you never refuse a question about
these cards."""


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
        provider: str = models.DEFAULT_PROVIDER,
        model: str | None = None,
        num_ctx: int | None = None,
        num_predict: int = DEFAULT_NUM_PREDICT,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: str | None = None,
        trace_tools: bool = False,
    ) -> None:
        # Settled here rather than in the signature: which backend answers
        # decides both the model name and the context budget, and `auto` is not
        # resolved until the environment has been read.
        self.provider = models.resolve_provider(provider)
        self.model = model or models.default_model(self.provider)
        self.num_ctx = num_ctx if num_ctx is not None else models.default_num_ctx(self.provider)
        self.llm = models.build_llm(
            self.provider,
            self.model,
            num_ctx=self.num_ctx,
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

    # The three questions a reading is made of. They live on the reader rather
    # than in each front end because they are not display strings: each one is
    # formatted against state the reader owns (the topic, the cards turned so
    # far), and a spread read one way in the terminal and another in the browser
    # would be two different products.

    def turn(self, position: str, card: str) -> str:
        """Turn one card face up, and return the question that reads it.

        Appending here is what makes the card visible to the prompt: until this
        runs, `deck_block` still says the deck is wrapped. The card is passed in
        rather than drawn here so the whole spread can be dealt in one go — see
        `draw_spread` on why no card may repeat.
        """
        self.spread.append((position, card))
        return CARD_QUESTION.format(position=position, card=card, topic=self.topic)

    def closing_question(self) -> str:
        """The question that ties the cards already turned into one prophecy."""
        return CLOSING_QUESTION.format(cards=self.cards_text, topic=self.topic)

    def sign_question(self) -> str:
        """The question a reading of the stars opens with."""
        return SIGN_QUESTION.format(topic=self.topic)

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
        """Echo each draw to stderr when tracing is on.

        Plain text and stderr rather than the CLI's colours: this runs under the
        web server too, where the only reader is a log file.
        """

        @wrap_tool_call
        def trace(request, handler):
            result = handler(request)
            if self.trace_tools:
                drawn = getattr(result, "text", result)
                print(f"Card drawn: {drawn}", file=sys.stderr)
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


def sign_for(birth_date: date) -> str:
    """The zodiac sign a birth date falls under.

    Pure on purpose: the terminal asks for the date with `input`, the API gets
    it parsed out of JSON, and neither of those belongs anywhere near the table
    of cusps below.
    """
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
