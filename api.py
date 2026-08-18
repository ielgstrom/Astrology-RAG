"""The reader over HTTP.

The shape of a session is the same one the terminal has always had: pick a
reading, say what you come to ask about, and then the prophecy arrives a word at
a time. What changes is that the words travel over Server-Sent Events instead of
straight to stdout, and that several querents may be mid-reading at once.

Run it with:

    uvicorn api:app --reload
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from typing import Callable, Iterable, Iterator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import models
from FutureReader import (
    DEFAULT_SOURCES,
    READING_CARDS,
    READING_SIGN,
    ZODIAC_GLYPHS,
    FutureReader,
    sign_for,
)
from sources import load_sources
from tools import draw_spread

# Read before anything asks which provider is available: `resolve_provider`
# decides on the key being in the environment, and the CLI's own load_dotenv
# never runs in this process.
load_dotenv()

app = FastAPI(title="FutureReader")

# The built-in sources, parsed once. Every session gets the same Document
# objects rather than its own copy: the PDF never changes, and reparsing it per
# querent would put a second of pypdf in front of every reading.
DOCUMENTS = load_sources(DEFAULT_SOURCES, quiet=True)


@dataclass
class Session:
    """One querent's reading, from the opening menu to the last question."""

    reader: FutureReader
    reading: str
    # Dealt at once when the session opens, so no card can repeat, but handed to
    # the reader one at a time while the prophecy is streamed. The browser gets
    # them here too — face down is still something to draw on screen.
    cards: list[tuple[str, str]] = field(default_factory=list)
    # One reading at a time per session. The prompt is rebuilt from `reader`
    # state on every model call, so two requests streaming through the same
    # session would read each other's cards.
    lock: threading.Lock = field(default_factory=threading.Lock)


# Sessions live in memory, which is fine while this is one process on a laptop:
# a restart drops every reading, and nothing here expires. Both are worth fixing
# before anyone else can reach it.
SESSIONS: dict[str, Session] = {}


class NewSession(BaseModel):
    reading: Literal["cards", "sign"]
    topic: str = Field(min_length=1)
    # Only the stars need it. A card reading is given through the sign as well,
    # but the terminal has never asked for a date on that path, so neither does
    # this: without one the sign stays "unknown" exactly as it does there.
    birth_date: date | None = None


def _session(session_id: str) -> Session:
    try:
        return SESSIONS[session_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="No such reading.") from None


def _event(event: str, **data) -> dict[str, str]:
    """One SSE frame. The browser switches on `event`, so every frame is named."""
    return {"event": event, "data": json.dumps(data)}


# One step of a reading: the card it turns, if any, and the question to put once
# it is turned. The question is a callable rather than a string because turning
# a card is what builds it, and that has to happen in order, under the lock,
# with the previous answer already in the history.
Step = tuple[tuple[str, str] | None, Callable[[], str]]


def _stream(session: Session, steps: Iterable[Step]) -> Iterator[dict[str, str]]:
    """Walk the steps, putting each question to the reader as its turn comes.

    A plain generator rather than an async one, deliberately: `reader.stream` is
    blocking, and sse-starlette runs a sync iterator in a threadpool, so the
    event loop stays free while the model thinks. Made async, one slow reading
    would freeze every other querent.
    """
    with session.lock:
        try:
            for card, question_for in steps:
                # Naming the card before its question goes out keeps the browser
                # in step with the prompt: the text that follows is about this
                # card, and the ones after it are still face down.
                if card is not None:
                    yield _event("card", position=card[0], card=card[1])
                for chunk in session.reader.stream(question_for()):
                    yield _event("chunk", text=chunk)
                yield _event("answer_end")
        except Exception as exc:  # noqa: BLE001 - the browser gets what the server said
            hint = models.error_hint(exc, session.reader.provider, session.reader.model)
            yield _event("error", message=str(exc), hint=hint or "")
            return
    yield _event("end")


@app.get("/health")
def health() -> dict[str, str]:
    """Which backend would answer if a question arrived right now."""
    provider = models.resolve_provider()
    logging.info("Health check: nonsense")
    return {"status": "ok", "provider": provider, "model": models.default_model(provider)}


@app.post("/sessions")
def open_session(new: NewSession) -> dict:
    """Open a reading: settle the sign, deal the cards, and hand back an id.

    Nothing is asked of the model here. The cards come back in the response so
    the browser can lay them out face down before a single token arrives.
    """
    reader = FutureReader()
    reader.documents = list(DOCUMENTS)
    reader.topic = new.topic

    session = Session(reader=reader, reading=new.reading)
    if new.reading == READING_SIGN:
        if new.birth_date is None:
            raise HTTPException(
                status_code=422, detail="A reading of the stars needs a birth date."
            )
        reader.sign = sign_for(new.birth_date)
    else:
        session.cards = draw_spread()

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = session
    return {
        "session_id": session_id,
        "reading": session.reading,
        "sign": reader.sign,
        "glyph": ZODIAC_GLYPHS.get(reader.sign, "✦"),
        "cards": [{"position": p, "card": c} for p, c in session.cards],
    }


@app.get("/sessions/{session_id}/reading")
def reading(session_id: str) -> EventSourceResponse:
    """Stream the reading the querent came for.

    The cards are turned one at a time — each read on its own, and only then
    admitted to the prompt — and the spread is tied together at the end. That
    order is the reading, so it lives in `FutureReader`; all this does is walk
    it and name each card to the browser as it goes.
    """
    session = _session(session_id)

    if session.reading == READING_CARDS:
        steps: list[Step] = [
            (card, partial(session.reader.turn, *card)) for card in session.cards
        ]
        steps.append((None, session.reader.closing_question))
    else:
        steps = [(None, session.reader.sign_question)]

    return EventSourceResponse(_stream(session, steps))


@app.get("/sessions/{session_id}/ask")
def ask(session_id: str, q: str) -> EventSourceResponse:
    """A question of the querent's own, once the reading has been given.

    A GET with the question in the query string rather than a POST, because
    `EventSource` in the browser only ever sends GET.
    """
    session = _session(session_id)
    return EventSourceResponse(_stream(session, [(None, lambda: q)]))
