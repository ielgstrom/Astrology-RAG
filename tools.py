"""Tools the model can call in the middle of an answer.

Only put something here when its value depends on what happens during the
conversation. Anything already known at startup belongs in the system prompt
instead: it costs one round-trip less, and small local models are unreliable
at deciding when to reach for a tool.

Drawing a card qualifies. Language models are poor random generators — asked
for a card they gravitate to the same handful, usually the Tower or Death —
so the draw has to happen in Python.
"""

from __future__ import annotations

import random
import re
import unicodedata

from langchain_core.tools import tool

# Major Arcana only. Twenty-two cards is plenty of variety for a prophecy, and
# a short table keeps the tool result small enough to sit in an already tight
# context window.
MAJOR_ARCANA: dict[str, str] = {
    "The Fool": "beginnings, innocence, a leap taken blind",
    "The Magician": "will, skill, something made from nothing",
    "The High Priestess": "secrets, intuition, what is not yet spoken",
    "The Empress": "abundance, nurture, slow growth",
    "The Emperor": "authority, structure, the weight of order",
    "The Hierophant": "tradition, teaching, inherited rules",
    "The Lovers": "union, choice, two roads",
    "The Chariot": "drive, control, a hard-won advance",
    "Strength": "patience, quiet courage, the tamed beast",
    "The Hermit": "solitude, searching, a lantern in the dark",
    "Wheel of Fortune": "turning luck, cycles, what comes around",
    "Justice": "balance, consequence, the scales settling",
    "The Hanged Man": "suspension, surrender, the reversed view",
    "Death": "endings, transformation, the door closing",
    "Temperance": "moderation, blending, slow alchemy",
    "The Devil": "bondage, appetite, the chain one chooses",
    "The Tower": "sudden ruin, revelation, the fall",
    "The Star": "hope, healing, light after the storm",
    "The Moon": "illusion, dreams, the path half-seen",
    "The Sun": "clarity, joy, warmth without shadow",
    "Judgement": "reckoning, awakening, the call answered",
    "The World": "completion, return, the circle closed",
}

# Roughly the usual split: reversed cards are the minority in a real spread.
REVERSED_ODDS = 0.35


@tool
def draw_tarot_card() -> str:
    """Draw one random card from the tarot's Major Arcana.

    Call this ONLY when the querent explicitly asks for a card, a reading, or a
    tarot spread. Do not call it for ordinary questions about the future, and do
    not call it more than once per question. Returns the card's name, whether it
    fell upright or reversed, and its keywords. When a card is called for, it
    must come from here: never invent one or guess which one was drawn.
    """
    name = random.choice(list(MAJOR_ARCANA))
    meaning = MAJOR_ARCANA[name]
    if random.random() < REVERSED_ODDS:
        return (
            f"{name}, reversed — {meaning}; reversed, so read it as blocked, "
            "delayed, resisted, or turned inward."
        )
    return f"{name}, upright — {meaning}."


# Everything bound to the model by default. Import this rather than the
# individual tools so adding one is a single-line change here.
DEFAULT_TOOLS = [draw_tarot_card]

# Small local models reach for whatever tool they are handed, however the
# instructions are worded, so whether the deck is even offered is decided here
# instead of being left to the model. Both languages the reader is used in, and
# only words that mean a card and nothing else: "sign" is deliberately absent,
# since every querent has a zodiac one.
CARD_REQUEST_WORDS = frozenset(
    {
        "card", "cards", "tarot", "arcana", "arcanum", "deck", "spread", "draw",
        "reading", "carta", "cartas", "baraja", "tirada", "arcano", "arcanos",
        "mazo", "echame", "tirame",
    }
)


def asks_for_card(question: str) -> bool:
    """True when the question actually asks for a card to be drawn."""
    # Accents are stripped rather than spelled out in the table, so "tírame"
    # and a keyboard-lazy "tirame" both land on the same entry. Splitting on
    # letters keeps punctuation from gluing itself to the last word, which
    # "¿una carta?" would otherwise do.
    plain = unicodedata.normalize("NFKD", question.lower())
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch))
    return any(
        word in CARD_REQUEST_WORDS for word in re.findall(r"[^\W\d_]+", plain)
    )
