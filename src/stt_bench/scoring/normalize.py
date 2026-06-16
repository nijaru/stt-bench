"""Text normalization for scoring.

Conservative policy: lowercase, collapse whitespace, strip punctuation (keep apostrophes).
No number normalization. No abbreviation normalization.
"""

import re
import unicodedata

# Punctuation to strip (keep apostrophes for contractions)
_PUNCT_RE = re.compile(r"[^\w\s']")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize text for scoring. Idempotent."""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Strip punctuation (keep apostrophes)
    text = _PUNCT_RE.sub(" ", text)
    # Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text
