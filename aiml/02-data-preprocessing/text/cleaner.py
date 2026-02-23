"""
Text cleaning utilities.

Covers:
- Unicode normalization
- Whitespace / punctuation normalization
- HTML and URL stripping
- Stop-word & short token removal
- Contraction expansion
- Lemmatization (spaCy)
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import List, Optional

# Optional heavy deps — lazy-imported to not force install in every context
try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
}


class TextCleaner:
    """
    Configurable text cleaning pipeline.

    Example:
        cleaner = TextCleaner(remove_stopwords=True, lemmatize=True)
        clean = cleaner("Check out https://example.com — it's amazing!!")
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = False,
        expand_contractions: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        min_token_length: int = 1,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length
        self._nlp = None
        self._spacy_model = spacy_model

    def _get_nlp(self):
        if self._nlp is None:
            if not _SPACY_AVAILABLE:
                raise ImportError(
                    "spaCy is required: pip install spacy && python -m spacy download en_core_web_sm"
                )
            self._nlp = spacy.load(self._spacy_model, exclude=["parser", "ner"])
        return self._nlp

    # ── Individual cleaning steps ─────────────────────────────────────

    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def strip_html(text: str) -> str:
        text = _HTML_TAG_RE.sub(" ", text)
        return html.unescape(text)

    @staticmethod
    def strip_urls(text: str) -> str:
        return _URL_RE.sub(" ", text)

    @staticmethod
    def expand_contractions_text(text: str) -> str:
        for contraction, expansion in _CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text).strip()

    # ── Main pipeline ─────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        text = self.normalize_unicode(text)
        if self.remove_html:
            text = self.strip_html(text)
        if self.remove_urls:
            text = self.strip_urls(text)
        if self.expand_contractions:
            text = self.expand_contractions_text(text)
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = _PUNCT_RE.sub(" ", text)
        text = self.normalize_whitespace(text)

        if self.remove_stopwords or self.lemmatize:
            nlp = self._get_nlp()
            doc = nlp(text)
            tokens = []
            for tok in doc:
                if self.remove_stopwords and tok.is_stop:
                    continue
                if len(tok.text) < self.min_token_length:
                    continue
                tokens.append(tok.lemma_ if self.lemmatize else tok.text)
            text = " ".join(tokens)

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    def __call__(self, text: str) -> str:
        return self.clean(text)
