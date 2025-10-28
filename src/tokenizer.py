# src/tokenizer.py
"""
Tokenizer module.

Here we are building the very first stage of the Information Retrieval (IR) pipeline:
    -> converting raw document text or queries into clean, normalized tokens.

Our tokenizer ensures:
  - Lowercasing
  - Keeping only alphanumeric words and apostrophes (e.g. "don't")
  - Removing pure digits (like 2020)
  - Removing common stopwords (words that add little meaning, like "the", "of", "is")
  - Optional word stemming using Porter Stemmer if requested

This module will be reused both in indexing (during document parsing)
and in query processing (when we tokenize user queries).

Design principles:
  - Lightweight and easily extensible
  - Deterministic output (always same tokens for same text)
  - Fully self-contained (does not depend on heavy NLP libraries unless optional)
"""

import re
from typing import List, Iterable, Optional


_DEFAULT_STOPWORDS = {
    "an", "the", "and", "or", "not", "in", "on", "at", "for", "of", "to",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "as", "by", "with", "from", "but", "if", "then", "else",
    "when", "while", "which", "who", "whom", "what", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "about", "into", "through"
}

# Import stemming if available
try:
    from nltk.stem.porter import PorterStemmer  # type: ignore
    _HAS_PORTER = True
except Exception:
    PorterStemmer = None
    _HAS_PORTER = False


class Tokenizer:
    """
    The Tokenizer class.

    Here we are defining a clean and reusable tokenizer object.

    Responsibilities:
      - Normalize and tokenize text.
      - Optionally stem words (reduce to root form).
      - Exclude numbers and stopwords.
    
    Typical usage in the IR pipeline:
      1. The Indexer uses Tokenizer to process document text.
      2. The QueryProcessor uses the same Tokenizer for query input.
      -> This guarantees consistent tokenization across indexing and querying.

    Parameters:
        stopwords : Optional[Iterable[str]]
            A custom list of stopwords (if provided, overrides defaults).
        use_stemmer : bool
            Whether to apply stemming using Porter Stemmer.
    """
    def __init__(self, stopwords: Optional[Iterable[str]] = None, use_stemmer: bool = False):
        self.stopwords = set(stopwords) if stopwords is not None else _DEFAULT_STOPWORDS

        # Detect tokens using regex patterns.
        # Tokens consists of letters, digits, apostrophes and hyphens
        self._token_re = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", flags=re.UNICODE)

        # Enable or disable stemming functionality
        self.use_stemmer = use_stemmer
        if self.use_stemmer:
            if not _HAS_PORTER:
                raise ImportError("PorterStemmer requested but nltk not installed.")
            self.stemmer = PorterStemmer()


    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize an input string into normalized tokens.

        Here we are performing the full normalization pipeline step-by-step:
          1. Convert text to lowercase (for case-insensitivity)
          2. Extract all words using the regex
          3. Filter out:
               - Numeric-only tokens
               - Stopwords
               - HTML blocks
               - URLs
               - 1 character tokens
          4. Apply stemming if requested
          5. Return the final clean list of tokens

        Parameters
        ----------
        text : str
            The raw text (document body or query string).

        Returns
        -------
        List[str]
            A list of cleaned, normalized tokens ready for indexing or querying.
        """
        if text is None:
            return []

        text = str(text).lower()
        text = re.compile(r"<[^>]+>").sub(" ", text) # html
    
        # remove urls starting with <protocol>://<url> or www.<url>
        text = re.compile(r"""(?xi)
            \b(
                (?:https?://|ftp://|file://|mailto:|www\.)
                [^\s<>'"]+
            )
        """).sub(" ", text)

        raw = self._token_re.findall(text)

        tokens = []
        for t in raw:
            if len(t) == 1:
                continue

            if t.isdigit():
                continue

            if t in self.stopwords:
                continue

            # Apply stemming if enabled.
            if self.use_stemmer:
                t = self.stemmer.stem(t)

            tokens.append(t)

        return tokens
