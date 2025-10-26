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

# ------------------------------------------------------------
# Here we are defining a small but representative STOPWORD list.
# These are common English words that typically don't help in retrieval,
# since they occur in nearly all documents.
# ------------------------------------------------------------
_DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "not", "in", "on", "at", "for", "of", "to",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "as", "by", "with", "from", "but", "if", "then", "else",
    "when", "while", "which", "who", "whom", "what", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "about", "into", "through"
}

# ------------------------------------------------------------
# Here we are trying to import an optional stemmer (PorterStemmer).
# If available, it helps to reduce words to their "root" form.
# Example: "running", "runs", "ran" -> "run"
# If not available, we simply skip this functionality gracefully.
# ------------------------------------------------------------
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
        # Here we are setting the stopwords.
        # If the user provides a custom list, we convert it to a set.
        # Otherwise, we use the default list defined above.
        self.stopwords = set(stopwords) if stopwords is not None else _DEFAULT_STOPWORDS

        # Here we are compiling a regular expression pattern to detect tokens.
        # The pattern \b[0-9A-Za-z']+\b means:
        #   - \b: word boundary
        #   - [0-9A-Za-z']+: one or more alphanumeric characters or apostrophes
        #   - \b: word boundary
        self._token_re = re.compile(r"\b[0-9A-Za-z']+\b", flags=re.UNICODE)

        # Here we are enabling or disabling stemming functionality.
        self.use_stemmer = use_stemmer
        if self.use_stemmer:
            # If stemming requested but nltk is missing, raise an explicit error.
            if not _HAS_PORTER:
                raise ImportError("PorterStemmer requested but nltk not installed.")
            # Otherwise, instantiate the stemmer.
            self.stemmer = PorterStemmer()

    # ------------------------------------------------------------
    # Core method: tokenize(text)
    # ------------------------------------------------------------
    def tokenize(self, text: Optional[str]) -> List[str]:
        """
        Tokenize an input string into normalized tokens.

        Here we are performing the full normalization pipeline step-by-step:
          1. Convert text to lowercase (for case-insensitivity)
          2. Extract all words using the regex
          3. Filter out:
               - Numeric-only tokens
               - Stopwords
          4. Apply stemming if requested
          5. Return the final clean list of tokens

        Parameters
        ----------
        text : Optional[str]
            The raw text (document body or query string).

        Returns
        -------
        List[str]
            A list of cleaned, normalized tokens ready for indexing or querying.
        """
        # Step 1: handle None input safely.
        if text is None:
            return []

        # Step 2: convert to lowercase to make search case-insensitive.
        text = str(text).lower()

        # Step 3: find all tokens using our regex pattern.
        raw = self._token_re.findall(text)

        # Step 4: initialize an empty list for valid tokens.
        tokens = []

        # Step 5: iterate through each candidate token and clean it.
        for t in raw:
            # Skip if the token is purely numeric (e.g. "1234").
            if t.isdigit():
                continue

            # Skip if token is a stopword.
            if t in self.stopwords:
                continue

            # Apply stemming if enabled.
            if self.use_stemmer:
                t = self.stemmer.stem(t)

            # Keep the cleaned token.
            tokens.append(t)

        # Step 6: return the fully processed token list.
        return tokens
