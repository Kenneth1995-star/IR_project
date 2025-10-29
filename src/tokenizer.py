# src/tokenizer.py
"""
Tokenizer module.

Using nltk, strings are tokenized and stemmed for use in
the Information Retrieval system.

This module also preprocesses the input by removing URLs and HTML.
"""

import os, re, nltk
from typing import List, Iterable, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Tokenizer:
    """
    The Tokenizer class.

    Here we are defining a clean and reusable tokenizer object.

    Responsibilities:
      - Normalize and tokenize text.
      - Optionally stem words (reduce to root form).
      - Exclude numbers and stopwords.
      - Remove URLs and HTML

    Parameters:
        stopwords : Optional[Iterable[str]]
            A custom list of stopwords (if provided, overrides defaults).
        use_stemmer : bool
            Whether to apply stemming using Porter Stemmer.
    """
    def __init__(self, custom_stopwords: Optional[Iterable[str]] = None, use_stemmer: bool = True, nltk_dir="nltk_data"):

        # nltk stopwords and stemmer
        os.makedirs(nltk_dir, exist_ok=True)
        nltk.data.path.append(nltk_dir)
        nltk.download("stopwords", download_dir=nltk_dir, quiet=True)
        nltk.download("punkt", download_dir=nltk_dir, quiet=True)
        self.stopwords = set(custom_stopwords) if custom_stopwords is not None else stopwords.words("english")

        # Detect tokens using regex patterns.
        # Tokens consists of letters, digits and hyphens
        self._token_re = re.compile(r"[A-Za-z0-9]+(?:[-][A-Za-z0-9]+)*", flags=re.UNICODE)

        # Detect URLs
        self._url_re = re.compile(r"""(?xi)
            \b(
                (?:https?://|ftp://|file://|mailto:|www\.)
                [^\s<>'"]+
            )
        """)

        # Detect HTML
        self._html_re = re.compile(r"<[^>]+>")

        # Enable or disable stemming functionality
        self.use_stemmer = use_stemmer
        if self.use_stemmer:
            from nltk.stem.porter import PorterStemmer
            self.stemmer = PorterStemmer()


    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize an input string into normalized tokens.

        Here we are performing the full normalization pipeline step-by-step:
          1. Convert text to lowercase (for case-insensitivity)
          2. Tokenize using nltk word_tokenize
          3. Filter out:
               - Stopwords
               - HTML blocks
               - URLs
               - 1 character tokens
          4. Apply stemming if requested

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

        text = text.lower()

        # HTML
        text = self._html_re.sub(" ", text)
    
        # Remove urls starting with <protocol>://<url> or www.<url>
        text = self._url_re.sub(" ", text)

        tokens = []
        for token in self._token_re.findall(text):
            if len(token) == 1:
                continue

            if token.isdigit():
                continue

            if token in self.stopwords:
                continue

            # Apply stemming if enabled.
            if self.use_stemmer:
                token = self.stemmer.stem(token)

            tokens.append(token)

        return tokens

    def token_stream(self, stream):
        """
        Tokenize, but with i/o stream
        """
        while text := next(stream, None):
            for token in self.tokenize(text):
                yield token


if __name__ == "__main__":
    import glob
    def filestream():
        sample_dir = os.path.join("data", "wikipedia-movies")
        paths = sorted(glob.glob(os.path.join(sample_dir, "*")))
        for path in paths:
            yield path

    tokenizer = Tokenizer()
    import DocumentManager
    gen = tokenizer.token_stream(DocumentManager.DocumentManager().read_document_stream(filestream()))
    while x := next(gen, None):
        print(x)