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
from queue import Queue
from threading import Thread
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

# Worker for multi processing
_worker_tokenizer = None

def _init_worker(stopwords_list, use_stemmer, nltk_dir):
    # Build a Tokenizer in each process without redownloading resources.
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer(
        custom_stopwords=stopwords_list,
        use_stemmer=use_stemmer,
        nltk_dir=nltk_dir,
        nltk_download=False
    )

def _tokenize_batch(texts):
    # texts: list[str]
    return [ _worker_tokenizer.tokenize(t) for t in texts if t is not None ]

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
    def __init__(self, custom_stopwords: Optional[Iterable[str]] = None, use_stemmer: bool = True, 
                 nltk_dir="nltk_data", nltk_download: bool = True):

        # nltk stopwords and stemmer
        if nltk_download:
            os.makedirs(nltk_dir, exist_ok=True)
            nltk.data.path.append(nltk_dir)
            nltk.download("stopwords", download_dir=nltk_dir)
            nltk.download("punkt", download_dir=nltk_dir)
            self.nltk_dir = nltk_dir
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

    def token_stream_mp(self, stream: Iterable[str], workers: Optional[int] = None, batch_size: int = 512, prefetch_batches: int = 8):
        """
        Yield tokens using multiple processes.
        - stream: an iterator/generator of strings
        - workers: number of processes (defaults to os.cpu_count())
        - batch_size: how many texts per task sent to a worker
        - prefetch_batches: how many batches to keep in-flight
        """
        # Helper: turn the (possibly infinite) stream into fixed-size lists
        # Shipping one short string per process call is dominated by IPC overhead. Batching amortizes that.
        # Or so the AI says
        def batched(it, n):
            it = iter(it)
            while True:
                chunk = list(islice(it, n))
                if not chunk:
                    return
                yield chunk

        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(self.stopwords, self.use_stemmer, self.nltk_dir)) as ex:

            # Submit batches and keep a small queue of futures to avoid unbounded memory
            in_flight = []
            batches = batched(stream, batch_size)

            # Prime the pump
            for _ in range(prefetch_batches):
                try:
                    b = next(batches)
                except StopIteration:
                    break
                in_flight.append(ex.submit(_tokenize_batch, b))

            while in_flight:
                # Pop the oldest future to preserve order
                fut = in_flight.pop(0)
                token_lists = fut.result()  # [[tokens for doc 1], [tokens for doc 2], ...]
                # Immediately submit the next batch (if any)
                if b := next(batches, None):
                    in_flight.append(ex.submit(_tokenize_batch, b))

                # Stream results to the caller
                for tokens in token_lists:
                    for tok in tokens:
                        yield tok


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