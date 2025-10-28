from tokenizer import Tokenizer
from DocumentManager import DocumentManager
from Vocabulary import Vocabulary
from collections import defaultdict
from typing import Optional
from functools import partial


class Indexer:
    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.manager = DocumentManager()
        self.posting_list = defaultdict(list)

    def build_index_in_memory(self, filestream):
        """
        In-memory posting list building with term frequency and positions
        """
        while filepath := next(filestream, None):
            doc_id, content = self.manager.read_document(filepath)

            temp = defaultdict(list)
            for position, token in enumerate(self.tokenizer.tokenize(content)):
                temp[token].append(position)

            for token, data in temp.items():
                self.posting_list[token].append({
                    "doc_id": doc_id,
                    "frequency": len(data),
                    "positions": data
                })


    def build_index(self, filestream, posting_limit=1_000_000):
        """
        SPIMI implementation
        """
        block_count = 0
        posting_count = 0

        dictionary: dict[int, dict[int, list[int]]] = defaultdict(partial(defaultdict, list))
        vocabulary: Vocabulary = Vocabulary()

        while filepath := next(filestream, None):
            doc_id, content = self.manager.read_document(filepath)

            for position, token in enumerate(self.tokenizer.tokenize(content)):
                term_id = vocabulary.get_id(token)
                posting_list = dictionary[term_id]
                posting_list[doc_id].append(position)
                posting_count += 1

                if posting_count >= posting_limit:
                    self._writeblock(dictionary, vocabulary)
                    block_count += 1
                    posting_count = 0
                    dictionary = defaultdict(partial(defaultdict, list))
                    vocabulary = Vocabulary()
        
        self._writeblock(dictionary, vocabulary)

    def _writeblock(self, dictinary, vocabulary: Vocabulary):
        pass



def filestream():
    import os, glob

    sample_dir = os.path.join("data", "sample_docs")
    paths = sorted(glob.glob(os.path.join(sample_dir, "*.txt")))
    for path in paths:
        yield path

if __name__ == "__main__":
    indexer = Indexer()

    indexer.build_index(filestream())

    print(indexer.posting_list)
