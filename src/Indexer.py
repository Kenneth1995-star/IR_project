from tokenizer import Tokenizer
from DocumentManager import DocumentManager
from collections import defaultdict
from typing import Optional


class Indexer:
    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.manager = DocumentManager()
        self.posting_list = defaultdict(list)

    def build_index(self, filestream):
        while filepath := next(filestream, None):
            doc_id, content = self.manager.read_document(filepath)

            temp = defaultdict(lambda: {"frequency": 0, "positions": []})
            for position, token in enumerate(self.tokenizer.tokenize(content)):
                temp[token]["frequency"] += 1
                temp[token]["positions"].append(position)

            for token, data in temp.items():
                self.posting_list[token].append({
                    "doc_id": doc_id,
                    "frequency": data["frequency"],
                    "positions": data["positions"]
                })


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
