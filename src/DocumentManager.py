import numpy as np
import marisa_trie
import os
import json
from typing import List


class DocumentManager:
    def __init__(self, path="index/"):
        self.path = path
        self.ids_path = os.path.join(path, "docids.marisa")
        self.lengths_path = os.path.join(path, "doclengths.npy")
        self.stats_path = os.path.join(path, "stats.json")

    def initialize(self, filepaths):
        """
        Initialize mapping and return generator for files
        """
        # Assign IDs
        # Marisa Tries do not have the fastest lookups, O(|key|)
        # but they are very simple to use and mmap.
        # Also supports reverse lookups, so no need to store reverse list
        # For fully in-memory implementations, a normal dict will be better.
        trie = marisa_trie.Trie(filepaths)
        trie.save(self.ids_path)
        self.N = len(trie)
        self.lengths = np.zeros(self.N, dtype=np.uint32)
        self.path_to_id = trie.mmap(self.ids_path) # mmap
        for key in self.path_to_id.keys():
            yield key

    def finalize(self):
        self.mean = np.mean(self.lengths, dtype=np.float64)
        np.save(self.lengths_path, self.lengths_path)
        self.lengths = np.load(self.lengths_path, mmap_mode="r")

        with open(self.stats_path, "w", encoding="utf8") as f:
            json.dump({"N": self.N, "mean": self.mean}, f, ensure_ascii=False)

    def load(self):
        self.path_to_id = marisa_trie.Trie().mmap(self.ids_path)
        self.lengths = np.load(self.lengths_path, mmap_mode="r")

        with open(self.stats_path, "r", encoding="utf8") as lf:
            stats = json.load(lf)
        self.N = stats["N"]
        self.mean = stats["mean"]

    def get_keys(self) -> List[str]:
        return self.path_to_id.keys()

    def get_key(self, doc_id: int) -> str:
        return self.path_to_id.restore_key(doc_id)

    def get_values(self) -> List[int]:
        """
        This only works because the index is static.
        """
        return list(range(self.N))
    
    def get_id(self, key) -> int:
        return self.path_to_id.get(key)

    def get_id_and_read_doc(self, filepath):
        return self.get_id(filepath), self.read_document_stream(filepath)

    def set_length(self, doc_id: int, length: int) -> None:
        self.lengths[doc_id] = length

    def get_length(self, doc_id: int) -> int:
        return self.lengths[doc_id]
    
    def get_average_length(self) -> int:
        return self.mean

    def read_document_stream(self, filepath):
        with open(filepath, "r") as f:
            for line in f:
                yield line