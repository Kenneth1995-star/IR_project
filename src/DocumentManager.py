import numpy as np
import marisa_trie
import os
import json


class DocumentManager:
    def __init__(self, path="index/"):
        self.path = path
        self.ids_path = os.path.join(path, "docids.marisa")
        self.ints_path = os.path.join(path, "ints.npy")
        self.stats_path = os.path.join(path, "stats.json")
        self.floats_path = os.path.join(path, "floats.npy")
        self.norms_path = os.path.join(path, "norms.npy")

    def initialize(self, filepaths):
        """
        Initialize mapping and return generator for files
        """
        # Assign IDs
        trie = marisa_trie.Trie(filepaths)
        trie.save(self.ids_path)
        self.N = len(trie)
        self.lengths = np.zeros(self.N, dtype=np.uint32)
        self.path_to_id = trie.mmap(self.ids_path) # mmap
        for key in self.path_to_id.keys():
            yield key

    def finalize(self, max_tf, unique_terms, floats, norms):
        self.mean = np.mean(self.lengths, dtype=np.float64)

        # 0: lengths, 1: max_tf, 2: unique_terms
        np.save(self.ints_path, np.stack((self.lengths, max_tf, unique_terms), dtype=np.uint32))

        # avg_tf
        np.save(self.floats_path, floats)

        # norms. 0: ltc, 14: Lpc [lnabL - tnp - c]
        np.save(self.norms_path, norms)

        stats = {
            "N": self.N, 
            "mean": self.mean
        }
        with open(self.stats_path, "w", encoding="utf8") as f:
            json.dump(stats, f, ensure_ascii=False)

        self.load()

    def load(self):
        self.path_to_id = marisa_trie.Trie().mmap(self.ids_path)

        ints =  np.load(self.ints_path, mmap_mode="r")
        self.lengths = ints[0]
        self.max_tf = ints[1]
        self.unique_terms = ints[2]

        self.avg_tf = np.load(self.floats_path, mmap_mode="r")

        self.norms = np.load(self.norms_path, mmap_mode="r")

        with open(self.stats_path, "r", encoding="utf8") as lf:
            stats = json.load(lf)
        self.N = stats["N"]
        self.mean = stats["mean"]
<<<<<<< HEAD
=======

    def get_keys(self) -> List[str]:
        return self.path_to_id.keys()

    def get_key(self, doc_id: int) -> str:
        return self.path_to_id.restore_key(doc_id)

    def get_values(self) -> List[int]:
        """
        Get all Doc ids in order
        """
        return list(range(self.N))
>>>>>>> ba6744e4
    
    def get_id(self, key) -> int:
        return self.path_to_id.get(key)

    def get_id_and_read_doc(self, filepath):
        return self.get_id(filepath), self.read_document_stream(filepath)

    def set_length(self, doc_id: int, length: int) -> None:
        self.lengths[doc_id] = length

    def get_length(self, doc_id: int) -> int:
        return self.lengths[doc_id]

    def get_byte_length(self, doc_id: int) -> int:
        with open(self.get_key(doc_id), "rb") as f:
            f.seek(0, 2)
            return f.tell()
    
    def get_average_length(self) -> int:
        return self.mean

    def read_document_stream(self, filepath):
        with open(filepath, "r") as f:
            for line in f:
                yield line