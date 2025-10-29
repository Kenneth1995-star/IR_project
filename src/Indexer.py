from tokenizer import Tokenizer
from DocumentManager import DocumentManager
from Lexicon import Lexicon
from collections import defaultdict
from typing import Optional, Any
from functools import partial
import msgpack, os, glob
import zstandard as zstd


class Indexer:
    def __init__(self, tokenizer: Optional[Tokenizer] = None, path="index/"):
        self.tokenizer = tokenizer or Tokenizer()
        self.manager = DocumentManager()
        self.posting_list = defaultdict(list)
        self.lexicon: Lexicon = Lexicon()
        self.path = path
        self.index_file = os.path.join(self.path, "index")

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

    def build_index(self, filestream, posting_limit=5_000_000):
        """
        BSBI implementation for reverse indexing
        """
        block_count = 0
        posting_count = 0

        dictionary: dict[int, dict[int, list[int]]] = defaultdict(partial(defaultdict, list))

        os.makedirs(self.path, exist_ok=True)
        for zst in glob.glob(os.path.join(self.path, "*.zst")):
            os.remove(zst)

        while filepath := next(filestream, None):
            doc_id, content = self.manager.read_document(filepath)

            for position, token in enumerate(self.tokenizer.tokenize(content)):
                term_id = self.lexicon.get_id(token)
                posting_list = dictionary[term_id]
                posting_list[doc_id].append(position)
                posting_count += 1

                if posting_count >= posting_limit:
                    if dictionary:
                        self._write_block(dictionary, block_count)
                        block_count += 1
                    posting_count = 0
                    dictionary = defaultdict(partial(defaultdict, list))
        
        if dictionary:
            self._write_block(dictionary, block_count)

        self._merge_blocks()

    def _write_block(self, dictionary, block_id: int) -> None:
        """
        Encode block with msgpack term by term and compress with zstandard
        """
        filepath = os.path.join(self.path, f"block{block_id}.zst")
        with open(filepath, "wb") as f, zstd.ZstdCompressor().stream_writer(f) as zf:
            pack = msgpack.Packer().pack
            for term_id in sorted(dictionary.keys()):
                postings = dictionary[term_id]
                rec = (term_id, [(doc_id, postings[doc_id]) for doc_id in sorted(postings)])
                zf.write(pack(rec))

    def _read_block(self, filepath):
        """
        Read a file written in _write_block, returning a generator
        Returns:
            (term_id, postings), postings is a list of [doc_id, positions]
        """
        with open(filepath, "rb") as f, zstd.ZstdDecompressor().stream_reader(f) as zf:
            unpacker = msgpack.Unpacker(zf)
            for term_id, postings in unpacker:
                yield term_id, postings

    def _merge_postings(self, l_postings, r_postings):
        """
        Merge 2 postings of the same term, concatinating positions of the same document
        Returns:
            The merged postings
        """
        l_idx, r_idx = 0, 0
        out = []

        ln, rn = len(l_postings), len(r_postings)
        while l_idx < ln and r_idx < rn:
            l_doc, l_positions = l_postings[l_idx]
            r_doc, r_positions = r_postings[r_idx]

            if l_doc == r_doc:
                out.append([l_doc, sorted(l_positions + r_positions)])
                l_idx += 1
                r_idx += 1
            
            elif l_doc < r_doc:
                out.append([l_doc, l_positions])
                l_idx += 1

            else:
                out.append([r_doc, r_positions])
                r_idx += 1

        if l_idx < ln:
            out.extend(l_postings[l_idx:])
        
        if r_idx < rn:
            out.extend(r_postings[r_idx:])

        return out

    def _merge_blocks(self, level=0):
        """
        Merge blocks logarithmically
        """
        filepaths = sorted(glob.glob(os.path.join(self.path, "*.zst")))

        if not filepaths:
            return

        if len(filepaths) == 1:
            os.rename(filepaths[0], self.index_file)
            return
        idx = 0
        while idx < len(filepaths) - 1:
            l_filepath = filepaths[idx]
            r_filepath = filepaths[idx + 1]
            l_it = self._read_block(l_filepath)
            r_it = self._read_block(r_filepath)

            l_content = next(l_it, None)
            r_content = next(r_it, None)

            outpath = os.path.join(self.path, f"{level}-{idx}.zst")
            with open(outpath, "wb") as f, zstd.ZstdCompressor().stream_writer(f) as zf:
                pack = msgpack.Packer().pack

                while l_content and r_content:
                    l_id, l_postings = l_content
                    r_id, r_postings = r_content

                    if l_id == r_id:
                        zf.write(pack((l_id, self._merge_postings(l_postings, r_postings))))
                        l_content = next(l_it, None)
                        r_content = next(r_it, None)
                    
                    elif l_id < r_id:
                        zf.write(pack(l_content))
                        l_content = next(l_it, None) 
                    
                    else:
                        zf.write(pack(r_content))
                        r_content = next(r_it, None)
                
                while l_content:
                    zf.write(pack(l_content))
                    l_content = next(l_it, None)
                    
                while r_content:
                    zf.write(pack(r_content))
                    r_content = next(r_it, None)

            os.remove(l_filepath)
            os.remove(r_filepath)
            idx += 2
        self._merge_blocks(level + 1)

def filestream():
    sample_dir = os.path.join("data", "wikipedia-movies")
    paths = sorted(glob.glob(os.path.join(sample_dir, "*")))
    for path in paths:
        yield path

if __name__ == "__main__":
    indexer = Indexer()

    indexer.build_index(filestream())
    gen = indexer._read_block(indexer.index_file)
    while x := next(gen, None):
        print(x)
