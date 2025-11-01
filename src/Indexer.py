from tokenizer import Tokenizer
from DocumentManager import DocumentManager
from Lexicon import Lexicon
from collections import defaultdict
from typing import Optional, Any, Literal, Dict, List, Tuple, Iterable
from functools import partial
from Extras import vbyte_decode, vbyte_encode, delta_decode, delta_encode
import msgpack, os, glob, io, mmap, sys
import zstandard as zstd
import marisa_trie


class Indexer:
    def __init__(self, tokenizer: Optional[Tokenizer] = None, path="index/"):
        self.tokenizer = tokenizer or Tokenizer(use_stemmer=True)
        self.path = path
        self.manager = DocumentManager(path)
        self.posting_list = defaultdict(list)

        # Disk index
        self.postings_path = os.path.join(self.path, "postings.bin") 
        self.postings_fd = None
        self.mm = None

        # Disk lexicon
        self._record_fmt = "<QII" # 8 + 4 + 4 bytes
        self.lexicon_path = os.path.join(self.path, "lexicon.marisa")
        self.trie = None

    def build_index(self, filepaths, bytes_limit=50_000_000):
        """
        SPIMI implementation for reverse indexing, storing term positions. 
        """
        
        # Term to Posting list, which consists of DocID to term positions
        dictionary: Dict[Any, Dict[int, List[int]]] = defaultdict(partial(defaultdict, list))

        block_count = 0
        estimated_bytes = sys.getsizeof(dictionary)

        # Remove all index files
        os.makedirs(self.path, exist_ok=True)
        for zst in glob.glob(os.path.join(self.path, "*.zst")):
            os.remove(zst)

        filestream = self.manager.initialize(filepaths)
        while filepath := next(filestream, None):
            position = 0
            doc_id, input_stream = self.manager.get_id_and_read_doc(filepath)
            tokenstream = self.tokenizer.token_stream_mp(input_stream)

            while token := next(tokenstream, None):
                # Increase estimation
                estimated_bytes += 0 if token in dictionary else sys.getsizeof(token)
                estimated_bytes += sys.getsizeof(position)
                posting_list = dictionary[token]
                posting_list[doc_id].append(position)

                # Flush
                if estimated_bytes >= bytes_limit:
                    if dictionary:
                        self._write_block(dictionary, block_count)
                        block_count += 1
                    dictionary = defaultdict(partial(defaultdict, list))
                    estimated_bytes = sys.getsizeof(dictionary)
                position += 1
            self.manager.set_length(doc_id, position)
        
        if dictionary:
            self._write_block(dictionary, block_count)

        self._merge_blocks()
        self.manager.finalize()

    def get_doc_keys(self) -> List[str]:
        return self.manager.get_keys()

    def get_doc_ids(self) -> List[int]:
        return self.manager.get_values()

    def get_meta(self, term: str):
        """
        Returns (off, length, df) or None.
        """
        res = self.trie.get(term)
        return res[0] if res else None

    def get_postings(self, term: str) -> List[Tuple[int, List[int]]]:
        meta = self.get_meta(term)
        if not meta:
            return []
        off, ln, _df = meta
        raw = self.mm[off:off+ln]
        return self._decode_term_block(raw)

    def iter_prefix(self, prefix: str, limit: int = 50):
        """
        Predictive/prefix lookup for terms beginning with `prefix`.
        """
        count = 0
        for key in self.trie.iterkeys(prefix):
            yield key, self.trie[key][0]
            count += 1
            if count >= limit:
                break

    def load(self):
        """
        Loads previously saved postings and lexicon file.
        """
        self.postings_fd = open(self.postings_path, "rb")
        self.mm = mmap.mmap(self.postings_fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.trie = marisa_trie.RecordTrie(self._record_fmt)
        self.trie.mmap(self.lexicon_path)
        self.manager.load()

    def close(self):
        self.mm.close()
        self.postings_fd.close()

    def _write_block(self, dictionary, block_id: int) -> None:
        """
        Encode block with msgpack term by term and compress with zstandard
        Term by term writing allows for term by term reading
        """
        filepath = os.path.join(self.path, f"{0}-{block_id}.zst")
        with open(filepath, "wb") as f, zstd.ZstdCompressor().stream_writer(f) as zf:
            pack = msgpack.Packer().pack
            # Sort keys
            for term_id in sorted(dictionary.keys()):
                postings = dictionary[term_id]
                # Sorted by doc_id through insertion order
                rec = (term_id, [(doc_id, postings[doc_id]) for doc_id in postings])
                zf.write(pack(rec))

    def _read_block(self, filepath):
        """
        Read a file written in _write_block or _merge_blocks, returning a generator.
        File stream reduces memory usage.
        Yields:
            (term_id, postings), postings is a list of [doc_id, positions]
        """
        with open(filepath, "rb") as f, zstd.ZstdDecompressor().stream_reader(f) as zf:
            unpacker = msgpack.Unpacker(zf)
            for term_id, postings in unpacker:
                yield term_id, postings

    def _merge_postings(self, l_postings, r_postings):
        """
        Merge 2 postings of the same term, concatinating positions of the same document.
        Uses 2-pointer merge
        Returns:
            The merged postings
        """
        l_idx, r_idx = 0, 0
        out = []

        # Sorted by doc_id
        ln, rn = len(l_postings), len(r_postings)
        while l_idx < ln and r_idx < rn:
            l_doc, l_positions = l_postings[l_idx]
            r_doc, r_positions = r_postings[r_idx]

            if l_doc == r_doc:
                # Order of positions not guaranteed, so we sort
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

    def _merge_blocks(self, level=1):
        """
        Merge blocks logarithmically
        """
        filepaths = glob.glob(os.path.join(self.path, "*.zst"))

        if not filepaths:
            return

        if len(filepaths) == 1:
            self._write_posting_bin(self._read_block(filepaths[0]))
            return

        idx = 0
        while idx < len(filepaths) - 1:
            l_filepath = filepaths[idx]
            r_filepath = filepaths[idx + 1]
            l_it = self._read_block(l_filepath)
            r_it = self._read_block(r_filepath)

            l_content = next(l_it, None)
            r_content = next(r_it, None)

            # 2 pointer merge
            outpath = os.path.join(self.path, f"{level}-{idx / 2}.zst")
            with open(outpath, "wb") as f, zstd.ZstdCompressor().stream_writer(f) as zf:
                pack = msgpack.Packer().pack

                # Keys are sorted
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

    def _encode_term_block(self, postings: List[Tuple[int, List[int]]]) -> bytes:
        """
        postings: list of (doc_id, positions[]) with doc_ids sorted, positions sorted
        """
        buf = io.BytesIO()
        # docIDs (delta + vbyte)
        docids = [doc for doc, _ in postings]
        buf.write(vbyte_encode([len(docids)]))
        buf.write(vbyte_encode(list(delta_encode(docids))))

        # positions (per doc: gap-encode then vbyte)
        for _, pos in postings:
            gaps = [pos[0]] + [pos[i]-pos[i-1] for i in range(1, len(pos))]
            buf.write(vbyte_encode([len(gaps)]))
            buf.write(vbyte_encode(gaps))
        return buf.getvalue()

    def _decode_term_block(self, raw: bytes) -> List[Tuple[int, List[int]]]:
        """
        """
        it = iter(vbyte_decode(raw))
        # number of docs
        num_docs = next(it)
        # docids
        docid_deltas = [next(it) for _ in range(num_docs)]
        docids = list(delta_decode(docid_deltas))
        postings = []
        for d in docids:
            plen = next(it)
            gaps = [next(it) for _ in range(plen)]
            # rebuild positions
            pos = [] 
            acc = 0
            for g in gaps:
                acc += g
                pos.append(acc)
            postings.append((d, pos))
        return postings

    def _build_marisa_lexicon(self, items: Iterable[Tuple[str, int, int, int]]):
        """
        items: iterable of (term, off, length, df)
        """
        keys, recs = [], []
        for term, off, length, df in items:
            keys.append(term)
            recs.append((off, length, df))
        self.trie = marisa_trie.RecordTrie(self._record_fmt, zip(keys, recs))
        self.trie.save(self.lexicon_path)
        self.trie.mmap(self.lexicon_path)

    def _write_posting_bin(self, posting_stream):
        """
        spimi_terms_to_postings must be sorted by term; each value is postings list.
        Writes postings and lexicon files
        """
        # Stream-write postings; collect (term, off, len, df) for lexicon
        meta: List[Tuple[str, int, int, int]] = []
        off = 0
        with open(self.postings_path, "wb") as f:
            while x := next(posting_stream, None):
                term, postings = x
                block = self._encode_term_block(postings)
                ln = len(block)
                f.write(block)
                df = len(postings)
                meta.append((term, off, ln, df))
                off += ln

        # Build mmappable lexicon
        self._build_marisa_lexicon(meta)

        # Make querying available
        self.postings_fd = open(self.postings_path, "rb")
        self.mm = mmap.mmap(self.postings_fd.fileno(), 0, access=mmap.ACCESS_READ)

def filestream():
    sample_dir = os.path.join("data", "wikipedia-movies")
    paths = sorted(glob.glob(os.path.join(sample_dir, "*")))
    for path in paths:
        yield path

if __name__ == "__main__":
    indexer = Indexer()

    indexer.build_index(filestream(), bytes_limit=2_000_000_000)
    # indexer._merge_blocks()

    # indexer.load()
    print(indexer.get_meta("000-meter"))
    print(indexer.get_postings("000-meter"))
    # gen = indexer.iter_prefix("ab")
    # while x := next(gen, None):
    #     print(x)
