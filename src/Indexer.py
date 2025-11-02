import atexit
import glob
import io
import math
import mmap
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import marisa_trie
import msgpack
import numpy as np
import zstandard as zstd

from src.DocumentManager import DocumentManager
from src.Extras import delta_decode, delta_encode, vbyte_decode, vbyte_encode
from src.Tokenizer import Tokenizer


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
        self._record_fmt = "<QII"  # 8 + 4 + 4 bytes
        self.lexicon_path = os.path.join(self.path, "lexicon.marisa")
        self.trie = None

        # smart scheme offset map
        self.smart_offset_map = {
            scheme: i
            for i, scheme in enumerate(
                a + b + "c" for b in "tnp" for a in "lnabL"
            )
        }
        atexit.register(self.close)

    def build_index(self, filepaths, bytes_limit=50_000_000):
        """
        SPIMI implementation for reverse indexing, storing term positions.
        """

        # Term to Posting list, which consists of DocID to term positions
        dictionary: Dict[Any, Dict[int, List[int]]] = defaultdict(
            partial(defaultdict, list)
        )

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
                estimated_bytes += (
                    0 if token in dictionary else sys.getsizeof(token)
                )
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
        self.manager.finalize(*self._compute_stats())

    # --- Getters --- #

    def N(self) -> int:
        return self.manager.N

    def get_doc_keys(self) -> List[str]:
        return self.manager.get_keys()

    def get_doc_ids(self) -> List[int]:
        return self.manager.get_values()

    def doc_mean(self) -> float:
        return self.manager.mean

    def get_doc_length(self, doc_id) -> int:
        return self.manager.lengths[doc_id]

    def get_max_tf(self, doc_id) -> int:
        return self.manager.max_tf[doc_id]

    def get_avg_tf(self, doc_id) -> float:
        return self.manager.avg_tf[doc_id]

    def get_unique_terms(self, doc_id: int) -> int:
        return self.manager.unique_terms[doc_id]

    def get_avg_unique(self) -> float:
        return np.mean(self.manager.unique_terms)

    def get_byte_length(self, doc_id: int) -> int:
        return self.manager.get_byte_length(doc_id)

    def get_norm(self, doc_id: int, scheme: str) -> float:
        return self.manager.norms[self.smart_offset_map[scheme]][doc_id]

    def get_meta(self, term: str):
        """
        Returns (offset, length, df) or None.
        For retrieving postings, tf or df
        """
        res = self.trie.get(term)
        return res[0] if res else None

    def get_postings(self, term: str) -> List[Tuple[int, List[int]]]:
        """
        Load postings from disk
        """
        meta = self.get_meta(term)
        if not meta:
            return []
        off, ln, _df = meta
        raw = self.mm[off:off + ln]
        return self._decode_term_block(raw)

    def get_tfs(self, term: str) -> List[Tuple[int, int]]:
        """
        Load term frequencies without reading the whole postings
        """
        meta = self.get_meta(term)
        if not meta:
            return []
        off, ln, _df = meta
        raw = self.mm[off:off + ln]
        return self._decode_term_block(raw, False)

    def iter_prefix(self, prefix: str, limit: int = 50):
        """
        Predictive/prefix lookup for terms beginning with `prefix`.
        Used with wildcard queries, but is not implemented.
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
        self.close()
        self.postings_fd = open(self.postings_path, "rb")
        self.mm = mmap.mmap(
            self.postings_fd.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.trie = marisa_trie.RecordTrie(self._record_fmt)
        self.trie.mmap(self.lexicon_path)
        self.manager.load()

    def close(self):
        """
        Close open files
        """
        if self.mm:
            self.mm.close()
            self.mm = None
        if self.postings_fd:
            self.postings_fd.close()
            self.postings_fd = None

    def _compute_stats(self):
        """
        Compute all document norms, lengths, averages, etc.
        Useful as the index is static.
        """
        # Traverse all terms twice: for max tf and for computing the stats
        N = self.N()
        max_tf = np.zeros(N, dtype=np.uint32)
        sum_tf = np.zeros(N, dtype=np.float32)
        unique_terms = np.zeros(N, dtype=np.uint32)
        for term in self.trie.keys():
            for doc_id, tf in self.get_tfs(term):
                if tf > max_tf[doc_id]:
                    max_tf[doc_id] = tf
                sum_tf[doc_id] += tf
                unique_terms[doc_id] += 1

        avg_tf = np.zeros(self.N(), dtype=np.float32)
        nonzero_mask = unique_terms > 0
        avg_tf[nonzero_mask] = (
            sum_tf[nonzero_mask] / unique_terms[nonzero_mask]
        )

        # Calculate all variations of cosine norm
        # It's fine because index is static
        # lnabL - tnp - c, so 0: ltc, 14:Lpc
        norms = np.zeros((15, N), dtype=np.float32)
        for term, meta in self.trie.iteritems():
            df = meta[2]
            if df <= 0 or df >= N:
                continue
            idf_t = math.log10(N / df)
            idf_p = max(0.0, math.log10((N - df) / df))
            for doc_id, tf in self.get_tfs(term):
                tf_l = 1.0 + math.log10(tf)
                tf_a = 0.5 + 0.5 * tf / max_tf[doc_id]
                tf_b = 1.0 if tf > 0 else 0.0
                tf_L = (1.0 + math.log10(tf)) / (
                    1.0 + math.log10(avg_tf[doc_id])
                )
                norms[0][doc_id] += (tf_l * idf_t) ** 2
                norms[1][doc_id] += (tf * idf_t) ** 2
                norms[2][doc_id] += (tf_a * idf_t) ** 2
                norms[3][doc_id] += (tf_b * idf_t) ** 2
                norms[4][doc_id] += (tf_L * idf_t) ** 2
                norms[5][doc_id] += (tf_l) ** 2
                norms[6][doc_id] += (tf) ** 2
                norms[7][doc_id] += (tf_a) ** 2
                norms[8][doc_id] += (tf_b) ** 2
                norms[9][doc_id] += (tf_L) ** 2
                norms[10][doc_id] += (tf_l * idf_p) ** 2
                norms[11][doc_id] += (tf * idf_p) ** 2
                norms[12][doc_id] += (tf_a * idf_p) ** 2
                norms[13][doc_id] += (tf_b * idf_p) ** 2
                norms[14][doc_id] += (tf_L * idf_p) ** 2

        norms = 1 / np.sqrt(norms)
        return max_tf, unique_terms, avg_tf, norms

    def _write_block(self, dictionary, block_id: int) -> None:
        """
        Encode block with msgpack term by term and compress with zstandard
        Term by term writing allows for term by term reading
        """
        filepath = os.path.join(self.path, f"{0}-{block_id}.zst")
        with open(filepath, "wb") as f, zstd.ZstdCompressor().stream_writer(
            f
        ) as zf:
            pack = msgpack.Packer().pack
            # Sort keys
            for term_id in sorted(dictionary.keys()):
                postings = dictionary[term_id]
                # Sorted by doc_id through insertion order
                rec = (
                    term_id,
                    [(doc_id, postings[doc_id]) for doc_id in postings],
                )
                zf.write(pack(rec))

    def _read_block(self, filepath):
        """
        Read a file written in _write_block or _merge_blocks,
        returning a generator. File stream reduces memory usage.
        Yields:
            (term_id, postings), postings is a list of [doc_id, positions]
        """
        with open(filepath, "rb") as f, zstd.ZstdDecompressor().stream_reader(
            f
        ) as zf:
            unpacker = msgpack.Unpacker(zf)
            for term_id, postings in unpacker:
                yield term_id, postings

    def _merge_postings(self, l_postings, r_postings):
        """
        Merge 2 postings of the same term, concatinating positions of the same
        document. Uses 2-pointer merge
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
            with open(outpath, "wb") as f, zstd.ZstdCompressor().stream_writer(
                f
            ) as zf:
                pack = msgpack.Packer().pack

                # Keys are sorted
                while l_content and r_content:
                    l_id, l_postings = l_content
                    r_id, r_postings = r_content

                    if l_id == r_id:
                        zf.write(
                            pack(
                                (
                                    l_id,
                                    self._merge_postings(
                                        l_postings, r_postings
                                    ),
                                )
                            )
                        )
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

    def _encode_term_block(
        self, postings: List[Tuple[int, List[int]]]
    ) -> bytes:
        """
        postings: list of (doc_id, positions[]) with doc_ids sorted, positions
        sorted. Stores docIDs and TFs upfront so scoring can skip the
        positions blob.
        """
        buf = io.BytesIO()

        docids = [doc for doc, _ in postings]
        tfs = [len(pos) for _, pos in postings]

        buf.write(vbyte_encode([len(docids)]))
        buf.write(vbyte_encode(list(delta_encode(docids))))
        buf.write(vbyte_encode(tfs))

        for _, pos in postings:
            # gap-encode positions
            gaps = [pos[0]] + [pos[i] - pos[i - 1] for i in range(1, len(pos))]
            buf.write(vbyte_encode(gaps))

        return buf.getvalue()

    def _decode_term_block(self, raw: bytes, include_positions: bool = True):
        """
        If include_positions=True (default), returns
        List[(doc_id, positions[])]. If include_positions=False,
        returns List[(doc_id, tf)] and skips positions decoding.
        """
        it = iter(vbyte_decode(raw))
        num_docs = next(it)

        # docIDs
        docid_deltas = [next(it) for _ in range(num_docs)]
        docids = list(delta_decode(docid_deltas))

        # tfs
        tfs = [next(it) for _ in range(num_docs)]

        if not include_positions:
            return zip(docids, tfs)

        # Decode positions using tfs counts
        postings = []
        for d, tf in zip(docids, tfs):
            gaps = [next(it) for _ in range(tf)]
            # rebuild absolute positions
            acc = 0
            pos = []
            for g in gaps:
                acc += g
                pos.append(acc)
            postings.append((d, pos))

        return postings

    def _build_marisa_lexicon(
        self, items: Iterable[Tuple[str, Tuple[int, int, int]]]
    ):
        """
        Builds the lexicon with byte offset, and df.
        items: iterable of (term, off, length, df)
        """
        self.trie = marisa_trie.RecordTrie(self._record_fmt, items)
        self.trie.save(self.lexicon_path)
        self.trie.mmap(self.lexicon_path)

    def _write_posting_bin(self, posting_stream):
        """
        spimi_terms_to_postings must be sorted by term; each value is postings
        list. Writes postings and lexicon files
        """
        # Stream-write postings; collect (term, off, len, df) for lexicon
        meta: List[Tuple[str, Tuple[int, int, int]]] = []
        off = 0
        with open(self.postings_path, "wb") as f:
            while x := next(posting_stream, None):
                term, postings = x
                block = self._encode_term_block(postings)
                ln = len(block)
                f.write(block)
                df = len(postings)
                meta.append((term, (off, ln, df)))
                off += ln

        # Build mmappable lexicon
        self._build_marisa_lexicon(meta)

        # Make querying available
        self.postings_fd = open(self.postings_path, "rb")
        self.mm = mmap.mmap(
            self.postings_fd.fileno(), 0, access=mmap.ACCESS_READ
        )
