""" Offline payload-conflict cleaning (Option C1).

In packet-level classification where the model only sees the 1,500 payload bytes, identical or template-identical
payload byte vectors appearing under multiple traffic labels act as contradictory label noise. Specifically, when
the Payload-Byte tool matched CIC-IDS2017 Wednesday PCAP packets to flows using the 5-tuple without timestamps,
ephemeral port reuse across the four HTTP DoS attacks mixed randomized GET/POST templates across classes.

Rule (C1):
  1. Compute a canonical payload key for each 1,500-byte row (normalizing the randomized HTTP query/body tokens
     of Hulk/GoldenEye/Slowhttptest templates, and MD5-hashing exact bytes otherwise).
  2. Count label frequencies per canonical key across the dataset.
  3. Keep a row iff its label is the strict majority label for that key (ties are dropped).
  4. Within-class duplicates are preserved (e.g. DDoS consists of a single payload repeated ~10,000 times).

Note / Limitation for results.md:
  Because C1 runs across the full sample before known/unknown partitioning (matching the paper's offline
  deduplication stage in Section 3.1.1), it uses the labels of all classes during offline cleaning.
"""
import hashlib
import re
from collections import Counter, defaultdict
from typing import Dict, Tuple

import numpy as np

_HULK_RE = re.compile(
    rb"^GET /\?[A-Z]+=[A-Z]+ HTTP/1\.1\r\nAccept-Encoding: identity\r\nHost: 205\.174\.165\.68\r\nKeep-Alive: \d+\r\n"
)


def _canonical_payload_key(b: bytes) -> bytes:
    """Return a canonical key for conflict resolution across the 1,500-byte payload."""
    if _HULK_RE.match(b[:150]):
        return b"__TPL_HULK_GET_UPPER_IDENTITY__"
    if b.startswith(b"GET /?"):
        return b"__TPL_GOLDENEYE_GET_MIXED__"
    if b.startswith(b"&") and b"=" in b[:30]:
        return b"__TPL_SLOWHTTPTEST_BODY_PARAM__"
    return hashlib.md5(b).digest()


def resolve_payload_conflicts(X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """Drop rows whose canonical 1,500-byte payload key appears under another label with >= count.
    Within-class duplicates of the strict-majority label are kept.
    Returns (X_kept, labels_kept, stats_dict) where stats_dict[label] = (before_count, after_count)."""
    if len(X) != len(labels):
        raise ValueError(f"X ({len(X)}) and labels ({len(labels)}) must have the same length")

    X_contig = np.ascontiguousarray(X)
    hashes = [_canonical_payload_key(row.tobytes()) for row in X_contig]

    cnt: Dict[bytes, Counter] = defaultdict(Counter)
    for h, lab in zip(hashes, labels):
        cnt[h][lab] += 1

    top2 = {h: c.most_common(2) for h, c in cnt.items()}
    keep = np.fromiter(
        (t[0][0] == lab and (len(t) == 1 or t[0][1] > t[1][1]) for t, lab in ((top2[h], lab) for h, lab in zip(hashes, labels))),
        dtype=bool,
        count=len(labels),
    )

    stats = {
        str(name): (int((labels == name).sum()), int((labels[keep] == name).sum()))
        for name in sorted(set(labels))
    }
    return X_contig[keep], labels[keep], stats


def print_dedup_stats(stats: Dict[str, Tuple[int, int]]) -> None:
    before_total = sum(b for b, _ in stats.values())
    after_total = sum(a for _, a in stats.values())
    print(f"dedup (C1 strict-majority payload cleaning): {before_total} -> {after_total} rows "
          f"(removed {before_total - after_total})")
    for name, (b, a) in stats.items():
        print(f"  {name:<30s} {b:6d} -> {a:6d}")
