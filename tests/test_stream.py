"""Unit tests for StreamCollector and stream rule (Plan Week 4.1 / P2)."""
import numpy as np
import torch

import pipeline as pl
import skeleton as sk


def test_stream_collector_run_3_discarded_run_4_kept():
    col = pl.StreamCollector(min_len=4)
    # Run of 3 -> discarded when a non-unknown arrives
    for i in range(3):
        assert col.push(i, True) is None
    assert col.current_len == 3
    assert col.push(99, False) is None
    assert col.current_len == 0
    assert col.discarded_runs == 1
    assert col.discarded_packets == 3
    assert col.kept_runs == 0

    # Run of 4 -> emitted when a non-unknown arrives
    for i in range(10, 14):
        assert col.push(i, True) is None
    run = col.push(100, False)
    assert run == [10, 11, 12, 13]
    assert col.kept_runs == 1
    assert col.discarded_runs == 1


def test_stream_collector_flush_at_end_of_stream():
    col = pl.StreamCollector(min_len=4)
    # Short run at end of stream -> discarded on flush
    for i in range(2):
        col.push(i, True)
    assert col.flush() is None
    assert col.discarded_runs == 1
    assert col.discarded_packets == 2

    # Valid run at end of stream -> returned on flush
    for i in range(5):
        col.push(i, True)
    run = col.flush()
    assert run == [0, 1, 2, 3, 4]
    assert col.kept_runs == 1


def test_stream_collector_interleaved_runs():
    col = pl.StreamCollector(min_len=4)
    flags = [
        True, True, False,                          # 2 -> discard
        True, True, True, True, False,              # 4 -> keep
        False, True, True, True, False,             # 3 -> discard
        True, True, True, True, True, True          # 6 at end -> keep on flush
    ]
    runs = []
    for idx, f in enumerate(flags):
        r = col.push(idx, f)
        if r is not None:
            runs.append(r)
    r = col.flush()
    if r is not None:
        runs.append(r)

    assert len(runs) == 2
    assert runs[0] == [3, 4, 5, 6]
    assert runs[1] == [13, 14, 15, 16, 17, 18]
    assert col.kept_runs == 2
    assert col.discarded_runs == 2
    assert col.discarded_packets == 5
