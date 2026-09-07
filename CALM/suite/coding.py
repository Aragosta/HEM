#!/usr/bin/env python3
"""T6, the part that fits: does bits-per-byte equal the compressed size?

`metrics.bits_per_byte` has been used in this repo as though it were the size of
the file an arithmetic coder would emit. That is the claim compression-as-metric
rests on, and it had never been checked. Checking it needs no training and no
GPU: an arithmetic coder driven by *any* model will do, because the claim is
about the coder and the arithmetic, not about the model's quality.

So the model here is deliberately trivial -- an order-1 (bigram) byte model with
add-one smoothing, fitted on the wikitext2 train split. Using a trained network
would test the same identity more expensively and confound a coder bug with a
model bug.

Three things are checked, and each can fail independently:

1. **Round trip.** Decode must reproduce the input byte for byte. A coder that
   compresses well and decodes wrongly is not a coder.
2. **Rate identity.** Emitted bits / input bytes must equal the model's analytic
   cross-entropy in bits/byte. This is the claim. Agreement to <1% licenses the
   metric; a gap means "bits per byte" is a loss, not a file size, and the suite
   must rename it.
3. **Order sensitivity.** The order-1 model must beat the order-0 model on the
   same text, in both the analytic and the emitted number, and by the same
   margin. This is the control: it shows the pipeline responds to the model
   rather than to the coder's own overhead.

Implementation is Witten-Neal-Cleary with 32-bit registers and underflow
counting. Frequency tables are rescaled so every row totals <= 4096, keeping
`total < 2**(precision-2)` as the algorithm requires; the *same* quantised table
drives the analytic figure, so the two numbers are comparable by construction
rather than by luck.

Usage::

    python CALM/suite/coding.py --bytes 65536
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

DATA = Path(__file__).resolve().parents[1] / "data"

PRECISION = 32
TOP = 1 << PRECISION
HALF = TOP >> 1
QUARTER = TOP >> 2
THREE_QUARTER = 3 * QUARTER
MASK = TOP - 1
ROW_TOTAL = 4096


class Model:
    """Static order-0 or order-1 byte model as cumulative frequency tables."""

    def __init__(self, text: bytes, order: int = 1):
        self.order = order
        rows = 256 if order == 1 else 1
        counts = [[1] * 256 for _ in range(rows)]
        previous = 0
        for byte in text:
            counts[previous if order == 1 else 0][byte] += 1
            previous = byte
        self.cum: List[List[int]] = []
        for row in counts:
            total = sum(row)
            # Rescale to ROW_TOTAL, never letting a symbol reach zero: a symbol
            # with zero probability cannot be coded at all, and the held-out
            # split will contain bytes the train split did not.
            scaled = [max(1, (c * ROW_TOTAL) // total) for c in row]
            cumulative = [0]
            for value in scaled:
                cumulative.append(cumulative[-1] + value)
            self.cum.append(cumulative)

    def row(self, previous: int) -> List[int]:
        return self.cum[previous if self.order == 1 else 0]

    def analytic_bits(self, text: bytes) -> float:
        """Cross-entropy in bits per byte under the same quantised table."""
        bits = 0.0
        previous = 0
        for byte in text:
            cum = self.row(previous)
            probability = (cum[byte + 1] - cum[byte]) / cum[256]
            bits -= math.log2(probability)
            previous = byte
        return bits / len(text)


def encode(model: Model, text: bytes) -> bytes:
    low, high, pending = 0, MASK, 0
    bits: List[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        bits.append(bit)
        bits.extend([1 - bit] * pending)
        pending = 0

    previous = 0
    for byte in text:
        cum = model.row(previous)
        total = cum[256]
        span = high - low + 1
        high = low + (span * cum[byte + 1]) // total - 1
        low = low + (span * cum[byte]) // total
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low = (low << 1) & MASK
            high = ((high << 1) | 1) & MASK
        previous = byte

    pending += 1
    emit(0 if low < QUARTER else 1)
    while len(bits) % 8:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)


def decode(model: Model, blob: bytes, length: int) -> bytes:
    bits = [(b >> (7 - i)) & 1 for b in blob for i in range(8)]
    cursor = 0

    def read() -> int:
        nonlocal cursor
        bit = bits[cursor] if cursor < len(bits) else 0
        cursor += 1
        return bit

    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | read()
    low, high = 0, MASK
    out = bytearray()
    previous = 0
    for _ in range(length):
        cum = model.row(previous)
        total = cum[256]
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        # Linear scan: 256 symbols, and clarity is worth more here than speed.
        symbol = 0
        while cum[symbol + 1] <= target:
            symbol += 1
        out.append(symbol)
        high = low + (span * cum[symbol + 1]) // total - 1
        low = low + (span * cum[symbol]) // total
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER
            else:
                break
            low = (low << 1) & MASK
            high = ((high << 1) | 1) & MASK
            value = ((value << 1) | read()) & MASK
        previous = symbol
    return bytes(out)


def check(order: int, train: bytes, held_out: bytes) -> Dict[str, float]:
    model = Model(train, order=order)
    analytic = model.analytic_bits(held_out)
    blob = encode(model, held_out)
    restored = decode(model, blob, len(held_out))
    emitted = 8 * len(blob) / len(held_out)
    return {
        "order": order,
        "analytic_bpb": analytic,
        "emitted_bpb": emitted,
        "gap_pct": 100 * (emitted - analytic) / analytic,
        "round_trip": restored == held_out,
        "bytes_in": len(held_out),
        "bytes_out": len(blob),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=65536)
    parser.add_argument("--corpus", default="wikitext2")
    options = parser.parse_args()

    train = (DATA / f"{options.corpus}.train.txt").read_bytes()
    held_out = (DATA / f"{options.corpus}.valid.txt").read_bytes()[:options.bytes]

    print("T6  is bits-per-byte the compressed size?")
    print(f"    {options.corpus}: {len(train):,} train bytes, "
          f"{len(held_out):,} held-out bytes coded\n")
    print(f"    {'model':>8s} {'analytic':>10s} {'emitted':>10s} {'gap':>8s} "
          f"{'round trip':>11s}")
    rows = []
    for order in (0, 1):
        row = check(order, train, held_out)
        rows.append(row)
        print(f"    order-{order:<2d} {row['analytic_bpb']:10.4f} "
              f"{row['emitted_bpb']:10.4f} {row['gap_pct']:7.2f}% "
              f"{'EXACT' if row['round_trip'] else 'MISMATCH':>11s}")

    print()
    ok_trip = all(r["round_trip"] for r in rows)
    ok_rate = all(abs(r["gap_pct"]) < 1.0 for r in rows)
    ok_order = (rows[1]["analytic_bpb"] < rows[0]["analytic_bpb"]
                and rows[1]["emitted_bpb"] < rows[0]["emitted_bpb"])
    print(f"    1 round trip     {'PASS' if ok_trip else 'FAIL'}")
    print(f"    2 rate identity  {'PASS' if ok_rate else 'FAIL'} "
          f"(worst gap {max(abs(r['gap_pct']) for r in rows):.2f}%)")
    print(f"    3 order control  {'PASS' if ok_order else 'FAIL'} "
          f"(order-1 saves {rows[0]['analytic_bpb'] - rows[1]['analytic_bpb']:.4f} "
          f"bits/byte analytically, "
          f"{rows[0]['emitted_bpb'] - rows[1]['emitted_bpb']:.4f} emitted)")
    verdict = ("bits-per-byte IS the compressed size; the metric is licensed"
               if ok_trip and ok_rate and ok_order else
               "the identity does not hold here; bits-per-byte is a loss, not a "
               "file size, and must be reported as such")
    print(f"\n    {verdict}")
    return 0 if (ok_trip and ok_rate and ok_order) else 1


if __name__ == "__main__":
    sys.exit(main())
