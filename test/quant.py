#!/usr/bin/env python3
"""
Cross-check the C++ GGUF embedding-row dequant against an independent
Python implementation, using the model's real tokenizer.

Generates test-case data for a whole sentence: for each token, prints
the first N, last N, and several evenly-spaced middle windows of the
dequantized embedding row, so you can spot-check your C++ output across
the full width of the vector (not just the first few dims).

    pip install gguf llama-cpp-python
"""

import struct
import numpy as np
from gguf import GGUFReader

MODEL_PATH = r"C:\Users\zezo_\.lmstudio\models\lmstudio-community\Qwen2.5-3B-Instruct-GGUF\Qwen2.5-3B-Instruct-Q8_0.gguf"

# The sentence to tokenize and embed. Pick something with a mix of
# common words, a leading-space edge case, punctuation, etc.
SENTENCE = "héllo wörld 你好世界 😀🚀"

N_EDGE = 16          # how many dims to show at start/end
N_MID_WINDOWS = 3    # how many windows to sample from the middle
N_MID_WIDTH = 8       # how many dims per middle window


def get_tokens(model_path: str, sentence: str):
    """Tokenizes with the model's own tokenizer via llama.cpp bindings,
    so ids/boundaries match a real forward pass exactly."""
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
    ids = llm.tokenize(sentence.encode("utf-8"), add_bos=False)

    # Recover the piece text for each id for readability/debugging.
    pieces = []
    for tid in ids:
        piece = llm.detokenize([tid]).decode("utf-8", errors="replace")
        pieces.append(piece)

    return ids, pieces


def dequant_q8_0_row(reader: GGUFReader, tensor, token_id: int) -> np.ndarray:
    """Independent reimplementation of embed_token(), straight off the
    raw GGUF bytes -- deliberately not reusing llama.cpp dequant code."""
    n_embd = tensor.shape[0]
    assert n_embd % 32 == 0, "q8_0 rows must be a multiple of 32"
    n_blocks_per_row = n_embd // 32
    block_size = 2 + 32  # fp16 scale + 32 int8 values

    raw = bytes(tensor.data)
    row_offset = token_id * n_blocks_per_row * block_size
    row_bytes = raw[row_offset: row_offset + n_blocks_per_row * block_size]

    out = np.empty(n_embd, dtype=np.float32)
    for b in range(n_blocks_per_row):
        block = row_bytes[b * block_size: (b + 1) * block_size]
        scale = struct.unpack("<e", block[0:2])[0]
        qs = np.frombuffer(block[2:], dtype=np.int8)
        out[b * 32: (b + 1) * 32] = qs.astype(np.float32) * scale

    return out


def middle_window_starts(n_embd: int, n_windows: int, width: int):
    """Evenly-spaced starting indices for middle sample windows,
    avoiding the edge regions already covered by N_EDGE."""
    lo, hi = N_EDGE, n_embd - N_EDGE - width
    if n_windows == 1:
        return [(lo + hi) // 2]
    step = (hi - lo) / (n_windows - 1)
    return [int(round(lo + i * step)) for i in range(n_windows)]


def main():
    reader = GGUFReader(MODEL_PATH)
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    n_embd = tensor.shape[0]

    ids, pieces = get_tokens(MODEL_PATH, SENTENCE)
    print(f"Sentence: {SENTENCE!r}")
    print(f"Tokens ({len(ids)}): {list(zip(ids, pieces))}\n")

    mid_starts = middle_window_starts(n_embd, N_MID_WINDOWS, N_MID_WIDTH)

    for tid, piece in zip(ids, pieces):
        row = dequant_q8_0_row(reader, tensor, tid)

        print("=" * 70)
        print(f"token_id = {tid}   piece = {piece!r}")
        print(f"first {N_EDGE} dims [0:{N_EDGE}]:")
        print(np.array2string(row[:N_EDGE], precision=6, floatmode="fixed"))

        for start in mid_starts:
            end = start + N_MID_WIDTH
            print(f"middle dims [{start}:{end}]:")
            print(np.array2string(row[start:end], precision=6, floatmode="fixed"))

        print(f"last {N_EDGE} dims [{n_embd - N_EDGE}:{n_embd}]:")
        print(np.array2string(row[-N_EDGE:], precision=6, floatmode="fixed"))
        print()


if __name__ == "__main__":
    main()


# --- fallback if llama-cpp-python won't build --------------------------------
# Pull tokenizer.ggml.tokens yourself and match pieces by hand:
#
#   reader = GGUFReader(MODEL_PATH)
#   tokens = reader.fields["tokenizer.ggml.tokens"].parts
#   for candidate in ["queen", " queen", "Ġqueen", "▁queen"]:
#       for i, t in enumerate(tokens):
#           if t.tobytes().decode("utf-8", errors="ignore") == candidate:
#               print(candidate, "->", i)
#
# Whichever candidate appears tells you this model's leading-space
# convention -- build your id list from that instead of get_tokens().