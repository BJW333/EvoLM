"""
================================================================================
prepare_data.py -- Tokenize raw text into the .bin format that train.py reads.
================================================================================

WHAT THIS DOES:
    Takes a single .txt file (or a directory of .txt files), tokenizes it
    via tokenizer.py (GPT-2 encoder by default), and writes:
        <output-dir>/train.bin  -- np.uint16 array of token IDs
        <output-dir>/val.bin    -- same, for held-out validation

DEPENDENCY:
    pip install tiktoken numpy

USAGE:
    python prepare_data.py --input-file my_corpus.txt --output-dir data/
    python prepare_data.py --input-dir my_text_files/ --output-dir data/ --val-frac 0.05

WHY UINT16?
    The tokenizer's GPT-2 vocab is 50257 tokens. uint16 fits up to 65535. So we
    save half the disk space vs int32. train.py reads back as uint16 then casts
    to int64 for nn.Embedding.

VOCAB / PAD CONVENTIONS:
    See tokenizer.py for the rationale. Briefly: the model uses vocab_size=50258
    (one larger than tiktoken's vocab) and pad_token_id=50257 (a sentinel that
    the tokenizer never emits). train.py's defaults match this.

NOTE ON STREAMING:
    For very large corpora that don't fit in RAM, this script does naive
    full-file reads. If you have multi-GB text, swap the read loop for a
    chunked tokenize-and-append. Out of scope for the default implementation.
"""

import argparse
from pathlib import Path

import numpy as np

from tokenizer import encode


def read_text(input_file: Path = None, input_dir: Path = None) -> str:
    """
    Reads input text from either a single file or a directory of .txt files.
    Concatenates files with a single newline between them (so the tokenizer
    sees them as separate documents).
    """
    if input_file is not None:
        return Path(input_file).read_text(encoding="utf-8")
    if input_dir is not None:
        chunks = []
        for p in sorted(Path(input_dir).rglob("*.txt")):
            chunks.append(p.read_text(encoding="utf-8"))
        if not chunks:
            raise SystemExit(f"No .txt files found under {input_dir}")
        return "\n".join(chunks)
    raise SystemExit("Must provide --input-file or --input-dir")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-file", type=str, help="single .txt file")
    src.add_argument("--input-dir", type=str, help="directory of .txt files (recursive)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val-frac", type=float, default=0.005,
                        help="fraction of tokens reserved for validation")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="cap total tokens (useful for quick smoke tests)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read raw text ----
    print("[prepare] reading text...")
    text = read_text(
        input_file=Path(args.input_file) if args.input_file else None,
        input_dir=Path(args.input_dir) if args.input_dir else None,
    )
    print(f"[prepare] {len(text):,} chars")

    # ---- Tokenize ----
    print("[prepare] tokenizing...")
    ids = encode(text)
    print(f"[prepare] {len(ids):,} tokens")

    if args.max_tokens is not None:
        ids = ids[: args.max_tokens]
        print(f"[prepare] truncated to {len(ids):,} tokens")

    ids_np = np.array(ids, dtype=np.uint16)

    # Sanity check: any token id >= 65535 means uint16 won't hold it.
    # tiktoken's gpt2 encoding maxes at 50256; this should never trip.
    if ids_np.max() > np.iinfo(np.uint16).max:
        raise SystemExit("Token ID exceeded uint16 range -- pick a wider dtype.")

    # ---- Train / val split ----
    n_val = max(1, int(len(ids_np) * args.val_frac))
    n_train = len(ids_np) - n_val
    train_ids = ids_np[:n_train]
    val_ids = ids_np[n_train:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"[prepare] wrote {train_path} ({len(train_ids):,} tokens)")
    print(f"[prepare] wrote {val_path}   ({len(val_ids):,} tokens)")
    print(f"[prepare] use vocab_size=50258, pad_token_id=50257 in train.py")


if __name__ == "__main__":
    main()
