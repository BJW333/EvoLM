"""
================================================================================
tokenizer.py -- Shared tokenizer wrapping (tiktoken's GPT-2 encoder).
================================================================================

Both prepare_data.py and generate.py go through this module, so they're
guaranteed to use the same vocab and id mapping. If you ever swap to a
different tokenizer (BPE-trained on your own corpus, sentencepiece, etc.),
this is the single place to change it.

VOCAB CONVENTION:
    Tiktoken's GPT-2 encoding has 50257 tokens (ids 0..50256).
    We add ONE extra reserved id at 50257 to act as the pad token, giving
    the model an effective vocab_size of 50258. The tokenizer never emits
    50257, so we can use it as a sentinel that's safely outside the data
    distribution. The model's embedding has row 50257 zeroed (via
    nn.Embedding's padding_idx).

FUNCTIONS:
    get_tokenizer()       -- returns the cached tiktoken encoder
    encode(text)          -- text -> list[int]
    decode(token_ids)     -- list[int] -> text
    VOCAB_SIZE = 50258
    PAD_TOKEN_ID = 50257
"""

from typing import List, Iterable, Union

# Constants used elsewhere (train.py defaults match these).
VOCAB_SIZE = 50258
PAD_TOKEN_ID = 50257
EOT_TOKEN_ID = 50256          # tiktoken's <|endoftext|> -- used as eos in generation


# Lazily-instantiated singleton so we don't pay tokenizer load cost just
# from importing this module.
_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import tiktoken
        except ImportError:
            raise SystemExit(
                "tiktoken not installed. Run: pip install tiktoken"
            )
        _TOKENIZER = tiktoken.get_encoding("gpt2")
    return _TOKENIZER


def encode(text: str) -> List[int]:
    """Text -> token ids. Uses encode_ordinary so special-token strings in
    the text don't get auto-promoted to special ids."""
    return get_tokenizer().encode_ordinary(text)


def decode(token_ids: Union[Iterable[int], "torch.Tensor"]) -> str:
    """Token ids -> text. Filters out the pad sentinel (50257) since
    tiktoken's decoder doesn't know about it."""
    # Accept torch tensors / numpy arrays as well as plain lists.
    try:
        ids_list = token_ids.tolist()
    except AttributeError:
        ids_list = list(token_ids)
    # Drop the pad sentinel; tiktoken would error on it.
    ids_list = [i for i in ids_list if i != PAD_TOKEN_ID]
    return get_tokenizer().decode(ids_list)
