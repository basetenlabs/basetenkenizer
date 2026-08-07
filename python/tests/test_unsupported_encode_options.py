# Copyright 2026 Baseten
# All rights reserved.
#
# This file is proprietary to Baseten and is not licensed under the
# Apache License, Version 2.0.

import pickle

import pytest

from basetenkenizer._compat import _TokenizerShim
from test_async_stub import TOKENIZER_JSON


HELLO_TOKEN_ID = 11


def test_encode_accepts_split_special_tokens_true_as_noop_without_added_tokens() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    plain = tokenizer.encode("hello")
    split = tokenizer.encode("hello", split_special_tokens=True)

    assert split.ids == plain.ids


def test_encode_batch_accepts_split_special_tokens_true_as_noop_without_added_tokens() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    plain = tokenizer.encode_batch(["hello"])
    split = tokenizer.encode_batch(["hello"], split_special_tokens=True)

    assert split[0].ids == plain[0].ids


def test_encode_accepts_split_special_tokens_false_as_noop() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    plain = tokenizer.encode("hello")
    explicit_false = tokenizer.encode("hello", split_special_tokens=False)

    assert explicit_false.ids == plain.ids


def test_encode_rejects_unknown_kwargs() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    with pytest.raises(TypeError, match="return_offsets_mapping"):
        tokenizer.encode("hello", return_offsets_mapping=True)


def test_encode_special_tokens_true_is_supported() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    tokenizer.encode_special_tokens = True

    assert tokenizer.encode_special_tokens is True
    assert tokenizer.encode("hello").ids == [HELLO_TOKEN_ID]


def test_encode_special_tokens_false_is_noop() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    tokenizer.encode_special_tokens = False

    assert tokenizer.encode_special_tokens is False


def test_pickle_restores_stored_encode_special_tokens_true() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)
    state = (
        tokenizer.to_str(),
        tokenizer.truncation,
        tokenizer.padding,
        True,
    )

    payload = pickle.dumps(state)
    restored = _TokenizerShim.from_str(TOKENIZER_JSON)

    restored.__setstate__(pickle.loads(payload))

    assert restored.encode_special_tokens is True
