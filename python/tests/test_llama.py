# Copyright 2026 fastokens contributors
# Copyright 2026 Baseten
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Python loader against the real offline Llama 2 fixture."""

from pathlib import Path

import pytest
import tokenizers

from basetenkenizer import Tokenizer

FIXTURE = Path(__file__).resolve().parents[2] / "tests/fixtures/llama-2-7b-chat/tokenizer.json"


@pytest.mark.parametrize("add_special", [False, True])
def test_llama2_python_parity(add_special):
    native = Tokenizer.from_file(str(FIXTURE))
    reference = tokenizers.Tokenizer.from_file(str(FIXTURE))
    texts = ["", " hello ", "café 数学 🫠", "\0\U0010ffff", "<s>[INST] Hello! [/INST]Sure!</s>"]
    for text in texts:
        expected = reference.encode(text, add_special_tokens=add_special)
        actual = native.encode(text, add_special_tokens=add_special)
        assert actual.ids == expected.ids
        assert actual.type_ids == expected.type_ids
        assert actual.attention_mask == expected.attention_mask
        for skip in [False, True]:
            assert native.decode(actual.ids, skip_special_tokens=skip) == reference.decode(expected.ids, skip_special_tokens=skip)
    actual_batch = native.encode_batch(texts, add_special_tokens=add_special)
    assert [encoding.ids for encoding in actual_batch] == [encoding.ids for encoding in reference.encode_batch(texts, add_special_tokens=add_special)]
