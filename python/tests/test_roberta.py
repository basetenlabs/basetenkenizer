# Copyright 2026 Baseten
# All rights reserved.
#
# This file is proprietary to Baseten and is not licensed under the
# Apache License, Version 2.0.

import json
import unittest

from tokenizers import Tokenizer as ReferenceTokenizer
from tokenizers.pre_tokenizers import ByteLevel

from basetenkenizer._native import Tokenizer


def roberta_config(prefix_space=False):
    vocab = {
        token: index
        for index, token in enumerate(
            ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
            + sorted(ByteLevel.alphabet())
        )
    }
    added = []
    for content, special, lstrip in [
        ("<mask>", True, True),
        ("    ", False, False),
        ("Ġ literal", False, False),
        ("Ġ👩🏽‍💻", False, False),
    ]:
        token_id = vocab.setdefault(content, len(vocab))
        added.append(
            dict(id=token_id, content=content, single_word=False,
                 lstrip=lstrip, rstrip=False, normalized=False, special=special)
        )
    return dict(
        version="1.0", added_tokens=added,
        pre_tokenizer=dict(type="ByteLevel", add_prefix_space=prefix_space,
                           trim_offsets=True, use_regex=True),
        post_processor=dict(type="RobertaProcessing", sep=["</s>", 2],
                            cls=["<s>", 0], trim_offsets=True,
                            add_prefix_space=prefix_space),
        decoder=dict(type="ByteLevel", add_prefix_space=True,
                     trim_offsets=True, use_regex=True),
        model=dict(type="BPE", vocab=vocab, merges=[]),
    )


class RobertaTests(unittest.TestCase):
    def test_single_and_batch_encoding_matches_hugging_face(self):
        texts = ["", "Hello world!", " café 数学 🫠", "hello   <mask> world",
                 "\n\t\u00a0<mask>", "<mask> <mask>", "x    y",
                 "Ġ literal", "Ġ👩🏽‍💻"]
        for prefix in [False, True]:
            for legacy in [False, True]:
                config = roberta_config(prefix)
                if legacy:
                    del config["model"]["type"]
                raw = json.dumps(config)
                tokenizer = Tokenizer.from_json_str(raw)
                reference = ReferenceTokenizer.from_str(raw)
                for split in [False, True]:
                    reference.encode_special_tokens = split
                    for special in [False, True]:
                        expected = reference.encode_batch(texts, add_special_tokens=special)
                        actual = tokenizer.encode_batch(
                            texts, add_special_tokens=special, split_special_tokens=split,
                        )
                        for text, encoding, wanted in zip(texts, actual, expected):
                            with self.subTest(text=text, prefix=prefix, legacy=legacy,
                                              special=special, split=split):
                                self.assertEqual(encoding.ids, wanted.ids)
                                self.assertEqual(encoding.attention_mask, wanted.attention_mask)
                                self.assertEqual(encoding.type_ids, wanted.type_ids)
                                self.assertEqual(tokenizer.encode(
                                    text, add_special_tokens=special,
                                    split_special_tokens=split,
                                ).ids, wanted.ids)
                                for skip in [False, True]:
                                    self.assertEqual(
                                        tokenizer.decode(encoding.ids, skip_special_tokens=skip),
                                        reference.decode(wanted.ids, skip_special_tokens=skip),
                                    )

    def test_batch_padding_matches_hugging_face(self):
        raw = json.dumps(roberta_config())
        tokenizer = Tokenizer.from_json_str(raw)
        reference = ReferenceTokenizer.from_str(raw)
        for direction in ["left", "right"]:
            options = dict(direction=direction, pad_id=1, pad_token="<pad>",
                           pad_type_id=0, pad_to_multiple_of=8)
            tokenizer.enable_padding(**options)
            reference.enable_padding(**options)
            for actual, expected in zip(
                tokenizer.encode_batch(["", "hello <mask> world"], add_special_tokens=True),
                reference.encode_batch(["", "hello <mask> world"], add_special_tokens=True),
            ):
                self.assertEqual(actual.ids, expected.ids)
                self.assertEqual(actual.attention_mask, expected.attention_mask)
                self.assertEqual(actual.type_ids, expected.type_ids)


if __name__ == "__main__":
    unittest.main()
