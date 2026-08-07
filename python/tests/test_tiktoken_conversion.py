# Copyright 2026 Baseten
# All rights reserved.
#
# This file is proprietary to Baseten and is not licensed under the
# Apache License, Version 2.0.

import base64
import builtins
from collections import Counter
import gzip
import json
import logging
from pathlib import Path
import sys
import types

import pytest

from basetenkenizer import (
    Tokenizer,
    tiktoken_model_to_tokenizer_json,
    tiktoken_to_tokenizer_json,
)

VENDORED_DIR = Path(__file__).parents[2] / "vendored_tokenizers"


@pytest.fixture
def hide_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_tiktoken(name, *args, **kwargs):
        if name == "tiktoken" or name.startswith("tiktoken."):
            raise ImportError("No module named 'tiktoken'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tiktoken)


def _create_test_encoding():
    tiktoken = pytest.importorskip("tiktoken")
    return tiktoken.Encoding(
        name="basetenkenizer-test",
        pat_str=r"(?s).+",
        mergeable_ranks={
            b"a": 0,
            b"b": 1,
            b"c": 2,
            b"d": 3,
            b"ab": 4,
            b"cd": 5,
            b"abcd": 6,
            b" ": 7,
            b"!": 8,
            b"c!": 9,
        },
        special_tokens={"<|end|>": 100},
    )


def _write_tiktoken_model(path: Path, encoding) -> None:
    path.write_text(
        "\n".join(
            f"{base64.b64encode(token).decode()} {rank}"
            for token, rank in encoding._mergeable_ranks.items()
        )
    )


def _merge_counter(merges: list[list[str]]) -> Counter[tuple[str, str]]:
    return Counter(tuple(merge) for merge in merges)


def _special_token_map(added_tokens: list[dict[str, object]]) -> dict[str, int]:
    return {t["content"]: t["id"] for t in added_tokens if t.get("special")}


def test_tiktoken_to_tokenizer_json_matches_encoding() -> None:
    encoding = _create_test_encoding()
    tokenizer = Tokenizer.from_json_str(tiktoken_to_tokenizer_json(encoding))

    texts = [
        "abcd",
        "ab cd",
        "abc!",
    ]
    for text in texts:
        assert tokenizer.encode(text, add_special_tokens=False).ids == encoding.encode(
            text
        )


def test_tiktoken_to_tokenizer_json_with_encoding_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    encoding = _create_test_encoding()
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)

    tokenizer_json = tiktoken_to_tokenizer_json("basetenkenizer-test")
    tokenizer = Tokenizer.from_json_str(tokenizer_json)

    assert tokenizer.encode("abcd", add_special_tokens=False).ids == [6]


def test_tiktoken_to_tokenizer_json_preserves_special_tokens() -> None:
    encoding = _create_test_encoding()
    tokenizer_json = tiktoken_to_tokenizer_json(encoding)
    config = json.loads(tokenizer_json)
    special_token = "<|end|>"
    special_id = 100

    assert {
        "id": special_id,
        "content": special_token,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    } in config["added_tokens"]

    tokenizer = Tokenizer.from_json_str(tokenizer_json)
    assert tokenizer.encode(special_token, add_special_tokens=False).ids == [special_id]


def test_tiktoken_to_tokenizer_json_returns_none_without_optional_tiktoken(
    hide_tiktoken,
) -> None:
    assert tiktoken_to_tokenizer_json("cl100k_base") is None


def test_tiktoken_model_to_tokenizer_json_matches_model_file(tmp_path) -> None:
    pytest.importorskip("tiktoken")
    encoding = _create_test_encoding()
    model_path = tmp_path / "tiktoken.model"
    _write_tiktoken_model(model_path, encoding)

    tokenizer_json = tiktoken_model_to_tokenizer_json(
        model_path,
        pattern=encoding._pat_str,
        special_tokens={"<|end|>": 100},
    )
    tokenizer = Tokenizer.from_json_str(tokenizer_json)

    for text in ["abcd", "ab cd", "abc!"]:
        assert tokenizer.encode(text, add_special_tokens=False).ids == encoding.encode(
            text
        )
    assert tokenizer.encode("<|end|>", add_special_tokens=False).ids == [100]


def test_tiktoken_model_to_tokenizer_json_reads_model_directory(tmp_path) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    encoding = _create_test_encoding()
    _write_tiktoken_model(tmp_path / "tiktoken.model", encoding)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pat_str": encoding._pat_str,
                "added_tokens_decoder": {
                    "100": {
                        "content": "<|end|>",
                        "special": True,
                    }
                },
            }
        )
    )

    tokenizer_json = tiktoken_model_to_tokenizer_json(tmp_path)
    tokenizer = Tokenizer.from_json_str(tokenizer_json)

    assert tokenizer.encode("abcd", add_special_tokens=False).ids == [6]
    assert tokenizer.encode("<|end|>", add_special_tokens=False).ids == [100]


def test_tiktoken_model_to_tokenizer_json_expands_sparse_kimi_added_tokens(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mergeable_ranks = {b"a": 0, b"b": 1, b"c": 2, b"d": 3}
    tiktoken_module = types.ModuleType("tiktoken")
    tiktoken_load_module = types.ModuleType("tiktoken.load")
    tiktoken_load_module.load_tiktoken_bpe = lambda path: mergeable_ranks
    tiktoken_module.load = tiktoken_load_module
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken_module)
    monkeypatch.setitem(sys.modules, "tiktoken.load", tiktoken_load_module)

    (tmp_path / "tiktoken.model").write_text("")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {
                    "AutoTokenizer": "tokenization_kimi.TikTokenTokenizer",
                },
                "added_tokens_decoder": {
                    "6": {
                        "content": "<|tool_call_begin|>",
                        "special": False,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                    },
                    "8": {
                        "content": "[PAD]",
                        "special": True,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                    },
                },
            }
        )
    )

    tokenizer_json = tiktoken_model_to_tokenizer_json(tmp_path)
    config = json.loads(tokenizer_json)
    added = config["added_tokens"]

    assert [token["id"] for token in added] == [4, 5, 6, 7, 8]
    assert [token["content"] for token in added] == [
        "<|reserved_token_4|>",
        "<|reserved_token_5|>",
        "<|tool_call_begin|>",
        "<|reserved_token_7|>",
        "[PAD]",
    ]
    assert [token["special"] for token in added] == [True, True, False, True, True]


def test_kimi_k2_5_config_conversion_preserves_reserved_added_token_range(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = VENDORED_DIR / "kimi-k2.5" / "tokenizer_config.json"
    if not config_path.exists():
        pytest.skip("vendored kimi-k2.5 tokenizer_config.json not found")

    class FakeMergeableRanks(dict):
        def __len__(self) -> int:
            return 163584

    mergeable_ranks = FakeMergeableRanks({b"a": 0, b"b": 1, b"c": 2, b"d": 3})
    tiktoken_module = types.ModuleType("tiktoken")
    tiktoken_load_module = types.ModuleType("tiktoken.load")
    tiktoken_load_module.load_tiktoken_bpe = lambda path: mergeable_ranks
    tiktoken_module.load = tiktoken_load_module
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken_module)
    monkeypatch.setitem(sys.modules, "tiktoken.load", tiktoken_load_module)

    (tmp_path / "tiktoken.model").write_text("")
    (tmp_path / "tokenizer_config.json").write_text(config_path.read_text())

    tokenizer_json = tiktoken_model_to_tokenizer_json(tmp_path)
    config = json.loads(tokenizer_json)
    added_by_id = {token["id"]: token for token in config["added_tokens"]}
    vendored = json.loads((VENDORED_DIR / "kimi-k2.5" / "tokenizer.json").read_text())
    vendored_pattern = vendored["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]

    assert list(added_by_id) == list(range(163584, 163840))
    assert added_by_id[163589] == {
        "id": 163589,
        "content": "<|reserved_token_163589|>",
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }
    assert added_by_id[163595]["content"] == "<|tool_calls_section_begin|>"
    assert added_by_id[163595]["special"] is False
    assert added_by_id[163606]["content"] == "<think>"
    assert added_by_id[163606]["special"] is False
    assert added_by_id[163838]["content"] == "[UNK]"
    assert added_by_id[163839]["content"] == "[PAD]"
    assert (
        config["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        == vendored_pattern
    )


def test_kimi_k2_5_conversion_encode_matches_hf_tokenizers(tmp_path) -> None:
    pytest.importorskip("tiktoken")
    hf_tokenizers = pytest.importorskip("tokenizers")

    kimi_dir = VENDORED_DIR / "kimi-k2.5"
    gz_path = kimi_dir / "tiktoken.model.gz"
    config_path = kimi_dir / "tokenizer_config.json"
    if not gz_path.exists() or not config_path.exists():
        pytest.skip("vendored kimi-k2.5 assets not found")

    model_path = tmp_path / "tiktoken.model"
    model_path.write_bytes(gzip.decompress(gz_path.read_bytes()))
    (tmp_path / "tokenizer_config.json").write_text(config_path.read_text())

    tokenizer_json = tiktoken_model_to_tokenizer_json(tmp_path)
    assert tokenizer_json is not None, "tiktoken_model_to_tokenizer_json returned None"

    tokenizer_config = json.loads(tokenizer_json)
    tokenizer_config.pop("decoder", None)
    tokenizer_json = json.dumps(tokenizer_config, ensure_ascii=False)
    basetenkenizer_tok = Tokenizer.from_json_str(tokenizer_json)
    hf_tok = hf_tokenizers.Tokenizer.from_str(tokenizer_json)

    corpus = [
        "[BOS]",
        "<|im_user|>",
        "<|tool_calls_section_begin|>",
        "<think>",
        "<|reserved_token_163589|>",
        "<|im_user|>user<|im_middle|>Hello<|im_end|>",
    ]
    for text in corpus:
        basetenkenizer_ids = basetenkenizer_tok.encode(text, add_special_tokens=False).ids
        hf_ids = hf_tok.encode(text, add_special_tokens=False).ids
        assert basetenkenizer_ids == hf_ids, (
            f"encoding mismatch for {text!r}:\n"
            f"  basetenkenizer : {basetenkenizer_ids}\n"
            f"  tokenizers: {hf_ids}"
        )


def test_kimi_k3_conversion_encode_matches_hf_tokenizers(tmp_path) -> None:
    pytest.importorskip("tiktoken")
    hf_tokenizers = pytest.importorskip("tokenizers")

    kimi_dir = VENDORED_DIR / "kimi-k3"
    gz_path = kimi_dir / "tiktoken.model.gz"
    config_path = kimi_dir / "tokenizer_config.json"
    if not gz_path.exists() or not config_path.exists():
        pytest.skip("vendored kimi-k3 assets not found")

    model_path = tmp_path / "tiktoken.model"
    model_path.write_bytes(gzip.decompress(gz_path.read_bytes()))
    (tmp_path / "tokenizer_config.json").write_text(config_path.read_text())

    tokenizer_json = tiktoken_model_to_tokenizer_json(tmp_path)
    assert tokenizer_json is not None, "tiktoken_model_to_tokenizer_json returned None"

    tokenizer_config = json.loads(tokenizer_json)
    tokenizer_config.pop("decoder", None)
    tokenizer_json = json.dumps(tokenizer_config, ensure_ascii=False)
    basetenkenizer_tok = Tokenizer.from_json_str(tokenizer_json)
    hf_tok = hf_tokenizers.Tokenizer.from_str(tokenizer_json)

    corpus = [
        "[BOS]",
        "<|open|>",
        "<|close|>",
        "<|sep|>",
        "<|end_of_msg|>",
        "<osagent_mode>",
        "<|kimi_image_placeholder|>",
        "<|reserved_token_163589|>",
        "<|open|>message<|sep|>Hello<|close|><|end_of_msg|>",
        "你好，世界",
    ]
    for text in corpus:
        basetenkenizer_ids = basetenkenizer_tok.encode(
            text, add_special_tokens=False
        ).ids
        hf_ids = hf_tok.encode(text, add_special_tokens=False).ids
        assert basetenkenizer_ids == hf_ids, (
            f"encoding mismatch for {text!r}:\n"
            f"  basetenkenizer: {basetenkenizer_ids}\n"
            f"  tokenizers: {hf_ids}"
        )


def test_kimi_k3_vendored_tokenizer_json_matches_on_the_fly_conversion(
    tmp_path,
) -> None:
    pytest.importorskip("tiktoken")

    kimi_dir = VENDORED_DIR / "kimi-k3"
    gz_path = kimi_dir / "tiktoken.model.gz"
    vendored_json_path = kimi_dir / "tokenizer.json"
    if not gz_path.exists() or not vendored_json_path.exists():
        pytest.skip("vendored kimi-k3 assets not found")

    model_path = tmp_path / "tiktoken.model"
    model_path.write_bytes(gzip.decompress(gz_path.read_bytes()))
    (tmp_path / "tokenizer_config.json").write_text(
        (kimi_dir / "tokenizer_config.json").read_text()
    )

    converted_json = tiktoken_model_to_tokenizer_json(tmp_path)
    assert converted_json is not None
    converted = json.loads(converted_json)
    vendored = json.loads(vendored_json_path.read_text())

    assert converted["model"]["vocab"] == vendored["model"]["vocab"]
    assert converted["model"]["merges"] == vendored["model"]["merges"]
    assert {t["id"]: t["content"] for t in converted["added_tokens"]} == {
        t["id"]: t["content"] for t in vendored["added_tokens"]
    }


def test_kimi_k3_encode_segments_match_tiktoken_chat_rendering(tmp_path) -> None:
    """basetenkenizer ``encode_segments`` must reproduce per-segment tiktoken
    encoding that ``tokenization_kimi.TikTokenTokenizer._encode_chat_segments``
    performs for Kimi K3's Python-rendered chat template (``encoding_k3.py``).
    """
    tiktoken = pytest.importorskip("tiktoken")

    kimi_dir = VENDORED_DIR / "kimi-k3"
    if not (kimi_dir / "tiktoken.model.gz").exists():
        pytest.skip("vendored kimi-k3 assets not found")

    import importlib.util

    encoding_k3_path = kimi_dir / "encoding_k3.py"
    spec = importlib.util.spec_from_file_location("encoding_k3", encoding_k3_path)
    assert spec is not None and spec.loader is not None
    encoding_k3 = importlib.util.module_from_spec(spec)
    sys.modules["encoding_k3"] = encoding_k3
    try:
        spec.loader.exec_module(encoding_k3)
    finally:
        sys.modules.pop("encoding_k3", None)

    model_path = tmp_path / "tiktoken.model"
    model_path.write_bytes(gzip.decompress((kimi_dir / "tiktoken.model.gz").read_bytes()))
    mergeable_ranks = tiktoken.load.load_tiktoken_bpe(str(model_path))
    base = len(mergeable_ranks)
    special_tokens = {f"<|reserved_token_{i}|>": i for i in range(base, base + 256)}
    config = json.loads((kimi_dir / "tokenizer_config.json").read_text())
    for token_id, token_config in config["added_tokens_decoder"].items():
        special_tokens[token_config["content"]] = int(token_id)

    pat_str = "|".join(
        [
            r"[\p{Han}]+",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}",
            r" ?[^\s\p{L}\p{N}]+[\r\n]*",
            r"\s*[\r\n]+",
            r"\s+(?!\S)",
            r"\s+",
        ]
    )
    encoding = tiktoken.Encoding(
        name="kimi-k3",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    def tiktoken_chat_ids(segments) -> list[int]:
        ids: list[int] = []
        for segment in segments:
            if segment.allow_special:
                ids.extend(encoding.encode(segment.text, allowed_special="all"))
            else:
                ids.extend(encoding.encode(segment.text, disallowed_special=()))
        return ids

    fast_tokenizer = Tokenizer.from_json_str((kimi_dir / "tokenizer.json").read_text())

    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hello, world! 你好世界"},
        {
            "role": "assistant",
            "content": "Hi there.",
            "reasoning_content": "private reasoning",
        },
    ]
    segments = encoding_k3.build_chat_segments(
        messages, add_generation_prompt=True, thinking=True
    )
    segment_tuples = [(s.text, s.allow_special) for s in segments]

    fast_ids = fast_tokenizer.encode_segments(segment_tuples, tiktoken_safe=True).ids
    ref_ids = tiktoken_chat_ids(segments)
    assert fast_ids == ref_ids, (
        f"encode_segments diverged from tiktoken chat rendering:\n"
        f"  basetenkenizer: {fast_ids}\n"
        f"  tiktoken : {ref_ids}"
    )


def test_tiktoken_model_to_tokenizer_json_rejects_unknown_patternless_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tiktoken_module = types.ModuleType("tiktoken")
    tiktoken_load_module = types.ModuleType("tiktoken.load")
    tiktoken_load_module.load_tiktoken_bpe = lambda path: {b"a": 0}
    tiktoken_module.load = tiktoken_load_module
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken_module)
    monkeypatch.setitem(sys.modules, "tiktoken.load", tiktoken_load_module)

    (tmp_path / "tiktoken.model").write_text("")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {
                    "AutoTokenizer": [
                        "tokenization_glm.TikTokenTokenizer",
                        None,
                    ],
                },
                "added_tokens_decoder": {},
            }
        )
    )

    with caplog.at_level(logging.WARNING, logger="basetenkenizer.tiktoken"):
        assert tiktoken_model_to_tokenizer_json(tmp_path) is None
    assert "does not support converting tiktoken tokenizer config" in caplog.text
    assert "does not define pat_str or pattern" in caplog.text


def test_tiktoken_model_to_tokenizer_json_returns_none_without_optional_tiktoken(
    hide_tiktoken,
) -> None:
    assert tiktoken_model_to_tokenizer_json("tiktoken.model") is None


def test_tiktoken_model_to_tokenizer_json_raises_for_missing_file(tmp_path) -> None:
    pytest.importorskip("tiktoken")

    with pytest.raises(FileNotFoundError):
        tiktoken_model_to_tokenizer_json(tmp_path / "missing.model")


def test_tiktoken_model_to_tokenizer_json_raises_for_invalid_file(tmp_path) -> None:
    pytest.importorskip("tiktoken")
    model_path = tmp_path / "tiktoken.model"
    model_path.write_text("not-a-valid-model-line")

    with pytest.raises(ValueError, match="invalid tiktoken model"):
        tiktoken_model_to_tokenizer_json(model_path)


def test_kimi_k2_5_tiktoken_gz_conversion_matches_vendored_tokenizer_json(
    tmp_path,
) -> None:
    """Convert the vendored Kimi K2.5 tiktoken.model.gz on the fly and verify
    that encoding results match the vendored tokenizer.json for a corpus of
    test strings. Requires tiktoken; skipped otherwise."""
    pytest.importorskip("tiktoken")

    kimi_dir = VENDORED_DIR / "kimi-k2.5"
    gz_path = kimi_dir / "tiktoken.model.gz"
    vendored_json_path = kimi_dir / "tokenizer.json"

    if not gz_path.exists() or not vendored_json_path.exists():
        pytest.skip("vendored kimi-k2.5 assets not found")

    # Decompress the tiktoken model to a temporary plain file.
    model_path = tmp_path / "tiktoken.model"
    model_path.write_bytes(gzip.decompress(gz_path.read_bytes()))

    # Extract the pre-tokenizer pattern and special tokens from the vendored
    # tokenizer.json so the conversion uses exactly the same configuration.
    vendored = json.loads(vendored_json_path.read_text())
    pattern: str = vendored["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    special_tokens: dict[str, int] = {
        t["content"]: t["id"] for t in vendored["added_tokens"] if t.get("special")
    }

    # Convert the tiktoken model to a tokenizer JSON and build a Tokenizer.
    converted_json = tiktoken_model_to_tokenizer_json(
        model_path,
        pattern=pattern,
        special_tokens=special_tokens,
    )
    assert converted_json is not None, "tiktoken_model_to_tokenizer_json returned None"
    converted_tok = Tokenizer.from_json_str(converted_json)
    converted = json.loads(converted_json)

    # Also build a Tokenizer directly from the vendored tokenizer.json.
    vendored_tok = Tokenizer.from_json_str(vendored_json_path.read_text())

    # Rough structural parity checks (order-insensitive).
    converted_model = converted["model"]
    vendored_model = vendored["model"]
    assert converted_model["type"] == vendored_model["type"] == "BPE"
    assert len(converted_model["vocab"]) == len(vendored_model["vocab"])
    assert converted_model["vocab"] == vendored_model["vocab"]
    assert len(converted_model["merges"]) == len(vendored_model["merges"])
    assert _merge_counter(converted_model["merges"]) == _merge_counter(
        vendored_model["merges"]
    )

    converted_special_tokens = _special_token_map(converted["added_tokens"])
    vendored_special_tokens = _special_token_map(vendored["added_tokens"])
    assert converted_special_tokens == vendored_special_tokens

    corpus = [
        "Hello, world!",
        "1 + 2 = 3",
        "The quick brown fox jumps over the lazy dog.",
        "你好，世界",
        "basetenkenizer converts tiktoken models on the fly",
    ]
    for text in corpus:
        converted_ids = converted_tok.encode(text, add_special_tokens=False).ids
        vendored_ids = vendored_tok.encode(text, add_special_tokens=False).ids
        assert converted_ids == vendored_ids, (
            f"encoding mismatch for {text!r}:\n"
            f"  converted : {converted_ids}\n"
            f"  vendored  : {vendored_ids}"
        )


def test_converted_tiktoken_tokenizer_encodes_segments_with_special_control() -> None:
    encoding = _create_test_encoding()
    tokenizer = Tokenizer.from_json_str(tiktoken_to_tokenizer_json(encoding))

    structural_ids = tokenizer.encode_segments(
        [("ab", False), ("<|end|>", True), ("cd", False)]
    ).ids
    text_ids = tokenizer.encode_segments([("ab cd", False)]).ids

    assert structural_ids == [4, 100, 5]
    assert text_ids == encoding.encode("ab cd", disallowed_special=())


def test_converted_tiktoken_tokenizer_tiktoken_safe_segments_match_manual_chunks() -> (
    None
):
    encoding = _create_test_encoding()
    tokenizer = Tokenizer.from_json_str(tiktoken_to_tokenizer_json(encoding))
    text = "aaaa"

    safe_ids = tokenizer.encode_segments(
        [(text, False)],
        tiktoken_safe=True,
    ).ids

    assert safe_ids == encoding.encode(text, disallowed_special=())
