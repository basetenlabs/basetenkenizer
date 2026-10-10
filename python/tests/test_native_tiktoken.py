# Copyright 2026 Baseten
# SPDX-License-Identifier: Apache-2.0
"""Offline native model-loading tests; no tiktoken dependency is required."""
import base64
import json

import pytest
from basetenkenizer import Tokenizer


def write_model(path):
    ranks = [(bytes([byte]), byte) for byte in range(256)]
    ranks.extend([(b"ab", 256), (b"abc", 257)])
    path.write_text("".join(f"{base64.b64encode(token).decode()} {rank}\n" for token, rank in ranks))


def test_native_tiktoken_explicit_pattern_and_specials(tmp_path):
    path = tmp_path / "tiktoken.model"
    write_model(path)
    tokenizer = Tokenizer.from_tiktoken_model(str(path), pattern=r"\S+|\s+", special_tokens={"<special>": 258})
    assert tokenizer.encode("ababc", add_special_tokens=False).ids == [256, 257]
    text = "abc é 🐈\n"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    assert tokenizer.encode("<special>ab").ids == [258, 256]
    assert tokenizer.decode([258, 256], skip_special_tokens=True) == "ab"


def test_native_tiktoken_directory_uses_declared_pattern_and_added_flags(tmp_path):
    write_model(tmp_path / "tiktoken.model")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "ExampleTokenizer", "pat_str": r"\S+|\s+",
        "added_tokens_decoder": {"258": {"content": "<special>", "special": True}, "259": {"content": "<mask>", "special": False, "lstrip": True}},
    }))
    tokenizer = Tokenizer.from_file(str(tmp_path))
    assert tokenizer.encode("ababc").ids == [256, 257]
    assert tokenizer.encode("ab   <mask>").ids == [256, 259]
    assert tokenizer.decode([258, 259], skip_special_tokens=True) == "<mask>"


def test_native_tiktoken_exact_filename_and_missing_path(tmp_path):
    write_model(tmp_path / "tiktoken.model")
    (tmp_path / "custom.model").write_text("Yg== 0\n")
    tokenizer = Tokenizer.from_tiktoken_model(str(tmp_path / "custom.model"))
    assert tokenizer.encode("b").ids == [0]
    with pytest.raises(ValueError):
        Tokenizer.from_tiktoken_model(str(tmp_path / "missing.model"))


def test_native_tiktoken_unknown_family_requires_pattern(tmp_path):
    write_model(tmp_path / "tiktoken.model")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "UnknownTokenizer"}))
    with pytest.raises(ValueError, match="unrecognized tiktoken family"):
        Tokenizer.from_file(str(tmp_path))


def test_native_tiktoken_kimi_reserved_tokens_and_flags(tmp_path):
    write_model(tmp_path / "tiktoken.model")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "auto_map": {"AutoTokenizer": [None, "tokenization_kimi.TikTokenTokenizer"]},
        "added_tokens_decoder": {"258": {"content": "<|end_of_msg|>", "special": True}, "259": {"content": "<mask>", "special": False}},
    }))
    tokenizer = Tokenizer.from_file(str(tmp_path))
    assert tokenizer.token_to_id("<|reserved_token_260|>") == 260
    assert tokenizer.encode("ab<|end_of_msg|>").ids == [256, 258]
    assert tokenizer.decode([258, 259], skip_special_tokens=True) == "<mask>"


@pytest.mark.parametrize("contents", ["YQ==", "YQ== x", "YQ== 0 extra", "YQ=== 0", "YR== 0", "YQ== 0\nYg== 0\n"])
def test_native_tiktoken_rejects_malformed_models(tmp_path, contents):
    path = tmp_path / "tiktoken.model"
    path.write_text(contents)
    with pytest.raises(ValueError):
        Tokenizer.from_tiktoken_model(str(path))


def test_native_kimi_matches_vendored_json(tmp_path):
    import gzip
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[2] / "vendored_tokenizers" / "kimi-k3"
    if not (fixtures / "tiktoken.model.gz").exists():
        pytest.skip("vendored Kimi ranks not available")
    (tmp_path / "tiktoken.model").write_bytes(gzip.decompress((fixtures / "tiktoken.model.gz").read_bytes()))
    (tmp_path / "tokenizer_config.json").write_bytes((fixtures / "tokenizer_config.json").read_bytes())
    native = Tokenizer.from_file(str(tmp_path))
    reference = Tokenizer.from_file(str(fixtures / "tokenizer.json"))
    for text in ["Hello, world!", "另一个测试 with mixed 内容", "def f(x):\n    return x * 2\n", "camelCase HTTPRequest O'Brien 12345", "a<|end_of_msg|>b", "é 🐈\n\t"]:
        assert native.encode(text).ids == reference.encode(text).ids
        ids = native.encode(text).ids
        assert native.decode(ids, skip_special_tokens=False) == reference.decode(ids, skip_special_tokens=False)


def test_from_file_loads_raw_model_without_config_or_tiktoken(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def import_without_tiktoken(name, *args, **kwargs):
        if name == "tiktoken" or name.startswith("tiktoken."):
            raise ImportError("tiktoken deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tiktoken)
    path = tmp_path / "tiktoken.model"
    write_model(path)
    tokenizer = Tokenizer.from_file(str(path))
    assert tokenizer.encode("abc").ids == [257]
    assert tokenizer.decode(tokenizer.encode("abc é\n").ids) == "abc é\n"
