// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Gemma 4 pipeline regression derived from fastokens commit 3e0d837.
// The vocabulary below is synthetic; no Google model vocabulary is distributed.

use basetenkenizer::Tokenizer;
use serde_json::json;

fn assert_parity(config: serde_json::Value) {
    let fast = Tokenizer::from_json(config.clone()).expect("load Gemma 4 pipeline");
    let reference = tokenizers::Tokenizer::from_bytes(config.to_string())
        .expect("load reference Gemma 4 pipeline");
    for text in [
        "",
        "Hello, world!",
        " multiple   spaces\nand a newline ",
        "数学と café 🚀",
        "\0\u{10ffff}",
        "<|turn>user\nWhat is 84 * 3 / 2?<turn|>",
        "<bos>Hello<eos>",
        "a▁b",
    ] {
        for special in [false, true] {
            let expected = reference.encode(text, special).unwrap();
            let actual = fast.encode_with_special_tokens(text, special).unwrap();
            assert_eq!(actual, expected.get_ids(), "encoding {text:?}");
            for skip in [false, true] {
                assert_eq!(
                    fast.decode(&actual, skip).unwrap(),
                    reference.decode(expected.get_ids(), skip).unwrap(),
                    "decoding {text:?}, skip_special_tokens={skip}"
                );
            }
        }
    }
}

#[test]
fn gemma4_pipeline_offline() {
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for token in ["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>"] {
        let id = vocab.len();
        vocab.insert(token.into(), json!(id));
        added.push(json!({"id": id, "content": token, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true}));
    }
    for byte in 0..=255 {
        vocab.insert(format!("<0x{byte:02X}>"), json!(vocab.len()));
    }
    for token in [
        "▁", "H", "e", "l", "o", "w", "r", "d", "!", ",", "He", "Hel", "Hell", "Hello",
    ] {
        vocab.insert(token.into(), json!(vocab.len()));
    }
    // Matches google/gemma-4-31B-it's tokenizer pipeline at revision
    // 842da3794eaa0b77d5f08bae87a17459d91ff475, with a small synthetic BPE.
    assert_parity(json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "},
            "behavior": "MergedWithPrevious", "invert": false},
        "post_processor": {"type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>",
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
            "vocab": vocab, "merges": [["H", "e"], ["He", "l"], ["Hel", "l"], ["Hell", "o"]]}
    }));
}

#[test]
#[ignore = "requires vendored Gemma 4 tokenizer; set GEMMA4_TOKENIZER_JSON to override its path"]
fn gemma4_full_vocabulary_parity() {
    let path = std::env::var_os("GEMMA4_TOKENIZER_JSON")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("vendored_tokenizers/gemma-4-31b-it/tokenizer.json")
        });
    let config = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_parity(config);
}
