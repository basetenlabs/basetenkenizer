// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0

use basetenkenizer::Tokenizer;
use serde_json::json;

#[test]
fn llama3_pipeline_offline() {
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for ch in alphabet {
        vocab.insert(ch.to_string(), json!(vocab.len()));
    }
    // An orphan token checks Llama 3's ignore_merges whole-pretoken lookup.
    for token in ["hello", "1234", "<|begin_of_text|>"] {
        vocab.insert(token.into(), json!(vocab.len()));
    }
    let bos = vocab["<|begin_of_text|>"].clone();
    let config = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "normalizer": null,
        "added_tokens": [{"id": bos, "content": "<|begin_of_text|>", "special": true,
            "normalized": false, "single_word": false, "lstrip": false, "rstrip": false}],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"},
                "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        ]},
        "post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<|begin_of_text|>": {"id": "<|begin_of_text|>", "ids": [bos], "tokens": ["<|begin_of_text|>"]}}},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "ignore_merges": true}
    });
    let fast = Tokenizer::from_json(config.clone()).unwrap();
    let reference = tokenizers::Tokenizer::from_bytes(config.to_string()).unwrap();
    for text in [
        "",
        "hello",
        "1234",
        "hello1234",
        "WE'RE café 🫠\n\n",
        "<|begin_of_text|>hello",
    ] {
        for special in [false, true] {
            let expected = reference.encode(text, special).unwrap();
            let actual = fast.encode_with_special_tokens(text, special).unwrap();
            assert_eq!(actual, expected.get_ids(), "{text:?}");
            assert_eq!(
                fast.decode(&actual, false).unwrap(),
                reference.decode(expected.get_ids(), false).unwrap()
            );
        }
    }
}

#[test]
fn llama2_pipeline_offline() {
    let source = include_str!("fixtures/llama-2-7b-chat/tokenizer.json");
    // Missing fallback IDs must fail only when encountered, not at construction.
    let mut incomplete: serde_json::Value = serde_json::from_str(source).unwrap();
    let vocab = incomplete["model"]["vocab"].as_object_mut().unwrap();
    let id = vocab.remove("<0xF0>").unwrap();
    vocab.insert("<unused-fallback>".into(), id);
    let fast = Tokenizer::from_json(incomplete).unwrap();
    assert!(fast.encode_with_special_tokens("hello", false).is_ok());
    assert!(
        fast.encode_with_special_tokens("🫠", false)
            .unwrap_err()
            .to_string()
            .contains("fallback byte F0 not in vocabulary")
    );
    for normalized in [true, false] {
        let mut config: serde_json::Value = serde_json::from_str(source).unwrap();
        // First test the unmodified fixture, then mixed raw/normalized tokens.
        config["added_tokens"][1]["normalized"] = json!(normalized);
        let fast = Tokenizer::from_json(config.clone()).unwrap();
        let reference = tokenizers::Tokenizer::from_bytes(config.to_string()).unwrap();
        // ByteFallback replaces every byte in an invalid UTF-8 run.
        for bytes in [
            vec![0xF0, 0x9F],
            vec![0x61, 0xFF],
            vec![0xF0, 0x9F, 0xAB, 0xA0],
        ] {
            let ids: Vec<_> = bytes
                .into_iter()
                .map(|byte| reference.token_to_id(&format!("<0x{byte:02X}>")).unwrap())
                .collect();
            assert_eq!(
                fast.decode(&ids, false).unwrap(),
                reference.decode(&ids, false).unwrap()
            );
        }
        for text in [
            "",
            "ab",
            " ab ",
            "a🫠b",
            "\0\u{10ffff}",
            "<s>ab<s>",
            "a <s> b",
            "<s>ab</s><s>ab",
            "hello</s>",
            "hello </s>",
            "<s>[INST] Tell me a joke. [/INST]Sure!</s>",
        ] {
            for special in [false, true] {
                let expected = reference.encode(text, special).unwrap();
                let actual = fast.encode_with_special_tokens(text, special).unwrap();
                assert_eq!(
                    actual,
                    expected.get_ids(),
                    "{text:?}, normalized={normalized}"
                );
                for skip in [false, true] {
                    assert_eq!(
                        fast.decode(&actual, skip).unwrap(),
                        reference.decode(expected.get_ids(), skip).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn llama2_segments_preserve_literal_user_text() {
    let source = include_str!("fixtures/llama-2-7b-chat/tokenizer.json");
    let config: serde_json::Value = serde_json::from_str(source).unwrap();
    let native = Tokenizer::from_json(config.clone()).unwrap();
    let mut literal_config = config.clone();
    literal_config["added_tokens"] = json!([]);
    let literal = tokenizers::Tokenizer::from_bytes(literal_config.to_string()).unwrap();
    let reference = tokenizers::Tokenizer::from_bytes(config.to_string()).unwrap();
    for text in ["<s>hello</s>", "a <s> 🫠", "</s><s>"] {
        for special in [false, true] {
            assert_eq!(
                native.encode_segments([(text, false)], special).unwrap(),
                literal.encode(text, special).unwrap().get_ids(),
                "literal content: {text:?}"
            );
            assert_eq!(
                native.encode_segments([(text, true)], special).unwrap(),
                reference.encode(text, special).unwrap().get_ids(),
                "trusted content: {text:?}"
            );
        }
    }
}
