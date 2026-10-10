// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0

use basetenkenizer::Tokenizer;
use serde_json::json;

fn roberta_config() -> serde_json::Value {
    let mut chars: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    chars.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for token in ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        .into_iter()
        .map(str::to_owned)
        .chain(chars.into_iter().map(|c| c.to_string()))
    {
        vocab.insert(token, json!(vocab.len()));
    }
    json!({
        "version":"1.0", "added_tokens":[
            {"id":4,"content":"<mask>","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":true}
        ],
        "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
        "post_processor":{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":false},
        "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        "model":{"type":"BPE","vocab":vocab,"merges":[]}
    })
}

#[test]
fn roberta_matches_hugging_face() {
    let config = roberta_config();
    for legacy in [false, true] {
        let mut config = config.clone();
        if legacy {
            config["model"].as_object_mut().unwrap().remove("type");
        }
        let fast = Tokenizer::from_json(config.clone()).unwrap();
        let mut reference = tokenizers::Tokenizer::from_bytes(config.to_string()).unwrap();
        for split_special in [false, true] {
            reference.set_encode_special_tokens(split_special);
            for special in [false, true] {
                for text in [
                    "",
                    "Hello world!",
                    " café 数学 🫠",
                    "hello   <mask> world",
                    "\n\t\u{a0}<mask>",
                    "<mask> <mask>",
                    "x<mask>  y",
                    "<mask>",
                ] {
                    let expected = reference.encode(text, special).unwrap();
                    let actual = fast
                        .encode_with_options(text, special, split_special)
                        .unwrap();
                    assert_eq!(
                        actual,
                        expected.get_ids(),
                        "{text:?}, legacy={legacy}, split={split_special}"
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
    let mut invalid = config;
    invalid["model"]["type"] = json!("NotBPE");
    assert!(Tokenizer::from_json(invalid).is_err());
}

#[test]
fn prefix_space_sequence_processor_and_literal_tokens_match_hugging_face() {
    for prefix_space in [false, true] {
        let mut config = roberta_config();
        config["pre_tokenizer"]["add_prefix_space"] = json!(prefix_space);
        config["post_processor"]["add_prefix_space"] = json!(prefix_space);
        let processor = config["post_processor"].clone();
        config["post_processor"] = json!({
            "type":"Sequence", "processors":[
                {"type":"ByteLevel","add_prefix_space":prefix_space,"trim_offsets":true,"use_regex":true},
                processor
            ]
        });
        for (offset, content) in ["    ", "Ġ literal", "Ġ👩🏽‍💻"].into_iter().enumerate()
        {
            let id = config["model"]["vocab"].as_object().unwrap().len() + offset;
            config["added_tokens"].as_array_mut().unwrap().push(json!({
                "id":id,"content":content,"single_word":false,"lstrip":false,
                "rstrip":false,"normalized":false,"special":false
            }));
        }
        let fast = Tokenizer::from_json(config.clone()).unwrap();
        let reference = tokenizers::Tokenizer::from_bytes(config.to_string()).unwrap();
        for add_special in [false, true] {
            for text in [
                "",
                "hello",
                " hello",
                "hello\nworld",
                "x    y",
                "xĠ literal y",
                "Ġ👩🏽‍💻",
                "a\u{2028} <mask>b",
                "<mask>    <mask>",
            ] {
                let expected = reference.encode(text, add_special).unwrap();
                let actual = fast.encode_with_options(text, add_special, false).unwrap();
                assert_eq!(
                    actual,
                    expected.get_ids(),
                    "{text:?}, prefix_space={prefix_space}"
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
