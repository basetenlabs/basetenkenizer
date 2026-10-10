// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0

use basetenkenizer::Tokenizer;
use serde_json::json;

#[test]
fn normalization_collisions_match_hugging_face() {
    for reversed in [false, true] {
        for special in [[false, false], [true, true], [false, true], [true, false]] {
            let mut tokens = vec![
                json!({"id": 1, "content": "é", "normalized": true, "special": special[0],
                    "single_word": false, "lstrip": false, "rstrip": false}),
                json!({"id": 2, "content": "e\u{301}", "normalized": true, "special": special[1],
                    "single_word": false, "lstrip": false, "rstrip": false}),
            ];
            if reversed {
                tokens.reverse();
            }
            let config = json!({
                "version": "1.0", "added_tokens": tokens,
                "normalizer": {"type": "NFC"},
                "model": {"type": "BPE", "vocab": {"x": 0, "é": 1, "e\u{301}": 2}, "merges": []}
            });
            let mut reference = tokenizers::Tokenizer::from_bytes(config.to_string()).unwrap();
            let fast = Tokenizer::from_json(config).unwrap();
            for split_special in [false, true] {
                reference.set_encode_special_tokens(split_special);
                for text in ["é", "e\u{301}", "xéxe\u{301}"] {
                    let expected = reference.encode(text, false).unwrap();
                    let actual = fast
                        .encode_with_options(text, false, split_special)
                        .unwrap();
                    assert_eq!(
                        actual,
                        expected.get_ids(),
                        "{text:?}, reversed={reversed}, special={special:?}, split={split_special}"
                    );
                }
            }
            for (id, content) in [(1, "é"), (2, "e\u{301}")] {
                let added = fast.added_tokens().unwrap();
                assert_eq!(added.id_to_token(id), Some(content));
                assert_eq!(added.token_to_id(content), Some(id));
            }
        }
    }
}
