use crate::pre_tokenizers::byte_level::BYTE_TO_CHAR;

/// Reverse mapping: Unicode char → original byte value.
///
/// All chars in `BYTE_TO_CHAR` are in the range U+0000..U+0143 (max codepoint
/// 323), so a flat 324-element array gives O(1) lookup.
const CHAR_TO_BYTE: [u8; 324] = build_char_to_byte();

const fn build_char_to_byte() -> [u8; 324] {
    let mut table = [0u8; 324];
    let mut i = 0u16;
    while i < 256 {
        let ch = BYTE_TO_CHAR[i as usize];
        table[ch as usize] = i as u8;
        i += 1;
    }
    table
}

/// ByteLevel decoder: reverses the GPT-2 byte-to-unicode mapping.
#[derive(Debug)]
pub struct ByteLevelDecoder;

impl ByteLevelDecoder {
    /// Apply byte-level decoding to a list of token strings.
    ///
    /// Joins all tokens, maps each char back to its byte value using the
    /// reverse GPT-2 table, and interprets the bytes as UTF-8.
    pub fn decode_chain(&self, tokens: Vec<String>) -> Vec<String> {
        let joined: String = tokens.into_iter().collect();
        let mut bytes: Vec<u8> = Vec::with_capacity(joined.len());
        for c in joined.chars() {
            let cp = c as usize;
            if cp < CHAR_TO_BYTE.len() {
                bytes.push(CHAR_TO_BYTE[cp]);
            } else {
                // Characters outside the GPT-2 table (e.g. ｜ U+FF5C, ▁ U+2581
                // in DeepSeek tokens): preserve their original UTF-8 encoding.
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
        vec![String::from_utf8_lossy(&bytes).into_owned()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii() {
        let dec = ByteLevelDecoder;
        let result = dec.decode_chain(vec!["Hello".to_string()]);
        assert_eq!(result, vec!["Hello"]);
    }

    #[test]
    fn roundtrip_space() {
        let dec = ByteLevelDecoder;
        // GPT-2 maps space (0x20) to Ġ (U+0120)
        let result = dec.decode_chain(vec!["\u{120}Hello".to_string()]);
        assert_eq!(result, vec![" Hello"]);
    }

    #[test]
    fn roundtrip_multibyte() {
        let dec = ByteLevelDecoder;
        // Euro sign €: UTF-8 bytes [0xE2, 0x82, 0xAC]
        // BYTE_TO_CHAR maps these to specific unicode chars
        let encoded: String = [0xE2u8, 0x82, 0xAC]
            .iter()
            .map(|&b| BYTE_TO_CHAR[b as usize])
            .collect();
        let result = dec.decode_chain(vec![encoded]);
        assert_eq!(result, vec!["€"]);
    }

    #[test]
    fn non_gpt2_chars_preserved() {
        let dec = ByteLevelDecoder;
        // DeepSeek uses ｜ (U+FF5C) and ▁ (U+2581) in token strings
        // like <｜begin▁of▁sentence｜>. These are outside the GPT-2
        // byte-to-char table and must pass through unchanged.
        let result = dec.decode_chain(vec![
            "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>".to_string(),
        ]);
        assert_eq!(result, vec!["<｜begin▁of▁sentence｜>"]);
    }
}
