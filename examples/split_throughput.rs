// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0

//! Isolated Split throughput; excludes construction and uses varying prefixes.
//! `cargo run --release --example split_throughput -- [fancy|pcre2] [threads]`
use basetenkenizer::{Split, pre_tokenized::PreTokenizedString};
use serde_json::json;
use std::{
    error::Error,
    hint::black_box,
    sync::Barrier,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "fancy".into());
    let threads: usize = args.next().unwrap_or_else(|| "1".into()).parse()?;
    if threads == 0 {
        return Err("threads must be positive".into());
    }
    let pattern = match mode.as_str() {
        // Intersection disables PCRE2; lookahead forces the Fancy VM.
        "fancy" => r"[\p{L}&&[^A-Z]]+(?!\S)|\p{N}{1,3}|\s+|[^\p{L}\p{N}\s]+",
        "pcre2" => r"\p{L}+|\p{N}+|\s+|[^\p{L}\p{N}\s]+",
        _ => return Err("mode must be fancy or pcre2".into()),
    };
    let split = Split::from_config(&json!({"Regex":pattern}), "Isolated", false)?;
    let inputs: Vec<_> = (0..16)
        .map(|index| {
            let block = format!(
                "{index} The tokenizer café 日本語 русский 12345 e\u{301} and symbols 🚀.\n"
            );
            block.repeat((256 * 1024 / block.len()) + 1)
        })
        .collect();
    let mut checksum = 0u64;
    for input in &inputs {
        let mut pts = PreTokenizedString::from_text(input);
        split.pre_tokenize(&mut pts)?;
        for segment in pts.splits() {
            checksum = checksum
                .wrapping_mul(1099511628211)
                .wrapping_add(segment.range.start as u64)
                .wrapping_add(segment.range.end as u64);
        }
    }
    let barrier = Barrier::new(threads);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|client| {
                let (split, inputs, barrier) = (&split, &inputs, &barrier);
                scope.spawn(move || {
                    let encode = |index: usize| {
                        let mut pts = PreTokenizedString::from_text(&inputs[index % inputs.len()]);
                        split.pre_tokenize(&mut pts).unwrap();
                        black_box(pts);
                    };
                    for index in 0..16 {
                        encode(index);
                    }
                    barrier.wait();
                    let start = Instant::now();
                    let mut count = 0;
                    while start.elapsed() < Duration::from_secs(2) {
                        encode(count + client);
                        count += 1;
                    }
                    (count, start.elapsed())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let count: usize = results.iter().map(|r| r.0).sum();
    let secs = results.iter().map(|r| r.1).max().unwrap().as_secs_f64();
    println!(
        "mode={mode} threads={threads} span_checksum={checksum} requests_per_second={:.1}",
        count as f64 / secs
    );
    Ok(())
}
