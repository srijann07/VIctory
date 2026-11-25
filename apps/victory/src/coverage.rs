//! Compare runtime translation failures against the offline container scan.
//! Only hash matches are translated; the command never writes shader data.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use victory_shader::emit;

use crate::shader::{MAX_FILE, Source, sources};

const FAILURE: &str = "shader does not translate; its pipelines' draws are dropped";
type Failures = BTreeMap<String, BTreeSet<String>>;
type Matches = BTreeMap<String, Result<(), String>>;

/// Strip terminal CSI colour sequences from redirected tracing output.
fn plain(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for part in chars.by_ref() {
                if ('@'..='~').contains(&part) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Group the same unsupported operation even at different code offsets.
fn reason(error: &str) -> &str {
    if let Some((offset, what)) = error
        .strip_prefix("GFX10 code at ")
        .and_then(|s| s.split_once(": "))
        && offset
            .strip_prefix("0x")
            .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return what;
    }
    error
}

fn failures(reader: impl BufRead) -> Result<(Failures, usize, usize)> {
    let mut found = Failures::new();
    let mut records = 0;
    let mut malformed = 0;
    for line in reader.lines() {
        let line = plain(&line?);
        let Some(fields) = line.split_once("victory_gpu::pipeline:").map(|(_, s)| s) else {
            continue;
        };
        if !fields.contains(FAILURE) {
            continue;
        }
        let hash = fields
            .split_whitespace()
            .find_map(|field| field.strip_prefix("hash="))
            .filter(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()));
        let error = fields
            .split_once(" error=")
            .map(|(_, s)| s.trim().trim_end_matches(FAILURE).trim())
            .filter(|s| !s.is_empty());
        match (hash, error) {
            (Some(hash), Some(error)) => {
                found
                    .entry(hash.to_ascii_lowercase())
                    .or_default()
                    .insert(reason(error).to_owned());
                records += 1;
            }
            _ => malformed += 1,
        }
    }
    Ok((found, records, malformed))
}

/// Raw embedded containers have a known declared extent but an unknown
/// runtime bytecode length. Standalone files and packed records also
/// have a known full extent, which may include padding.
fn candidates(bytes: &[u8], full_extent_known: bool) -> [Option<&[u8]>; 2] {
    let declared = bytes
        .get(0x18..0x1c)
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .map(|b| u32::from_le_bytes(b) as usize)
        .filter(|&len| len >= 0x20 && bytes.starts_with(b"DXBC"))
        .and_then(|len| bytes.get(..len));
    let full = declared
        .filter(|d| full_extent_known && d.len() != bytes.len())
        .map(|_| bytes);
    [declared, full]
}

fn probe(
    bytes: &[u8],
    wanted: &Failures,
    matched: &mut Matches,
    translate: impl FnOnce(&[u8]) -> Result<(), String>,
) {
    let hash = blake3::hash(bytes).to_hex()[..16].to_owned();
    if wanted.contains_key(&hash) {
        matched.entry(hash).or_insert_with(|| translate(bytes));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    OfflineEmitted,
    OfflineFailed,
    NotFound,
}

fn status(matched: &Matches, hash: &str) -> Status {
    match matched.get(hash) {
        Some(Ok(())) => Status::OfflineEmitted,
        Some(Err(_)) => Status::OfflineFailed,
        None => Status::NotFound,
    }
}

pub(crate) fn run(log: &Path, paths: Vec<PathBuf>) -> Result<()> {
    let file = std::fs::File::open(log).with_context(|| format!("open {}", log.display()))?;
    let (wanted, records, malformed) = failures(BufReader::new(file))
        .context("read runtime log as UTF-8 (use plain text redirection)")?;
    println!("runtime failure records: {records}");
    println!("unique failing hashes:   {}", wanted.len());
    println!("malformed failure rows:  {malformed}");
    if wanted.is_empty() {
        println!("No recognizable runtime translation failures; check logging and input.");
        return Ok(());
    }
    let paths = crate::default_paths(paths)?;
    let mut matched = Matches::new();
    let mut skipped = 0;
    for root in paths {
        for entry in walkdir::WalkDir::new(root) {
            let Ok(entry) = entry else {
                skipped += 1;
                continue;
            };
            if !entry.file_type().is_file() {
                continue;
            }
            if !entry.metadata().is_ok_and(|m| m.len() <= MAX_FILE) {
                skipped += 1;
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                skipped += 1;
                continue;
            };
            for source in sources(&data) {
                let full = matches!(&source, Source::Packed(_) | Source::Raw { offset: 0, .. });
                let Some(bytes) = source.bytes() else {
                    skipped += 1;
                    continue;
                };
                for bytes in candidates(&bytes, full).into_iter().flatten() {
                    probe(bytes, &wanted, &mut matched, |b| {
                        emit::translate(b).map(drop).map_err(|e| e.to_string())
                    });
                }
            }
        }
    }

    let emitted = matched.values().filter(|r| r.is_ok()).count();
    println!("matched / offline emitted: {emitted}");
    println!("matched / offline failed:  {}", matched.len() - emitted);
    println!("not found in scanned data:  {}", wanted.len() - matched.len());
    println!("skipped entries or records: {skipped}");
    println!("Not found means no matching byte extent was discovered, not that a shader is absent.");

    let mut reasons = BTreeMap::<&str, usize>::new();
    for (hash, errors) in &wanted {
        let label = match status(&matched, hash) {
            Status::OfflineEmitted => "offline emitted",
            Status::OfflineFailed => "offline failed",
            Status::NotFound => "not found",
        };
        println!("{hash}  {label}");
        for error in errors {
            println!("  runtime: {error}");
            *reasons.entry(error).or_default() += 1;
        }
        if let Some(Err(error)) = matched.get(hash) {
            println!("  offline: {error}");
        }
    }
    let mut reasons: Vec<_> = reasons.into_iter().collect();
    reasons.sort_by_key(|&(error, count)| (std::cmp::Reverse(count), error));
    println!("Runtime reasons (unique hashes per reason):");
    for (error, count) in reasons {
        println!("  {count:>7}  {error}");
    }
    println!("Emission is not DXIL validation, host pipeline creation or proof of playability.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_deduplicates_hashes_and_groups_offsets() {
        let log = format!(
            "WARN victory_gpu::pipeline: {FAILURE} stage=compute hash=0123456789ABCDEF error=GFX10 code at 0x12: unsupported\n\
             \x1b[33mWARN\x1b[0m victory_gpu::pipeline: {FAILURE} stage=compute hash=0123456789abcdef error=GFX10 code at 0x34: unsupported\n\
             WARN victory_gpu::pipeline: {FAILURE} stage=pixel hash=0123456789abcdef error=other reason\n\
             WARN unrelated: {FAILURE} hash=1111111111111111 error=ignored\n\
             WARN victory_gpu::pipeline: {FAILURE} hash=bad error=ignored\n"
        );
        let (found, records, malformed) = failures(log.as_bytes()).unwrap();
        assert_eq!((found.len(), records, malformed), (1, 3, 1));
        assert_eq!(
            found["0123456789abcdef"],
            BTreeSet::from(["unsupported".to_owned(), "other reason".to_owned()])
        );
    }


    #[test]
    fn plain_handles_csi_and_preserves_other_text() {
        for (input, expected) in [
            ("", ""),
            ("plain text", "plain text"),
            ("\x1b[1;33mWARN\x1b[0m", "WARN"),
            ("a\x1b[2Kb\x1b[0mc", "abc"),
            ("café\x1b[0m", "café"),
            ("text\x1b[33", "text"),
            ("a\x1bXb", "a\x1bXb"),
        ] {
            assert_eq!(plain(input), expected);
        }
    }

    #[test]
    fn reason_only_strips_valid_hex_offsets() {
        for (input, expected) in [
            ("GFX10 code at 0x0: unsupported", "unsupported"),
            ("GFX10 code at 0xAbCd: op: detail", "op: detail"),
            ("GFX10 code at 0x: unsupported", "GFX10 code at 0x: unsupported"),
            ("GFX10 code at 0xGG: unsupported", "GFX10 code at 0xGG: unsupported"),
            ("GFX10 code at 12: unsupported", "GFX10 code at 12: unsupported"),
            ("GFX10 code at 0x12", "GFX10 code at 0x12"),
            ("other reason", "other reason"),
            ("", ""),
        ] {
            assert_eq!(reason(input), expected);
        }
    }

    #[test]
    fn failures_rejects_malformed_rows_and_ignores_unrelated_logs() {
        let log = format!(
            "WARN victory_gpu::pipeline: {FAILURE} error=missing hash\n\
             WARN victory_gpu::pipeline: {FAILURE} hash=0123456789abcde error=short\n\
             WARN victory_gpu::pipeline: {FAILURE} hash=0123456789abcdef0 error=long\n\
             WARN victory_gpu::pipeline: {FAILURE} hash=0123456789abcdeg error=nonhex\n\
             WARN victory_gpu::pipeline: {FAILURE} hash=0123456789abcdef\n\
             WARN victory_gpu::pipeline: {FAILURE} hash=0123456789abcdef error=   \n\
             WARN unrelated: {FAILURE} hash=0123456789abcdef error=ignored\n\
             WARN victory_gpu::pipeline: unrelated hash=0123456789abcdef error=ignored\n\
             WARN victory_gpu::pipeline: {FAILURE} hash=FEDCBA9876543210 error=valid"
        );
        let (found, records, malformed) = failures(log.as_bytes()).unwrap();
        assert_eq!((found.len(), records, malformed), (1, 1, 6));
        assert_eq!(
            found["fedcba9876543210"],
            BTreeSet::from(["valid".to_owned()])
        );
    }

    #[test]
    fn failures_handles_empty_input_and_invalid_utf8() {
        let (found, records, malformed) = failures(&b""[..]).unwrap();
        assert!(found.is_empty());
        assert_eq!((records, malformed), (0, 0));
        assert!(failures(&b"\xff\n"[..]).is_err());
    }

    fn container() -> Vec<u8> {
        let mut bytes = vec![0; 32];
        bytes[..4].copy_from_slice(b"DXBC");
        bytes[0x18..0x1c].copy_from_slice(&32u32.to_le_bytes());
        bytes
    }

    #[test]
    fn candidate_extents_respect_embedded_boundaries() {
        let mut bytes = container();
        bytes.extend_from_slice(&[0; 8]);
        assert_eq!(candidates(&bytes, false), [Some(&bytes[..32]), None]);
        assert_eq!(candidates(&bytes, true), [Some(&bytes[..32]), Some(bytes.as_slice())]);
        assert_eq!(candidates(&bytes[..20], true), [None, None]);
        bytes[0x18..0x1c].copy_from_slice(&100u32.to_le_bytes());
        assert_eq!(candidates(&bytes, true), [None, None]);
    }

    #[test]
    fn only_requested_hashes_are_translated_once() {
        let bytes = container();
        let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
        let wanted = Failures::from([(hash.clone(), BTreeSet::new())]);
        let mut matched = Matches::new();
        probe(b"unrelated", &wanted, &mut matched, |_| panic!("not requested"));
        probe(&bytes, &wanted, &mut matched, |_| Ok(()));
        probe(&bytes, &wanted, &mut matched, |_| panic!("already matched"));
        assert_eq!(status(&matched, &hash), Status::OfflineEmitted);
        assert_eq!(status(&matched, "ffffffffffffffff"), Status::NotFound);
    }

    #[test]
    fn actual_translator_failure_is_not_an_offline_success() {
        let bytes = container();
        let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
        let wanted = Failures::from([(hash.clone(), BTreeSet::new())]);
        let mut matched = Matches::new();
        probe(&bytes, &wanted, &mut matched, |b| {
            emit::translate(b).map(drop).map_err(|e| e.to_string())
        });
        assert_eq!(status(&matched, &hash), Status::OfflineFailed);
    }
}
