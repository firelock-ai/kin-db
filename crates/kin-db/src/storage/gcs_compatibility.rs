// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

#[cfg(any(feature = "gcs", test))]
use crate::error::KinDbError;

#[cfg(any(feature = "gcs", test))]
const GCS_FULL_AUTHORITY_MAGIC_PREFIX: [u8; 6] = *b"KNGCSF";
#[cfg(any(feature = "gcs", test))]
const GCS_FULL_AUTHORITY_MAGIC_LEN: usize = 8;
#[cfg(any(feature = "gcs", test))]
const GCS_FULL_AUTHORITY_MAX_ENCODED_VERSION: u32 = 99;

/// Versions of the GCS full-authority envelope this KinDB reader accepts.
///
/// This metadata is available even when the `gcs` feature is disabled so a
/// caller can advertise the compatibility of the KinDB build it ships without
/// also linking the GCS client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcsFullAuthorityEnvelopeCompatibility {
    pub min_supported_version: u32,
    pub current_version: u32,
}

impl GcsFullAuthorityEnvelopeCompatibility {
    /// Whether `version` is accepted by this build's GCS envelope reader.
    pub const fn supports(&self, version: u32) -> bool {
        version >= self.min_supported_version && version <= self.current_version
    }
}

/// Compatibility range for the GCS full-authority envelope reader.
///
/// The current wire magic is `KNGCSF02`, so the reader and writer both derive
/// their encoded version from this one public value.
pub const GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY: GcsFullAuthorityEnvelopeCompatibility =
    GcsFullAuthorityEnvelopeCompatibility {
        min_supported_version: 2,
        current_version: 2,
    };

#[cfg(any(feature = "gcs", test))]
pub(crate) fn full_authority_envelope_magic(version: u32) -> Result<[u8; 8], KinDbError> {
    if version > GCS_FULL_AUTHORITY_MAX_ENCODED_VERSION {
        return Err(KinDbError::StorageError(format!(
            "GCS full-authority envelope version {version} does not fit its two-digit magic"
        )));
    }

    let mut magic = [0_u8; GCS_FULL_AUTHORITY_MAGIC_LEN];
    magic[..GCS_FULL_AUTHORITY_MAGIC_PREFIX.len()]
        .copy_from_slice(&GCS_FULL_AUTHORITY_MAGIC_PREFIX);
    magic[6] = b'0' + (version / 10) as u8;
    magic[7] = b'0' + (version % 10) as u8;
    Ok(magic)
}

#[cfg(any(feature = "gcs", test))]
pub(crate) fn full_authority_envelope_version(bytes: &[u8]) -> Result<u32, KinDbError> {
    let magic = bytes
        .get(..GCS_FULL_AUTHORITY_MAGIC_LEN)
        .ok_or_else(not_current_full_authority_envelope)?;
    if !magic.starts_with(&GCS_FULL_AUTHORITY_MAGIC_PREFIX)
        || !magic[6].is_ascii_digit()
        || !magic[7].is_ascii_digit()
    {
        return Err(not_current_full_authority_envelope());
    }

    Ok(u32::from(magic[6] - b'0') * 10 + u32::from(magic[7] - b'0'))
}

#[cfg(any(feature = "gcs", test))]
fn not_current_full_authority_envelope() -> KinDbError {
    KinDbError::StorageError(
        "GCS snapshot object is not a current full-authority envelope".to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_compatibility_accepts_only_its_inclusive_range() {
        let compatibility = GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY;
        assert!(compatibility.min_supported_version <= compatibility.current_version);
        assert!(compatibility.current_version <= GCS_FULL_AUTHORITY_MAX_ENCODED_VERSION);
        assert!(compatibility.supports(compatibility.min_supported_version));
        assert!(compatibility.supports(compatibility.current_version));
        assert!(!compatibility.supports(compatibility.min_supported_version - 1));
        assert!(!compatibility.supports(compatibility.current_version + 1));
    }

    #[test]
    fn magic_roundtrips_the_public_current_version() {
        let compatibility = GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY;
        let magic = full_authority_envelope_magic(compatibility.current_version).unwrap();
        assert_eq!(&magic, b"KNGCSF02");
        assert_eq!(
            full_authority_envelope_version(&magic).unwrap(),
            compatibility.current_version
        );
    }

    /// Keep the codec on the public compatibility value. This fails if the GCS
    /// implementation restores a copied `KNGCSF02` or replaces either codec
    /// read with a private numeric constant.
    #[test]
    fn gcs_codec_has_one_public_version_authority() {
        let source = include_str!("gcs.rs");
        assert_codec_source_is_bound(source).unwrap();

        let copied_writer = source.replacen(
            "GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.current_version,",
            "2,",
            1,
        );
        let copied_writer_with_comment_decoy = copied_writer.replacen(
            "#[cfg(test)]",
            "// GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.current_version,\n#[cfg(test)]",
            1,
        );
        assert!(assert_codec_source_is_bound(&copied_writer_with_comment_decoy).is_err());
        let copied_writer_with_block_comment_decoy = copied_writer.replacen(
            "#[cfg(test)]",
            "/* GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.current_version, */\n#[cfg(test)]",
            1,
        );
        assert!(assert_codec_source_is_bound(&copied_writer_with_block_comment_decoy).is_err());

        let copied_reader = source.replacen(
            "GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.supports(version)",
            "GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.supports(2)",
            1,
        );
        let copied_reader_with_comment_decoy = copied_reader.replacen(
            "#[cfg(test)]",
            "// GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.supports(version)\n#[cfg(test)]",
            1,
        );
        assert!(assert_codec_source_is_bound(&copied_reader_with_comment_decoy).is_err());
        let copied_reader_with_block_comment_decoy = copied_reader.replacen(
            "#[cfg(test)]",
            "/* GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.supports(version) */\n#[cfg(test)]",
            1,
        );
        assert!(assert_codec_source_is_bound(&copied_reader_with_block_comment_decoy).is_err());
    }

    fn assert_codec_source_is_bound(source: &str) -> Result<(), String> {
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("the production source precedes its test module");
        let uncommented = strip_rust_comments(production)?;
        let compact = uncommented
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();

        if compact.contains("KNGCSF02") {
            return Err("production GCS codec restored a hard-coded current magic".to_string());
        }
        let writer = concat!(
            "full_authority_envelope_magic(",
            "GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.current_version,",
            ")"
        );
        if compact.matches(writer).count() != 1 {
            return Err(
                "writer is not bound exactly once to the public current version".to_string(),
            );
        }
        let reader = "GCS_FULL_AUTHORITY_ENVELOPE_COMPATIBILITY.supports(version)";
        if compact.matches(reader).count() != 1 {
            return Err("reader is not bound exactly once to the public range".to_string());
        }
        Ok(())
    }

    fn strip_rust_comments(source: &str) -> Result<String, String> {
        let mut characters = source.chars().peekable();
        let mut uncommented = String::with_capacity(source.len());

        while let Some(character) = characters.next() {
            if character == '"' {
                uncommented.push(character);
                let mut escaped = false;
                let mut closed = false;
                for string_character in characters.by_ref() {
                    uncommented.push(string_character);
                    if escaped {
                        escaped = false;
                    } else if string_character == '\\' {
                        escaped = true;
                    } else if string_character == '"' {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err("unterminated string while inspecting GCS codec source".to_string());
                }
                continue;
            }

            if character != '/' {
                uncommented.push(character);
                continue;
            }

            match characters.peek().copied() {
                Some('/') => {
                    characters.next();
                    for comment_character in characters.by_ref() {
                        if comment_character == '\n' {
                            uncommented.push('\n');
                            break;
                        }
                    }
                }
                Some('*') => {
                    characters.next();
                    let mut depth = 1_u32;
                    while depth > 0 {
                        let Some(comment_character) = characters.next() else {
                            return Err(
                                "unterminated block comment while inspecting GCS codec source"
                                    .to_string(),
                            );
                        };
                        match (comment_character, characters.peek().copied()) {
                            ('/', Some('*')) => {
                                characters.next();
                                depth += 1;
                            }
                            ('*', Some('/')) => {
                                characters.next();
                                depth -= 1;
                            }
                            ('\n', _) => uncommented.push('\n'),
                            _ => {}
                        }
                    }
                }
                _ => uncommented.push(character),
            }
        }

        Ok(uncommented)
    }
}
