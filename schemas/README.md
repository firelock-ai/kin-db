# JSON schemas

These files are generated, not written by hand. Every one of them is
`schemars` output for a public type in
[`kin-model`](https://github.com/firelock-ai/kin-model), which owns the wire
contract that `kin-db` persists and serves. Nothing in `kin-db` reads them at
runtime. They are checked in so a reader can see the exact shape of the graph
payloads without building the crate.

The version they were generated from is recorded in
[`kin-model-version.txt`](kin-model-version.txt), and `scripts/check-schema-provenance.sh`
fails the build when that version stops matching the `kin-model` pin in
`crates/kin-db/Cargo.toml`. That guard is what catches the failure this
directory already suffered once: the pin moved from `0.5.0` to `=0.7.8` while
the snapshot stayed behind, so eleven types the newer `kin-model` defines had
no schema here at all and two types it had dropped still did.

The guard compares versions, not bytes. It cannot tell you that a file was
edited by hand, so do not edit one. Regenerate instead.

## Regenerating

The generator lives in `kin-model`, so run it from a checkout of that repo at
the tag matching the pin, and point it at this directory:

```bash
cd /path/to/kin-model
git checkout v0.7.8
cargo run --example generate_schemas -- /path/to/kin-db/schemas
```

Then update `kin-model-version.txt` to the same version and commit both. The
generator writes one file per type in its own list, so a type it has stopped
emitting leaves a stale file behind. Delete anything the run did not write.
