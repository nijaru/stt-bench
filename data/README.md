# Data

Audio files and model weights are not stored in this repo. They are downloaded on demand from Hugging Face and other sources.

## Manifests

- `manifests/sources-v0.jsonl` — source clip metadata (IDs, URIs, transcripts, licenses)
- `manifests/conditions-v0.jsonl` — generated condition variant metadata

## Generated files

`data/generated/` contains audio files produced by `stt-bench prepare`. These are gitignored and reproducible from manifests + downloaded assets.

## Assets

Noise recordings and room impulse responses are downloaded to `~/.cache/stt-bench/assets/` (or a configured path) by the preparation script. Checksums verified on download.
