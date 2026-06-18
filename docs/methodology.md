# STT-Bench v0 Methodology

## Principles

- Reproducibility over dataset size
- Real recordings over synthetic speech
- Real noise and real impulse responses over synthetic alternatives
- Documented licenses over convenient data
- Raw artifacts retained, normalized artifacts derived
- Model-independent scoring before model-runner complexity

## Pipeline overview

```
Source clips (HF datasets, downloaded on demand)
    → Condition generation (real noise + real RIRs + DSP transforms)
    → Model runners (HF models with pinned revisions)
    → Scoring (jiwer, conservative normalization)
    → Reporting (tables, markdown, CSV)
```

No audio or model files stored in the repo. Everything downloaded on demand with checksums.

## Source speech

### Requirements

Each source clip must have:

- Stable clip ID
- Audio URI (Hugging Face dataset path + split + index)
- Verified human reference transcript
- License metadata
- Duration, sample rate, channel count
- Notes for transcript quirks (false starts, named entities, numbers)

### Target mix

30 clips from real recordings. No TTS-generated speech.

| Bucket | Target count | Source |
| --- | ---: | --- |
| Clean read speech | 8 | LibriSpeech test-clean |
| Narration / audiobook | 6 | LibriSpeech test-other |
| Conversational / spontaneous | 8 | Mozilla Common Voice |
| Accented English | 8 | Common Voice (diverse speakers) |

### Why real recordings only

TTS-generated speech is deterministic and license-clean, but has known prosodic artifacts and doesn't represent real speaker variation. Real recordings from LibriSpeech (public domain) and Common Voice (CC-0) are redistributable, diverse, and ecologically valid.

Some clips may have minor existing background noise. This is acceptable: the controlled variable is the conditions we add, not the source cleanliness.

## Condition generation

### Design: real noise + programmatic transforms

Conditions use real-world noise recordings and real impulse responses, applied programmatically. This avoids synthetic artifacts while keeping the pipeline deterministic and reproducible.

Each condition maps to a real-world STT use case. We test 13 conditions across 4 categories:

| # | Condition ID | Type | Real-world scenario | Assets |
|---|--------------|------|---------------------|--------|
| 1 | `clean` | Baseline | Quiet recording environment | — |
| 2 | `noise_cafe_snr_15` | Noise | Coffee shop (moderate) | MUSAN noise recordings |
| 3 | `noise_cafe_snr_10` | Noise | Busy restaurant (noisy) | MUSAN |
| 4 | `noise_traffic_snr_15` | Noise | Walking near road | MUSAN |
| 5 | `noise_traffic_snr_10` | Noise | Busy street | MUSAN |
| 6 | `reverb_office` | Reverb | Small room (office, home) | OpenSLR-28 RIRs |
| 7 | `reverb_hall` | Reverb | Large space (conference hall) | OpenSLR-28 RIRs |
| 8 | `codec_telephony` | Codec | Phone call (G.711 mu-law) | — |
| 9 | `codec_opus_low` | Codec | Voice message (Opus 6kbps) | — |
| 10 | `codec_aac_low` | Codec | Video call (AAC 32kbps) | — |
| 11 | `mic_phone` | Microphone | Smartphone recording | — |
| 12 | `mic_laptop` | Microphone | Laptop internal mic | — |
| 13 | `noise_hvac` | Noise | Office HVAC/fan | MUSAN |

Two noise levels per type (e.g., cafe at 15dB and 10dB SNR) give us a degradation curve to see how models degrade as noise increases.

### SNR measurement

SNR targets (15 dB, 10 dB) are goals. Each variant records the measured achieved SNR. SNR is computed as:

```
SNR_dB = 10 * log10(signal_power / noise_power)
```

Signal power computed over speech-active regions only (using VAD or transcript timestamps if available).

### Reverb conditions

Use real room impulse responses from OpenSLR-28 (~417 RIRs). Select RIRs by RT60 metadata:

- `reverb_office`: RT60 0.4-0.6s (small-medium room)
- `reverb_hall`: RT60 0.8-1.2s (large room/hall)

Convolution via torchaudio `fftconvolve` or scipy. Output peak-normalized.

### Codec conditions

- `codec_telephony`: Resample to 8kHz, apply mu-law companding (G.711), resample back to 16kHz
- `codec_lowbitrate`: MP3 compression at 32kbps via pedalboard, decode back to WAV

### Microphone condition

- `mic_cheap`: Bandpass 200-6000Hz + slight mid-frequency dip simulating cheap electret mic response

### Condition manifest format

Each generated variant stores full provenance:

```json
{
    "variant_id": "src_0001__noise_cafe_snr_15",
    "clip_id": "src_0001",
    "condition_id": "noise_cafe_snr_15",
    "source_uri": "hf://librispeech_asr/test-clean/...",
    "noise_asset_id": "musan_noise_0042",
    "transforms": [
        {"type": "add_background_noise", "snr_target_db": 15.0, "snr_achieved_db": 14.7, "seed": 42}
    ],
    "reference_text": "raw human transcript",
    "checksum_sha256": "...",
    "sample_rate": 16000,
    "duration_seconds": 18.42
}
```

## Noise and room assets

| Asset | Source | License | Content |
|-------|--------|---------|---------|
| MUSAN | OpenSLR | Apache 2.0 (research) | Cafe, street, office, babble, music |
| OpenSLR-28 | OpenSLR | Open | 417 real room impulse responses |
| OpenSLR-26 | OpenSLR | Open | Simulated RIRs (backup) |

Assets downloaded on demand via preparation script with checksum verification. Not stored in repo.

### Download-on-demand flow

1. `stt-bench prepare` reads source manifest
2. Downloads required noise/RIR assets to `~/.cache/stt-bench/assets/`
3. Verifies checksums
4. Generates condition variants to `data/generated/` (gitignored)
5. Writes condition manifest with full provenance

## Model runners

### Runner protocol

Each runner:

1. Reads condition manifest
2. Loads/downloads model from Hugging Face (pinned revision)
3. Transcribes each variant
4. Writes hypothesis JSONL

### Runner output format

```json
{
    "run_id": "2026-06-17_whisper_v3",
    "model_id": "openai/whisper-large-v3",
    "model_revision": "main",
    "runner": "whisper_local",
    "runner_version": "0.1.0",
    "variant_id": "src_0001__noise_cafe_snr_15",
    "hypothesis_text": "model output transcript",
    "runtime_seconds": 2.41,
    "started_at": "2026-06-17T00:00:00Z",
    "config_hash": "..."
}
```

### v0 runners

1. **Whisper** — simplest to implement, reference model, openai/whisper-large-v3 via transformers
2. **Cohere Transcribe** — HF transformers with CohereAsrForConditionalGeneration
3. **Qwen3-ASR** — qwen_asr library or transformers
4. **Parakeet TDT** — nemo toolkit or transformers

## Scoring

### Protocol

For each hypothesis:

1. Normalize reference text (conservative policy)
2. Normalize hypothesis text (same policy)
3. Compute WER, CER, insertions, deletions, substitutions via jiwer
4. Compute clean-baseline delta: `wer_delta = degraded_wer - clean_wer` (same clip, same model)

Missing clean runs → deltas marked unavailable, not estimated.

### Text normalization

Conservative, inspectable:

1. Unicode normalization (NFC)
2. Lowercase
3. Collapse whitespace
4. Strip punctuation (keep apostrophes)
5. No number normalization
6. No abbreviation normalization

Raw text retained. Normalized text computed at scoring time.

### Aggregation

Report at multiple levels:

- Per sample
- Per source clip across conditions
- Per condition across source clips
- Per model
- Whole benchmark aggregate

Primary metric: macro-averaged WER across samples (so long clips don't dominate).

## Reproducibility

Each run records:

- STT-Bench git commit
- Source manifest checksum
- Condition manifest checksum
- Model HF repo ID + revision
- Runner config hash
- Python version, platform
- Dependency versions

## Directory conventions

```
data/
    manifests/
        sources-v0.jsonl       # source clip metadata (committed)
        conditions-v0.jsonl    # condition variant metadata (committed)
    generated/                 # generated audio files (gitignored)
    assets/                    # downloaded noise/RIR assets (gitignored, or ~/.cache/stt-bench/)
results/
    <run-id>/
        config.json
        hypotheses.jsonl
        scores.jsonl
        summary.csv
        summary.md
        failures.jsonl
```

## Open research steps

1. Verify MUSAN redistribution license terms for public benchmark
2. Verify OpenSLR-28 redistribution terms
3. Select 30 source clips with good variety (pace, accent, recording quality)
4. Implement condition generation pipeline
5. Implement scoring pipeline
6. Implement first runner (Whisper)
7. End-to-end validation on 5 clips
