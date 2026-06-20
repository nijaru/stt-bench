# STT-Bench

Real-world robustness benchmarks for speech-to-text systems.

## The problem

Existing STT benchmarks (Open ASR Leaderboard, LibriSpeech, FLEURS) evaluate clean audio. Production audio is noisy, reverberant, compressed, and recorded on cheap hardware. STT-Bench measures this gap.

## What it does

Tests 4 SOTA STT models under 13 acoustic conditions:

| # | Condition | Real-world scenario |
|---|-----------|---------------------|
| 1 | `clean` | Quiet recording environment |
| 2 | `noise_cafe_snr_15` | Coffee shop (moderate) |
| 3 | `noise_cafe_snr_10` | Busy restaurant (noisy) |
| 4 | `noise_traffic_snr_15` | Walking near road |
| 5 | `noise_traffic_snr_10` | Busy street |
| 6 | `reverb_office` | Small room (office, home) |
| 7 | `reverb_hall` | Large space (conference hall) |
| 8 | `codec_telephony` | Phone call (G.711 mu-law) |
| 9 | `codec_opus_low` | Voice message (Opus 6kbps) |
| 10 | `codec_aac_low` | Video call (AAC 32kbps) |
| 11 | `mic_phone` | Smartphone recording |
| 12 | `mic_laptop` | Laptop internal mic |
| 13 | `noise_hvac` | Office HVAC/fan |

All transforms use real noise recordings (MUSAN) and real room impulse responses (OpenSLR-28). No synthetic artifacts.

## Models (v0)

| Model | Params | Clean WER | License |
|-------|--------|-----------|---------|
| Whisper Large V3 | 1.55B | 7.44% | MIT |
| Cohere Transcribe | 2B | 5.42% | Apache 2.0 |
| Qwen3-ASR | 1.7B | 5.76% | Apache 2.0 |
| Parakeet TDT 1.1B | 1.1B | ~8.0% | CC-BY-4.0 |

## Quickstart

```bash
# Install core benchmark tools
git clone https://github.com/nijaru/stt-bench.git
cd stt-bench
uv sync

# Download assets
uv run stt-bench fetch-assets

# Select source clips
uv run stt-bench select-sources --n-clips 30

# Generate condition variants
uv run stt-bench prepare \
  --manifest data/manifests/sources-v0.jsonl \
  --output data/manifests/conditions-v0.jsonl

# Run a model through its isolated model environment
scripts/run-model whisper \
  --manifest data/manifests/conditions-v0.jsonl \
  --model openai/whisper-large-v3 \
  --output results/whisper-v3

# Score results from the core environment
uv run stt-bench score \
  --results-dir results/whisper-v3 \
  --manifest data/manifests/conditions-v0.jsonl
```

## How it works

```
Source clips (LibriSpeech, downloaded on demand)
    → Condition generation (real noise + real RIRs + DSP transforms)
    → Model runners (HF models with pinned revisions)
    → Scoring (jiwer WER/CER, conservative normalization)
    → Reporting (tables, markdown)
```

No audio or model weights stored in the repo. Everything downloaded on demand.

## Model environments

Model packages have incompatible dependency constraints, so STT-Bench keeps the
core package lightweight and runs each model family in its own uv project under
`model-envs/`.

Use the wrapper from the repo root:

```bash
scripts/run-model <env> --manifest <conditions.jsonl> --model <model-id> --output <results-dir>
```

Available envs:

| Env | Model family | Notes |
|-----|--------------|-------|
| `whisper` | Whisper / Transformers | Reference local runner |
| `cohere` | Cohere Transcribe | Requires Hugging Face auth and model access |
| `qwen3` | Qwen3-ASR | Uses `qwen-asr` in an isolated env |
| `parakeet` | NVIDIA Parakeet | NeMo env; Linux x86_64 recommended |

Direct uv form:

```bash
uv run --project model-envs/cohere stt-bench run ...
```

## Project structure

```
src/stt_bench/
    cli.py              # Click CLI
    manifest.py         # JSONL manifest schemas
    scoring/            # WER/CER with jiwer
    conditions/         # Audio transforms (noise, reverb, codec, mic)
    runners/            # Model runners (Whisper, Cohere, Qwen3, Parakeet)
    data/               # Asset downloader, source selector
    reports/            # Tables and plots
data/
    manifests/          # Source + condition metadata (committed)
results/                # Ad hoc outputs ignored; curated releases in results/release/
model-envs/             # Isolated uv projects for incompatible model deps
scripts/run-model       # Uniform wrapper for model-specific environments
docs/
    methodology.md      # Full methodology and reproducibility docs
```

## Adding a model

1. Create `src/stt_bench/runners/<model>.py`
2. Implement the runner protocol (read manifest, transcribe, write hypothesis JSONL)
3. Register in CLI
4. Run against existing condition manifest

See `docs/methodology.md` for full protocol specification.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- [MUSAN](https://www.openslr.org/17/) — noise corpus (CC BY 4.0)
- [OpenSLR-28](https://www.openslr.org/28/) — room impulse responses (Apache 2.0)
- [LibriSpeech](https://www.openslr.org/12/) — source speech (CC BY 4.0)
- [jiwer](https://github.com/jitsi/jiwer) — WER/CER computation
