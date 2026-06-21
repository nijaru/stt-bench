# STT-Bench

How do speech-to-text models hold up when the audio isn't clean?

Existing benchmarks (Open ASR Leaderboard, LibriSpeech, FLEURS) report word error rates on studio-quality recordings. Real audio is noisy, reverberant, compressed, and captured on built-in mics. STT-Bench measures that gap by running each source clip through 13 acoustic conditions and scoring the transcripts.

## Results

30 clips from LibriSpeech test-clean, each degraded 13 ways (390 samples per model). All models were run on a single RTX 4090.

| Model | Params | Overall WER | Clean | Reverb (hall) | Reverb (office) | Latency |
|-------|-------:|------------:|------:|--------------:|----------------:|--------:|
| Cohere Transcribe | 2.0B | 2.5% [2.1–3.0] | 1.5% [0.8–2.3] | 6.0% [4.0–8.0] | 9.6% [7.0–12.6] | 0.10s |
| Parakeet TDT 1.1B | 1.1B | 3.9% [3.1–4.8] | 1.7% [0.9–2.5] | 8.4% [6.0–11.0] | 23.0% [16.1–31.0] | 0.05s |
| Qwen3-ASR | 1.7B | 4.6% [3.8–5.4] | 2.0% [1.1–2.9] | 11.3% [8.7–14.1] | 25.5% [21.2–29.8] | 0.40s |
| Whisper Large V3 | 1.55B | 5.5% [4.6–6.3] | 3.0% [1.8–4.4] | 15.1% [11.5–18.8] | 25.7% [21.7–29.6] | 0.53s |

*WER ranges are 95% bootstrap CIs (2000 resamples). Latency is average seconds per sample on the RTX 4090.*

**Findings:**

Among 13 conditions tested, only reverb meaningfully degrades WER. Noise, codecs, and mic profiles stay within 1.5× of clean across every model. Hall reverb multiplies clean WER by 4–6×; office reverb reaches 6–14× despite receiving half the wet signal (5% vs 10%). Dense early reflections in small rooms smear phoneme boundaries more than the diffuse tail of a large space. Cohere handles reverb best at 6× (office), where Parakeet and Qwen3 reach 13–14×.

## Conditions

Each condition maps to a real recording scenario. All effects use real noise (MUSAN) and real room impulse responses (OpenSLR-28), not synthetic approximations.

| Condition | Scenario |
|-----------|----------|
| `clean` | Quiet recording environment |
| `noise_cafe_snr_15` / `_snr_10` | Coffee shop, moderate to busy |
| `noise_traffic_snr_15` / `_snr_10` | Street, walking to busy |
| `noise_hvac` | Office HVAC / fan hum |
| `reverb_office` | Small room (office, home) |
| `reverb_hall` | Large space (conference hall) |
| `codec_telephony` | Phone call (G.711 mu-law) |
| `codec_opus_low` | Voice message (Opus 6 kbps) |
| `codec_aac_low` | Video call (AAC 32 kbps) |
| `mic_phone` | Smartphone recording |
| `mic_laptop` | Laptop internal mic |

## Usage

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nijaru/stt-bench.git
cd stt-bench
uv sync

# Download noise and RIR assets
uv run stt-bench fetch-assets

# Select source clips and generate degraded variants
uv run stt-bench select-sources --n-clips 30
uv run stt-bench prepare \
  --manifest data/manifests/sources-v0.jsonl \
  --output data/manifests/conditions-v0.jsonl

# Run a model in its isolated environment, then score
scripts/run-model whisper \
  --manifest data/manifests/conditions-v0.jsonl \
  --model openai/whisper-large-v3 \
  --output results/whisper-v0
uv run stt-bench score \
  --results-dir results/whisper-v0 \
  --manifest data/manifests/conditions-v0.jsonl
```

No audio or model weights live in the repo. Source clips and assets download on demand from Hugging Face and OpenSLR.

### Model environments

Model families pin conflicting versions of `transformers` (Qwen3 pins 4.57, Cohere needs 5.0+, Parakeet needs NeMo on Linux). Each model runs in its own uv project under `model-envs/`, invoked through a single wrapper:

```bash
scripts/run-model <env> --manifest <conditions.jsonl> --model <model-id> --output <dir>
```

| Env | Model | Notes |
|-----|-------|-------|
| `whisper` | Whisper / Transformers | Reference runner |
| `cohere` | Cohere Transcribe | Needs HF auth + model access |
| `qwen3` | Qwen3-ASR | `qwen-asr`, pinned transformers |
| `parakeet` | NVIDIA Parakeet TDT | NeMo; Linux x86_64 |

## Adding a model

1. Write `src/stt_bench/runners/<model>.py` implementing the runner protocol (load model, transcribe a variant, return a hypothesis).
2. Register the runner in the CLI.
3. Add a `model-envs/<model>/` uv project if the deps conflict with existing ones.
4. Run against an existing condition manifest and compare.

`docs/methodology.md` covers the full protocol, scoring, and reproducibility notes.

## License

Apache 2.0. See [LICENSE](LICENSE).

Built on [LibriSpeech](https://www.openslr.org/12/) (CC BY 4.0), [MUSAN](https://www.openslr.org/17/) (CC BY 4.0), [OpenSLR-28](https://www.openslr.org/28/) (Apache 2.0), and [jiwer](https://github.com/jitsi/jiwer).
