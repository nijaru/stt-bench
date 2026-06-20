# Model-specific uv environments

STT model packages currently require incompatible dependency sets. The core
`stt-bench` package stays lightweight; each runner gets a standalone uv project
with its own lockfile and virtual environment.

Use the wrapper from the repo root:

```bash
scripts/run-model whisper --manifest data/manifests/conditions-test.jsonl \
  --model openai/whisper-large-v3 \
  --output results/whisper-test \
  --audio-dir data/generated-test
```

Available environments:

| Env | Purpose | Notes |
| --- | --- | --- |
| `whisper` | Whisper via Transformers | General local reference runner |
| `cohere` | Cohere Transcribe | Requires HF auth and model access |
| `qwen3` | Qwen3-ASR | Uses `qwen-asr`, pins Transformers independently |
| `parakeet` | NVIDIA Parakeet | Linux x86_64 / 4090 env; NeMo deps do not install on macOS ARM64 |

Direct form:

```bash
uv run --project model-envs/<env> stt-bench run ...
```

The `.venv/` directories under these projects are local artifacts and are not
committed. Commit `pyproject.toml`, `uv.lock`, and `.python-version`.
