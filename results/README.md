# Results

Each benchmark run produces a directory under `results/<run-id>/` containing:

- `config.json` — run configuration, model IDs, benchmark version
- `hypotheses.jsonl` — model outputs per condition variant
- `scores.jsonl` — WER/CER per sample
- `summary.csv` — aggregate results table
- `summary.md` — human-readable report
- `failures.jsonl` — failed runs

Results are committed for published runs. Working runs can be gitignored.
