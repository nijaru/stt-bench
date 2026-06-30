"""CLI entrypoint for stt-bench."""

import json
import sys
from pathlib import Path

import click

from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="stt-bench")
def main():
    """STT-Bench: Real-world robustness benchmarks for speech-to-text systems."""


@main.command()
@click.option(
    "--output-dir",
    default="~/.cache/stt-bench/assets",
    type=click.Path(),
    help="Directory to store downloaded assets.",
)
@click.option("--n-noise", default=50, type=int, help="Number of noise files to download.")
@click.option("--n-rir", default=20, type=int, help="Number of RIR files to download.")
def fetch_assets(output_dir, n_noise, n_rir):
    """Download noise and RIR assets from Hugging Face."""
    from .data.fetch_assets import fetch_all_assets

    output_path = Path(output_dir).expanduser()
    click.echo(f"Downloading assets to {output_path}")
    paths = fetch_all_assets(output_path, n_noise=n_noise, n_rir=n_rir)
    click.echo(f"Noise: {paths['noise']}")
    click.echo(f"RIRs: {paths['rir']}")


@main.command()
@click.option("--n-clips", default=30, type=int, help="Number of clips to select.")
@click.option("--min-duration", default=10.0, type=float, help="Min clip duration (seconds).")
@click.option("--max-duration", default=30.0, type=float, help="Max clip duration (seconds).")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option(
    "--output",
    default="data/manifests/sources-v0.jsonl",
    type=click.Path(),
    help="Output manifest path.",
)
def select_sources(n_clips, min_duration, max_duration, seed, output):
    """Select source clips from LibriSpeech test-clean."""
    from .data.select_sources import select_librispeech_clips, write_source_manifest

    click.echo(f"Selecting {n_clips} clips from LibriSpeech test-clean...")
    clips = select_librispeech_clips(
        n_clips=n_clips,
        min_duration=min_duration,
        max_duration=max_duration,
        seed=seed,
    )
    write_source_manifest(clips, Path(output))
    click.echo(f"Speakers: {len(set(c.speaker_id for c in clips if c.speaker_id))}")
    click.echo(f"Duration: {sum(c.duration_seconds for c in clips):.0f}s total")


@main.command()
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True),
    help="Source manifest JSONL.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path for condition manifest.",
)
@click.option(
    "--conditions",
    default=None,
    help="Comma-separated condition IDs (default: all v0).",
)
@click.option(
    "--output-dir",
    default="data/generated",
    type=click.Path(),
    help="Directory for generated audio.",
)
@click.option(
    "--assets-dir",
    default="~/.cache/stt-bench/assets",
    type=click.Path(),
    help="Directory with noise/RIR assets.",
)
@click.option("--seed", default=42, type=int, help="Random seed.")
def prepare(manifest, output, conditions, output_dir, assets_dir, seed):
    """Generate condition variants from a source manifest."""
    from .conditions.generator import CONDITIONS, generate_condition_manifest

    manifest_path = Path(manifest)
    output_path = Path(output)
    output_dir_path = Path(output_dir).expanduser()
    assets_dir_path = Path(assets_dir).expanduser()

    if conditions:
        cond_ids = [c.strip() for c in conditions.split(",")]
    else:
        cond_ids = list(CONDITIONS.keys())

    click.echo(f"Generating {len(cond_ids)} conditions from {manifest}")
    click.echo(f"Assets: {assets_dir_path}")
    click.echo(f"Output: {output_dir_path}")

    variants = generate_condition_manifest(
        manifest_path,
        output_path,
        cond_ids,
        output_dir_path,
        assets_dir_path,
        seed=seed,
    )

    click.echo(f"Generated {len(variants)} variants -> {output}")


@main.command()
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True),
    help="Condition manifest JSONL.",
)
@click.option("--model", required=True, help="Model ID (e.g. openai/whisper-large-v3).")
@click.option("--output", required=True, type=click.Path(), help="Output directory.")
@click.option(
    "--audio-dir",
    default="data/generated",
    type=click.Path(),
    help="Directory containing generated audio files.",
)
@click.option("--device", default="auto", help="Device: auto, cpu, cuda, mps.")
def run(manifest, model, output, audio_dir, device):
    """Run a model against a condition manifest."""
    from .manifest import ConditionVariant, iter_manifest
    from .runners import get_runner

    manifest_path = Path(manifest)
    output_path = Path(output)
    audio_dir_path = Path(audio_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runner = get_runner(model, device=device)
    click.echo(f"Running {model} on {manifest}")

    hypotheses = []
    for variant in iter_manifest(manifest_path, ConditionVariant):
        click.echo(f"  {variant.variant_id}...", nl=False)
        hyp = runner.transcribe(variant, audio_dir=audio_dir_path)
        hypotheses.append(hyp)
        click.echo(f" {hyp.runtime_seconds:.1f}s")

    hyp_path = output_path / "hypotheses.jsonl"
    with open(hyp_path, "w") as f:
        for h in hypotheses:
            f.write(h.to_json() + "\n")

    config = {
        "model_id": model,
        "manifest": str(manifest_path),
        "n_samples": len(hypotheses),
        "runner": runner.name,
        "runner_version": runner.version,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    click.echo(f"Wrote {len(hypotheses)} hypotheses -> {hyp_path}")


@main.command()
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with hypothesis JSONL.",
)
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True),
    help="Condition manifest JSONL.",
)
def score(results_dir, manifest):
    """Score precomputed hypotheses against references."""
    from .manifest import ConditionVariant, Hypothesis, iter_manifest, read_manifest
    from .reports.summary import generate_summary_md
    from .scoring.score import score_sample

    results_path = Path(results_dir)
    manifest_path = Path(manifest)

    variants = {v.variant_id: v for v in iter_manifest(manifest_path, ConditionVariant)}

    hyp_path = results_path / "hypotheses.jsonl"
    hypotheses = read_manifest(hyp_path, Hypothesis)

    click.echo(f"Scoring {len(hypotheses)} hypotheses against {len(variants)} variants")

    scores = []
    for hyp in hypotheses:
        variant = variants.get(hyp.variant_id)
        if not variant:
            click.echo(f"  WARNING: no reference for {hyp.variant_id}", err=True)
            continue

        sample_score = score_sample(
            variant_id=hyp.variant_id,
            model_id=hyp.model_id,
            reference=variant.reference_text,
            hypothesis=hyp.hypothesis_text,
        )
        scores.append(sample_score)

    scores_path = results_path / "scores.jsonl"
    with open(scores_path, "w") as f:
        for s in scores:
            f.write(s.to_json() + "\n")

    summary_path = results_path / "summary.md"
    generate_summary_md(scores_path, output_path=summary_path)

    click.echo(f"Scored {len(scores)} samples -> {scores_path}")
    click.echo(f"Summary -> {summary_path}")


@main.command()
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with scores.jsonl.",
)
def report(results_dir):
    """Generate summary report from scored results."""
    from .reports.summary import generate_summary_md

    results_path = Path(results_dir)
    scores_path = results_path / "scores.jsonl"

    if not scores_path.exists():
        click.echo(f"No scores.jsonl found in {results_dir}", err=True)
        sys.exit(1)

    summary_path = results_path / "summary.md"
    generate_summary_md(scores_path, output_path=summary_path)
    click.echo(f"Report generated -> {summary_path}")


if __name__ == "__main__":
    main()
