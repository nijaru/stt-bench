"""CLI entrypoint for stt-bench."""

import click


@click.group()
@click.version_option(package_name="stt-bench")
def main():
    """STT-Bench: Real-world robustness benchmarks for speech-to-text systems."""


@main.command()
@click.option("--manifest", required=True, help="Path to condition manifest JSONL.")
@click.option("--model", required=True, help="Model ID (e.g. openai/whisper-large-v3).")
@click.option("--output", required=True, help="Output directory for results.")
def run(manifest: str, model: str, output: str):
    """Run a model against a condition manifest."""
    click.echo(f"Running {model} on {manifest} -> {output}")
    # TODO: implement
    raise SystemExit(1)


@main.command()
@click.option("--results-dir", required=True, help="Directory with hypothesis JSONL files.")
@click.option("--manifest", required=True, help="Path to condition manifest JSONL.")
@click.option("--output", required=True, help="Output directory for scores.")
def score(results_dir: str, manifest: str, output: str):
    """Score precomputed hypotheses against references."""
    click.echo(f"Scoring {results_dir} against {manifest} -> {output}")
    # TODO: implement
    raise SystemExit(1)


@main.command()
@click.option("--scores-dir", required=True, help="Directory with score JSONL files.")
@click.option("--output", required=True, help="Output directory for report.")
def report(scores_dir: str, output: str):
    """Generate tables and plots from scores."""
    click.echo(f"Generating report from {scores_dir} -> {output}")
    # TODO: implement
    raise SystemExit(1)


@main.command()
@click.option("--manifest", required=True, help="Source manifest to prepare conditions for.")
@click.option("--output", required=True, help="Output path for condition manifest.")
def prepare(manifest: str, output: str):
    """Generate condition variants from a source manifest."""
    click.echo(f"Preparing conditions from {manifest} -> {output}")
    # TODO: implement
    raise SystemExit(1)


if __name__ == "__main__":
    main()
