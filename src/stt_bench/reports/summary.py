"""Report generation: tables and markdown from score JSONL."""

from __future__ import annotations

from pathlib import Path

from ..manifest import iter_manifest
from ..scoring.score import SampleScore, aggregate_scores


def generate_summary_md(scores_path: Path, output_path: Path | None = None) -> str:
    """Generate a summary markdown report from a scores JSONL file.

    Returns the markdown string. Writes to output_path if given.
    """
    scores = list(iter_manifest(scores_path, SampleScore))

    if not scores:
        lines = ["# STT-Bench Results", "", "No scores found.", ""]
        text = "\n".join(lines)
        if output_path:
            output_path.write_text(text)
        return text

    model_id = scores[0].model_id
    condition_ids = sorted(set(s.variant_id.split("__", 1)[1] for s in scores))

    lines = [f"# STT-Bench Results: {model_id}", ""]
    lines.append(f"**Samples scored:** {len(scores)}")
    lines.append("")

    overall = aggregate_scores(scores, model_id)
    lines.append(f"**Overall WER:** {overall.macro_wer:.1%}")
    lines.append(f"**Overall CER:** {overall.macro_cer:.1%}")
    lines.append(f"**Total insertions:** {overall.total_insertions}")
    lines.append(f"**Total deletions:** {overall.total_deletions}")
    lines.append(f"**Total substitutions:** {overall.total_substitutions}")
    lines.append("")

    lines.append("## Per-condition WER")
    lines.append("")
    lines.append("| Condition | Samples | WER | CER | Worst WER |")
    lines.append("|-----------|---------|-----|-----|-----------|")

    for cond_id in condition_ids:
        agg = aggregate_scores(scores, model_id, condition_id=cond_id)
        lines.append(
            f"| {cond_id} | {agg.n_samples} "
            f"| {agg.macro_wer:.1%} | {agg.macro_cer:.1%} "
            f"| {agg.worst_wer:.1%} |"
        )

    lines.append("")

    text = "\n".join(lines)
    if output_path:
        output_path.write_text(text)
    return text
