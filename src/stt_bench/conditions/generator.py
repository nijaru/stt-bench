"""Condition generator: applies transforms to source clips to produce condition variants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ..audio_io import save_audio
from ..manifest import ConditionVariant, SourceClip, write_manifest
from .codec import apply_mp3_codec, apply_mulaw_codec
from .noise import apply_noise_condition
from .reverb import apply_reverb_condition

# Condition definitions: (condition_id, transform_type, params)
# Each condition maps to a real-world STT use case.
CONDITIONS = {
    # Baseline
    "clean": None,
    # Background noise (2 types × 2 SNR levels = degradation curve)
    "noise_cafe_snr_15": {"type": "noise", "noise_type": "cafe", "snr_db": 15},
    "noise_cafe_snr_10": {"type": "noise", "noise_type": "cafe", "snr_db": 10},
    "noise_traffic_snr_15": {"type": "noise", "noise_type": "traffic", "snr_db": 15},
    "noise_traffic_snr_10": {"type": "noise", "noise_type": "traffic", "snr_db": 10},
    # Room acoustics (2 room sizes)
    "reverb_office": {"type": "reverb", "rir_type": "office"},
    "reverb_hall": {"type": "reverb", "rir_type": "hall"},
    # Codec compression (3 common codecs)
    "codec_telephony": {"type": "codec_mulaw"},
    "codec_opus_low": {"type": "codec_opus", "bitrate_kbps": 6},
    "codec_aac_low": {"type": "codec_aac", "bitrate_kbps": 32},
    # Microphone quality (2 device types)
    "mic_phone": {"type": "mic", "mic_type": "phone"},
    "mic_laptop": {"type": "mic", "mic_type": "laptop"},
    # Environmental (HVAC/office noise)
    "noise_hvac": {"type": "noise", "noise_type": "hvac", "snr_db": 20},
}

SAMPLE_RATE = 16000


def _hash_audio(waveform: torch.Tensor) -> str:
    """SHA256 hash of audio tensor for checksums."""
    return hashlib.sha256(waveform.numpy().tobytes()).hexdigest()[:16]


def _load_audio(uri: str, cache_dir: Path) -> torch.Tensor:
    """Load audio from HTTP URL, HF dataset URI, or local path.

    URI format:
    - https://...     : direct download (cached to cache_dir)
    - hf://dataset/split/index : HF dataset streaming
    - local path     : soundfile load
    """
    from ..audio_io import load_audio

    if uri.startswith("http://") or uri.startswith("https://"):
        # Download with caching
        cache_dir.mkdir(parents=True, exist_ok=True)
        import hashlib

        url_hash = hashlib.sha256(uri.encode()).hexdigest()[:16]
        cached_path = cache_dir / f"source_{url_hash}.flac"

        if not cached_path.exists():
            import requests

            response = requests.get(uri, timeout=60)
            response.raise_for_status()
            with open(cached_path, "wb") as f:
                f.write(response.content)

        return load_audio(str(cached_path), target_sr=SAMPLE_RATE)

    elif uri.startswith("hf://"):
        # Durable HF dataset URI:
        # hf://<org>/<dataset>/<config>/<split>/<row_idx>
        parts = uri.replace("hf://", "").split("/")
        if len(parts) != 5:
            raise ValueError(
                "HF source URI must be hf://<org>/<dataset>/<config>/<split>/<row_idx>; "
                f"got {uri!r}"
            )
        dataset = "/".join(parts[:2])
        config = parts[2]
        split = parts[3]
        row_idx = int(parts[4])

        import requests

        rows_response = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": row_idx,
                "length": 1,
            },
            timeout=30,
        )
        rows_response.raise_for_status()
        rows = rows_response.json().get("rows", [])
        if not rows:
            raise FileNotFoundError(f"No HF dataset row found for {uri}")

        audio_list = rows[0]["row"].get("audio", [])
        if not audio_list:
            raise FileNotFoundError(f"No audio asset found for {uri}")
        audio_url = audio_list[0]["src"]
        return _load_audio(audio_url, cache_dir)
    else:
        return load_audio(uri, target_sr=SAMPLE_RATE)


def _pick_noise_asset(noise_type: str, assets_dir: Path) -> str | None:
    """Pick a noise recording for the given noise type. Returns path or None.

    Noise types:
    - cafe: restaurant/cafe ambiance (MUSAN noise category)
    - traffic: road/traffic noise (MUSAN noise category)
    - hvac: office HVAC/fan noise (MUSAN noise category)
    """
    # MUSAN has noise files in noise/free-sound/ and noise/sound-bible/
    noise_dir = assets_dir / "noise" / "musan"
    if not noise_dir.exists():
        return None

    # For v0, we use the same noise source for all types
    # In v1, we'd filter by metadata or use separate noise corpora
    for f in sorted(noise_dir.iterdir()):
        if f.suffix in (".wav", ".flac", ".mp3"):
            return str(f)
    return None


def _pick_rir(rir_type: str, assets_dir: Path) -> str | None:
    """Pick a room impulse response. Returns path or None.

    For v0, picks RIRs by index (office=earlier, hall=later).
    TODO: filter by RT60 metadata when available.
    """
    rir_dir = assets_dir / "rir" / "rirs-noises"
    if not rir_dir.exists():
        return None

    # Support both .wav and .flac
    rirs = sorted(list(rir_dir.glob("*.wav")) + list(rir_dir.glob("*.flac")))
    if not rirs:
        return None

    # For v0, pick by index. office=first half, hall=second half
    # TODO: filter by RT60 metadata (office: 0.4-0.6s, hall: 0.8-1.2s)
    mid = len(rirs) // 2
    if rir_type == "office":
        return str(rirs[0])
    else:  # hall
        return str(rirs[mid])


def generate_variants(
    source_clip: SourceClip,
    condition_ids: list[str],
    output_dir: Path,
    assets_dir: Path,
    seed: int = 42,
) -> list[ConditionVariant]:
    """Generate all condition variants for a source clip.

    Returns list of ConditionVariant records. Audio files written to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source audio
    waveform = _load_audio(source_clip.audio_uri, assets_dir)

    variants = []
    for cond_id in condition_ids:
        variant_id = f"{source_clip.clip_id}__{cond_id}"
        output_path = output_dir / f"{variant_id}.wav"

        if cond_id == "clean":
            # Clean: just copy/convert
            save_audio(str(output_path), waveform, SAMPLE_RATE)
            transforms = []
        else:
            cond_def = CONDITIONS[cond_id]
            if cond_def is None:
                continue

            result = waveform.clone()
            transforms = []

            if cond_def["type"] == "noise":
                noise_path = _pick_noise_asset(cond_def["noise_type"], assets_dir)
                if not noise_path:
                    raise FileNotFoundError(
                        f"No noise assets found for condition '{cond_id}'. "
                        f"Run 'stt-bench fetch-assets' first. "
                        f"Expected assets in: {assets_dir / 'noise' / 'musan'}"
                    )
                result, param = apply_noise_condition(
                    result,
                    noise_path,
                    cond_def["snr_db"],
                    sample_rate=SAMPLE_RATE,
                    seed=seed,
                )
                transforms.append(param)

            elif cond_def["type"] == "reverb":
                rir_path = _pick_rir(cond_def["rir_type"], assets_dir)
                if not rir_path:
                    raise FileNotFoundError(
                        f"No RIR assets found for condition '{cond_id}'. "
                        f"Run 'stt-bench fetch-assets' first. "
                        f"Expected assets in: {assets_dir / 'rir' / 'rirs-noises'}"
                    )
                # Office: shorter RIR (0.3-0.5s), Hall: longer RIR (1-2s)
                max_rir_seconds = 0.5 if cond_def["rir_type"] == "office" else 2.0
                # Office: subtle reverb, Hall: moderate reverb
                wet_dry_ratio = 0.05 if cond_def["rir_type"] == "office" else 0.10
                result, param = apply_reverb_condition(
                    result,
                    rir_path,
                    sample_rate=SAMPLE_RATE,
                    max_rir_seconds=max_rir_seconds,
                    wet_dry_ratio=wet_dry_ratio,
                )
                transforms.append(param)

            elif cond_def["type"] == "codec_mulaw":
                result, param = apply_mulaw_codec(result, sample_rate=SAMPLE_RATE)
                transforms.append(param)

            elif cond_def["type"] == "codec_mp3":
                result, param = apply_mp3_codec(
                    result,
                    sample_rate=SAMPLE_RATE,
                    bitrate_kbps=cond_def["bitrate_kbps"],
                )
                transforms.append(param)

            elif cond_def["type"] == "codec_opus":
                from .codec import apply_opus_codec

                result, param = apply_opus_codec(
                    result,
                    sample_rate=SAMPLE_RATE,
                    bitrate_kbps=cond_def["bitrate_kbps"],
                )
                transforms.append(param)

            elif cond_def["type"] == "codec_aac":
                from .codec import apply_aac_codec

                result, param = apply_aac_codec(
                    result,
                    sample_rate=SAMPLE_RATE,
                    bitrate_kbps=cond_def["bitrate_kbps"],
                )
                transforms.append(param)

            elif cond_def["type"] == "mic":
                from .mic import apply_mic_profile

                result, param = apply_mic_profile(
                    result,
                    mic_type=cond_def["mic_type"],
                    sample_rate=SAMPLE_RATE,
                )
                transforms.append(param)

            # Peak normalize to prevent clipping
            peak = result.abs().max()
            if peak > 0.99:
                result = result / peak * 0.95

            save_audio(str(output_path), result, SAMPLE_RATE)

        # Fail loudly if a non-clean condition produced no transforms
        if cond_id != "clean" and len(transforms) == 0:
            raise RuntimeError(
                f"Condition '{cond_id}' produced no transforms for clip "
                f"'{source_clip.clip_id}'. This indicates a missing condition handler."
            )

        variant = ConditionVariant(
            variant_id=variant_id,
            clip_id=source_clip.clip_id,
            condition_id=cond_id,
            source_uri=source_clip.audio_uri,
            reference_text=source_clip.reference_text,
            transforms=transforms,
            checksum_sha256=_hash_audio(waveform if cond_id == "clean" else result),
            sample_rate=SAMPLE_RATE,
            duration_seconds=waveform.shape[-1] / SAMPLE_RATE,
        )
        variants.append(variant)

    return variants


def generate_condition_manifest(
    source_manifest_path: Path,
    output_manifest_path: Path,
    condition_ids: list[str],
    output_dir: Path,
    assets_dir: Path,
    seed: int = 42,
) -> list[ConditionVariant]:
    """Generate condition variants from a source manifest.

    Reads source clips, generates variants, writes condition manifest.
    """
    from ..manifest import read_manifest

    sources = read_manifest(source_manifest_path, SourceClip)
    all_variants = []

    for source in sources:
        variants = generate_variants(
            source,
            condition_ids,
            output_dir,
            assets_dir,
            seed=seed,
        )
        all_variants.extend(variants)

    write_manifest(output_manifest_path, all_variants)
    return all_variants
