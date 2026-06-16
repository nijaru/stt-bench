# stt-bench

Real-world speech-to-text benchmark across voices, acoustic conditions, microphones, and models.

## Goal

Build a practical benchmark for local and API speech-to-text systems that answers questions standard WER leaderboards miss:

- Which voices and accents degrade first?
- How much do busy rooms, music beds, reverb, phone codecs, and cheap microphones hurt accuracy?
- Which models are fastest at acceptable accuracy on local hardware?
- Do quantized models preserve transcription quality under bad conditions?

## Evaluation axes

### Voices

- accents and dialects
- speaking pace
- pitch range
- age brackets where licensing allows
- read speech, conversational speech, and narration

### Conditions

- clean close-mic reference
- busy room beds layered under speech
- background music and crowd noise
- room reverb and far-field recordings
- microphone EQ curves and bandwidth limits
- codec compression and 8 kHz telephony
- controlled SNR sweeps for calibration

### Metrics

- word error rate (WER)
- character error rate (CER)
- real-time factor / throughput (RTFx)
- latency where available
- peak VRAM / memory
- hallucination and omission counts where measurable

## Initial scope

Start with a small, reproducible harness:

1. Prepare clean reference clips with transcripts.
2. Generate condition variants from licensed noise beds and DSP transforms.
3. Run multiple STT backends against the same manifest.
4. Produce per-condition and aggregate reports.

Python is the primary implementation language because the audio and ML tooling ecosystem is strongest there.
