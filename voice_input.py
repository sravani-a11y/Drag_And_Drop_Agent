"""Optional voice input, for both the CLI and the FastAPI /api/agent/voice endpoint.

Speech-to-text is provided by openai/whisper-large-v3-turbo (via
transformers), loaded once per process and reused - not once per request.
Two entry points, sharing one transcription implementation:

    - get_voice_input(): CLI only. Captures live audio from a local
      microphone via speech_recognition's Microphone - used here purely
      for audio *capture*, not recognition - then transcribes it the same
      way transcribe_audio_bytes() does. Falls back to a plain `input()`
      prompt on any failure (package/model unavailable, no microphone,
      transcription failure), so main.py never needs to know which one
      actually happened.
    - transcribe_audio_bytes(): server-safe. Transcribes an already-recorded
      audio clip (e.g. an uploaded file). Returns None on any failure - no
      raise, no input() fallback - there is no terminal to prompt in a
      server process; api/server.py turns None into a clean HTTP error
      response instead.

Nothing here is imported unless one of these functions is actually called,
so a normal text-only run never needs torch/transformers/soundfile
installed, and the (multi-GB) model is only downloaded/loaded the first
time transcription is actually attempted, not at import time.

No component-name correction of any kind happens in this module - that is
command_normalizer.py's job, after this. This module's only responsibility
is producing the most accurate *raw* transcript it can.
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
_TARGET_SAMPLE_RATE = 16000

# Explicit - this deployment only ever expects English commands ("add
# ESP32", "connect X to Y"). Whisper otherwise auto-detects language per
# clip, which is one less source of variance to worry about for a
# single-language product.
_LANGUAGE = "english"

# Audio shorter than this almost never contains a real command - Whisper
# tends to hallucinate plausible-sounding filler ("Thank you.", "Add yes
# and don't do it.") when asked to transcribe something too short/quiet to
# actually contain speech. Rejecting it here means one fewer thing
# command_normalizer.py has to reject after the fact.
_MIN_DURATION_SECONDS = 0.3

# RMS energy below this is treated as near-silent. Deliberately low: a
# true digital-silence clip measures exactly 0.0, while even a
# deliberately very-quiet real recording (measured at 5% synthesized
# volume) still measures ~0.003 - this sits well below that, so it only
# ever catches genuinely empty/silent audio, not just quiet speech.
_MIN_RMS_ENERGY = 0.001

# Peak amplitude every accepted clip is normalized to before Whisper sees
# it, so a quiet recording is treated the same as a loud one. Deliberately
# not 1.0, to leave a little headroom.
_TARGET_PEAK_AMPLITUDE = 0.95

# Leading/trailing silence trimming: a real microphone recording almost
# always has a beat of silence (or room noise) before the speaker starts
# and after they finish - a synthesized test clip typically doesn't. That
# silence contributes nothing useful, dilutes the peak-normalization step
# below (the true speech content ends up quieter than it should be
# relative to the padding), and is a well-documented trigger for Whisper
# hallucinating content to "explain" audio that doesn't contain speech.
# Frames are 20ms; a frame counts as speech once its RMS is at least this
# fraction of the clip's own loudest frame (relative, not absolute, so it
# adapts to how loud/quiet the recording is) - only leading/trailing
# silence is trimmed, never a pause in the middle of a real command.
_SILENCE_TRIM_FRAME_MS = 20
_SILENCE_TRIM_RELATIVE_THRESHOLD = 0.08
_SILENCE_TRIM_PADDING_SECONDS = 0.15

_pipeline = None  # lazily built, cached module-level singleton - see _get_pipeline()


def _get_pipeline():
    """Build (once) and return the cached Whisper ASR pipeline.

    Loading openai/whisper-large-v3-turbo is expensive (a multi-GB
    download the first time this runs, then several seconds to load into
    memory) - this must happen once per process, not once per request.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from transformers import pipeline as hf_pipeline

    logger.info("Loading %s (first use only - this can take a while)...", _WHISPER_MODEL_ID)
    _pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model=_WHISPER_MODEL_ID,
        dtype=torch.float32,
        device="cpu",
    )
    logger.info("%s loaded", _WHISPER_MODEL_ID)
    return _pipeline


class _InvalidAudio(Exception):
    """Raised by _decode_and_prepare() for empty/too-short/near-silent audio - never sent to Whisper."""


def _trim_silence(data, sample_rate: int):
    """Trim leading/trailing near-silence from `data`, keeping a small margin either side.

    General signal-processing on whatever samples are present - nothing
    here depends on what's actually being said, so it applies identically
    to any command. A pause in the *middle* of real speech is never
    touched, only the lead-in/trail-off outside the first and last active
    frame. Returns `data` unchanged if it's too short to frame, or if no
    frame ever clears the (relative) activity threshold.
    """
    import numpy as np

    frame_length = max(1, int(sample_rate * _SILENCE_TRIM_FRAME_MS / 1000))
    num_frames = len(data) // frame_length
    if num_frames < 2:
        return data

    frames = data[: num_frames * frame_length].reshape(num_frames, frame_length)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))

    peak_rms = float(frame_rms.max())
    if peak_rms <= 0:
        return data

    active = np.flatnonzero(frame_rms >= (peak_rms * _SILENCE_TRIM_RELATIVE_THRESHOLD))
    if active.size == 0:
        return data

    padding_frames = max(1, int(_SILENCE_TRIM_PADDING_SECONDS * sample_rate / frame_length))
    start = max(0, (active[0] - padding_frames) * frame_length)
    end = min(len(data), (active[-1] + 1 + padding_frames) * frame_length)

    return data[start:end]


def _decode_and_prepare(audio_bytes: bytes):
    """Decode `audio_bytes`, validate it, and prepare it for Whisper: mono, 16kHz, float32, trimmed, peak-normalized.

    Raises _InvalidAudio for anything too short or too quiet to plausibly
    contain speech, and for anything soundfile/torchaudio can't decode at
    all - callers are expected to catch both and treat them the same way
    (never send it to Whisper). Logs the diagnostic line requested for
    debugging audio-quality-vs-decoding issues; never logs the audio data
    itself, only scalar measurements of it.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    channels = 1 if data.ndim == 1 else data.shape[1]
    duration = (len(data) / sample_rate) if sample_rate else 0.0
    energy = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0

    logger.info(
        "STT DEBUG:\nduration=%.2fs\nsample_rate=%d\nchannels=%d\naudio_energy=%.5f",
        duration,
        sample_rate,
        channels,
        energy,
    )

    if data.size == 0 or duration < _MIN_DURATION_SECONDS:
        raise _InvalidAudio(f"audio too short ({duration:.2f}s < {_MIN_DURATION_SECONDS}s minimum)")
    if energy < _MIN_RMS_ENERGY:
        raise _InvalidAudio(f"audio is near-silent (energy={energy:.5f} < {_MIN_RMS_ENERGY} minimum)")

    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono

    if sample_rate != _TARGET_SAMPLE_RATE:
        tensor = torch.from_numpy(data).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, sample_rate, _TARGET_SAMPLE_RATE)
        data = tensor.squeeze(0).numpy()

    data = _trim_silence(data, _TARGET_SAMPLE_RATE)
    if len(data) / _TARGET_SAMPLE_RATE < _MIN_DURATION_SECONDS:
        # Trimming ate almost the whole clip - the untrimmed version passed
        # the earlier whole-clip energy check only because of a brief loud
        # moment (a click, a cough) surrounded by silence, not real speech.
        raise _InvalidAudio("no sustained speech-like audio found after trimming silence")

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0:
        data = data * (_TARGET_PEAK_AMPLITUDE / peak)

    return data


def _transcribe_array(audio_array) -> Optional[str]:
    """Run the Whisper pipeline over an already-decoded, 16kHz mono float32 array.

    Deterministic, prompt-free decoding:
      - No initial_prompt/prompt_ids. A prior version of this module
        primed Whisper with the full component vocabulary (ESP32, STM32,
        DHT11, SPI, ...) as context - testing showed this made no
        measurable difference on clean audio, but under even mild noise it
        biased the model into repeating that vocabulary in a loop (e.g.
        "STM32, STM33, STM33, STM33, ..." dozens of times). Removing it
        entirely eliminated that failure mode in the same test conditions.
      - do_sample=False: greedy decoding (this was already the default,
        since temperature/sampling are only used when do_sample=True - this
        just makes that deterministic behavior explicit rather than
        implicit).
      - no_repeat_ngram_size=3: a standard, model-agnostic decoding
        constraint (not specific to "long-form" Whisper chunking, unlike
        no_speech_threshold/compression_ratio_threshold/logprob_threshold/
        condition_on_prev_tokens - the transformers docs describe all four
        of those as "only relevant for long-form transcription", and this
        module never chunks audio, so none of them would do anything here).
        Blocks any 3-token sequence from being generated twice in the same
        output, which directly targets repetition-loop hallucination
        regardless of what triggers it.
    Returns the raw transcript exactly as Whisper produced it - no
    hard-coded correction/substitution of any kind happens here or
    anywhere else in this module; that is command_normalizer.py's job,
    after this.
    """
    try:
        asr = _get_pipeline()
        result = asr(
            {"raw": audio_array, "sampling_rate": _TARGET_SAMPLE_RATE},
            generate_kwargs={
                "language": _LANGUAGE,
                "task": "transcribe",
                "do_sample": False,
                "no_repeat_ngram_size": 3,
            },
        )
        text = (result.get("text") or "").strip()
        logger.info('STT DEBUG:\nraw_transcript="%s"', text)
        return text or None
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return None


def get_voice_input(prompt: str = "Speak your request now...") -> str:
    """Capture one spoken request (microphone) and transcribe it; fall back to typed input on any failure."""
    try:
        import speech_recognition as sr
    except ImportError:
        logger.info("speech_recognition is not installed - falling back to text input")
        return input("(voice unavailable - type your request instead) > ").strip()

    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print(prompt)
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    except Exception as exc:
        logger.warning("Microphone capture failed (%s) - falling back to text input", exc)
        return input("(voice failed - type your request instead) > ").strip()

    text = transcribe_audio_bytes(audio.get_wav_data())
    if text:
        print(f"Heard: {text}")
        return text

    logger.warning("Voice input failed to transcribe - falling back to text input")
    return input("(voice failed - type your request instead) > ").strip()


def transcribe_audio_bytes(audio_bytes: bytes) -> Optional[str]:
    """Transcribe an already-recorded audio clip to text via openai/whisper-large-v3-turbo, or return None on failure.

    Never raises and never falls back to input() - this is meant for a
    server context with no terminal and no microphone. Accepts any format
    libsndfile can decode (WAV, FLAC, AIFF) - a browser's default
    webm/ogg/mp3 recording still needs converting to one of these first.

    Returns None (never calling Whisper at all) for empty, too-short, or
    near-silent audio, in addition to the existing "Whisper itself failed"
    case - both are surfaced identically to the caller today (api/server.py
    turns any None into the same clean error response); which specific
    reason it happened for is only visible in the logs.
    """
    if not audio_bytes:
        return None

    try:
        audio_array = _decode_and_prepare(audio_bytes)
    except _InvalidAudio as exc:
        logger.warning("Rejecting audio before sending it to Whisper: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Audio decoding failed: %s", exc)
        return None

    return _transcribe_array(audio_array)
