"""Optional voice input, for both the CLI and the FastAPI /api/agent/voice endpoint.

Speech-to-text is provided by Groq's hosted Whisper API
(whisper-large-v3-turbo - the same Whisper model this module previously
ran locally), so the server needs no PyTorch/transformers and no multi-GB
model in memory. The API key is read from the GROQ_API_KEY environment
variable; it is never hard-coded or logged. Two entry points, sharing one
transcription implementation:

    - get_voice_input(): CLI only. Captures live audio from a local
      microphone via speech_recognition's Microphone - used here purely
      for audio *capture*, not recognition - then transcribes it the same
      way transcribe_audio_bytes() does. Falls back to a plain `input()`
      prompt on any failure (package unavailable, no microphone,
      transcription failure), so main.py never needs to know which one
      actually happened.
    - transcribe_audio_bytes(): server-safe. Transcribes an already-recorded
      audio clip (e.g. an uploaded file). Returns None on any failure - no
      raise, no input() fallback - there is no terminal to prompt in a
      server process; api/server.py turns None into a clean HTTP error
      response instead.

No component-name correction of any kind happens in this module - that is
command_normalizer.py's job, after this. This module's only responsibility
is producing the most accurate *raw* transcript it can.
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_GROQ_API_KEY_ENV = "GROQ_API_KEY"
_WHISPER_MODEL_ID = "whisper-large-v3-turbo"
_REQUEST_TIMEOUT_SECONDS = 30

# Explicit - this deployment only ever expects English commands ("add
# ESP32", "connect X to Y"). Whisper otherwise auto-detects language per
# clip, which is one less source of variance to worry about for a
# single-language product.
_LANGUAGE = "en"

# Groq's upload limit for audio files.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Audio shorter than this almost never contains a real command - Whisper
# tends to hallucinate plausible-sounding filler when asked to transcribe
# something too short to actually contain speech.
_MIN_DURATION_SECONDS = 0.3

# What Whisper reliably "hears" in silent or near-silent audio (measured:
# a silent clip comes back as "Thank you." with no_speech_prob 0, so the
# API's own scores can't flag it). None of these is a canvas command, so a
# transcript that is exactly one of them is treated as no speech.
_SILENCE_HALLUCINATIONS = frozenset({"thank you", "thanks for watching", "thank you for watching", "you", "bye"})

# Groq picks the decoder from the file extension, but the endpoint only
# receives raw bytes - so the format is recognised from its first bytes.
# Covers what browsers record: WebM (Chrome/Edge), OGG (Firefox), MP4
# (Safari), plus WAV/FLAC/MP3.
_FORMAT_SIGNATURES = (
    (b"\x1a\x45\xdf\xa3", "webm"),
    (b"OggS", "ogg"),
    (b"RIFF", "wav"),
    (b"fLaC", "flac"),
    (b"ID3", "mp3"),
)


def _guess_extension(audio_bytes: bytes) -> str:
    """Return the file extension Groq needs for `audio_bytes`, from its magic bytes."""
    for signature, extension in _FORMAT_SIGNATURES:
        if audio_bytes.startswith(signature):
            return extension
    if audio_bytes[4:8] == b"ftyp":
        return "mp4"
    if audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    # Unknown header: let Groq try it as WebM (the browser default) and
    # report its own error if it can't decode it.
    return "webm"


def _is_silence_hallucination(text: str) -> bool:
    """True if `text` has no letters at all, or is one of Whisper's known silence phrases."""
    words = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return not words or words in _SILENCE_HALLUCINATIONS


def _transcribe_with_groq(audio_bytes: bytes) -> Optional[str]:
    """Send `audio_bytes` to Groq's Whisper API and return the raw transcript, or None on failure.

    Deterministic, prompt-free decoding (temperature 0, no prompt), the same
    settings the local pipeline used: priming Whisper with the component
    vocabulary previously made it repeat that vocabulary in a loop under
    noise, so no prompt is sent.
    """
    api_key = os.environ.get(_GROQ_API_KEY_ENV)
    if not api_key:
        logger.error("Speech-to-text is not configured: the %s environment variable is not set", _GROQ_API_KEY_ENV)
        return None

    import httpx

    extension = _guess_extension(audio_bytes)
    logger.info(
        "STT: sending %d bytes (detected format: %s) to Groq %s", len(audio_bytes), extension, _WHISPER_MODEL_ID
    )

    try:
        response = httpx.post(
            _GROQ_TRANSCRIPTION_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": _WHISPER_MODEL_ID,
                "language": _LANGUAGE,
                "temperature": "0",
                "response_format": "verbose_json",
            },
            files={"file": (f"audio.{extension}", audio_bytes)},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException:
        logger.error("STT failed: Groq did not respond within %d s", _REQUEST_TIMEOUT_SECONDS)
        return None
    except httpx.HTTPError as exc:
        logger.error("STT failed: could not reach Groq (%s: %s)", type(exc).__name__, exc)
        return None

    if response.status_code != 200:
        try:
            error = response.json().get("error", {})
            detail = f"{error.get('code') or error.get('type')}: {error.get('message')}"
        except ValueError:
            detail = response.text[:300]
        logger.error("STT failed: Groq returned HTTP %d (%s)", response.status_code, detail)
        return None

    try:
        body = response.json()
    except ValueError:
        logger.error("STT failed: Groq returned a non-JSON response: %r", response.text[:300])
        return None

    text = (body.get("text") or "").strip()
    duration = body.get("duration")
    logger.info('STT DEBUG:\nduration=%ss\nraw_transcript="%s"', duration, text)

    if isinstance(duration, (int, float)) and duration < _MIN_DURATION_SECONDS:
        logger.warning("Rejecting transcript: audio too short (%.2fs < %ss minimum)", duration, _MIN_DURATION_SECONDS)
        return None
    if _is_silence_hallucination(text):
        logger.warning("Rejecting transcript %r: no speech detected (silent or near-silent audio)", text)
        return None

    return text


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
    """Transcribe an already-recorded audio clip to text via Groq's Whisper API, or return None on failure.

    Never raises and never falls back to input() - this is meant for a
    server context with no terminal and no microphone. Accepts the formats
    browsers record (WebM/Opus, OGG/Opus, MP4/AAC) plus WAV, FLAC and MP3,
    sent to Groq as-is - no local decoding or conversion.

    Returns None for empty, oversized, too-short or silent audio, and when
    the API call fails - all surfaced identically to the caller
    (api/server.py turns any None into the same clean error response); the
    specific reason is in the logs.
    """
    if not audio_bytes:
        logger.warning("Rejecting audio: the upload is empty (0 bytes)")
        return None
    if len(audio_bytes) > _MAX_UPLOAD_BYTES:
        logger.warning(
            "Rejecting audio: %d bytes is over the %d-byte speech-to-text upload limit", len(audio_bytes), _MAX_UPLOAD_BYTES
        )
        return None

    try:
        return _transcribe_with_groq(audio_bytes)
    except Exception:
        logger.exception("STT failed with an unexpected error")
        return None
