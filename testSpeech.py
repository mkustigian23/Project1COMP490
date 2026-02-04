import pytest
import os
from speechToText import transcribe_file   # adjust import if filename changed


@pytest.fixture(scope="session")
def model_path():
    path = "vosk-model-small-en-us-0.15"
    if not os.path.exists(path):
        pytest.skip(f"Model folder missing: {path}")
    return path


def test_transcribe_known_audio(model_path):
    audio_file = "sample_audio.wav"

    assert os.path.exists(audio_file), f"Audio file not found: {audio_file}"

    print(f"Starting transcription on {audio_file}...")
    text = transcribe_file(audio_file, model_path)
    text = text.lower().strip()

    print("Recognized:", text)
    print("Length:", len(text))

    # Very loose check to start – adjust words to what YOU actually said
    assert len(text) > 3, "No transcription produced"

    # Example – change to your spoken words
    expected_phrases = ["hello", "hello this is", "he low this in a short sample audio file for transcription testing"]
    found = any(phrase in text for phrase in expected_phrases)
    assert found, f"None of {expected_phrases} found in: {text!r}"
