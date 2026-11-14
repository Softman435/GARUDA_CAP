"""A stand-in for the `faster_whisper` package used ONLY by the mocked transcription tests.

It copies the real library's public shapes (class names, NamedTuple fields, argument names of
WhisperModel.__init__ and WhisperModel.transcribe as of faster-whisper 1.1.x) so the REAL app function
`app.transcribe.transcribe()` can be exercised end to end without downloading a model.
It proves the app's logic. It does NOT prove that the genuine library works on your machine: that is what
`scripts\\diagnose.bat` (deep test) and the real end-to-end test are for.
"""
from __future__ import annotations

import contextlib
import sys
import types
from typing import NamedTuple, Optional


class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float


class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[list]
    temperature: Optional[float]


class TranscriptionInfo(NamedTuple):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[list]
    transcription_options: object
    vad_options: object


def seg(text, start, end, words=None):
    return Segment(1, 0, start, end, text, [], -0.2, 1.0, 0.01, words, 0.0)


class FakeWhisperModel:
    """`behavior(calls_options) -> (segments, info)` is set by each test."""
    behavior = None
    init_calls: list = []
    transcribe_calls: list = []
    init_error: Optional[BaseException] = None

    def __init__(self, model_size_or_path, device="auto", device_index=0, compute_type="default", cpu_threads=0,
                 num_workers=1, download_root=None, local_files_only=False):
        if FakeWhisperModel.init_error:
            raise FakeWhisperModel.init_error
        FakeWhisperModel.init_calls.append(dict(model=model_size_or_path, device=device, compute_type=compute_type,
                                                cpu_threads=cpu_threads, download_root=download_root,
                                                local_files_only=local_files_only))

    def transcribe(self, audio, language=None, task="transcribe", beam_size=5, word_timestamps=False,
                   vad_filter=False, condition_on_previous_text=True, initial_prompt=None, multilingual=False,
                   vad_parameters=None):
        options = dict(audio=audio, language=language, task=task, beam_size=beam_size, word_timestamps=word_timestamps,
                       vad_filter=vad_filter, condition_on_previous_text=condition_on_previous_text,
                       initial_prompt=initial_prompt, multilingual=multilingual, vad_parameters=vad_parameters)
        FakeWhisperModel.transcribe_calls.append(options)
        return FakeWhisperModel.behavior(options)


@contextlib.contextmanager
def install(behavior=None, init_error=None, drop_multilingual=False):
    """Make `import faster_whisper` return the fake. Restores everything afterwards."""
    FakeWhisperModel.behavior = staticmethod(behavior or (lambda o: ([], info())))
    FakeWhisperModel.init_calls, FakeWhisperModel.transcribe_calls = [], []
    FakeWhisperModel.init_error = init_error
    module = types.ModuleType("faster_whisper")
    model_cls = FakeWhisperModel
    if drop_multilingual:           # simulates an OLDER faster-whisper without `multilingual`
        class Older(FakeWhisperModel):
            def transcribe(self, audio, language=None, task="transcribe", beam_size=5, word_timestamps=False,
                           vad_filter=False, condition_on_previous_text=True, initial_prompt=None,
                           vad_parameters=None):
                return FakeWhisperModel.transcribe(self, audio, language, task, beam_size, word_timestamps, vad_filter,
                                                   condition_on_previous_text, initial_prompt,
                                                   vad_parameters=vad_parameters)
        model_cls = Older
    module.WhisperModel = model_cls
    previous = sys.modules.get("faster_whisper")
    sys.modules["faster_whisper"] = module
    try:
        yield FakeWhisperModel
    finally:
        if previous is None:
            sys.modules.pop("faster_whisper", None)
        else:
            sys.modules["faster_whisper"] = previous


def info(language="kn", duration=10.0):
    return TranscriptionInfo(language, 0.9, duration, duration, None, None, None)
