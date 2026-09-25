"""Minimal stand-in for deepspeed, installed only in the EgoGPT inference venv.

EgoGPT's speech encoder (egogpt/model/speech_encoder/speech_encoder.py) imports deepspeed solely for
deepspeed.zero.GatheredParameters around loading the Whisper weights. For parameters that were never ZeRO-3 partitioned
(plain single-GPU inference) the real context manager is a no-op, so this stub reproduces its behaviour exactly.
The real package cannot be built on openvla (its setup imports triton, which needs a GPU driver).
"""
from . import zero  # noqa: F401

__version__ = "0.0.0-egogpt-inference-stub"
