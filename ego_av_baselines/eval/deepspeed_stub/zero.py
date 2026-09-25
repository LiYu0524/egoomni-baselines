import contextlib


class GatheredParameters(contextlib.nullcontext):
    """No-op: nothing is partitioned without ZeRO-3 (real deepspeed also returns immediately for plain tensors)."""

    def __init__(self, params=None, modifier_rank=None, fwd_module=None, enabled=True):
        super().__init__()


class Init(contextlib.nullcontext):
    def __init__(self, *args, **kwargs):
        super().__init__()
