"""Test harness stub: grind.py is absent from this checkout. It is only
reached via fair_benchmark.session (fair battery); the official gate and
handcraft batteries drive ValhallaBase directly and never call it."""
from __future__ import annotations


def grind_text(*args, **kwargs):  # noqa: ANN002, ANN003
    raise RuntimeError("grind missing from checkout (stub)")


def grind_shard_for_native_qa(*args, **kwargs):  # noqa: ANN002, ANN003
    raise RuntimeError("grind missing from checkout (stub)")
