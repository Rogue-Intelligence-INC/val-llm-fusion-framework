"""Test harness stub: this repo checkout ships tools/fair_benchmark/
run_fair_benchmark.py which imports `artifact` at module level, but the
module itself is absent from the checkout. The functions are only used by
run_fair_benchmark's CLI main(), never by load_suite/select_prompts which
the official tests import. These stubs fail loudly if actually called so
they can never silently fake a result."""
from __future__ import annotations


def _missing(name: str):
    raise RuntimeError(f"artifact.{name} unavailable in this checkout (stub)")


def load_artifact(path):  # noqa: ANN001
    _missing("load_artifact")


def write_artifact(*args, **kwargs):  # noqa: ANN002, ANN003
    _missing("write_artifact")


def attach_eval_report(*args, **kwargs):  # noqa: ANN002, ANN003
    _missing("attach_eval_report")
