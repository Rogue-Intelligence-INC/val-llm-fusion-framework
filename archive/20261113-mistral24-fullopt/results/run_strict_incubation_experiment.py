"""Test harness stub: run_strict_incubation_experiment is absent from
this checkout. Its only symbol used by the official tests,
strict_user_text(prompt), is called solely inside the fair battery, which
additionally needs the absent incubation_benchmark_suite.json data.
Raising here keeps that dependency explicit instead of faking results."""
from __future__ import annotations


def strict_user_text(prompt):  # noqa: ANN001
    raise RuntimeError(
        "run_strict_incubation_experiment missing from checkout; "
        "fair battery unavailable (stub)"
    )
