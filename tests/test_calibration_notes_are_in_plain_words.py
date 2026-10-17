"""The calibration notes users read at run time state their evidence in plain words, with no internal reference.

`infer_tensions_with_uncertainty` stores a calibration note on every result. Up to 0.7.0 the `LineBootstrap` note
justified itself by "the per-edge Young-Dupre work", a name for a piece of development no user can look up. Pins:
no note names a piece of work, a development item or a phase; the `LineBootstrap` note keeps its claim and its
numbers (a factor of about 4 in the weights between lines, two orders of magnitude in the error).
"""

from __future__ import annotations

import re

import pytest

from foambryo.uncertainty import UncertaintyModel, _calibration_note

INTERNAL_REFERENCE = re.compile(r"\bwork\b|\bR\d{3}\b|\bphase\b", re.IGNORECASE)


@pytest.mark.parametrize("model", list(UncertaintyModel), ids=lambda model: model.name)
def test_no_note_names_internal_work(model: UncertaintyModel) -> None:
    note = _calibration_note(model)
    assert not INTERNAL_REFERENCE.search(note), note


def test_the_line_bootstrap_note_keeps_its_evidence() -> None:
    note = _calibration_note(UncertaintyModel.LineBootstrap)
    assert "factor of about 4" in note
    assert "two orders of magnitude" in note
    assert "Diagnostic only" in note
