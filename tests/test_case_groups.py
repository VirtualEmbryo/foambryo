"""The equilibrium split is correct, and no aggregate can silently cross it.

A recorded project decision (H. Turlier) excludes the five non-equilibrium `benchmarking-dataset`
cases from every inference-accuracy and attribution claim and keeps them as a documented negative
control (`foambryo_benchmarks/case_groups.py`). Three things are pinned here:

1. the hard-coded id list still matches what the committed dataset-audit records measure — so the
   decision cannot silently drift away from the data it rests on;
2. `require_single_group` raises on a mixed list, which is the mechanism that turns "report
   the two groups separately" from a convention into a structural property;
3. the two report generators (`run_dataset_audit.summarise`, `run_baseline.summarise_methods`)
   produce per-group output and no unlabelled all-47 aggregate.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from foambryo_benchmarks import case_groups
from foambryo_benchmarks.run_baseline import markdown_method_table, summarise_methods
from foambryo_benchmarks.run_dataset_audit import markdown_table, summarise

AUDIT_DIR = Path(__file__).resolve().parent.parent / "foambryo_benchmarks" / "baseline" / "dataset_audit"
BASELINE_DIR = Path(__file__).resolve().parent.parent / "foambryo_benchmarks" / "baseline"


def _load(directory: Path, pattern: str = "case_*.json") -> list[dict]:
    if not directory.is_dir():
        pytest.skip(f"committed baseline directory not found: {directory}")
    paths = sorted(directory.glob(pattern))
    if not paths:
        message = f"{directory} exists but has no files matching {pattern!r} -- a real gap, not a missing worktree"
        raise AssertionError(message)
    return [json.loads(path.read_text()) for path in paths]


def test_the_five_non_equilibrium_cases_are_the_ones_b1_measured():
    """The hard-coded list is re-derived from the committed audit records, not trusted.

    The dataset audit separated the two populations by `rho` **and** by where the residual sits: a case is
    non-equilibrium when no tensions balance it (`rho_free ~= rho`, both large) *and* the
    residual is spread over the interface interiors rather than concentrated at the triple
    lines. If a future re-run of the audit moves a case across either threshold, this test
    fails rather than letting `NON_EQUILIBRIUM_CASE_IDS` quietly become wrong.
    """
    records = _load(AUDIT_DIR)

    derived = {
        r["case"]
        for r in records
        if r["rho"] > case_groups.B1_RHO_THRESHOLD
        and r["residual_localisation"]["fraction_interface_interior"] > case_groups.B1_INTERIOR_FRACTION_THRESHOLD
    }

    assert derived == set(case_groups.NON_EQUILIBRIUM_CASE_IDS)


def test_the_two_populations_do_not_overlap_in_rho():
    """The split rests on a gap between two populations, not on a tuned cutoff.

    Recorded as a measurement so the *size* of the gap is visible: if it ever narrows, the
    threshold in `case_groups` becomes a judgement call and should be revisited rather than
    trusted. The dataset audit measured `rho` 0.027-0.100 on the 42 against 0.264-0.277 on the five.
    """
    grouped = case_groups.split_records(_load(AUDIT_DIR))
    equilibrium = np.array([r["rho"] for r in grouped[case_groups.EQUILIBRIUM]])
    non_equilibrium = np.array([r["rho"] for r in grouped[case_groups.NON_EQUILIBRIUM]])

    assert equilibrium.max() < non_equilibrium.min()
    assert non_equilibrium.min() / equilibrium.max() > 2.0


def test_case_004_is_kept_in_the_equilibrium_group_with_its_flag():
    """`004` fails integrity but is usable: the equilibrium-split decision keeps it, flagged."""
    assert case_groups.group_of("004") == case_groups.EQUILIBRIUM
    assert "004" in case_groups.INTEGRITY_FLAGGED_CASE_IDS

    records = {r["case"]: r for r in _load(AUDIT_DIR)}
    if "004" in records:
        assert records["004"]["integrity"]["intact"] is False
        assert records["004"]["rho"] < case_groups.B1_RHO_THRESHOLD


def test_require_single_group_rejects_a_mixed_list():
    """The guard that makes "never aggregate across the split" structural."""
    mixed = [{"case": "000"}, {"case": "005"}]

    with pytest.raises(ValueError, match="refusing to aggregate across the equilibrium split"):
        case_groups.require_single_group(mixed)


def test_require_single_group_accepts_a_pure_list_and_names_the_group():
    assert case_groups.require_single_group([{"case": "000"}, {"case": "004"}]) == case_groups.EQUILIBRIUM
    assert case_groups.require_single_group([{"case": "005"}, {"case": "043"}]) == case_groups.NON_EQUILIBRIUM


def test_require_single_group_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty record list"):
        case_groups.require_single_group([])


def test_split_case_ids_covers_every_case_exactly_once():
    ids = [f"{i:03d}" for i in range(47)]
    groups = case_groups.split_case_ids(ids)

    assert set(groups) == set(case_groups.GROUP_ORDER)
    assert sorted(groups[case_groups.EQUILIBRIUM] + groups[case_groups.NON_EQUILIBRIUM]) == ids
    assert len(groups[case_groups.NON_EQUILIBRIUM]) == len(case_groups.NON_EQUILIBRIUM_CASE_IDS)
    assert len(groups[case_groups.EQUILIBRIUM]) == 42


def test_dataset_audit_summary_is_split_and_has_no_mixed_aggregate():
    """`summarise` reports per group; the only dataset-wide block is integrity counts."""
    summary = summarise(_load(AUDIT_DIR))

    assert set(summary["groups"]) == set(case_groups.GROUP_ORDER)
    assert summary["groups"][case_groups.EQUILIBRIUM]["n_cases"] == 42
    assert summary["groups"][case_groups.NON_EQUILIBRIUM]["n_cases"] == len(case_groups.NON_EQUILIBRIUM_CASE_IDS)

    # Every quantity that could be read as an accuracy or equilibrium claim lives under a
    # group, not at the top level.
    assert set(summary["dataset"]) == {"n_cases", "n_by_group", "group_notes", "integrity"}
    for key in ("rho", "rho_free", "residual_localisation", "identifiability", "effective_tension_ratio_pooled"):
        assert key not in summary["dataset"]
        assert key in summary["groups"][case_groups.EQUILIBRIUM]

    # And each group's numbers really are computed from its own cases only.
    grouped = case_groups.split_records(_load(AUDIT_DIR))
    for group, items in grouped.items():
        expected_max = max(r["rho"] for r in items)
        assert summary["groups"][group]["rho"]["max"] == pytest.approx(expected_max)


def test_dataset_audit_table_separates_the_groups():
    table = markdown_table(_load(AUDIT_DIR))

    assert f"| **{case_groups.NON_EQUILIBRIUM}** |" in table
    separator_line = next(i for i, line in enumerate(table.splitlines()) if "| **non_equilibrium** |" in line)
    lines = table.splitlines()
    for case_id in sorted(case_groups.NON_EQUILIBRIUM_CASE_IDS):
        row = next(i for i, line in enumerate(lines) if line.startswith(f"| {case_id} |"))
        assert row > separator_line, f"case {case_id} must appear below the group separator"


def test_method_comparison_summary_is_split_by_group():
    """The method table — the one that carries every accuracy claim — is per group."""
    records = _load(BASELINE_DIR)
    summary = summarise_methods(records)

    assert set(summary["groups"]) == set(case_groups.GROUP_ORDER)
    assert summary["dataset"]["n_by_group"][case_groups.NON_EQUILIBRIUM] == len(
        case_groups.NON_EQUILIBRIUM_CASE_IDS,
    )
    # No method statistic at the top level, only counts and provenance notes.
    assert "methods" not in summary["dataset"]

    equilibrium = summary["groups"][case_groups.EQUILIBRIUM]
    assert set(equilibrium["cases"]).isdisjoint(case_groups.NON_EQUILIBRIUM_CASE_IDS)

    table = markdown_method_table(summary)
    assert f"## {case_groups.EQUILIBRIUM}" in table
    assert f"## {case_groups.NON_EQUILIBRIUM}" in table


def test_summarise_methods_refuses_a_hand_built_mixed_group():
    """Calling the per-group reducer directly on mixed records raises, not returns."""
    from foambryo_benchmarks.run_baseline import _summarise_methods_for_group

    records = _load(BASELINE_DIR)
    mixed = [r for r in records if r["case"] in {"000", "005"}]
    if len(mixed) < 2:
        pytest.skip("needs both case 000 and case 005 records")

    with pytest.raises(ValueError, match="refusing to aggregate"):
        _summarise_methods_for_group(mixed)
