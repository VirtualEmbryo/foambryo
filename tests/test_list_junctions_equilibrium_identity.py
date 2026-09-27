"""The junction-edge classification fix: `list_junctions` is bitwise identical on clean meshes.

The fix's own prediction: on a mesh with no abnormal (malformed triple-line) edges,
`regions_per_edge == 3` and `(interfaces_per_edge == 3) & (triangles_per_edge == 3)`
coincide, so the fix must return bitwise-identical output to the pre-fix function. Tested
here by hashing `list_junctions`' full return value (both the region-triple list and the
edge-to-triple map) from the pre-fix implementation (frozen below, verbatim, as it read at
`v0.4.1` commit `6b701ce` -- not reimported, since the fix has already replaced it in
`foambryo.validation.mesh_io`) against the current, fixed one.

**The literal "all 42" prediction was falsified and is not asserted here.** Case `004` is
kept in the 42-case equilibrium group (`foambryo_benchmarks.case_groups`) despite being
flagged with 2 malformed triple-line edges (`INTEGRITY_FLAGGED_CASE_IDS`) -- i.e. it is NOT
a mesh with no abnormal edges, so the fix changes its `list_junctions` output by
construction, exactly as intended. That is asserted explicitly below, not silently skipped:
the conditional claim ("no abnormal edges implies bitwise identical") is what is actually
tested, and case 004 is the one case in this small sample that tests the *other* branch --
that the fix's change is exactly the malformed count, not any other edge.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
import pytest

from foambryo.validation.mesh_io import list_junctions, mesh_integrity, read_rec
from tests.conftest import DATASET_DIR

# A handful of the 42-case equilibrium group (`foambryo_benchmarks.case_groups`,
# `NON_EQUILIBRIUM_CASE_IDS = {"005", "015", "018", "021", "043"}`), not all 47 -- the
# full-cohort measurement is recorded in the development repository's research history, not
# here. "000"-"003" have zero malformed triple-line edges;
# "004" is the one documented exception (see module docstring).
CLEAN_CASES = ("000", "001", "002", "003")
FLAGGED_CASE = "004"


def _list_junctions_pre_fix_v0_4_1(f: np.ndarray, regions: np.ndarray):
    """Frozen, verbatim copy of `list_junctions` as it read at `v0.4.1` (commit `6b701ce`).

    Do not "fix" or refactor this copy: its entire purpose is to be the historical
    pre-fix behaviour to hash against, including the bug the junction-edge classification fix removed.
    """
    edge_regs: dict[tuple[int, int], set[int]] = defaultdict(set)
    for i in range(len(f)):
        a, b, c = f[i]
        ra, rb = regions[i]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_regs[(u, v) if u < v else (v, u)] |= {int(ra), int(rb)}
    junctions: set[tuple[int, int, int]] = set()
    edge_to_triple: dict[tuple[int, int], tuple[int, int, int]] = {}
    for e, rs in edge_regs.items():
        if len(rs) == 3:
            tri = tuple(sorted(rs))
            junctions.add(tri)
            edge_to_triple[e] = tri
    return sorted(junctions), edge_to_triple


def _hash_output(junctions: list, edge_to_triple: dict) -> str:
    edge_items = sorted(([int(x) for x in k], [int(x) for x in v]) for k, v in edge_to_triple.items())
    junctions_plain = [[int(x) for x in t] for t in junctions]
    payload = json.dumps({"junctions": junctions_plain, "edges": edge_items}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _require_case(case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh_path = DATASET_DIR / f"{case}_mesh.rec"
    if not mesh_path.exists():
        pytest.skip(f"{mesh_path} not found (sibling benchmarking-dataset/ missing)")
    return read_rec(mesh_path)


@pytest.mark.parametrize("case", CLEAN_CASES)
def test_bitwise_identical_on_clean_equilibrium_cases(case: str) -> None:
    v, f, regions = _require_case(case)
    integrity = mesh_integrity(v, f, regions)
    assert integrity["n_malformed_triple_line_edges"] == 0, (
        f"case {case} was expected to have no abnormal edges; re-check CLEAN_CASES"
    )

    junctions_pre, edge_to_triple_pre = _list_junctions_pre_fix_v0_4_1(f, regions)
    junctions_post, edge_to_triple_post = list_junctions(f, regions)

    assert _hash_output(junctions_pre, edge_to_triple_pre) == _hash_output(junctions_post, edge_to_triple_post)
    assert sorted(junctions_pre) == sorted(junctions_post)
    assert edge_to_triple_pre == edge_to_triple_post


def test_flagged_case_004_changes_by_exactly_its_malformed_count() -> None:
    """The falsified "all 42" claim, made precise: case 004 differs, by exactly 2 edges."""
    v, f, regions = _require_case(FLAGGED_CASE)
    integrity = mesh_integrity(v, f, regions)
    assert integrity["n_malformed_triple_line_edges"] == 2, (
        "case 004's known malformed-edge count changed; the measurement this test "
        "documents no longer matches the data -- re-measure before trusting this assertion"
    )

    junctions_pre, edge_to_triple_pre = _list_junctions_pre_fix_v0_4_1(f, regions)
    junctions_post, edge_to_triple_post = list_junctions(f, regions)

    assert _hash_output(junctions_pre, edge_to_triple_pre) != _hash_output(junctions_post, edge_to_triple_post)
    assert len(edge_to_triple_pre) - len(edge_to_triple_post) == 2
    # Every post-fix edge is still present pre-fix, unchanged: the fix only ever removes
    # edges, never relabels or adds one.
    assert edge_to_triple_post.items() <= edge_to_triple_pre.items()
