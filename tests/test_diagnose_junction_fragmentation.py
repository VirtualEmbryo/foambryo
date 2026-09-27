"""Tests for `foambryo_benchmarks.diagnose_junction_fragmentation` (backs the converged-cohort fragmentation claim).

`test_fragmentation_matches_converged_cohort_stages` reproduces, from the committed `.rec`
meshes directly, the exact per-stage component counts behind `YoungDupre`'s error pinning at 1.0 on
the converged cohort -- so that claim is checked against code, not only against the interactive
analysis that found it. (Fragmentation was once proposed as the mechanism behind the pooled
`rho_free` prediction test's anomaly and was ruled out: fixing it leaves that test unchanged.)
"""

from __future__ import annotations

import numpy as np

from foambryo_benchmarks.diagnose_junction_fragmentation import triple_junction_component_sizes
from foambryo.validation import mesh_io
from tests.conftest import WORKSPACE_ROOT

CONVERGED_MESHES_DIR = WORKSPACE_ROOT / "converged-meshes"


def _require_converged_meshes():
    import pytest

    if not CONVERGED_MESHES_DIR.exists():
        pytest.skip(f"{CONVERGED_MESHES_DIR} not found (sibling converged-meshes/ missing)")


def test_fully_connected_triple_junction_graph_returns_one_component():
    """A single doublet-like triple line touching one shared vertex is one component."""
    f = np.array([[0, 1, 2]])
    regions = np.array([[0, 1]])
    # No triple line at all (only one triangle, one interface) -> one component per node,
    # but there is only one node (interface), so the graph is trivially "one component".
    sizes = triple_junction_component_sizes(f, regions)
    assert sizes == [1]


def test_fragmentation_matches_converged_cohort_stages():
    _require_converged_meshes()
    folder = CONVERGED_MESHES_DIR / "165LumenA_AllEqu_Base1B_pTEiup_ICMup_mTo3_pTEmTEsplit15_pTEoup_V2_mTEo_inversion"
    expected = [1, 1, 1, 1, 1, 1, 3, 29, 141, 143, 143, 143, 143, 143, 143, 143, 143, 143]
    counts = []
    for stage in range(18):
        _v, f, regions = mesh_io.read_rec(folder / "converged" / f"mesh{stage:06d}.rec")
        counts.append(len(triple_junction_component_sizes(f, regions)))
    assert counts == expected


def test_90lumen_a_and_165lumen_a_y27_stay_fully_connected_every_stage():
    """The two folders the converged-mesh-cohort work treats as "clean" mesh-to-mesh-variability groups, verified."""
    _require_converged_meshes()
    for folder_name in (
        "90LumenA_AllEqu_Base1B_pTEidown_mToup",
        "165LumenA_AllEqu_Base1B_pTEiup_ICMup_mTo3_pTEmTEsplit15_pTEoup_V2_Y27",
    ):
        folder = CONVERGED_MESHES_DIR / folder_name
        for stage in range(18):
            _v, f, regions = mesh_io.read_rec(folder / "converged" / f"mesh{stage:06d}.rec")
            assert len(triple_junction_component_sizes(f, regions)) == 1, (folder_name, stage)
