"""Tests for `foambryo_benchmarks.converged_mesh_parser` (the converged-mesh-cohort work).

Two kinds of check, per the phase's instruction not to assume the parser is right:

1. **Hand-computed values**, derived independently from
   `converged-meshes/90LumenA_.../multitensions.opt`'s eight rules and checked against the
   parser's output at stages 0, 10 and 17 -- 17 is chosen specifically because every
   variable-length rule in that file (5 and 11 tokens against 18 stages) has already
   exhausted its token list by stage 10, so stage 17 is pure hold-last-value.
2. **Coverage against real meshes**: every interface `list_interfaces` finds in the actual
   `.rec` mesh for a given stage must have a resolved tension (`validate_coverage`), on all
   four converged-mesh folders and all 18 stages of each -- this is the check the phase asks
   for directly ("assert every interface actually present in the .rec has a tension").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foambryo_benchmarks.converged_mesh_parser import (
    parse_multitensions,
    parse_region_range,
    stage_groups,
    validate_coverage,
)
from foambryo.validation import mesh_io
from tests.conftest import WORKSPACE_ROOT

CONVERGED_MESHES_DIR = WORKSPACE_ROOT / "converged-meshes"
NINETY_LUMEN_A = CONVERGED_MESHES_DIR / "90LumenA_AllEqu_Base1B_pTEidown_mToup"


def _require_converged_meshes() -> None:
    if not CONVERGED_MESHES_DIR.exists():
        pytest.skip(f"{CONVERGED_MESHES_DIR} not found (sibling converged-meshes/ missing)")


def test_parse_region_range():
    assert parse_region_range("5") == [5]
    assert parse_region_range("2-4") == [2, 3, 4]
    assert parse_region_range("2-23,40,55-57") == (
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 40, 55, 56, 57]
    )
    assert parse_region_range("1,3,2") == [1, 2, 3]  # de-duplicated and sorted


def test_token_list_longer_than_n_stages_is_a_parse_error(tmp_path: Path):
    bad = tmp_path / "multitensions.opt"
    bad.write_text("# Ncells NOptStages\n3 2\n\n1:0 1.0_2.0_3.0\n")
    with pytest.raises(ValueError, match="file error"):
        parse_multitensions(bad)


def test_90lumen_a_header_and_stage_groups():
    _require_converged_meshes()
    parsed = parse_multitensions(NINETY_LUMEN_A / "multitensions.opt")
    assert parsed.n_cells == 91
    assert parsed.n_regions == 92
    assert parsed.n_stages == 18
    # Every rule in this file has either 1 token (constant) or 11 tokens against 18 stages,
    # so every group after the first 10 singleton stages holds the last value together.
    groups = stage_groups(parsed)
    assert groups[-1] == list(range(10, 18))
    assert all(len(g) == 1 for g in groups[:-1])


@pytest.mark.parametrize(
    ("stage", "a", "b", "expected"),
    [
        # Rule "1:0 6.0" -- one token, constant at every stage.
        (0, 0, 1, 6.0),
        (10, 0, 1, 6.0),
        (17, 0, 1, 6.0),
        # Rule "2-23:0 6.0" -- one token, constant.
        (0, 0, 5, 6.0),
        (17, 0, 5, 6.0),
        # Rule "24-91:0 1.0_..._2.0" -- 11 tokens over 18 stages: stage 0 is the first token,
        # stage 10 is the last (index 10), stage 17 holds that same last value.
        (0, 0, 30, 1.0),
        (10, 0, 30, 2.0),
        (17, 0, 30, 2.0),
        # Rule "2-23:1 1.0" / "24-91:1 1.0" -- constant.
        (0, 1, 5, 1.0),
        (0, 1, 30, 1.0),
        (17, 1, 90, 1.0),
        # Rule "2-23:2-23 0.618" -- constant.
        (0, 5, 10, 0.618),
        (17, 5, 10, 0.618),
        # Rule "24-91:24-91 0.232_..._0.463" -- 11 tokens: same hold-last-value pattern.
        (0, 30, 40, 0.232),
        (10, 30, 40, 0.463),
        (17, 30, 40, 0.463),
        # Rule "2-23:24-91 0.9195" -- constant.
        (0, 5, 30, 0.9195),
        (17, 5, 30, 0.9195),
    ],
)
def test_90lumen_a_hand_computed_values(stage, a, b, expected):
    _require_converged_meshes()
    parsed = parse_multitensions(NINETY_LUMEN_A / "multitensions.opt")
    key = (min(a, b), max(a, b))
    assert parsed.tensions_by_stage[stage][key] == pytest.approx(expected)


def test_90lumen_a_stage0_total_key_count_matches_hand_derivation():
    """Independent hand count: 1 + 22 + 68 + 22 + 68 + C(22,2) + C(68,2) + 22*68 = 4186."""
    _require_converged_meshes()
    parsed = parse_multitensions(NINETY_LUMEN_A / "multitensions.opt")
    expected = 1 + 22 + 68 + 22 + 68 + (22 * 21 // 2) + (68 * 67 // 2) + 22 * 68
    assert len(parsed.tensions_by_stage[0]) == expected == 4186
    # Stages 10-17 share the same map (hold-last), so the key *count* is unchanged too.
    assert len(parsed.tensions_by_stage[10]) == expected


@pytest.mark.parametrize(
    "folder_name",
    [
        "165LumenA_AllEqu_Base1B_pTEiup_ICMup_mTo3_pTEmTEsplit15_pTEoup_V2_Y27",
        "165LumenA_AllEqu_Base1B_pTEiup_ICMup_mTo3_pTEmTEsplit15_pTEoup_V2_mTEo_inversion",
        "90LumenA_AllEqu_Base1B_pTEidown_mToup",
        "90LumenC_AllEqu_Base1B_pTEidown_ICMdown_mTo08_pTEmTEsplit14_pTEICMup_CN03+ICM+ap",
    ],
)
def test_every_mesh_interface_is_covered_by_a_tension(folder_name: str):
    """No interface a real mesh contains is left without a resolved tension, any stage."""
    _require_converged_meshes()
    folder = CONVERGED_MESHES_DIR / folder_name
    parsed = parse_multitensions(folder / "multitensions.opt")
    for stage in range(parsed.n_stages):
        mesh_path = folder / "converged" / f"mesh{stage:06d}.rec"
        _v, _f, regions = mesh_io.read_rec(mesh_path)
        interfaces = mesh_io.list_interfaces(regions)
        coverage = validate_coverage(parsed.tensions_by_stage[stage], interfaces)
        assert coverage["n_uncovered"] == 0, (
            f"{folder_name} stage {stage}: {coverage['n_uncovered']} interfaces have no rule: "
            f"{coverage['uncovered'][:10]}"
        )
