"""Regression tests for the malformed-triple-line-edge crash.

`foambryo.validation.mesh_io.list_junctions` used to admit an edge as a genuine triple line
whenever exactly three distinct region labels touched it (`regions_per_edge == 3`), without
also requiring three interfaces on exactly three incident triangles
(`mesh_integrity`'s own `triple_line_ok`). A non-manifold edge with a fourth (or more)
incident triangle could satisfy the weaker rule while synthesizing a region-triple whose
derived interface pair never occurs as any triangle's actual region pair in the mesh --
`junction_graph.partition_by_junction_graph`'s union-find has no key for that pair and used
to raise a bare `KeyError`.

The mesh below reproduces this with the minimum possible edge: one edge shared by four
triangles (`triangles_per_edge == 4`) whose region labels span `{0, 1, 2}`
(`regions_per_edge == 3`), but which only ever pairs `0` with `1` or `0` with `2` -- never
`1` with `2` directly. So the mesh's only two interfaces are `(0, 1)` and `(0, 2)`;
`(1, 2)` never occurs. Deliberately does not build a closed/watertight solid (the other
edges of these four triangles are left as boundary edges): `list_junctions` and
`partition_by_junction_graph` operate on `(f, regions)` alone and have no watertightness
requirement, and keeping the mesh to exactly the six vertices and four triangles this needs
is what keeps it "small enough to read in the test file, not a fixture blob."
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from foambryo.junction_graph import MalformedTripleLineError, partition_by_junction_graph
from foambryo.validation.mesh_io import (
    MalformedTripleLineWarning,
    list_interfaces,
    list_junctions,
    mesh_integrity,
)


def _one_malformed_triple_line_edge() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(v, f, regions)`: edge `(0, 1)` touches labels `{0, 1, 2}` on four triangles.

    Triangles 0 and 1 both separate `(0, 1)`; triangles 2 and 3 both separate `(0, 2)`.
    `(1, 2)` is never a triangle's region pair.
    """
    v = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    )
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]], dtype=np.int64)
    regions = np.array([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=np.int64)
    return v, f, regions


def _no_abnormal_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A single genuine triple line: edge `(0, 1)` on exactly three triangles, three interfaces."""
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    regions = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    return v, f, regions


def test_malformed_edge_is_flagged_by_mesh_integrity() -> None:
    v, f, regions = _one_malformed_triple_line_edge()
    report = mesh_integrity(v, f, regions)
    assert report["n_triple_line_edges"] == 1
    assert report["n_malformed_triple_line_edges"] == 1
    assert report["intact"] is False


def test_list_junctions_excludes_the_malformed_edge() -> None:
    """The fix: a malformed triple-line edge is no longer admitted as a genuine one."""
    _v, f, regions = _one_malformed_triple_line_edge()
    junctions, edge_to_triple = list_junctions(f, regions)
    assert junctions == []
    assert edge_to_triple == {}


def test_list_junctions_still_admits_a_genuine_triple_line() -> None:
    """Null control: a well-formed triple line is unaffected by the stricter rule."""
    _v, f, regions = _no_abnormal_edges()
    junctions, edge_to_triple = list_junctions(f, regions)
    assert junctions == [(0, 1, 2)]
    assert edge_to_triple == {(0, 1): (0, 1, 2)}


def test_partition_by_junction_graph_no_longer_raises_key_error() -> None:
    """The crash itself: this used to raise a bare `KeyError` before the fix.

    `(1, 2)` was never a mesh interface, but pre-fix `list_junctions` synthesized triple
    `(0, 1, 2)` from the malformed edge anyway, and `partition_by_junction_graph`'s
    union-find tried to look up `(1, 2)` and raised. Post-fix, the edge is excluded before
    it ever reaches the union-find, so this must complete without exception.
    """
    _v, f, regions = _one_malformed_triple_line_edge()

    partition = partition_by_junction_graph(f, regions)

    assert set(partition.interfaces) == {(0, 1), (0, 2)}
    assert partition.components == ()
    assert set(partition.isolated) == {(0, 1), (0, 2)}


def test_partition_by_junction_graph_warns_on_the_dropped_edge() -> None:
    """Fail loudly, never silently: dropping the edge must not be invisible."""
    _v, f, regions = _one_malformed_triple_line_edge()

    with pytest.warns(MalformedTripleLineWarning, match=r"1 triple-line edge\(s\) excluded"):
        partition_by_junction_graph(f, regions)


def test_no_warning_on_a_mesh_with_no_abnormal_edges() -> None:
    """Null control: a healthy mesh produces no new warning (Safeguard 3)."""
    _v, f, regions = _no_abnormal_edges()

    with warnings.catch_warnings():
        warnings.simplefilter("error", MalformedTripleLineWarning)
        partition_by_junction_graph(f, regions)  # must not raise/warn


def test_malformed_triple_line_error_carries_edge_and_triple() -> None:
    """The named exception (defensive backstop) carries the offending edge and triple.

    Constructed directly rather than triggered through `partition_by_junction_graph`: after
    the junction-edge classification fix, every triple `list_junctions` admits provably has all three of its
    sub-interfaces present (see that function's docstring), so this path should not be
    reachable through the public API on any real mesh -- it exists as a safety net.
    """
    error = MalformedTripleLineError(edge=(0, 1), triple=(0, 1, 2), missing_interface=(1, 2))
    assert error.edge == (0, 1)
    assert error.triple == (0, 1, 2)
    assert error.missing_interface == (1, 2)
    assert "(0, 1, 2)" in str(error)
    assert "(1, 2)" in str(error)


def test_list_interfaces_unaffected_by_the_fix() -> None:
    """`list_interfaces` classifies by triangle region pairs directly; the fix does not touch it."""
    _v, _f, regions = _one_malformed_triple_line_edge()
    assert list_interfaces(regions) == [(0, 1), (0, 2)]
