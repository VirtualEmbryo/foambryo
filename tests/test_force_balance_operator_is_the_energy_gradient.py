"""Every shipped assembly of the force-balance gradient is the gradient of the energy it claims.

`foambryo.validation.force_balance.force_balance_operator` assembles

    M = [ dA_k/dv | dL_j/dv | dV_c/dv ],   one column per interface k, junction line j, cell c,

the discrete gradients of the capillary energy `E = sum_k gamma_k A_k + sum_j t_j L_j - sum_c p_c V_c`
(the pressure enters with a minus, so the solved pressure entries are `-p` and
`foambryo.force_balance_inference` negates them). Nothing tested that these columns are what they
say they are: that a column is the derivative of *its own* functional, with the stated sign, and not,
say, half of it. A missing factor on one block -- for instance line tensions recovered at half their
value on lines at the free surface -- could come from the mesh that was analysed or from the operator;
only a comparison against the energy itself can tell the two apart.

This module computes every functional **from first principles**, without calling any gradient code:

* `A_k` -- the summed area of the triangles whose sorted region pair is `k`;
* `L_j` -- the summed length of the junction edges of triple `j`, each edge counted once (an edge is
  a junction edge when exactly three triangles share it and they carry three distinct interfaces);
* `V_c` -- the divergence theorem, `sum det(v0, v1, v2) / 6` over the triangles whose first-listed
  region is `c`, minus the same over those whose second-listed region is `c`. The docstring
  convention is that each triangle's right-hand normal points away from its first-listed region, so
  every bounded cell must come out with `V_c > 0` -- which is asserted, checking the convention
  against the geometry.

and compares each column of every shipped assembly with the central-difference derivative of its
functional, block by block and, for the line block, separately for lines touching the exterior and
for internal lines. The comparison is made twice: as a relative error bound, and as the fitted scale
between the two (`<M_col, D_col> / <D_col, D_col>`), so that a constant factor -- 1/2 or 2 in
particular -- would show as exactly that. A physical sign check closes it: a single closed cell (an
icosphere against the exterior) with `gamma > 0` must be inferred at a **positive** pressure
`p ~ 2 gamma / R` (Laplace), by every pressure solve that ships.

Tolerances: central differences with step `h = 1e-6` of the mesh's bounding-box diagonal; the
truncation error is `O(h^2)` and the round-off `O(eps * |F| / h)`, both far below `RTOL = 1e-6`
relative to the column's largest entry. The scale must equal 1 to `RTOL`.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pytest

from foambryo import PressureComputationMethod, infer_pressures
from foambryo.dcel import DcelData
from foambryo.force_balance_inference import build_force_balance_operator, infer_force_balance
from foambryo.geometry import (
    compute_area_derivative_dict,
    compute_area_derivative_sparse,
    compute_length_derivative_dict,
    compute_volume_derivative_dict,
    compute_volume_derivative_sparse,
)
from foambryo.validation import mesh_io
from foambryo.validation.force_balance import force_balance_operator
from tests.conftest import require_substitute_mesh

RTOL = 1e-6
STEP_FRACTION = 1e-6
#: Vertices of the golden mesh whose rows are checked: every vertex of a junction edge, plus this
#: many others drawn with a fixed seed (the full 11 357 would only repeat the same check).
N_OTHER_VERTICES = 150
SEED = 0


# ---------------------------------------------------------------------------------------------
# first-principles functionals
# ---------------------------------------------------------------------------------------------
def junction_edges(f: np.ndarray, regions: np.ndarray) -> dict[tuple[int, int, int], list[tuple[int, int]]]:
    """Junction edges by region triple: three triangles on the edge, three distinct interfaces."""
    faces_of_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face, (a, b, c) in enumerate(f.tolist()):
        for u, w in ((a, b), (b, c), (c, a)):
            faces_of_edge[(min(u, w), max(u, w))].append(face)
    out: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for edge, faces in faces_of_edge.items():
        if len(faces) != 3:
            continue
        pairs = {tuple(sorted(regions[face].tolist())) for face in faces}
        labels = {label for pair in pairs for label in pair}
        if len(pairs) == 3 and len(labels) == 3:
            out[tuple(sorted(labels))].append(edge)
    return dict(out)


def functionals(v, f, regions, interfaces, triples_edges, cells):
    """`(A_k..., L_j..., V_c...)` in the operator's column order, from first principles."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    sorted_regions = np.sort(regions, axis=1)
    out = [area[(sorted_regions[:, 0] == k[0]) & (sorted_regions[:, 1] == k[1])].sum() for k in interfaces]
    out.extend(
        np.linalg.norm(v[np.asarray(edges)[:, 0]] - v[np.asarray(edges)[:, 1]], axis=1).sum()
        for _triple, edges in triples_edges
    )
    det = np.einsum("ij,ij->i", p0, np.cross(p1, p2))
    out.extend((det[regions[:, 0] == c].sum() - det[regions[:, 1] == c].sum()) / 6.0 for c in cells)
    return np.asarray(out, dtype=np.float64)


def central_difference(v, vertices, evaluate):
    """`D[3*i + a, col]` for the listed vertices: d(functional col)/d(v[i, a])."""
    h = STEP_FRACTION * float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    rows = []
    for i in vertices:
        for axis in range(3):
            plus, minus = v.copy(), v.copy()
            plus[i, axis] += h
            minus[i, axis] -= h
            rows.append((evaluate(plus) - evaluate(minus)) / (2 * h))
    return np.asarray(rows)


def operator_rows(m: np.ndarray, vertices) -> np.ndarray:
    return m.reshape(-1, 3, m.shape[1])[np.asarray(vertices)].reshape(-1, m.shape[1])


def compare(assembled: np.ndarray, numerical: np.ndarray) -> tuple[float, float]:
    """Relative error (to the column's largest entry) and fitted scale of one column."""
    scale_ref = float(np.abs(numerical).max())
    error = float(np.abs(assembled - numerical).max()) / scale_ref
    fitted = float(assembled @ numerical / (numerical @ numerical))
    return error, fitted


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------
def _two_glued_cubes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells 1 and 2, two unit cubes sharing the face x = 1, against the exterior 0.

    Every triangle is wound so its right-hand normal points out of its first-listed region. The one
    triple line (0, 1, 2) is the square boundary of the shared face: a line on the exterior.
    """
    grid = {(x, y, z): i for i, (x, y, z) in enumerate(
        (x, y, z) for x in range(3) for y in range(2) for z in range(2))}
    v = np.asarray(list(grid), dtype=np.float64)
    # A little asymmetry so no gradient is zero by symmetry alone.
    v = v + 0.07 * np.sin(np.arange(v.size).reshape(v.shape) * 1.7)
    faces, regions = [], []

    def quad(corners: list[tuple[int, int, int]], first: int, second: int, outward: np.ndarray) -> None:
        a, b, c, d = (grid[q] for q in corners)
        for tri in ((a, b, c), (a, c, d)):
            n = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
            faces.append(tri if n @ outward > 0 else (tri[0], tri[2], tri[1]))
            regions.append((first, second))

    for cell, x0 in ((1, 0), (2, 1)):
        x1 = x0 + 1
        quad([(x0, 0, 0), (x1, 0, 0), (x1, 0, 1), (x0, 0, 1)], cell, 0, np.array([0, -1, 0]))
        quad([(x0, 1, 0), (x1, 1, 0), (x1, 1, 1), (x0, 1, 1)], cell, 0, np.array([0, 1, 0]))
        quad([(x0, 0, 0), (x1, 0, 0), (x1, 1, 0), (x0, 1, 0)], cell, 0, np.array([0, 0, -1]))
        quad([(x0, 0, 1), (x1, 0, 1), (x1, 1, 1), (x0, 1, 1)], cell, 0, np.array([0, 0, 1]))
    quad([(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], 1, 0, np.array([-1, 0, 0]))
    quad([(2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 0, 1)], 2, 0, np.array([1, 0, 0]))
    quad([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], 1, 2, np.array([1, 0, 0]))
    return v, np.asarray(faces, dtype=np.int64), np.asarray(regions, dtype=np.int64)


def _golden_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mesh_io.read_mesh(require_substitute_mesh())


MESHES = {"two_glued_cubes": _two_glued_cubes, "golden_master_000": _golden_mesh}


@pytest.fixture(scope="module", params=list(MESHES))
def case(request):
    """Raw arrays, keys, the validation operator, and the central-difference Jacobian."""
    v, f, regions = MESHES[request.param]()
    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    junctions, edge_to_triple = mesh_io.list_junctions(f, regions)
    edges_by_triple = junction_edges(f, regions)
    triples_edges = [(t, edges_by_triple[t]) for t in junctions]
    m = force_balance_operator(v, f, regions, interfaces, junctions, edge_to_triple, cells)
    junction_vertices = sorted({u for edges in edges_by_triple.values() for e in edges for u in e})
    others = np.setdiff1d(np.arange(len(v)), junction_vertices)
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(others, min(N_OTHER_VERTICES, len(others)), replace=False).tolist()
    vertices = sorted(set(junction_vertices) | set(sampled))
    numerical = central_difference(
        v, vertices, lambda w: functionals(w, f, regions, interfaces, triples_edges, cells),
    )
    return {
        "name": request.param, "v": v, "f": f, "regions": regions, "interfaces": interfaces,
        "junctions": junctions, "edge_to_triple": edge_to_triple, "edges_by_triple": edges_by_triple,
        "cells": cells, "m": m, "vertices": vertices, "numerical": numerical,
    }


# ---------------------------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------------------------
def test_the_orientation_convention_gives_every_cell_a_positive_volume(case):
    values = functionals(case["v"], case["f"], case["regions"], [], [], case["cells"])
    assert (values > 0).all(), dict(zip(case["cells"], values, strict=True))


def test_the_junction_edges_agree_with_list_junctions(case):
    by_triple_from_list: dict[tuple[int, int, int], set] = defaultdict(set)
    for edge, triple in case["edge_to_triple"].items():
        by_triple_from_list[triple].add(tuple(sorted(edge)))
    assert {t: set(e) for t, e in case["edges_by_triple"].items()} == dict(by_triple_from_list)


def _blocks(case) -> dict[str, list[int]]:
    ni, nj = len(case["interfaces"]), len(case["junctions"])
    exterior = [ni + j for j, t in enumerate(case["junctions"]) if 0 in t]
    internal = [ni + j for j, t in enumerate(case["junctions"]) if 0 not in t]
    return {
        "surface": list(range(ni)),
        "line_on_the_exterior": exterior,
        "line_internal": internal,
        "pressure": list(range(ni + nj, ni + nj + len(case["cells"]))),
    }


@pytest.mark.parametrize("block", ["surface", "line_on_the_exterior", "line_internal", "pressure"])
def test_each_operator_column_is_the_derivative_of_its_own_functional(case, block):
    """Relative error and fitted scale per column; a factor of 1/2 or 2 would read as exactly that."""
    columns = _blocks(case)[block]
    if not columns:
        pytest.skip(f"{case['name']} has no {block} column")
    assembled = operator_rows(case["m"], case["vertices"])
    for col in columns:
        error, fitted = compare(assembled[:, col], case["numerical"][:, col])
        assert abs(fitted - 1.0) < RTOL, f"{block} column {col}: scale {fitted}"
        assert error < RTOL, f"{block} column {col}: relative error {error:.3e}"


@pytest.fixture(scope="module")
def dcel_case(case):
    """The same mesh as `DcelData` (label order and winding normalised), with its own Jacobian."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = DcelData(case["v"], case["f"], case["regions"])
    v, f, regions = np.asarray(mesh.v, dtype=np.float64), mesh.f[:, :3], mesh.f[:, 3:5]
    interfaces, cells = mesh_io.list_interfaces(regions), mesh_io.cell_labels(regions)
    edges_by_triple = junction_edges(f, regions)
    triples = sorted(edges_by_triple)
    materials = [0, *cells]
    vertices = case["vertices"] if len(v) == len(case["v"]) else list(range(len(v)))
    numerical = central_difference(
        v, vertices,
        lambda w: functionals(w, f, regions, interfaces, [(t, edges_by_triple[t]) for t in triples], materials),
    )
    return {"mesh": mesh, "interfaces": interfaces, "cells": cells, "triples": triples,
            "materials": materials, "vertices": vertices, "numerical": numerical}


def _check(assembled_rows: np.ndarray, numerical: np.ndarray, label: str) -> None:
    error, fitted = compare(assembled_rows, numerical)
    assert abs(fitted - 1.0) < RTOL, f"{label}: scale {fitted}"
    assert error < RTOL, f"{label}: relative error {error:.3e}"


def test_the_production_assemblies_are_the_same_gradients(dcel_case):
    """`build_force_balance_operator`, and the dict and sparse area/length/volume gradients.

    The production tension and pressure solves consume these, not the validation operator, so each
    is checked against the same first-principles derivatives: area columns first, then line, then
    volume (the exterior's own volume, a negative number, included as material 0).
    """
    d = dcel_case
    ni, nt = len(d["interfaces"]), len(d["triples"])
    rows = np.asarray(d["vertices"])
    num = d["numerical"]
    num_area, num_line, num_volume = num[:, :ni], num[:, ni:ni + nt], num[:, ni + nt:]

    m, interfaces, cells = build_force_balance_operator(d["mesh"])
    assert interfaces == d["interfaces"]
    assert cells == d["cells"]
    m_rows = operator_rows(m, rows)
    for k in range(ni):
        _check(m_rows[:, k], num_area[:, k], f"build_force_balance_operator surface {interfaces[k]}")
    for c in range(len(cells)):
        _check(m_rows[:, ni + c], num_volume[:, 1 + c], f"build_force_balance_operator pressure {cells[c]}")

    area_dict = compute_area_derivative_dict(d["mesh"])
    keys, area_sparse = compute_area_derivative_sparse(d["mesh"])
    assert [tuple(k) for k in keys] == d["interfaces"]
    dense_sparse = area_sparse.toarray().reshape(ni, -1, 3)
    for k, key in enumerate(d["interfaces"]):
        _check(area_dict[key][rows].reshape(-1), num_area[:, k], f"compute_area_derivative_dict {key}")
        _check(dense_sparse[k][rows].reshape(-1), num_area[:, k], f"compute_area_derivative_sparse {key}")

    length_dict = compute_length_derivative_dict(d["mesh"])
    assert sorted(length_dict) == d["triples"]
    for j, triple in enumerate(d["triples"]):
        _check(length_dict[triple][rows].reshape(-1), num_line[:, j], f"compute_length_derivative_dict {triple}")

    volume_dict = compute_volume_derivative_dict(d["mesh"])
    materials, volume_sparse = compute_volume_derivative_sparse(d["mesh"])
    assert [int(x) for x in materials] == d["materials"]
    dense_volume = volume_sparse.toarray().reshape(len(materials), -1, 3)
    for c, material in enumerate(d["materials"]):
        _check(volume_dict[material][rows].reshape(-1), num_volume[:, c], f"compute_volume_derivative_dict {material}")
        _check(dense_volume[c][rows].reshape(-1), num_volume[:, c], f"compute_volume_derivative_sparse {material}")


# ---------------------------------------------------------------------------------------------
# the physical sign: a single bubble is at a positive pressure 2 gamma / R
# ---------------------------------------------------------------------------------------------
def _icosphere(radius: float) -> DcelData:
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    v, f = np.asarray(sphere.vertices, dtype=np.float64), np.asarray(sphere.faces, dtype=np.int64)
    # trimesh winds the icosphere outward: the normal points out of the cell, the first label.
    regions = np.tile([1, 0], (len(f), 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DcelData(v, f, regions)


@pytest.mark.parametrize("radius", [1.0, 2.5])
def test_a_single_bubble_is_at_the_laplace_pressure(radius):
    """`p = 2 gamma / R`, positive, from the force-balance solve and from each shipped pressure solve.

    The icosphere is inscribed in the sphere of radius `R`, so its discrete `dA/dV` is not exactly
    `2/R`; 2 % covers that discretisation at three subdivisions.
    """
    mesh = _icosphere(radius)
    expected = 2.0 / radius
    result = infer_force_balance(mesh, mean_tension=1.0)
    assert result.pressures[0] > 0
    assert abs(result.pressures[0] / expected - 1) < 0.02, result.pressures
    for mode in (PressureComputationMethod.Variational, PressureComputationMethod.Laplace):
        pressure = infer_pressures(mesh, {(0, 1): 1.0}, mode=mode)[1]
        assert pressure > 0, (mode, pressure)
        assert abs(pressure / expected - 1) < 0.02, (mode, pressure)
