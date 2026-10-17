"""Exploding the view keeps the graph on the cells, and changes nothing that is computed.

What this pins, and why: the graph view (a node per cell, edges between them) is moved with the scattered
("exploded") surface by `update_graph_with_scattered_values`. It used to add each cell's displacement in place to
the node's array, which is `mesh.centroids[cell]` itself: every explode moved the mesh's centroids too, the next
graph was built from the moved ones, and the displacements accumulated. With the explode followed live (the
window re-prepares its scene at each coefficient), the nodes ran away from their cells: on benchmark case 000,
4.7 length units after 24 coefficients, for a tissue 1.37 across. So:

* for many coefficients up and down, every node is at its exploded cell's barycentre (the mean of the cell's
  vertices in `mesh.v_scattered`), an interface with the medium moves with its cell, and `mesh.centroids` is
  bitwise unchanged, whether the graph is rebuilt at each coefficient (as the viewers do) or moved again;
* the tensions, pressures, stress tensors and sphere-fit residuals computed after any number of explodes are
  bitwise those computed before (they were already: the defect was in where the viewer drew, not in a number);
* the sphere-fit residuals the viewer draws on an exploded surface are the unexploded ones, bitwise (fitted on
  the translated vertices of an interface with the medium, they differed by about 1e-12 relative);
* through `plot_force_inference` called again and again on one mesh, the nodes drawn are on the cells.
"""

from __future__ import annotations

import hashlib

import networkx as nx
import numpy as np
import pytest

from foambryo import dcel_mesh_from_file, infer_forces
from foambryo._cell_stress import cell_mean_stress_tensors
from foambryo.curvature import compute_sphere_fit_residues_dict
from foambryo.dcel import update_graph_with_scattered_values

from .conftest import require_substitute_mesh

#: Up, down, back to 0, and up again, twice: the live explode's slider.
COEFFICIENTS = [0.1, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0, 0.3, 0.8, 0.05, 0.6, 0.2] * 2


@pytest.fixture(scope="module")
def case_000():
    return dcel_mesh_from_file(require_substitute_mesh())


def _barycentres_of_exploded_cells(mesh) -> dict[int, np.ndarray]:
    cell_of_vertex = np.zeros(len(mesh.v_scattered), dtype=np.int64)
    cell_of_vertex[mesh.f_scattered[:, :3].ravel()] = np.repeat(mesh.cluster_idx_scattered.astype(np.int64), 3)
    return {cell: mesh.v_scattered[cell_of_vertex == cell].mean(axis=0) for cell in mesh.materials[1:]}


def _assert_on_the_cells(graph: nx.Graph, mesh, unexploded_verts: dict) -> None:
    tolerance = 1e-12 * float(np.ptp(mesh.v, axis=0).max())
    for cell, barycentre in _barycentres_of_exploded_cells(mesh).items():
        assert np.linalg.norm(graph.nodes[cell]["centroid"] - barycentre) <= tolerance, cell
    for (a, b), verts in unexploded_verts.items():
        moved = graph.edges[a, b]["verts"]
        expected = verts + mesh.clusters_displacements[b] if a == 0 else verts
        assert np.array_equal(moved, expected), (a, b)


def test_nodes_stay_on_their_cells_for_every_coefficient_with_the_graph_rebuilt(case_000):
    mesh = case_000
    centroids = {cell: value.copy() for cell, value in mesh.centroids.items()}
    unexploded_verts = {(a, b): v.copy() for a, b, v in mesh.compute_networkx_graph().edges.data("verts")}
    for coeff in COEFFICIENTS:
        graph = mesh.compute_networkx_graph()
        mesh.compute_scattered_arrays(coeff=coeff)
        graph = update_graph_with_scattered_values(graph, mesh)
        _assert_on_the_cells(graph, mesh, unexploded_verts)
        assert all(np.array_equal(mesh.centroids[cell], centroids[cell]) for cell in centroids)


def test_nodes_stay_on_their_cells_when_one_graph_is_moved_again(case_000):
    mesh = case_000
    centroids = {cell: value.copy() for cell, value in mesh.centroids.items()}
    graph = mesh.compute_networkx_graph()
    unexploded_verts = {(a, b): v.copy() for a, b, v in graph.edges.data("verts")}
    for coeff in COEFFICIENTS:
        mesh.compute_scattered_arrays(coeff=coeff)
        update_graph_with_scattered_values(graph, mesh)
        _assert_on_the_cells(graph, mesh, unexploded_verts)
    assert all(np.array_equal(mesh.centroids[cell], centroids[cell]) for cell in centroids)


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _computed(mesh) -> dict[str, str]:
    tensions, pressures = infer_forces(mesh)
    cells = [c for c in mesh.materials[1:] if np.isfinite(pressures[c])]
    stresses = cell_mean_stress_tensors(mesh, tensions, pressures, mesh.compute_volumes_cells(), cells)
    residues = compute_sphere_fit_residues_dict(mesh.compute_networkx_graph())
    return {
        "tensions": _digest(np.array([tensions[k] for k in sorted(tensions)])),
        "pressures": _digest(np.array([pressures[c] for c in sorted(pressures)])),
        "stress tensors": _digest(*[stresses[c] for c in sorted(stresses)]),
        "sphere-fit residuals": _digest(np.array([residues[k] for k in sorted(residues)])),
    }


def test_nothing_computed_changes_after_explodes():
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    before = _computed(mesh)
    for coeff in COEFFICIENTS:
        graph = mesh.compute_networkx_graph()
        mesh.compute_scattered_arrays(coeff=coeff)
        update_graph_with_scattered_values(graph, mesh)
    assert _computed(mesh) == before


@pytest.fixture
def mock_polyscope(monkeypatch):
    """Polyscope's mock backend, and a record of the residuals and graph nodes the viewer draws."""
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    monkeypatch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=1))
    recorded: dict[str, object] = {}
    original = ps.register_point_cloud

    def recording(name, points, *args: object, **kwargs: object):
        recorded[f"points: {name}"] = np.array(points, dtype=np.float64)
        return original(name, points, *args, **kwargs)

    monkeypatch.setattr(ps, "register_point_cloud", recording)
    import foambryo.viewing as viewing

    original_values = viewing.view_dict_values_on_mesh

    def values_recording(mesh, values, *args: object, **kwargs: object):
        recorded[f"values: {kwargs.get('name_values')}"] = dict(values)
        return original_values(mesh, values, *args, **kwargs)

    monkeypatch.setattr(viewing, "view_dict_values_on_mesh", values_recording)
    yield recorded
    ps.remove_all_structures()


def test_the_viewer_draws_the_nodes_on_the_cells_and_the_unexploded_residuals(mock_polyscope):
    import foambryo.viewing as viewing

    mesh = dcel_mesh_from_file(require_substitute_mesh())
    tensions, pressures = infer_forces(mesh)
    viewing.plot_force_inference(mesh, tensions, pressures, scalar_quantities=True)
    unexploded = mock_polyscope["values: Sphere fit residual"]
    graph = mesh.compute_networkx_graph()
    drawn_cells = [cell for cell, volume in graph.nodes.data("volume") if volume > 0 and cell != 0]
    for coeff in (0.2, 0.6, 0.2, 1.0, 0.1):
        viewing.plot_force_inference(mesh, tensions, pressures, scalar_quantities=True, scattered=True,
                                     scatter_coeff=coeff)
        assert mock_polyscope["values: Sphere fit residual"] == unexploded
        barycentres = _barycentres_of_exploded_cells(mesh)
        nodes = mock_polyscope["points: Graph Nodes"]
        # the medium (label 0) has volume 0 and no node; the drawn rows are the cells in graph order
        assert len(nodes) == len(drawn_cells)
        tolerance = 1e-12 * float(np.ptp(mesh.v, axis=0).max())
        for row, cell in enumerate(drawn_cells):
            assert np.linalg.norm(nodes[row] - barycentres[cell]) <= tolerance, (coeff, cell)
