"""The force viewer's mean stress tensor (`foambryo._cell_stress`), without a display.

Two groups:

* the principal split: real under numpy 2, each column scaled by its own eigenvalue, independent
  of eigenvalue order and eigenvector sign, and sorted by sign into compression (negative) and
  tension (non-negative) under the tension-positive convention;
* the physics of the tensor, on two analytic configurations: a single bubble at its Laplace
  pressure carries no mean stress, and a mirror-symmetric doublet gives its two cells mirror
  images of the same tensor.
"""

import itertools

import numpy as np
import pytest
import trimesh

from foambryo._cell_stress import cell_mean_stress_tensors, split_principal_stresses
from foambryo.dcel import DcelData


# ---------------------------------------------------------------------------------------------
# the principal split
# ---------------------------------------------------------------------------------------------
def _rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


def _columns(parts: np.ndarray) -> list[np.ndarray]:
    return [parts[:, k] for k in range(3) if np.any(parts[:, k] != 0)]


EIGENVALUE_SETS = [(-2.0, 0.5, 3.0), (-1.0, 2.0, 3.0), (-3.0, -1.0, 2.0), (-1.5, -0.5, -0.25), (0.25, 1.0, 4.0)]


@pytest.mark.parametrize("eigenvalues", EIGENVALUE_SETS)
@pytest.mark.parametrize("seed", [0, 1])
def test_split_scales_each_axis_by_its_own_eigenvalue(eigenvalues, seed):
    rotation = _rotation(seed)
    for order in itertools.permutations(eigenvalues):
        tensor = rotation @ np.diag(order) @ rotation.T
        tensor = (tensor + tensor.T) / 2  # exactly symmetric, as the viewer's tensors are
        axes = split_principal_stresses(tensor)
        negative, non_negative = axes.compression, axes.tension

        # Real, not complex: numpy.linalg.eig returned complex128 here under numpy 2.
        assert negative.dtype == np.float64
        assert non_negative.dtype == np.float64

        # Every eigenvalue lands in exactly one part, scaling its own column: a column c with
        # eigenvalue lam satisfies |c| = |lam| and tensor @ c = lam * c.
        recovered = []
        for part, sign in ((negative, -1.0), (non_negative, 1.0)):
            for column in _columns(part):
                lam = sign * np.linalg.norm(column)
                np.testing.assert_allclose(tensor @ column, lam * column, atol=1e-12)
                recovered.append(lam)
        expected = sorted(eigenvalues)
        np.testing.assert_allclose(sorted(recovered), expected, atol=1e-12)
        assert len(_columns(negative)) == sum(v < 0 for v in eigenvalues)

        # Order- and sign-independent reconstruction: sum over axes of lam * v v^T.
        rebuilt = sum(
            np.outer(c, c) / (s * np.linalg.norm(c))
            for part, s in ((negative, -1.0), (non_negative, 1.0))
            for c in _columns(part)
        )
        np.testing.assert_allclose(rebuilt, tensor, atol=1e-12)


def test_split_ignores_eigenvector_sign():
    rotation = _rotation(3)
    flipped = rotation * np.array([1.0, -1.0, -1.0])
    tensor = rotation @ np.diag([-1.0, 2.0, 3.0]) @ rotation.T
    same = flipped @ np.diag([-1.0, 2.0, 3.0]) @ flipped.T
    for a, b in zip(split_principal_stresses(tensor), split_principal_stresses(same), strict=True):
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_purely_tensile_tensor_is_all_tension(seed):
    rotation = _rotation(seed)
    tensor = rotation @ np.diag([0.5, 1.0, 2.0]) @ rotation.T  # positive definite
    axes = split_principal_stresses((tensor + tensor.T) / 2)
    assert not np.any(axes.compression)
    assert len(_columns(axes.tension)) == 3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_purely_compressive_tensor_is_all_compression(seed):
    rotation = _rotation(seed)
    tensor = rotation @ np.diag([-2.0, -1.0, -0.5]) @ rotation.T  # negative definite
    axes = split_principal_stresses((tensor + tensor.T) / 2)
    assert not np.any(axes.tension)
    assert len(_columns(axes.compression)) == 3


@pytest.mark.parametrize("eigenvalues", [(-2.0, 0.5, 3.0), (-3.0, -1.0, 2.0)])
def test_mixed_tensor_splits_by_sign(eigenvalues):
    rotation = _rotation(4)
    tensor = rotation @ np.diag(eigenvalues) @ rotation.T
    axes = split_principal_stresses((tensor + tensor.T) / 2)
    for part, sign in ((axes.compression, -1.0), (axes.tension, 1.0)):
        values = sorted(sign * np.linalg.norm(c) for c in _columns(part))
        np.testing.assert_allclose(values, sorted(v for v in eigenvalues if np.sign(v) == sign), atol=1e-12)


# ---------------------------------------------------------------------------------------------
# single bubble: (1/V) sum A (I - n n) -> (2/R) I, so -p I + gamma (2/R) I = 0 at p = 2 gamma / R
# ---------------------------------------------------------------------------------------------
def _bubble(subdivisions: int, radius: float) -> DcelData:
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    # trimesh winds outward: the normal points away from the cell (1) into the exterior (0).
    labels = np.tile([1, 0], (len(sphere.faces), 1))
    return DcelData(np.asarray(sphere.vertices), np.asarray(sphere.faces), labels)


def _bubble_stress(subdivisions: int, radius: float, gamma: float, pressure: float | None) -> tuple[np.ndarray, float]:
    mesh = _bubble(subdivisions, radius)
    volumes = mesh.compute_volumes_cells()
    assert volumes[1] > 0
    if pressure is None:  # the discrete Laplace pressure of this very mesh: (2/3) gamma A / V
        pressure = 2 * gamma * mesh.compute_areas_cells()[1] / (3 * volumes[1])
    stress = cell_mean_stress_tensors(mesh, {(0, 1): gamma}, {1: pressure}, volumes, [1])[1]
    return stress, pressure


@pytest.mark.parametrize("radius", [1.0, 7.5])
def test_single_bubble_at_the_discrete_laplace_pressure_carries_no_mean_stress(radius):
    # Icosahedral symmetry makes sum A (I - n n) isotropic on an icosphere, so at the mesh's own
    # Laplace pressure the tensor vanishes to rounding.
    stress, pressure = _bubble_stress(4, radius, gamma=1.3, pressure=None)
    assert np.abs(stress).max() <= 1e-12 * pressure


def test_single_bubble_at_the_continuum_laplace_pressure_converges_to_zero():
    # At p = 2 gamma / R the residual is the mesh's discretisation error, of order (h / R)^2:
    # stated tolerance 2e-3 of p at 5 120 faces (subdivisions = 4), and the error must shrink
    # by about 4 per subdivision (h halves).
    gamma, radius = 1.3, 2.0
    errors = []
    for subdivisions in (2, 3, 4):
        stress, pressure = _bubble_stress(subdivisions, radius, gamma, pressure=2 * gamma / radius)
        np.testing.assert_allclose(stress, stress[0, 0] * np.identity(3), atol=1e-12 * pressure)
        errors.append(np.abs(stress).max() / pressure)
    assert errors[-1] <= 2e-3
    for coarse, fine in itertools.pairwise(errors):
        assert 3.5 <= coarse / fine <= 4.5


# ---------------------------------------------------------------------------------------------
# symmetric doublet: two cells, mirror images of each other through x = 1, equal pressures
# ---------------------------------------------------------------------------------------------
def _mirror_doublet() -> DcelData:
    """Cells 1 (x in [0, 1]) and 2 (its mirror image through the plane x = 1), shared face x = 1.

    The vertices are perturbed by a displacement that commutes with the mirror (odd in x - 1
    along x, even across it), so the faces are not flat and the tensors are not diagonal; cell
    2's exterior triangles are cell 1's mirrored, so the two triangulations are mirror images too.
    """
    grid = {(x, y, z): i for i, (x, y, z) in enumerate((x, y, z) for x in range(3) for y in range(2) for z in range(2))}
    v = np.asarray(list(grid), dtype=np.float64)
    s, y, z = v[:, 0] - 1.0, v[:, 1], v[:, 2]
    v = v + np.column_stack(
        (
            0.06 * s * np.sin(y + 2 * z + 0.3),
            0.05 * np.sin(3 * y + 2 * z + s**2),
            0.04 * np.cos(2 * y - z + s**2),
        ),
    )
    mirror_index = {grid[(x, y, z)]: grid[(2 - x, y, z)] for (x, y, z) in grid}

    faces, labels = [], []

    def quad(corners, first, second, outward):
        a, b, c, d = (grid[q] for q in corners)
        for tri in ((a, b, c), (a, c, d)):
            n = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
            faces.append(tri if n @ outward > 0 else (tri[0], tri[2], tri[1]))
            labels.append((first, second))

    quad([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], 1, 0, np.array([0, -1, 0]))
    quad([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)], 1, 0, np.array([0, 1, 0]))
    quad([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], 1, 0, np.array([0, 0, -1]))
    quad([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], 1, 0, np.array([0, 0, 1]))
    quad([(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], 1, 0, np.array([-1, 0, 0]))
    exterior_of_cell_1 = list(faces)
    quad([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], 1, 2, np.array([1, 0, 0]))
    for a, b, c in exterior_of_cell_1:  # a mirror reverses orientation: swap two vertices
        faces.append((mirror_index[a], mirror_index[c], mirror_index[b]))
        labels.append((2, 0))
    return DcelData(v, np.asarray(faces, dtype=np.int64), np.asarray(labels, dtype=np.int64))


@pytest.mark.parametrize("shared_tension", [0.4, 1.0, 1.7])
def test_mirror_doublet_gives_both_cells_the_mirrored_tensor(shared_tension):
    mesh = _mirror_doublet()
    volumes = mesh.compute_volumes_cells()
    assert volumes[1] > 0
    np.testing.assert_allclose(volumes[1], volumes[2], rtol=1e-13)
    tensions = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): shared_tension}
    pressure = 0.3
    stresses = cell_mean_stress_tensors(mesh, tensions, {1: pressure, 2: pressure}, volumes, [1, 2])
    mirror = np.diag([-1.0, 1.0, 1.0])
    assert np.abs(stresses[1] - np.diag(np.diag(stresses[1]))).max() > 1e-3  # not trivially diagonal
    np.testing.assert_allclose(stresses[2], mirror @ stresses[1] @ mirror, atol=1e-13)


def test_an_interface_absent_from_the_tension_map_contributes_no_stress():
    """A crop interface (only crop faces) is absent from the tension map: it contributes no stress.

    Its faces used to be looked up as a membrane and raised a `KeyError`, so the stress viewer could not draw such
    a cropped mesh. Absent is exactly a tension of 0.
    """
    from foambryo import dcel_mesh_from_file, infer_forces

    from .conftest import require_substitute_mesh

    mesh = dcel_mesh_from_file(require_substitute_mesh())
    tensions, pressures = infer_forces(mesh)
    absent = next(key for key in tensions if key[0] == 0)
    without = {key: value for key, value in tensions.items() if key != absent}
    volumes = mesh.compute_volumes_cells()
    cells = [int(c) for c in mesh.materials[1:]]
    left_out = cell_mean_stress_tensors(mesh, without, pressures, volumes, cells)
    zero = cell_mean_stress_tensors(mesh, {**without, absent: 0.0}, pressures, volumes, cells)
    assert all(np.array_equal(left_out[c], zero[c]) for c in cells)
