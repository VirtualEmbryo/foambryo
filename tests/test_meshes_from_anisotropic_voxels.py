"""Meshes from masks with a voxel size: foambryo takes delaunay-watershed-3d's route and records the spacing.

What this pins, and why:

* **Without a spacing, nothing changes.** `dcel_mesh_from_segmentation_mask(mask)` is the plain
  reconstruction, bitwise (SHA-256 of the vertex, triangle and crop arrays), and its mesh records no
  source spacing. A mesh file without a physical frame is read as before.
* **With a spacing, the mesh is delaunay-watershed-3d's `construct_mesh_in_physical_units`**: the mask
  resampled to cubic voxels when one axis is longer, meshed, and scaled to the spacing's unit. The
  arrays are compared by SHA-256, with cubic and with anisotropic voxels, with and without a cell
  that reaches the image box; sums or counts would miss a changed mesh.
* **The recorded spacing changes no number**: no inference reads it (the quadric-reading default for
  such meshes was tried and dropped: it was measured to be worse at real cell size).
* `dcel_mesh_from_file` reads the spacing only from a `.rec` whose appendix says the mesh is in
  physical units; a voxel-index mesh with a spacing beside it was meshed on the stretched grid.

Synthetic masks (four cells in a ball) and the benchmark mesh 000 only.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.io import load_rec, save_rec

from foambryo import (
    TensionComputationMethod,
    dcel_mesh_from_file,
    dcel_mesh_from_segmentation_mask,
    infer_forces,
    infer_tensions,
)
from foambryo import io as foambryo_io
from foambryo.dcel import DcelData

from .conftest import require_substitute_mesh
from .test_command_line import MASKS


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _fingerprint(mesh: DcelData) -> tuple[str, str, str]:
    return _sha256(np.asarray(mesh.v, dtype=np.float64)), _sha256(mesh.f), _sha256(mesh.crop_faces)


def _quietly(function, *args: object, **kwargs: object):
    """Call a library function with its warnings silenced (the cut mask's volume warning, for one)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.mark.parametrize("mask_name", list(MASKS))
def test_without_a_spacing_the_mesh_is_the_plain_reconstruction_bitwise(mask_name):
    mask = MASKS[mask_name]()
    mesh = _quietly(dcel_mesh_from_segmentation_mask, mask)
    algorithm = get_default_mesh_reconstruction_algorithm(3, False)
    points, triangles, labels = _quietly(algorithm.construct_mesh_from_segmentation_mask, mask)
    expected = DcelData(points, triangles, labels, crop_faces=algorithm.crop_faces)
    assert _fingerprint(mesh) == _fingerprint(expected)
    assert mesh.source_spacing is None
    assert mesh.source_anisotropy is None


@pytest.mark.parametrize("spacing", [(0.5, 0.5, 0.5), (1.0, 0.5, 0.5), (0.3, 0.3, 0.75)],
                         ids=["cubic", "z twice as long", "x 2.5 times as long"])
@pytest.mark.parametrize("mask_name", list(MASKS))
def test_with_a_spacing_the_mesh_is_the_physical_units_entry_point(mask_name, spacing):
    mask = MASKS[mask_name]()
    mesh = _quietly(dcel_mesh_from_segmentation_mask, mask, spacing=spacing)
    algorithm = get_default_mesh_reconstruction_algorithm(3, False)
    points, triangles, labels = _quietly(algorithm.construct_mesh_in_physical_units, mask, spacing)
    expected = DcelData(points, triangles, labels, crop_faces=algorithm.crop_faces)
    assert _fingerprint(mesh) == _fingerprint(expected)
    assert mesh.source_spacing == spacing
    assert mesh.source_anisotropy == max(spacing) / min(spacing)
    assert bool(mesh.has_crop_faces) == (mask_name == "four cells cut by the image box")
    # the mesh is in physical units: it spans the mask's extent times the spacing, not its voxel count
    extent = np.ptp(np.asarray(mesh.v), axis=0)
    assert np.all(extent <= (np.array(mask.shape) - 1) * np.array(spacing) + 1e-9)
    if max(spacing) == min(spacing):  # cubic voxels: no resampling, the plain mesh scaled
        plain = get_default_mesh_reconstruction_algorithm(3, False)
        plain_points = np.asarray(_quietly(plain.construct_mesh_from_segmentation_mask, mask)[0], dtype=np.float64)
        assert _sha256(np.asarray(points, dtype=np.float64)) == _sha256(plain_points * spacing[0])


def test_a_spacing_with_two_different_short_axes_is_refused():
    with pytest.raises(ValueError, match="two different fine axes"):
        dcel_mesh_from_segmentation_mask(MASKS["four cells in a ball"](), spacing=(2.0, 1.0, 0.5))


def test_a_reconstruction_without_the_physical_units_entry_point_refuses_a_spacing(monkeypatch):
    class OlderReconstruction:
        def construct_mesh_from_segmentation_mask(self, _mask: np.ndarray) -> None:
            raise AssertionError

    monkeypatch.setattr(foambryo_io, "get_default_mesh_reconstruction_algorithm", lambda *_: OlderReconstruction())
    with pytest.raises(ValueError, match="construct_mesh_in_physical_units"):
        dcel_mesh_from_segmentation_mask(MASKS["four cells in a ball"](), spacing=(2.0, 1.0, 1.0))


@pytest.mark.parametrize("source_spacing", [(1.0, 1.0), (1.0, 0.0, 1.0), (1.0, math.nan, 1.0), (-1.0, 1.0, 1.0)])
def test_a_malformed_source_spacing_is_refused(source_spacing):
    points, triangles, labels = load_rec(require_substitute_mesh())
    with pytest.raises(ValueError, match="source_spacing must be three positive finite numbers"):
        DcelData(points, triangles, labels, source_spacing=source_spacing)


@pytest.mark.parametrize("mode", [TensionComputationMethod.YoungDupre, TensionComputationMethod.ForceBalance,
                                  TensionComputationMethod.Variational])
def test_a_recorded_source_spacing_changes_no_number(mode):
    points, triangles, labels = load_rec(require_substitute_mesh())
    plain = DcelData(points, triangles, labels)
    recorded = DcelData(points, triangles, labels, source_spacing=(3.0, 1.0, 1.0))
    if mode is TensionComputationMethod.YoungDupre:  # the default: tensions and pressures
        for a, b in zip(infer_forces(plain), infer_forces(recorded), strict=True):
            assert a.keys() == b.keys()
            assert all(np.float64(a[k]).tobytes() == np.float64(b[k]).tobytes() for k in a)
    else:
        a, b = infer_tensions(plain, mode=mode), infer_tensions(recorded, mode=mode)
        assert a.keys() == b.keys()
        assert all(np.float64(a[k]).tobytes() == np.float64(b[k]).tobytes() for k in a)


def _write(path: Path, metadata: dict | None) -> Path:
    points, triangles, labels = load_rec(require_substitute_mesh())
    save_rec(path, points, triangles, labels, metadata=metadata)
    return path


def test_the_file_reader_takes_the_source_spacing_only_from_a_physical_frame(tmp_path):
    physical = _write(tmp_path / "physical.rec", {"coordinate_frame": "physical", "spacing": (0.5, 0.3, 0.3)})
    voxels = _write(tmp_path / "voxels.rec", {"coordinate_frame": "voxel_index", "spacing": (0.5, 0.3, 0.3)})
    legacy = _write(tmp_path / "legacy.rec", None)
    assert dcel_mesh_from_file(physical).source_spacing == (0.5, 0.3, 0.3)
    assert dcel_mesh_from_file(physical).source_anisotropy == pytest.approx(5 / 3)
    assert dcel_mesh_from_file(voxels).source_spacing is None
    assert dcel_mesh_from_file(legacy).source_spacing is None
    assert _fingerprint(dcel_mesh_from_file(physical)) == _fingerprint(dcel_mesh_from_file(legacy))
