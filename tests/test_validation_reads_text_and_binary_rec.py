"""`foambryo.validation.read_rec` and `read_mesh` read a `.rec` file in either layout, text or binary.

Before 0.7.0 they read only the binary layout of the reference implementation, while delaunay-watershed-3d writes
`.rec` files as text by default and `foambryo mesh` writes them that way: a text `.rec` raised or returned garbage.
Pins: the text and binary files dw3d writes (with and without a metadata appendix and crop faces) and the file
`foambryo mesh` writes read back bitwise as `dw3d.io.load_rec` reads them; the binary path is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dw3d.io import load_rec, save_rec

from foambryo.io import reconstruct_mask
from foambryo.validation import read_mesh, read_rec

POINTS = np.array([[0.0, 0.0, 0.0], [1.5, 0.1, 0.0], [0.0, 1.0, 1 / 3], [0.3, 0.0, 1.0]])
TRIANGLES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
LABELS = np.array([[0, 1], [0, 1], [0, 2], [1, 2]], dtype=np.int64)


def assert_reads_as_dw3d(path: Path) -> None:
    expected = load_rec(path)
    for reader in (read_rec, read_mesh):
        v, f, regions = reader(path)
        assert v.dtype == np.float64
        assert np.array_equal(v, expected[0])
        assert np.array_equal(f, expected[1])
        assert np.array_equal(regions, expected[2])


@pytest.mark.parametrize("binary_mode", [False, True], ids=["text", "binary"])
@pytest.mark.parametrize("with_appendix", [False, True], ids=["no appendix", "appendix and crop faces"])
def test_dw3d_files_read_back_in_either_layout(tmp_path: Path, binary_mode: bool, with_appendix: bool) -> None:
    path = tmp_path / "mesh.rec"
    extra = (
        {
            "metadata": {"coordinate_frame": "physical", "spacing": (0.5, 0.25, 0.25)},
            "crop_faces": np.array([False, True, False, False]),
        }
        if with_appendix
        else {}
    )
    save_rec(path, POINTS, TRIANGLES, LABELS, binary_mode, **extra)
    assert_reads_as_dw3d(path)
    if not with_appendix:
        v, f, regions = read_rec(path)
        assert np.array_equal(v, POINTS)
        assert np.array_equal(f, TRIANGLES)
        assert np.array_equal(regions, LABELS)


def test_the_file_foambryo_mesh_writes_reads_back(tmp_path: Path) -> None:
    z, y, x = np.mgrid[:24, :24, :24].astype(float)
    mask = np.zeros((24, 24, 24), dtype=np.uint16)
    mask[(z - 12) ** 2 + (y - 8) ** 2 + (x - 12) ** 2 < 6**2] = 1
    mask[(z - 12) ** 2 + (y - 16) ** 2 + (x - 12) ** 2 < 5**2] = 2
    path = tmp_path / "from_the_command.rec"
    reconstruct_mask(mask, (1.0, 1.0, 1.0)).write_rec(path)
    assert path.read_bytes()[:1].isdigit()  # the command's layout is text
    assert_reads_as_dw3d(path)
