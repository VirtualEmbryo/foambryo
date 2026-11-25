"""The `foambryo` command on 2-D input: `mesh` and `infer` give the library's numbers; the rest refuse with a message.

Pinned: (1) `foambryo mesh` on a 2-D mask writes delaunay-watershed's 2-D mesh in physical units, with `dimension = 2`
in the `.rec` appendix; (2) `foambryo infer` on that file writes bitwise `infer_forces(dcel_mesh_from_file(MESH))`, and
a run record of schema 2 with `dimension = 2` and the 2-D mesh block; (3) `--tension-method equilibrium-tiling --mask`
writes `infer_forces_by_equilibrium_tiling`'s numbers; (4) exit 2, with the reason, for: a spacing of the wrong length,
anisotropic pixels, a 3-D pressure method or tension method on a 2-D mesh, `--media` on a 2-D mesh, `--mask` without
the global fit, the global fit on a 3-D mesh or without `--mask`, `infer-over-seeds` on 2-D input. (`view` takes 2-D
input since the 2-D window: `test_viewer_in_two_dimensions.py`.)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import foambryo
from foambryo import cli
from foambryo.results import read_pressure_table, read_tension_table


def _run(capsys: pytest.CaptureFixture[str], *arguments: object) -> tuple[int, str, str]:
    code = cli.main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture(scope="module")
def mask_file(tmp_path_factory: pytest.TempPathFactory) -> object:
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[10:50, 10:30] = 1
    labels[10:30, 30:50] = 2
    labels[30:50, 30:50] = 3
    path = tmp_path_factory.mktemp("two_dimensional") / "three_cells.npy"
    np.save(path, labels)
    return path


@pytest.fixture(scope="module")
def mesh_file(mask_file: object) -> object:
    path = mask_file.parent / "three_cells.rec"
    assert cli.main(["mesh", str(mask_file), "-o", str(path), "--spacing", "0.5", "0.5"]) == cli.EXIT_OK
    return path


def test_mesh_writes_the_library_s_2d_mesh_with_its_dimension(mesh_file, mask_file):
    import dw2d.io

    reconstruction = foambryo.reconstruct_mask(np.load(mask_file), (0.5, 0.5))
    assert mesh_file.read_bytes() == reconstruction.rec_bytes
    appendix = dw2d.io.load_rec_appendix(mesh_file)
    assert appendix["dimension"] == 2 and appendix["coordinate_frame"] == "physical"
    assert appendix["spacing"] == (0.5, 0.5) and appendix["spacing_unit"] == "um"
    assert reconstruction.min_distance == 4


def test_infer_writes_the_library_s_numbers_and_a_schema_2_record(capsys, tmp_path, mesh_file):
    code, _, err = _run(capsys, "infer", mesh_file, "-o", tmp_path)
    assert code == cli.EXIT_OK, err
    tensions, pressures = foambryo.infer_forces(foambryo.dcel_mesh_from_file(mesh_file))
    read_tensions = read_tension_table(tmp_path / "tensions.csv")
    read_pressures = read_pressure_table(tmp_path / "pressures.csv")
    assert read_tensions == tensions and read_pressures == pressures
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["schema"] == 2 and record["dimension"] == 2
    assert record["options"]["pressure_method"] == "laplace"
    assert set(record["mesh"]) == {"points", "segments", "materials", "cells", "media", "unresolved",
                                   "image_border_vertices", "cut_cells"}
    assert record["units"]["pressure"] == "tension per um"


def test_the_global_fit_from_the_command_line(capsys, tmp_path, mesh_file, mask_file):
    code, _, err = _run(capsys, "infer", mesh_file, "-o", tmp_path, "--tension-method", "equilibrium-tiling",
                        "--mask", mask_file, "--base-pressure", "1")
    assert code == cli.EXIT_OK, err
    solution, pressures = foambryo.infer_forces_by_equilibrium_tiling(foambryo.dcel_mesh_from_file(mesh_file),
                                                                      np.load(mask_file))
    assert read_tension_table(tmp_path / "tensions.csv") == solution.dict_tensions
    assert read_pressure_table(tmp_path / "pressures.csv") == {k: v + 1.0 for k, v in pressures.items()}
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["library_call"].startswith("infer_forces_by_equilibrium_tiling(mesh, labels")
    assert record["options"]["mask"] == str(mask_file)


@pytest.mark.parametrize(("arguments", "message"), [
    (("mesh", "{mask}", "-o", "{out}/m.rec", "--spacing", "1", "1", "1"), "takes 2 values"),
    (("mesh", "{mask}", "-o", "{out}/m.rec", "--spacing", "1", "2"), "square pixels only"),
    (("infer", "{mesh}", "-o", "{out}", "--pressure-method", "variational"), "laplace only"),
    (("infer", "{mesh}", "-o", "{out}", "--tension-method", "lami"), "not offered for a 2-D mesh"),
    (("infer", "{mesh}", "-o", "{out}", "--media", "2"), "3-D meshes only"),
    (("infer", "{mesh}", "-o", "{out}", "--mask", "{mask}"), "applies to --tension-method equilibrium-tiling"),
    (("infer", "{mesh}", "-o", "{out}", "--tension-method", "equilibrium-tiling"), "needs --mask"),
    (("infer-over-seeds", "{mask}", "-o", "{out}", "--spacing", "1", "1"), "is a 2-D mask"),
])
def test_what_does_not_apply_to_2d_input_exits_2_with_its_reason(capsys, tmp_path, mesh_file, mask_file, arguments,
                                                                  message):
    filled = [a.format(mask=mask_file, mesh=mesh_file, out=tmp_path) for a in arguments]
    code, _, err = _run(capsys, *filled)
    assert code == cli.EXIT_USAGE
    assert message in err


def test_the_global_fit_on_a_3d_mesh_exits_2(capsys, tmp_path, mask_file):
    mask = np.zeros((24, 24, 24), dtype=np.int32)
    mask[4:20, 4:12, 4:20] = 1
    mask[4:20, 12:20, 4:20] = 2
    np.save(tmp_path / "boxes.npy", mask)
    assert _run(capsys, "mesh", tmp_path / "boxes.npy", "-o", tmp_path / "boxes.rec", "--spacing", "1", "1", "1")[0] == 0
    code, _, err = _run(capsys, "infer", tmp_path / "boxes.rec", "-o", tmp_path / "out", "--tension-method",
                        "equilibrium-tiling", "--mask", mask_file)
    assert code == cli.EXIT_USAGE and "2-D meshes only" in err
    record = cli.main(["infer", str(tmp_path / "boxes.rec"), "-o", str(tmp_path / "out3")])
    assert record == cli.EXIT_OK
    assert json.loads((tmp_path / "out3" / "run.json").read_text())["dimension"] == 3


def test_the_version_names_both_reconstructions_and_iorec(capsys):
    code, out, _ = _run(capsys, "--version")
    assert code == cli.EXIT_OK
    out = out.replace("\n", "")  # argparse wraps the line at the terminal's width
    assert "delaunay-watershed-3d" in out and "delaunay-watershed-2d" in out and "iorec" in out
