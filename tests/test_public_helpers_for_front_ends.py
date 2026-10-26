"""The functions front ends share with the `foambryo` command give the library's numbers and the command's files.

The command and the napari plugin call the same public functions, so that one mask gives one answer in Python,
on the command line and in napari. What this pins:

* `reconstruct_mask` is the command's reconstruction: its mesh gives bitwise the tensions and pressures of
  `dcel_mesh_from_segmentation_mask(mask, spacing=...)`, its `.rec` bytes are `foambryo mesh`'s file, and it
  survives pickling (a front end reconstructs in a child process) with the same numbers;
* `infer_tensions_then_pressures` is `infer_forces`, bitwise, and keeps the tensions when the pressures are refused;
* `PressureComputationMethod.has_crop_face_formulation` foresees exactly the methods `infer_pressures` refuses on
  a cropped mesh;
* `infer_tensions_over_seeds(progress=...)` reports each seed once, in order, and changes no number;
* `segmentation_mask_report` counts what it is built to count, on masks made with exactly those defects, and
  changes nothing;
* the public `results` and `colour_scales` modules are the command's own functions, not copies.

Synthetic masks only (four cells in a ball, built in the command's tests).
"""

from __future__ import annotations

import hashlib
import pickle
import warnings

import numpy as np
import pytest

from foambryo import (
    PressureComputationMethod,
    TensionComputationMethod,
    _colour_scales,
    _result_tables,
    cli,
    colour_scales,
    dcel_mesh_from_segmentation_mask,
    infer_forces,
    infer_pressures,
    infer_tensions,
    infer_tensions_over_seeds,
    infer_tensions_then_pressures,
    reconstruct_mask,
    results,
)
from foambryo.segmentation_masks import (
    SegmentationMaskError,
    check_segmentation_mask,
    read_segmentation_mask,
    segmentation_mask_report,
)
from tests.test_command_line import MASKS, _same_floats


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


SPACINGS = {"cubic voxels": (0.5, 0.5, 0.5), "z twice as long": (1.0, 0.5, 0.5)}


@pytest.fixture(scope="module")
def reconstructions() -> dict[tuple[str, str], object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {(m, s): reconstruct_mask(MASKS[m](), SPACINGS[s]) for m in MASKS for s in SPACINGS}


# ---------------------------------------------------------------------------------------------
# reconstruct_mask
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("spacing_name", list(SPACINGS))
@pytest.mark.parametrize("mask_name", list(MASKS))
def test_reconstruct_mask_is_the_library_mesh_and_the_commands_file(
    reconstructions, tmp_path, capsys, mask_name, spacing_name,
):
    spacing = SPACINGS[spacing_name]
    reconstruction = reconstructions[mask_name, spacing_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        library = dcel_mesh_from_segmentation_mask(MASKS[mask_name](), spacing=spacing)
        mesh = reconstruction.mesh()
        assert _sha(mesh.v) == _sha(library.v)
        assert _sha(mesh.f) == _sha(library.f)
        assert _sha(mesh.crop_faces) == _sha(library.crop_faces)
        assert mesh.source_spacing == library.source_spacing == spacing
        expected = infer_forces(library)
        assert _same_floats(infer_forces(mesh)[0], expected[0])
        assert _same_floats(infer_forces(mesh)[1], expected[1])
        # building a mesh leaves the reconstruction as it was
        assert _sha(reconstruction.mesh().v) == _sha(mesh.v)

    np.save(tmp_path / "mask.npy", MASKS[mask_name]())
    code = cli.main(["mesh", str(tmp_path / "mask.npy"), "-o", str(tmp_path / "command.rec"),
                     "--spacing", *map(str, spacing)])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert (tmp_path / "command.rec").read_bytes() == reconstruction.rec_bytes
    reconstruction.write_rec(tmp_path / "written.rec")
    assert (tmp_path / "written.rec").read_bytes() == reconstruction.rec_bytes
    assert out == (f"{segmentation_mask_report(MASKS[mask_name]()).summary()}\n"
                   f"wrote {tmp_path / 'command.rec'}: {reconstruction.summary()}\n")
    # The cut cells are the cells with a triangle that has a vertex on a crop face (delaunay-watershed-3d's
    # rule), counted here on the reconstruction's own arrays, not pinned as a number: on the cut mask, which
    # cells carry such a vertex depends on how Qhull splits sets of cospherical lattice points, which differs
    # between machines (3 cut cells on macOS, 4 on Linux, with the same delaunay-watershed-3d 0.7.0).
    if mask_name == "four cells cut by the image box":
        on_crop = np.isin(reconstruction.triangles, np.unique(reconstruction.triangles[reconstruction.crop_faces]))
        cut = np.setdiff1d(reconstruction.labels[on_crop.any(axis=1)], [0])
        assert reconstruction.n_cut_cells == len(cut) > 0
    else:
        assert reconstruction.n_cut_cells == 0
    assert reconstruction.n_cells == int(np.count_nonzero(np.unique(library.f[:, 3:5])))


def test_a_reconstruction_survives_pickling_with_the_same_numbers(reconstructions):
    reconstruction = reconstructions["four cells cut by the image box", "z twice as long"]
    copy = pickle.loads(pickle.dumps(reconstruction))  # noqa: S301 - the test's own object
    assert copy.rec_bytes == reconstruction.rec_bytes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        here, there = infer_forces(reconstruction.mesh()), infer_forces(copy.mesh())
    assert _same_floats(here[0], there[0])
    assert _same_floats(here[1], there[1])


# ---------------------------------------------------------------------------------------------
# infer_tensions_then_pressures
# ---------------------------------------------------------------------------------------------
def test_infer_tensions_then_pressures_defaults_are_infer_forces_defaults():
    import inspect

    ours = inspect.signature(infer_tensions_then_pressures).parameters
    theirs = inspect.signature(infer_forces).parameters
    for name in ("mean_tension", "base_pressure", "mode_tension", "mode_pressure", "solver", "diagnostics"):
        assert ours[name].default == theirs[name].default, name


@pytest.mark.parametrize("mode_tension", [TensionComputationMethod.YoungDupre, TensionComputationMethod.Lami])
@pytest.mark.parametrize("diagnostics", [False, "spectrum"])
def test_infer_tensions_then_pressures_is_infer_forces(reconstructions, mode_tension, diagnostics):
    mesh = reconstructions["four cells in a ball", "cubic voxels"].mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = infer_tensions_then_pressures(mesh, 2.0, 0.5, mode_tension, diagnostics=diagnostics)
        expected = infer_forces(mesh, 2.0, 0.5, mode_tension, diagnostics=diagnostics)
    assert outcome.status == "complete"
    assert outcome.error is None
    assert _same_floats(outcome.tensions, expected[0])
    assert _same_floats(outcome.pressures, expected[1])
    assert len(outcome.records) == (0 if diagnostics is False else len(expected[2].records))
    assert outcome.library_call.startswith(f"infer_forces(mesh, mean_tension=2.0, base_pressure=0.5, "
                                           f"mode_tension=TensionComputationMethod.{mode_tension.name}")


def test_a_refused_pressure_method_keeps_the_tensions(reconstructions):
    mesh = reconstructions["four cells cut by the image box", "cubic voxels"].mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = infer_tensions_then_pressures(mesh, mode_pressure=PressureComputationMethod.WeightedLaplace)
        tensions = infer_tensions(mesh)
    assert outcome.status == "failed"
    assert "no formulation for crop faces" in outcome.error
    assert _same_floats(outcome.tensions, tensions)
    assert outcome.pressures is None


# ---------------------------------------------------------------------------------------------
# crop-face formulation (foreseeable refusals)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", list(PressureComputationMethod))
def test_the_crop_face_property_foresees_the_refusal(reconstructions, method):
    mesh = reconstructions["four cells cut by the image box", "cubic voxels"].mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensions = infer_tensions(mesh)
        if method.has_crop_face_formulation:
            infer_pressures(mesh, tensions, mode=method)
        else:
            with pytest.raises(ValueError, match="no formulation for crop faces"):
                infer_pressures(mesh, tensions, mode=method)
    assert method.has_crop_face_formulation == (method is PressureComputationMethod.Variational)


# ---------------------------------------------------------------------------------------------
# seed ensembles: progress
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("max_workers", [1, 2])
def test_progress_is_called_once_per_seed_in_order_and_changes_nothing(max_workers):
    mask = MASKS["four cells in a ball"]()
    calls = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_progress = infer_tensions_over_seeds(mask, [3, 1, 2], max_workers=max_workers,
                                                  progress=lambda *call: calls.append(call))
        without = infer_tensions_over_seeds(mask, [3, 1, 2], max_workers=max_workers)
    assert calls == [(1, 3, 3), (2, 3, 1), (3, 3, 2)]
    assert _same_floats(with_progress.tensions, without.tensions)
    assert _same_floats(with_progress.tension_seed_sd, without.tension_seed_sd)
    for seed in (3, 1, 2):
        assert _same_floats(with_progress.per_seed_tensions[seed], without.per_seed_tensions[seed])


# ---------------------------------------------------------------------------------------------
# segmentation masks
# ---------------------------------------------------------------------------------------------
def test_reading_and_checking_give_the_commands_messages(tmp_path):
    np.save(tmp_path / "flat.npy", np.zeros((4, 4), dtype=np.uint8))
    np.save(tmp_path / "floats.npy", np.zeros((4, 4, 4)))
    np.save(tmp_path / "negative.npy", -np.ones((4, 4, 4), dtype=np.int32))
    np.savez(tmp_path / "two.npz", a=np.zeros((2, 2, 2), dtype=np.uint8), b=np.zeros((2, 2, 2), dtype=np.uint8))
    expected = {
        "flat.npy": "flat.npy holds a 2-D array; a mask is 3-D",
        "floats.npy": "floats.npy holds float64 values; a mask holds integer labels",
        "negative.npy": "negative.npy holds negative labels; labels are 0 (medium) or positive",
        "two.npz": "two.npz holds the arrays ['a', 'b']: name the mask 'mask'",
    }
    for name, message in expected.items():
        with pytest.raises(SegmentationMaskError) as raised:
            read_segmentation_mask(tmp_path / name)
        assert str(raised.value) == message
    with pytest.raises(SegmentationMaskError, match="holds bool values"):
        check_segmentation_mask(np.zeros((2, 2, 2), dtype=bool))
    mask = np.zeros((2, 2, 2), dtype=np.uint16)
    np.savez(tmp_path / "with_spacing.npz", mask=mask, spacing=[2.0, 1.0, 1.0], spacing_unit="nm")
    read, spacing, unit = read_segmentation_mask(tmp_path / "with_spacing.npz")
    assert read.dtype == np.uint16
    assert (spacing, unit) == ((2.0, 1.0, 1.0), "nm")


def _mask_with_known_defects() -> np.ndarray:
    """Labels 5 (two pieces), 7 (encloses a hole of medium), 9 (touches the box), 11 (small); not consecutive."""
    mask = np.zeros((30, 30, 30), dtype=np.uint16)
    mask[2:8, 2:8, 2:8] = 5
    mask[20:26, 2:8, 2:8] = 5  # a second piece of 5
    mask[10:20, 10:20, 10:20] = 7
    mask[14:16, 14:16, 14:16] = 0  # medium enclosed by 7
    mask[0:6, 20:26, 20:26] = 9  # touches the face z = 0
    mask[25, 25, 25] = 11  # one voxel
    mask[22:28, 12:18, 22:28] = 13
    return mask


@pytest.mark.parametrize("dtype", [np.uint16, np.int64, np.uint64])
def test_the_mask_report_counts_the_defects_it_was_built_with(dtype):
    mask = _mask_with_known_defects().astype(dtype)
    before = mask.copy()
    report = segmentation_mask_report(mask)
    assert np.array_equal(mask, before)
    assert report.n_cells == 5
    assert report.cells_touching_the_image_box == (9,)
    assert report.cells_in_several_pieces == {5: 2}
    assert (report.enclosed_medium_pieces, report.enclosed_medium_voxels) == (1, 8)
    assert report.small_labels == {11: 1}
    assert report.median_cell_voxels == 216.0
    assert report.summary() == (
        "mask, for information (nothing is changed): 5 cells; 1 touch the image box (cut cells); 1 in several "
        "pieces; 1 pieces of medium enclosed by the tissue; 1 smaller than 5% of the median cell (10.8 voxels)"
    )
    assert report.as_record()["cells_in_several_pieces"] == {"5": 2}


def test_the_mask_report_reads_large_sparse_label_values():
    mask = _mask_with_known_defects().astype(np.uint64)
    mask[mask == 13] = 2**40
    report = segmentation_mask_report(mask)
    assert report.n_cells == 5
    assert report.cells_in_several_pieces == {5: 2}
    assert report.small_labels == {11: 1}


def test_pieces_touching_only_at_an_edge_are_two_pieces():
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask[3:5, 3:5, 1:3] = 1  # shares an edge with the first block, no face
    assert segmentation_mask_report(mask).cells_in_several_pieces == {1: 2}


# ---------------------------------------------------------------------------------------------
# the public modules are the command's own functions
# ---------------------------------------------------------------------------------------------
def test_the_public_modules_re_export_the_commands_functions():
    for name in colour_scales.__all__:
        assert getattr(colour_scales, name) is getattr(_colour_scales, name), name
    for name in ("write_tension_table", "read_tension_table", "format_float", "write_run_record", "RUN_RECORD_SCHEMA"):
        assert getattr(results, name) is getattr(_result_tables, name), name
    assert cli.tables is results
    assert cli.TENSION_METHODS is results.TENSION_METHODS
    assert results.option_name(TensionComputationMethod.ForceBalanceMAP) == "force-balance-map"


def test_the_warning_collector_counts_each_warning_once():
    with results.WarningCollector() as collector:
        for _ in range(3):
            warnings.warn("same", UserWarning, stacklevel=1)
        warnings.warn("other", RuntimeWarning, stacklevel=1)
        warnings.warn("hidden", DeprecationWarning, stacklevel=1)
    assert collector.summary == [
        {"category": "UserWarning", "message": "same", "count": 3},
        {"category": "RuntimeWarning", "message": "other", "count": 1},
    ]
