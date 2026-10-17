"""The Polyscope viewer from a mask, its explode and cut, and its inference choices: the library's numbers, headless.

What this pins, and why:

* **A child job is the library call.** `ChildJob` runs a library function in a child process that the window polls
  every frame: what it returns is bitwise what the call returns in this process (a reconstruction's `.rec` bytes, a
  seed ensemble's tensions); a refusal comes back as the library's message, a cancelled job keeps nothing and stops
  its process, and an ensemble's progress arrives seed by seed.
* **The mask workflow is `foambryo mesh` then `foambryo infer`.** Its reconstruction is `reconstruct_mask`'s, its
  one-mesh inference `infer_tensions_then_pressures`' (with or without pressures), its ensemble
  `infer_tensions_over_seeds`'; the command lines it shows, run in a folder, write the same numbers.
* **No default of its own.** The workflow's choices default to `infer_forces`' and `infer_tensions_over_seeds`'
  signatures; `foambryo view --scattered`'s coefficient is the viewer functions' default.
* **The mask on screen** is every cell voxel at `index * spacing` (strided above a size), the frame of the mesh.
* **Explode and cut** in the window: the scattered view's coefficient is `compute_scattered_arrays`'; the cut is a
  scene slice plane whose normal points to the side kept (measured in a real window, Polyscope 2.6.1); Recompute with
  the pressures unticked draws `infer_tensions`' tensions and no pressure.
* **`foambryo view MASK`** passes the command's mask checks and spacing rules, and refuses what needs a mesh.

Synthetic masks (four cells in a ball) and the benchmark mesh 000 only; Polyscope's mock backend, no display.
"""

from __future__ import annotations

import inspect
import os
import shlex
import time
from pathlib import Path

import numpy as np
import pytest

from foambryo import (
    TensionComputationMethod,
    cli,
    infer_forces,
    infer_tensions,
    infer_tensions_then_pressures,
    reconstruct_mask,
)
from foambryo import results as tables
from foambryo._child_runs import ChildJob
from foambryo._mask_workflow import (
    SEED_ENSEMBLE,
    MaskWorkflow,
    ViewerChoices,
    equivalent_commands,
    label_colours,
    mask_point_cloud,
    reconstruction_estimate,
)
from foambryo.seed_ensembles import infer_tensions_over_seeds
from tests.test_command_line import _four_cells_in_a_ball, _run, _same_floats

from .conftest import require_substitute_mesh

TIMEOUT = 300.0


def sleep_then_return(seconds: float) -> str:
    """A slow child target, for the cancel test (imported by the child from this module)."""
    time.sleep(seconds)
    return "slept"


def refuse(reason: str) -> None:
    """A child target that raises the library's kind of refusal."""
    raise ValueError(reason)


@pytest.fixture(scope="module")
def mask():
    return _four_cells_in_a_ball()


@pytest.fixture(scope="module")
def reconstruction(mask):
    return reconstruct_mask(mask, (1.0, 1.0, 1.0), spacing_unit="voxel")


# ---------------------------------------------------------------------------------------------
# the child job
# ---------------------------------------------------------------------------------------------
def test_a_child_job_returns_the_library_calls_value_bitwise(mask, reconstruction):
    job = ChildJob("foambryo.io:reconstruct_mask",
                   {"segmentation_mask": mask, "spacing": (1.0, 1.0, 1.0), "spacing_unit": "voxel"})
    assert job.poll() in ("running", "done")
    assert job.wait(TIMEOUT) == "done", job.error
    assert job.value.rec_bytes == reconstruction.rec_bytes
    assert np.array_equal(job.value.mesh().v, reconstruction.mesh().v)
    assert isinstance(job.warnings_summary, list)
    assert job.seconds > 0


def test_a_refusal_comes_back_as_the_librarys_message():
    job = ChildJob(f"{__name__}:refuse", {"reason": "no triple line"})
    assert job.wait(TIMEOUT) == "failed"
    assert job.error == "ValueError: no triple line"
    assert job.value is None


def test_cancel_stops_the_child_and_keeps_nothing():
    job = ChildJob(f"{__name__}:sleep_then_return", {"seconds": 60.0})
    time.sleep(1.0)
    started = time.perf_counter()
    job.cancel()
    assert time.perf_counter() - started < 10
    assert job.state == "cancelled"
    assert job.value is None
    assert not job._process.is_alive()


def test_an_ensembles_progress_arrives_and_its_tensions_are_the_librarys(mask):
    kwargs = {"mask": mask, "seeds": [0, 1], "spacing": (1.0, 1.0, 1.0)}
    job = ChildJob("foambryo.seed_ensembles:infer_tensions_over_seeds", kwargs, with_progress=True)
    assert job.wait(TIMEOUT) == "done", job.error
    assert job.progress is not None
    assert job.progress[0] == job.progress[1] == 2
    library = infer_tensions_over_seeds(**kwargs)
    assert _same_floats(job.value.tensions, library.tensions)
    assert _same_floats(job.value.tension_seed_sd, library.tension_seed_sd)


# ---------------------------------------------------------------------------------------------
# the workflow
# ---------------------------------------------------------------------------------------------
def test_the_choices_default_to_the_librarys_signatures():
    choices = ViewerChoices.library()
    forces = inspect.signature(infer_forces).parameters
    seeds = inspect.signature(infer_tensions_over_seeds).parameters
    assert tables.TENSION_METHODS[choices.tension_method] is forces["mode_tension"].default
    assert tables.PRESSURE_METHODS[choices.pressure_method] is forces["mode_pressure"].default
    assert (choices.mean_tension, choices.base_pressure) == (forces["mean_tension"].default,
                                                             forces["base_pressure"].default)
    assert choices.solver == forces["solver"].default
    assert (choices.combine, choices.angle_reading) == (seeds["combine"].default, seeds["angle_reading"].default)
    assert choices.max_workers == seeds["max_workers"].default
    from foambryo import DEFAULT_ENSEMBLE_SEEDS

    assert choices.seeds == len(DEFAULT_ENSEMBLE_SEEDS)
    assert choices.pressures is True


def _wait(workflow: MaskWorkflow) -> str | None:
    deadline = time.perf_counter() + TIMEOUT
    while time.perf_counter() < deadline:
        event = workflow.poll()
        if event is not None:
            return event
        time.sleep(0.05)
    raise AssertionError(workflow.status)


@pytest.mark.parametrize("pressures", [True, False])
def test_the_workflow_is_foambryo_mesh_then_foambryo_infer(mask, reconstruction, pressures):
    choices = ViewerChoices.library(tension_method="lami", pressures=pressures)
    workflow = MaskWorkflow(mask, (1.0, 1.0, 1.0), "voxel", "mask.npy", 3, choices)
    workflow.start_inference()
    assert workflow.status == "reconstruct the mask first"
    assert workflow.job is None
    workflow.start_reconstruction()
    assert "estimate" in workflow.status
    assert workflow.busy
    assert _wait(workflow) == "reconstructed"
    assert workflow.reconstruction.rec_bytes == reconstruction.rec_bytes
    workflow.start_inference()
    assert _wait(workflow) == "inferred", workflow.status
    library = infer_tensions_then_pressures(reconstruction.mesh(), mode_tension=TensionComputationMethod.Lami,
                                            with_pressures=pressures)
    assert _same_floats(workflow.outcome.tensions, library.tensions)
    assert (workflow.outcome.pressures is None) == (not pressures)
    if pressures:
        assert _same_floats(workflow.outcome.pressures, library.pressures)
    assert workflow.inferred_with == choices


def test_a_workflow_ensemble_is_infer_tensions_over_seeds(mask):
    choices = ViewerChoices.library(mode=SEED_ENSEMBLE, seeds=2)
    workflow = MaskWorkflow(mask, (1.0, 1.0, 1.0), "voxel", "mask.npy", 3, choices)
    workflow.start_inference()  # an ensemble reconstructs the mask itself: no reconstruction needed first
    assert "seed ensemble of 2 seeds" in workflow.status
    assert _wait(workflow) == "inferred", workflow.status
    library = infer_tensions_over_seeds(mask, [0, 1], spacing=(1.0, 1.0, 1.0))
    assert _same_floats(workflow.ensemble.tensions, library.tensions)
    assert workflow.outcome is None


def test_a_cancelled_reconstruction_keeps_nothing(mask):
    workflow = MaskWorkflow(mask, (1.0, 1.0, 1.0), "voxel", "mask.npy", 3)
    workflow.start_reconstruction()
    workflow.cancel()
    assert _wait(workflow) == "cancelled"
    assert workflow.reconstruction is None
    assert workflow.mesh is None
    assert "nothing was kept" in workflow.status


@pytest.mark.parametrize("pressures", [True, False])
def test_the_command_lines_shown_write_the_workflows_numbers(capsys, tmp_path, mask, pressures):
    choices = ViewerChoices.library(pressures=pressures, solver="dense")
    workflow = MaskWorkflow(mask, (1.0, 1.0, 1.0), "voxel", "mask.npy", 3, choices)
    workflow.start_reconstruction()
    assert _wait(workflow) == "reconstructed"
    workflow.start_inference()
    assert _wait(workflow) == "inferred"
    np.save(tmp_path / "mask.npy", mask)
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        for line in workflow.commands():
            assert cli.main(shlex.split(line)[1:]) == cli.EXIT_OK, line
    finally:
        os.chdir(cwd)
    capsys.readouterr()
    assert (tmp_path / "mesh.rec").read_bytes() == workflow.reconstruction.rec_bytes
    assert _same_floats(tables.read_tension_table(tmp_path / "results" / "tensions.csv"), workflow.outcome.tensions)
    assert (tmp_path / "results" / "pressures.csv").exists() == pressures


def test_the_ensemble_command_line_names_infer_over_seeds():
    choices = ViewerChoices.library(mode=SEED_ENSEMBLE, seeds=4, max_workers=2)
    (line,) = equivalent_commands(choices, "my mask.tif", (2.0, 0.5, 0.5), "um", 3)
    arguments = shlex.split(line)
    assert arguments[:5] == ["foambryo", "infer-over-seeds", "my mask.tif", "-o", "seed-ensemble"]
    args = cli.build_parser().parse_args(arguments[1:])
    assert (args.seeds, args.max_workers, tuple(args.spacing), args.spacing_unit) == (4, 2, (2.0, 0.5, 0.5), "um")


# ---------------------------------------------------------------------------------------------
# the mask on screen, the estimate
# ---------------------------------------------------------------------------------------------
def test_the_mask_cloud_is_every_cell_voxel_in_the_meshs_frame(mask):
    spacing = (2.0, 0.5, 0.5)
    cloud = mask_point_cloud(mask, spacing)
    assert cloud.stride == 1
    assert len(cloud.points) == np.count_nonzero(mask)
    index = np.argwhere(mask > 0)
    assert np.array_equal(cloud.points, index * np.array(spacing))
    assert np.array_equal(cloud.labels, mask[tuple(index.T)])
    assert (cloud.labels > 0).all()
    strided = mask_point_cloud(mask, spacing, max_points=1000)
    assert strided.stride > 1
    assert len(strided.points) <= 1000 * 1.5
    assert np.array_equal(strided.points % (np.array(spacing) * strided.stride), np.zeros_like(strided.points))


def test_label_colours_are_stable_and_distinct():
    a = label_colours(np.array([3, 5, 3, 9]))
    b = label_colours(np.array([9]))
    assert np.array_equal(a[0], a[2])
    assert np.array_equal(a[3], b[0])  # a label's colour does not depend on others
    assert len({tuple(c) for c in label_colours(np.arange(1, 40))}) == 39
    assert ((a >= 0) & (a <= 1)).all()


def test_the_estimate_is_the_librarys_of_the_voxels_after_resampling():
    """The window's estimate is `foambryo.reconstruction_cost`'s, the one every front end gives: no copy here."""
    from foambryo.reconstruction_cost import RECONSTRUCTION_COST_FIT, estimate_reconstruction_cost

    estimate = reconstruction_estimate((100, 200, 200), (2.0, 1.0, 1.0))
    voxels = 100 * 200 * 200 * 2.0
    assert estimate.voxels == voxels
    assert estimate == estimate_reconstruction_cost(voxels)
    fit = RECONSTRUCTION_COST_FIT["seconds"]
    assert estimate.seconds[1] == fit["prefactor"] * voxels ** fit["exponent"]
    assert estimate.seconds[0] < estimate.seconds[1] < estimate.seconds[2]
    assert "about" in estimate.text()


# ---------------------------------------------------------------------------------------------
# the window, on Polyscope's mock backend
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def mock_polyscope(monkeypatch):
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    monkeypatch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=1))
    yield ps
    ps.remove_all_structures()
    ps.remove_all_slice_planes()


def _force_window(kind: str = "force"):
    from foambryo import dcel_mesh_from_file
    from foambryo.viewing import _ForceWindow, _InferenceOptions

    mesh = dcel_mesh_from_file(require_substitute_mesh())
    tensions, pressures = infer_forces(mesh)
    options = _InferenceOptions(TensionComputationMethod.YoungDupre, inspect.signature(infer_forces)
                                .parameters["mode_pressure"].default, 1.0, 0.0, "auto")
    return _ForceWindow(mesh, kind, tensions, pressures if kind == "force" else None, options,
                        "inferred here: infer_forces(mesh)", mesh_name="000_mesh.rec")


@pytest.mark.usefixtures("mock_polyscope")
def test_explode_is_the_librarys_scattered_view():
    window = _force_window()
    window.draw()
    window.set_explode(True, 0.35)
    expected = window.mesh.v_scattered.copy()
    probe = __import__("copy").copy(window.mesh)
    probe.compute_scattered_arrays(0.35)
    assert np.array_equal(expected, probe.v_scattered)
    assert np.array_equal(window.drawn_box(), np.stack([expected.min(axis=0), expected.max(axis=0)]))
    window.set_explode(False)
    assert np.array_equal(window.drawn_box(), np.stack([window.mesh.v.min(axis=0), window.mesh.v.max(axis=0)]))


@pytest.mark.parametrize(("axis", "keep_higher"), [(0, False), (2, True)])
@pytest.mark.usefixtures("mock_polyscope")
def test_the_cut_is_a_slice_plane_pointing_to_the_side_kept(axis, keep_higher):
    window = _force_window()
    window.draw()
    window.cut = (axis, 0.25, keep_higher)
    window.apply_cut()
    point, normal = window.cut_pose()
    low, high = window.drawn_box()
    assert point[axis] == low[axis] + 0.25 * (high[axis] - low[axis])
    assert normal[axis] == (1.0 if keep_higher else -1.0)
    assert np.count_nonzero(normal) == 1
    assert window.slice_plane is not None
    assert window.slice_plane.get_active()
    assert np.allclose(window.slice_plane.get_normal(), normal)
    window.cut = None
    window.apply_cut()
    assert not window.slice_plane.get_active()


@pytest.mark.usefixtures("mock_polyscope")
def test_recompute_without_pressures_draws_infer_tensions():
    window = _force_window()
    window.draw()
    window.pressures_choice = False
    window.tension_choice = list(TensionComputationMethod).index(TensionComputationMethod.Lami)
    window.solver_choice = 1  # dense
    window.recompute()
    assert window.kind == "tension"
    assert window.pressures is None
    assert _same_floats(window.tensions, infer_tensions(window.mesh, mode=TensionComputationMethod.Lami,
                                                        solver="dense"))
    assert "--no-pressures" in window.command_line()
    assert "--solver dense" in window.command_line()
    args = cli.build_parser().parse_args(shlex.split(window.command_line())[1:])
    assert args.no_pressures
    assert args.tension_method == "lami"
    window.pressures_choice = True
    window.recompute()
    assert window.kind == "force"
    assert window.pressures is not None  # a force viewer draws its pressures again


def test_the_mask_window_shows_the_mask_then_the_values(mock_polyscope, mask):
    from foambryo.viewing import _MASK, _MaskWindow

    workflow = MaskWorkflow(mask, (1.0, 1.0, 1.0), "voxel", "mask.npy", 3)
    window = _MaskWindow(workflow)
    window.draw()
    assert mock_polyscope.has_point_cloud(_MASK)
    window.panel()  # one frame of the real panel, on the mock backend
    workflow.start_reconstruction()
    assert _wait(workflow) == "reconstructed"
    window.draw()
    workflow.start_inference()
    assert _wait(workflow) == "inferred"
    window.show_values()
    assert window.window is not None
    assert window.window.kind == "force"
    assert not window.show_mask  # hidden once the values are drawn: its points would speckle the surface
    assert mock_polyscope.has_point_cloud(_MASK)  # but kept in the scene
    window.panel()


# ---------------------------------------------------------------------------------------------
# foambryo view MASK
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def recorded_workflows(monkeypatch):
    pytest.importorskip("polyscope")
    import foambryo.viewing as viewing

    workflows: list = []
    monkeypatch.setattr(viewing, "_view_mask_workflow", lambda workflow, **kwargs: workflows.append((workflow, kwargs)))
    return workflows


def test_view_mask_opens_a_workflow_with_the_commands_spacing_and_defaults(capsys, tmp_path, mask, recorded_workflows):
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    code, out, err = _run(capsys, "view", path, "--spacing", "1", "1", "1", "--spacing-unit", "voxel",
                          "--no-pressures", "--scattered")
    assert code == cli.EXIT_OK, err
    assert out.startswith("mask, for information (nothing is changed): 4 cells")
    workflow, kwargs = recorded_workflows[-1]
    assert np.array_equal(workflow.mask, mask)
    assert (workflow.spacing, workflow.unit, workflow.mask_name) == ((1.0, 1.0, 1.0), "voxel", "mask.npy")
    from dw3d import get_default_mesh_reconstruction_algorithm

    assert workflow.min_distance == inspect.signature(get_default_mesh_reconstruction_algorithm).parameters[
        "min_distance"].default
    assert workflow.choices.pressures is False
    from foambryo.viewing import plot_force_inference

    assert kwargs["scattered"] is True
    assert kwargs["scatter_coeff"] == inspect.signature(plot_force_inference).parameters["scatter_coeff"].default
    assert _run(capsys, "view", path, "--spacing", "1", "1", "1", "--min-distance", "5")[0] == cli.EXIT_OK
    assert recorded_workflows[-1][0].min_distance == 5


@pytest.mark.parametrize("extra", [["--results", "out"], ["--figure", "f.png"], ["--colour-bars", "b.png"],
                                   ["--no-show"], ["--no-pressures", "--pressure-method", "laplace"]])
def test_view_mask_refuses_what_needs_a_mesh(capsys, tmp_path, mask, recorded_workflows, extra):
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    code, _, err = _run(capsys, "view", path, "--spacing", "1", "1", "1", *extra)
    assert code == cli.EXIT_USAGE, err
    assert not recorded_workflows


def test_view_mask_without_a_spacing_is_refused_as_mesh_refuses_it(capsys, tmp_path, mask, recorded_workflows):
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    code, _, err = _run(capsys, "view", path)
    assert code == cli.EXIT_USAGE
    assert "records no voxel spacing" in err
    assert not recorded_workflows


@pytest.mark.parametrize("extra", [["--spacing", "1", "1", "1"], ["--min-distance", "3"], ["--spacing-unit", "um"]])
@pytest.mark.usefixtures("recorded_workflows")
def test_mask_options_on_a_mesh_are_refused(capsys, extra):
    code, _, err = _run(capsys, "view", require_substitute_mesh(), *extra)
    assert code == cli.EXIT_USAGE
    assert "applies to a mask" in err
