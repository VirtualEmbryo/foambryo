"""The Polyscope viewer started from a mask: what it shows and runs, computed without Polyscope.

`foambryo view MASK --spacing DZ DY DX` opens the mask in the viewer's window first, reconstructs it there and
infers on the mesh, so that a user does not go back and forth between the terminal and the window. This module holds
everything of that workflow that is not drawing, so that it is tested without a display:

* **the mask on screen** (`mask_point_cloud`): the voxels of every cell (the medium left out) as points at their
  physical positions, one colour per label (`label_colours`); a large mask is strided so that the window stays
  responsive. Polyscope's volume grids cannot leave a value out (version 2.6.1 has no per-value culling), so the
  medium would wall the tissue in: points are what lets the cells be seen, and the scene's slice plane cuts them;
* **the estimate** of a reconstruction (`reconstruction_estimate`): `foambryo.reconstruction_cost`'s, the one every
  front end gives (power laws of the voxels handed to delaunay-watershed-3d; the band is in-sample, not a validated
  interval);
* **the choices** (`ViewerChoices`): tension method, pressure method or no pressures, solver, one mesh or a seed
  ensemble, each mapped to the library's argument with the library's default, and the equivalent command lines;
* **the workflow** (`MaskWorkflow`): the reconstruction and the inference run in child processes (`ChildJob`), the
  window polls them every frame, Cancel stops them; the numbers are the library calls' own.
"""

from __future__ import annotations

import inspect
import math
import shlex
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from foambryo._child_runs import CANCELLED, DONE, FAILED, RUNNING, ChildJob
from foambryo.reconstruction_cost import (
    ReconstructionCostEstimate,
    _about_duration,
    estimate_reconstruction_cost,
    voxels_handed_to_dw3d,
)
from foambryo.results import DIAGNOSTICS, PRESSURE_METHODS, TENSION_METHODS, option_name

if TYPE_CHECKING:
    from foambryo.dcel import DcelData

__all__ = [
    "MaskCloud",
    "MaskWorkflow",
    "ReconstructionEstimate",
    "ViewerChoices",
    "label_colours",
    "mask_point_cloud",
    "reconstruction_estimate",
]

#: The most points the mask is drawn with; a larger mask is strided along every axis.
MAX_MASK_POINTS = 3_000_000


# ---------------------------------------------------------------------------------------------
# the mask on screen
# ---------------------------------------------------------------------------------------------
def label_colours(labels: np.ndarray) -> np.ndarray:
    """A distinct, stable RGB colour per label (golden-ratio hues), as an `(n, 3)` array in [0, 1].

    The same label always has the same colour, whatever the other labels; neighbouring labels differ in hue.
    """
    import colorsys

    labels = np.asarray(labels, dtype=np.int64)
    unique, inverse = np.unique(labels, return_inverse=True)
    table = np.array([colorsys.hsv_to_rgb((0.618033988749895 * int(label)) % 1.0, 0.55 + 0.35 * (int(label) % 2),
                                          0.95) for label in unique], dtype=np.float64).reshape(-1, 3)
    return table[inverse.reshape(-1)]


@dataclass(frozen=True)
class MaskCloud:
    """The mask's cell voxels as points.

    Attributes:
        points: `(n, 3)` positions in the mesh's frame: voxel index (z, y, x) times the spacing.
        labels: `(n,)` the label of each point (never 0, the medium).
        colours: `(n, 3)` `label_colours(labels)`.
        stride: every `stride`-th voxel along each axis is drawn (1: every voxel).
        radius: the point radius, half the drawn voxels' smallest spacing, in the mesh's unit.
    """

    points: np.ndarray
    labels: np.ndarray
    colours: np.ndarray
    stride: int
    radius: float


def mask_point_cloud(
    mask: np.ndarray, spacing: tuple[float, float, float], max_points: int = MAX_MASK_POINTS,
) -> MaskCloud:
    """The cell voxels of `mask` as points, strided so that there are at most about `max_points`.

    Positions are `index * spacing`, the frame of `foambryo mesh`'s mesh (and of
    `dcel_mesh_from_segmentation_mask(mask, spacing=...)`), so the mesh is drawn over its mask.
    """
    mask = np.asarray(mask)
    cells = int(np.count_nonzero(mask))
    stride = max(1, math.ceil((cells / max_points) ** (1 / 3))) if cells > max_points else 1
    view = mask[::stride, ::stride, ::stride]
    index = np.argwhere(view > 0)
    labels = view[tuple(index.T)].astype(np.int64)
    points = (index * stride).astype(np.float64) * np.asarray(spacing, dtype=np.float64)
    return MaskCloud(points, labels, label_colours(labels), stride, 0.5 * stride * float(min(spacing)))


# ---------------------------------------------------------------------------------------------
# the estimate
# ---------------------------------------------------------------------------------------------
#: A reconstruction's estimated time and peak memory: `foambryo.reconstruction_cost`'s.
ReconstructionEstimate = ReconstructionCostEstimate


def reconstruction_estimate(shape: tuple[int, ...], spacing: tuple[float, float, float]) -> ReconstructionEstimate:
    """The estimate for a mask of this shape and voxel size (`foambryo.reconstruction_cost`).

    The voxels counted are those after resampling to cubic voxels.
    """
    return estimate_reconstruction_cost(voxels_handed_to_dw3d(shape, spacing))


# ---------------------------------------------------------------------------------------------
# the choices
# ---------------------------------------------------------------------------------------------
def _defaults() -> dict[str, Any]:
    """The library's defaults for every choice: `infer_forces`' and `infer_tensions_over_seeds`'."""
    from foambryo.force_inference import infer_forces
    from foambryo.seed_ensembles import DEFAULT_ENSEMBLE_SEEDS, infer_tensions_over_seeds

    forces = inspect.signature(infer_forces).parameters
    seeds = inspect.signature(infer_tensions_over_seeds).parameters
    return {
        "tension_method": option_name(forces["mode_tension"].default),
        "pressure_method": option_name(forces["mode_pressure"].default),
        "mean_tension": float(forces["mean_tension"].default),
        "base_pressure": float(forces["base_pressure"].default),
        "solver": forces["solver"].default,
        "diagnostics": "none",
        "seeds": len(DEFAULT_ENSEMBLE_SEEDS),
        "max_workers": int(seeds["max_workers"].default),
        "combine": seeds["combine"].default,
        "angle_reading": seeds["angle_reading"].default,
    }


ONE_MESH, SEED_ENSEMBLE = "one mesh", "seed ensemble"
SOLVERS = ("auto", "dense", "sparse")


@dataclass(frozen=True)
class ViewerChoices:
    """The inference the window runs, by the command's spellings; `ViewerChoices.library()` has every default.

    Attributes:
        tension_method, pressure_method: `foambryo infer`'s spellings (`young-dupre`, `variational`, ...).
        pressures: `False` is the tensions only (`foambryo infer --no-pressures`).
        mean_tension, base_pressure, solver: `infer_forces`' options.
        diagnostics: `foambryo infer`'s `--diagnostics` (`none`, `basic`, `spectrum`): the solve records the window
            exports as `solves.csv`.
        mode: `"one mesh"` (`infer_tensions_then_pressures` on the reconstruction) or `"seed ensemble"`
            (`infer_tensions_over_seeds` on the mask: tensions only, no single mesh carries pressures).
        seeds, max_workers, combine, angle_reading: `foambryo infer-over-seeds`' options.
    """

    tension_method: str
    pressure_method: str
    mean_tension: float
    base_pressure: float
    solver: str
    pressures: bool = True
    diagnostics: str = "none"
    mode: str = ONE_MESH
    seeds: int = 8
    max_workers: int = 1
    combine: str = "tensions"
    angle_reading: str = "chord"

    @classmethod
    def library(cls, **given: Any) -> ViewerChoices:  # noqa: ANN401
        """The library's defaults, with `given` replacing some."""
        return replace(cls(**{k: v for k, v in _defaults().items() if k in cls.__dataclass_fields__}), **given)

    def one_mesh_call(self, mesh: DcelData) -> tuple[str, dict[str, Any]]:
        """The child job of a one-mesh inference: `infer_tensions_then_pressures`, exactly `foambryo infer`'s call."""
        return "foambryo.force_inference:infer_tensions_then_pressures", {
            "mesh": mesh,
            "mean_tension": self.mean_tension,
            "base_pressure": self.base_pressure,
            "mode_tension": TENSION_METHODS[self.tension_method],
            "mode_pressure": PRESSURE_METHODS[self.pressure_method],
            "solver": self.solver,
            "diagnostics": DIAGNOSTICS[self.diagnostics],
            "with_pressures": self.pressures,
        }

    def ensemble_call(
        self, mask: np.ndarray, spacing: tuple[float, float, float], min_distance: int,
    ) -> tuple[str, dict[str, Any]]:
        """The child job of a seed ensemble: `infer_tensions_over_seeds`, exactly `foambryo infer-over-seeds`' call."""
        return "foambryo.seed_ensembles:infer_tensions_over_seeds", {
            "mask": mask,
            "seeds": list(range(self.seeds)),
            "min_distance": min_distance,
            "mode": TENSION_METHODS[self.tension_method],
            "angle_reading": self.angle_reading,
            "combine": self.combine,
            "max_workers": self.max_workers,
            "spacing": tuple(spacing),
        }

    def infer_arguments(self) -> list[str]:
        """`foambryo infer`'s options for these choices (`--no-pressures` instead of the pressure options)."""
        pressure = (["--pressure-method", self.pressure_method, "--base-pressure", repr(self.base_pressure)]
                    if self.pressures else ["--no-pressures"])
        diagnostics = ["--diagnostics", self.diagnostics] if self.diagnostics != "none" else []
        return ["--tension-method", self.tension_method, *pressure, "--mean-tension", repr(self.mean_tension),
                "--solver", self.solver, *diagnostics]

    def record_options(self) -> dict[str, Any]:
        """`run.json`'s `options` for these choices, as `foambryo infer` writes them."""
        options: dict[str, Any] = {"tension_method": self.tension_method, "pressure_method": self.pressure_method,
                                   "mean_tension": self.mean_tension, "base_pressure": self.base_pressure,
                                   "solver": self.solver, "diagnostics": self.diagnostics}
        if not self.pressures:
            options.update(pressure_method=None, base_pressure=None, no_pressures=True)
        return options

    def ensemble_arguments(self) -> list[str]:
        """`foambryo infer-over-seeds`' options for these choices."""
        return ["--seeds", str(self.seeds), "--tension-method", self.tension_method, "--combine", self.combine,
                "--angle-reading", self.angle_reading, "--max-workers", str(self.max_workers)]


def equivalent_commands(
    choices: ViewerChoices,
    mask_name: str,
    spacing: tuple[float, float, float],
    unit: str,
    min_distance: int,
) -> list[str]:
    """The command lines that give the window's numbers (run where the mask is; `mesh.rec` and `results` are names)."""
    mask = shlex.quote(mask_name)
    space = " ".join(repr(float(v)) for v in spacing)
    common = f"--spacing {space} --spacing-unit {shlex.quote(unit)} --min-distance {min_distance}"
    if choices.mode == SEED_ENSEMBLE:
        return [" ".join(["foambryo infer-over-seeds", mask, "-o", "seed-ensemble", common,
                          *choices.ensemble_arguments()])]
    return [f"foambryo mesh {mask} -o mesh.rec {common}",
            " ".join(["foambryo infer mesh.rec -o results", *choices.infer_arguments()])]


# ---------------------------------------------------------------------------------------------
# the workflow
# ---------------------------------------------------------------------------------------------
@dataclass
class MaskWorkflow:
    """A mask, its reconstruction and the last inference, with the child job running, if any.

    `start_reconstruction` and `start_inference` start a child job and return at once; the window calls `poll`
    every frame, which returns an event when a job ended (`"reconstructed"`, `"inferred"`, `"failed"`,
    `"cancelled"`), else `None`. Every number is the library call's own (`foambryo.io.reconstruct_mask`,
    `infer_tensions_then_pressures`, `infer_tensions_over_seeds`), sent back by pickling.

    Attributes:
        mask, spacing, unit, mask_name: the mask, its voxel size and unit (already accepted by the command's
            checks), and its file name.
        min_distance: the reconstruction's option (the command's `--min-distance`).
        choices: the inference choices.
        reconstruction: `foambryo.io.MaskReconstruction`, once reconstructed; `mesh` its `DcelData`.
        outcome: the last one-mesh inference (`InferenceOutcome`), or `None`; `ensemble` the last seed ensemble.
        inferred_with: the choices of the last inference; `inferred_date` when it started (ISO 8601) and
            `inferred_seconds` how long it took.
        job, job_kind: the running child job and what it is (`"reconstruct"` or `"infer"`), or `None`.
        status: what the window says about the last step; `warnings`: their warnings, each once with its count.
    """

    mask: np.ndarray
    spacing: tuple[float, float, float]
    unit: str
    mask_name: str
    min_distance: int
    choices: ViewerChoices = field(default_factory=ViewerChoices.library)
    reconstruction: Any = None
    mesh: Any = None
    outcome: Any = None
    ensemble: Any = None
    inferred_with: ViewerChoices | None = None
    inferred_date: str = ""
    inferred_seconds: float = 0.0
    running_choices: ViewerChoices | None = None
    running_date: str = ""
    job: ChildJob | None = None
    job_kind: str | None = None
    status: str = ""
    warnings: list = field(default_factory=list)

    @property
    def busy(self) -> bool:
        """A child job is running."""
        return self.job is not None and self.job.state == RUNNING

    def estimate(self) -> ReconstructionEstimate:
        """The reconstruction's estimate for this mask (`reconstruction_estimate`)."""
        return reconstruction_estimate(self.mask.shape, self.spacing)

    def start_reconstruction(self) -> None:
        """Reconstruct the mask in a child process (`foambryo.io.reconstruct_mask`, `foambryo mesh`'s call)."""
        if self.busy:
            return
        self.job = ChildJob("foambryo.io:reconstruct_mask", {
            "segmentation_mask": self.mask, "spacing": self.spacing, "spacing_unit": self.unit,
            "min_distance": self.min_distance,
        })
        self.job_kind = "reconstruct"
        self.status = f"reconstructing in a child process (estimate: {self.estimate().text()})"

    def start_inference(self) -> None:
        """Infer with `choices` in a child process: on the reconstruction (one mesh), or on the mask (an ensemble)."""
        if self.busy:
            return
        if self.choices.mode == SEED_ENSEMBLE:
            target, kwargs = self.choices.ensemble_call(self.mask, self.spacing, self.min_distance)
            self.job = ChildJob(target, kwargs, with_progress=True)
            seconds = self.estimate().seconds[1] * math.ceil(self.choices.seeds / self.choices.max_workers)
            self.status = (f"seed ensemble of {self.choices.seeds} seeds in a child process (estimate: "
                           f"{_about_duration(seconds)})")
        else:
            if self.mesh is None:
                self.status = "reconstruct the mask first"
                return
            target, kwargs = self.choices.one_mesh_call(self.mesh)
            self.job = ChildJob(target, kwargs)
            self.status = "inferring in a child process"
        self.job_kind = "infer"
        self.running_choices = self.choices
        self.running_date = datetime.now(UTC).isoformat(timespec="seconds")

    def cancel(self) -> None:
        """Stop the running job; nothing it computed is kept."""
        if self.job is not None:
            self.job.cancel()

    def progress_text(self) -> str:
        """The running job's elapsed time, and its progress for an ensemble."""
        if not self.busy:
            return ""
        text = f"{self.status}: {self.job.seconds:.0f} s"
        if self.job.progress is not None:
            done, total, seed = self.job.progress
            text += f", {done} of {total} seeds done (last: seed {seed})"
        return text + ". Cancel stops it."

    def poll(self) -> str | None:
        """Read the running job; returns an event when it ended, else `None`."""
        if self.job is None:
            return None
        state = self.job.poll()
        if state == RUNNING:
            return None
        job, kind = self.job, self.job_kind
        self.job, self.job_kind = None, None
        self.warnings = job.warnings_summary
        if state == CANCELLED:
            self.status = f"cancelled after {job.seconds:.0f} s: nothing was kept"
            return "cancelled"
        if state == FAILED:
            self.status = f"the library refused: {job.error}"
            return "failed"
        assert state == DONE
        if kind == "reconstruct":
            self.reconstruction, self.mesh = job.value, job.value.mesh()
            self.outcome = self.ensemble = self.inferred_with = None
            self.status = f"{job.value.summary()} in {job.seconds:.1f} s"
            return "reconstructed"
        self.inferred_with = self.running_choices
        self.inferred_date, self.inferred_seconds = self.running_date, round(job.seconds, 3)
        if self.inferred_with.mode == SEED_ENSEMBLE:
            self.ensemble, self.outcome = job.value, None
            flagged = int(sum(job.value.found_by_fewer_than_half_the_seeds.values()))
            self.status = (f"seed ensemble of {self.inferred_with.seeds} seeds in {job.seconds:.1f} s: "
                           f"{len(job.value.tensions)} interfaces, {flagged} found by fewer than half of the seeds")
        else:
            self.outcome, self.ensemble = job.value, None
            self.status = f"{job.value.status} in {job.seconds:.1f} s" + (f": {job.value.error}" if job.value.error
                                                                         else "")
        return "inferred"

    def commands(self) -> list[str]:
        """The command lines that give the window's numbers for the current choices."""
        return equivalent_commands(self.choices, self.mask_name, self.spacing, self.unit, self.min_distance)
