"""The `foambryo` command: mesh a segmentation mask, infer its tensions and pressures, view them.

Four sub-commands, each a thin layer over the library:

* `foambryo mesh MASK -o MESH.rec --spacing DZ DY DX`: delaunay-watershed-3d's default
  reconstruction of the mask in physical units (`construct_mesh_in_physical_units`: a mask whose
  voxels are longer along one axis is first resampled to cubic voxels), written as a `.rec` file
  whose metadata appendix records the spacing;
* `foambryo infer MESH -o OUTDIR`: `dcel_mesh_from_file`, then the tensions and the pressures exactly
  as `infer_forces` computes them (the tensions only with `--no-pressures`), written as `tensions.csv`,
  `pressures.csv` and a run record, `run.json` (and, with `--diagnostics`, the solves as a table,
  `solves.csv`, and with `spectrum` their singular values, `spectra.npz`);
* `foambryo infer-over-seeds MASK -o OUTDIR --spacing DZ DY DX [--seeds N] [--combine ...]`:
  `infer_tensions_over_seeds`, the mask reconstructed at seeds 0..N-1 and the tensions combined over them,
  written as `tensions.csv` (with the seed spread, how many seeds found each interface and a flag),
  `tensions_per_seed.csv` (every seed's own tensions) and `run.json`;
* `foambryo view MESH [--results OUTDIR] [--colour-bars FILE] [--figure FILE] [--no-show]
  [--scale-bar LENGTH | --no-scale-bar] [--scattered [COEFF]] [--no-pressures] [...]`: the Polyscope viewer
  (`foambryo[viewing]` and a display), with its colour bars and a length scale bar, the colour bars of its
  surface colours as a file (matplotlib only, no display), and the window's view with its bars as a figure; it
  infers with `infer`'s options (`--tension-method`, `--pressure-method`, `--mean-tension`, `--base-pressure`,
  `--solver`; no `--diagnostics`), or reads what `infer` or `infer-over-seeds` wrote. `foambryo view MASK
  --spacing DZ DY DX` opens the mask instead, and reconstructs and infers it in the window (`mesh`'s options,
  then `infer`'s or `infer-over-seeds`').

**The command has no default of its own.** Every default is read from the library function the
option is passed to, when the parser is built (`_library_default`), so the command line and a Python
call with no argument give the same numbers.

Exit codes: 0 success; 1 an unexpected error (with its traceback); 2 an invalid or incomplete
command line (argparse's own code), including a mask that records no spacing without `--spacing`;
3 a file that cannot be used as asked (missing, unreadable, of the wrong content, an existing output
without `--overwrite`, or an optional package not installed); 4 a reconstruction or an inference the
library refused, after writing what was computed before the refusal. Exit 2 also covers options that
contradict each other (`--no-pressures` with `--pressure-method` or `--base-pressure`, an inference option of
`view` with `--results`, `--no-show` with nothing to write). Errors are one line on standard
error, `foambryo <sub-command>: error: <message>`; warnings are relayed there once each,
`foambryo <sub-command>: warning: <Category>: <message>` (with ` (<n> times)` when raised more than once).

The behaviour, files and messages are specified in the README's "Command line" section and, in full, in
`docs/command_line_specification.md` of the source repository.
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from foambryo import results as tables
from foambryo.force_inference import infer_tensions_then_pressures
from foambryo.io import dcel_mesh_from_file, reconstruct_mask
from foambryo.results import (
    DIAGNOSTICS,
    PRESSURE_METHODS,
    TENSION_METHODS,
    WarningCollector,
    library_versions,
    option_name,
)
from foambryo.second_order_angle_reading import ANGLE_READINGS
from foambryo.segmentation_masks import (
    MASK_SUFFIXES,
    SPACING_UNIT_PATTERN,
    SegmentationMaskError,
    read_segmentation_mask,
    segmentation_mask_report,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from foambryo.dcel import DcelData

__all__ = [
    "DIAGNOSTICS",
    "MASK_SUFFIXES",
    "PRESSURE_METHODS",
    "TENSION_METHODS",
    "AnisotropicVoxelSpacingWarning",
    "CommandError",
    "build_parser",
    "main",
    "option_name",
    "read_mask",
]

PROGRAM = "foambryo"

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_FILE = 3
EXIT_REFUSED = 4

#: What a `--spacing-unit` may be: it is written into the `.rec` appendix as `key = value` text.
_UNIT_PATTERN = SPACING_UNIT_PATTERN


class AnisotropicVoxelSpacingWarning(UserWarning):
    """A mask's voxels are longer along one axis: `foambryo mesh` resamples it to cubic voxels first.

    Says what the command does, which changes the cost: the mask is resampled to cubic voxels of
    the shortest spacing by shape-based interpolation (delaunay-watershed-3d's
    `resample_to_isotropic`), so the reconstruction handles the anisotropy times as many voxels and
    `--min-distance` counts cubic voxels. Emitted whenever the voxels are not cubic. Above an
    anisotropy of 3, where the accuracy was not measured, delaunay-watershed-3d's own
    `UnmeasuredAnisotropyWarning` is relayed as well.
    """


class CommandError(Exception):
    """An error the command reports as one line and an exit code, without a traceback.

    Attributes:
        code: the exit code (`EXIT_USAGE` 2, `EXIT_FILE` 3 or `EXIT_REFUSED` 4).
        message: the line printed after `foambryo <sub-command>: error: `.
    """

    def __init__(self, code: int, message: str) -> None:
        """Hold the exit code and the message."""
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------------------------
# options (the methods' spellings, `option_name`, are `foambryo.results`')
# ---------------------------------------------------------------------------------------------
def _library_default(function: Callable[..., Any], parameter: str) -> Any:  # noqa: ANN401 - whatever the library's default is
    """The default value of `parameter` in the library's `function`: the command's only source of defaults."""
    return inspect.signature(function).parameters[parameter].default


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        message = f"must be a finite number, not {text!r}"
        raise argparse.ArgumentTypeError(message)
    return value


def _positive_integer(text: str) -> int:
    value = int(text)
    if value < 1:
        message = f"must be an integer of at least 1, not {text!r}"
        raise argparse.ArgumentTypeError(message)
    return value


def _positive_finite_float(text: str) -> float:
    value = _finite_float(text)
    if value <= 0:
        message = f"must be greater than 0, not {text!r}"
        raise argparse.ArgumentTypeError(message)
    return value


def _unit(text: str) -> str:
    if not _UNIT_PATTERN.fullmatch(text):
        message = f"must be one ASCII word (letters, digits, _ . - / ^), not {text!r}"
        raise argparse.ArgumentTypeError(message)
    return text


class _StoreAndNoteGiven(argparse.Action):
    """Store an option's value, and note that it was given: its value alone cannot say, once it has a default."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002 - argparse's signature
        namespace: argparse.Namespace,
        values: Any,  # noqa: ANN401 - whatever the option's type gives
        option_string: str | None = None,  # noqa: ARG002 - argparse's signature
    ) -> None:
        setattr(namespace, self.dest, values)
        namespace.given_options = {*getattr(namespace, "given_options", ()), self.option_strings[-1]}


def _versions() -> dict[str, str]:
    return library_versions()


#: `--media`'s help, shared by `infer` and `view`.
_MEDIA_HELP = (
    "the labels of the mesh that are media, not cells, besides the exterior 0 (an enclosed lumen, say): solved like "
    "cells, but reported (region_kind in pressures.csv, media in run.json) and drawn as media; added to the media "
    "the .rec file records"
)


def build_parser() -> argparse.ArgumentParser:
    """The command's argument parser, `foambryo {mesh,infer,infer-over-seeds,view}`; its help text is the interface.

    Every default is read from the library function the option is passed to (delaunay-watershed-3d's
    reconstruction, `infer_forces`, `infer_tensions_over_seeds`) when the parser is built.

    Returns:
        argparse.ArgumentParser: the parser, with one sub-parser per sub-command.
    """
    from dw3d import get_default_mesh_reconstruction_algorithm

    from foambryo.force_inference import infer_forces

    versions = _versions()
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Infer relative surface tensions and cell pressures from a 3D segmentation mask: "
            "'foambryo mesh' reconstructs the mask's multimaterial mesh, 'foambryo infer' infers the "
            "tensions and pressures, 'foambryo infer-over-seeds' combines the tensions of several "
            "reconstructions of the mask, 'foambryo view' shows them. Every default is the library's."
        ),
        epilog="Run 'foambryo <sub-command> --help' for the options of each sub-command.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"foambryo {versions['foambryo']} (delaunay-watershed-3d {versions['delaunay-watershed-3d']})",
    )
    commands = parser.add_subparsers(dest="command", metavar="{mesh,infer,infer-over-seeds,view}")
    commands.required = True

    default_min_distance = _library_default(get_default_mesh_reconstruction_algorithm, "min_distance")
    mesh = commands.add_parser(
        "mesh",
        help="reconstruct the multimaterial mesh of a segmentation mask",
        description=(
            "Reconstruct the multimaterial mesh of a segmentation mask with delaunay-watershed-3d's "
            "default reconstruction, in physical units, and write it as a .rec file whose metadata "
            "appendix records the spacing. A mask whose voxels are longer along one axis (a z step "
            "larger than the pixel size) is first resampled to cubic voxels of the shortest spacing "
            "by shape-based interpolation, which was measured to be more accurate than meshing the "
            "native voxels. The mask is a 3-D array of non-negative integer labels (0 is the medium), "
            "axis 0 being z, axis 1 y and axis 2 x. Every face of the image box is a crop: a cell "
            "that reaches it is a cut cell."
        ),
    )
    mesh.add_argument("mask", metavar="MASK", help="the mask: .npy, .npz (array 'mask', optionally "
                      "'spacing' and 'spacing_unit'), or .tif/.tiff (needs foambryo[cli])")
    mesh.add_argument("-o", "--output", required=True, metavar="MESH.rec", help="the mesh file to write")
    mesh.add_argument(
        "--spacing", nargs=3, type=_positive_finite_float, metavar=("DZ", "DY", "DX"),
        help="voxel size along axes 0, 1, 2 (z, y, x); one axis may be longer than the other two, which must "
             "be equal. Required unless the .npz records it: there is no default, because a wrong spacing "
             "changes every angle silently. --spacing 1 1 1 keeps voxel units",
    )
    mesh.add_argument("--spacing-unit", type=_unit, metavar="UNIT",
                      help="unit of the spacing, recorded in the mesh file (default: the .npz's, else um)")
    mesh.add_argument(
        "--min-distance", type=_positive_integer, default=default_min_distance, metavar="N",
        help="minimum distance between the reconstruction's points, in cubic voxels of the shortest spacing "
             "(default: %(default)s, the library's)",
    )
    mesh.add_argument("--overwrite", action="store_true", help="replace MESH.rec if it exists")

    infer = commands.add_parser(
        "infer",
        help="infer the tensions and pressures of a mesh",
        description=(
            "Infer the tensions, then the pressures, of a multimaterial mesh exactly as foambryo's "
            "infer_forces does with the same options (the tensions only with --no-pressures), and write "
            "tensions.csv, pressures.csv and a run record, run.json, in OUTDIR (and with --diagnostics the "
            "solves as a table, solves.csv, and with spectrum their singular values, spectra.npz). Numbers are "
            "written exactly (they read back as the same 64-bit floats); undetermined ones are nan."
        ),
    )
    infer.add_argument("mesh", metavar="MESH", help="the mesh: .rec/.arec, or a file meshio reads with label1/label2 "
                       "cell data (.vtk, .vtu, ...)")
    infer.add_argument("-o", "--output", required=True, metavar="OUTDIR",
                       help="the output directory (created if missing)")
    infer.add_argument(
        "--tension-method", choices=list(TENSION_METHODS), metavar="METHOD",
        default=option_name(_library_default(infer_forces, "mode_tension")),
        help="tension method, one of: " + ", ".join(TENSION_METHODS) + " (default: %(default)s)",
    )
    infer.add_argument(
        "--pressure-method", choices=list(PRESSURE_METHODS), metavar="METHOD", action=_StoreAndNoteGiven,
        default=option_name(_library_default(infer_forces, "mode_pressure")),
        help="pressure method, one of: " + ", ".join(PRESSURE_METHODS) + " (default: %(default)s)",
    )
    infer.add_argument("--no-pressures", action="store_true",
                       help="infer the tensions only (infer_tensions, with the same options): no pressures.csv, and "
                            "run.json says that the pressures were not computed")
    infer.add_argument("--mean-tension", type=_finite_float,
                       default=float(_library_default(infer_forces, "mean_tension")),
                       metavar="T", help="the mean the tensions are gauged to (default: %(default)s)")
    infer.add_argument("--base-pressure", type=_finite_float, action=_StoreAndNoteGiven,
                       default=float(_library_default(infer_forces, "base_pressure")), metavar="P",
                       help="the pressure of the medium, label 0 (default: %(default)s)")
    infer.add_argument("--solver", choices=["auto", "dense", "sparse"],
                       default=_library_default(infer_forces, "solver"),
                       help="how the linear systems are solved; auto is dense below 2 000 unknowns and sparse "
                            "above (default: %(default)s)")
    infer.add_argument(
        "--diagnostics", choices=list(DIAGNOSTICS),
        default=next(name for name, value in DIAGNOSTICS.items()
                     if value is _library_default(infer_forces, "diagnostics")),
        help="record every linear solve in run.json and in solves.csv (basic), and also the singular values in "
             "spectra.npz (spectrum) (default: %(default)s)",
    )
    infer.add_argument("--media", type=int, nargs="+", metavar="LABEL", help=_MEDIA_HELP)
    infer.add_argument("--overwrite", action="store_true",
                       help="replace the outputs of an earlier run in OUTDIR (only " + ", ".join(tables.INFER_OUTPUTS)
                            + " are deleted)")

    _add_infer_over_seeds_parser(commands, default_min_distance)

    _add_view_parser(commands)
    return parser


def _add_view_parser(commands: Any) -> None:  # noqa: ANN401 - argparse's subparsers
    """`foambryo view`: its options; the inference options are `infer`'s, their defaults read off `infer_forces`."""
    from dw3d import get_default_mesh_reconstruction_algorithm

    from foambryo._viewer_defaults import DEFAULT_SCATTER_COEFF
    from foambryo.force_inference import infer_forces

    view = commands.add_parser(
        "view",
        help="view a mesh's tensions and pressures, and write their colour bars; or open a mask and reconstruct it",
        description=(
            "Show a mesh's tensions and pressures in foambryo's Polyscope viewer (needs foambryo[viewing] "
            "and a display), with their colour bars and a length scale bar in the window, and write the colour "
            "bars of its surface colours, or the view with its bars, to a file. Without --results, the "
            "tensions and pressures are inferred exactly as 'foambryo infer' does with the same options. Given a "
            "mask (.npy, .npz, .tif, .tiff) with its voxel size, the window shows the mask, reconstructs it as "
            "'foambryo mesh' does and infers as 'foambryo infer' or 'foambryo infer-over-seeds' does, each in a "
            "child process that the window follows and can cancel."
        ),
    )
    view.add_argument("mesh", metavar="MESH|MASK", help="the mesh, as for 'foambryo infer'; or a mask, as for "
                      "'foambryo mesh', to reconstruct in the window")
    view.add_argument("--results", metavar="OUTDIR", help="read the tensions and pressures from what "
                      "'foambryo infer' wrote in OUTDIR (refused if its run.json names another mesh), or the "
                      "tensions 'foambryo infer-over-seeds' wrote there (matched to MESH's interfaces by label pair)")
    view.add_argument("--media", type=int, nargs="+", metavar="LABEL", help=_MEDIA_HELP)
    view.add_argument("--colour-bars", metavar="FILE", help="write the colour bars to FILE (.png, .pdf, .svg, ...); "
                      "needs matplotlib")
    view.add_argument("--figure", metavar="FILE", help="write the window's first view with its colour bars and scale "
                      "bar to FILE (.png, .pdf, .svg, ...); needs Polyscope and a display")
    view.add_argument("--no-show", action="store_true",
                      help="do not open the viewer: only write the colour bars (no display or Polyscope needed), "
                           "or the figure (the window closes once it is written)")
    scale = view.add_mutually_exclusive_group()
    scale.add_argument("--scale-bar", type=_positive_finite_float, metavar="LENGTH",
                       help="the length scale bar's length, in the mesh's unit (default: 1, 2 or 5 times a power "
                            "of ten, at most a quarter of the tissue's largest side; drawn when the mesh file records "
                            "its unit)")
    scale.add_argument("--no-scale-bar", action="store_true", help="draw no length scale bar")
    view.add_argument("--scattered", nargs="?", type=_positive_finite_float, const=DEFAULT_SCATTER_COEFF,
                      metavar="COEFF", help="explode the view: each cell moved away from the tissue's centre by COEFF "
                      "x (its centroid - the tissue's centroid) (COEFF default: %(const)s, the viewer's); the panel "
                      "changes it")
    mask = view.add_argument_group("mask options", "as for 'foambryo mesh', when MASK is a mask")
    mask.add_argument("--spacing", nargs=3, type=_positive_finite_float, metavar=("DZ", "DY", "DX"),
                      help="voxel size along axes 0, 1, 2 (z, y, x); required unless the .npz records it")
    mask.add_argument("--spacing-unit", type=_unit, metavar="UNIT",
                      help="unit of the spacing (default: the .npz's, else um)")
    mask.add_argument("--min-distance", type=_positive_integer, metavar="N",
                      help="minimum distance between the reconstruction's points, in cubic voxels (default: "
                           f"{_library_default(get_default_mesh_reconstruction_algorithm, 'min_distance')}, the "
                           "library's); the panel changes it")
    inference = view.add_argument_group(
        "inference options", "as for 'foambryo infer', used when the viewer infers (not with --results)")
    inference.add_argument(
        "--tension-method", choices=list(TENSION_METHODS), metavar="METHOD",
        help="tension method, one of: " + ", ".join(TENSION_METHODS)
             + f" (default: {option_name(_library_default(infer_forces, 'mode_tension'))})",
    )
    inference.add_argument(
        "--pressure-method", choices=list(PRESSURE_METHODS), metavar="METHOD",
        help="pressure method, one of: " + ", ".join(PRESSURE_METHODS)
             + f" (default: {option_name(_library_default(infer_forces, 'mode_pressure'))})",
    )
    inference.add_argument("--mean-tension", type=_finite_float, metavar="T",
                           help="the mean the tensions are gauged to "
                                f"(default: {float(_library_default(infer_forces, 'mean_tension'))})")
    inference.add_argument("--base-pressure", type=_finite_float, metavar="P",
                           help="the pressure of the medium, label 0 "
                                f"(default: {float(_library_default(infer_forces, 'base_pressure'))})")
    inference.add_argument("--solver", choices=["auto", "dense", "sparse"],
                           help="how the linear systems are solved "
                                f"(default: {_library_default(infer_forces, 'solver')})")
    inference.add_argument("--no-pressures", action="store_true",
                           help="infer the tensions only, as 'foambryo infer --no-pressures'")


def _add_infer_over_seeds_parser(commands: Any, default_min_distance: int) -> None:  # noqa: ANN401 - argparse's subparsers
    """`foambryo infer-over-seeds`: its options, every default read off `infer_tensions_over_seeds`."""
    from foambryo.seed_ensembles import COMBINE_OPTIONS, DEFAULT_ENSEMBLE_SEEDS, infer_tensions_over_seeds

    command = commands.add_parser(
        "infer-over-seeds",
        help="infer the tensions of several reconstructions of a mask, and combine them",
        description=(
            "Reconstruct a segmentation mask at the dither seeds 0..N-1 (delaunay-watershed-3d's default "
            "reconstruction, resampled to cubic voxels as 'foambryo mesh' does), infer the tensions of each mesh, "
            "and combine them per interface exactly as foambryo's infer_tensions_over_seeds does. Writes "
            "tensions.csv (the ensemble tension, its standard deviation over the seeds, how many seeds found the "
            "interface, and a flag for those found by fewer than half of the seeds: mostly interfaces that do not "
            "exist), tensions_per_seed.csv (every seed's own tensions) and run.json in OUTDIR; no pressures, since no "
            "single mesh carries them. The seed spread is a "
            "lower bound on the error, not an error bar. Fix N in advance: never choose seeds by their result."
        ),
    )
    command.add_argument("mask", metavar="MASK", help="the mask, as for 'foambryo mesh'")
    command.add_argument("-o", "--output", required=True, metavar="OUTDIR",
                         help="the output directory (created if missing)")
    command.add_argument(
        "--spacing", nargs=3, type=_positive_finite_float, metavar=("DZ", "DY", "DX"),
        help="voxel size along axes 0, 1, 2, as for 'foambryo mesh' (required unless the .npz records it)",
    )
    command.add_argument("--spacing-unit", type=_unit, metavar="UNIT",
                         help="unit of the spacing, recorded in run.json (default: the .npz's, else um)")
    command.add_argument(
        "--min-distance", type=_positive_integer, default=default_min_distance, metavar="N",
        help="minimum distance between the reconstruction's points, in cubic voxels of the shortest spacing "
             "(default: %(default)s, the library's)",
    )
    command.add_argument(
        "--seeds", type=_positive_integer, default=len(DEFAULT_ENSEMBLE_SEEDS), metavar="N",
        help="reconstruct at the seeds 0..N-1 (default: %(default)s, the library's seed list)",
    )
    command.add_argument(
        "--combine", choices=list(COMBINE_OPTIONS), default=_library_default(infer_tensions_over_seeds, "combine"),
        help="average the seeds' tensions, or (experimental, young-dupre only) their contact angles and infer once "
             "(default: %(default)s)",
    )
    command.add_argument(
        "--angle-reading", choices=list(ANGLE_READINGS),
        default=_library_default(infer_tensions_over_seeds, "angle_reading"),
        help="how the angle-reading methods read the trijunction angles (default: %(default)s)",
    )
    command.add_argument(
        "--tension-method", choices=list(TENSION_METHODS), metavar="METHOD",
        default=option_name(_library_default(infer_tensions_over_seeds, "mode")),
        help="tension method, as for 'foambryo infer' (default: %(default)s)",
    )
    command.add_argument(
        "--max-workers", type=_positive_integer, default=_library_default(infer_tensions_over_seeds, "max_workers"),
        metavar="K", help="seeds reconstructed at once, in separate processes, each holding one reconstruction in "
                          "memory; the result does not depend on it (default: %(default)s)",
    )
    command.add_argument("--overwrite", action="store_true",
                         help="replace the outputs of an earlier run in OUTDIR (only "
                              + ", ".join(tables.ENSEMBLE_OUTPUTS) + " are deleted)")


# ---------------------------------------------------------------------------------------------
# warnings: relayed once each, and recorded
# ---------------------------------------------------------------------------------------------
class _WarningRelay(WarningCollector):
    """Catch the warnings raised inside a block (`foambryo.results.WarningCollector`), then print each once.

    Each distinct warning is printed once on standard error, with its count.
    """

    def __init__(self, program: str) -> None:
        super().__init__()
        self.program = program

    def __exit__(self, *exc_info: object) -> None:
        super().__exit__(*exc_info)
        for entry in self.summary:
            times = f" ({entry['count']} times)" if entry["count"] > 1 else ""
            print(f"{self.program}: warning: {entry['category']}: {entry['message']}{times}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# foambryo mesh
# ---------------------------------------------------------------------------------------------
def _require_file(path: Path, given: str) -> None:
    if not path.is_file():
        raise CommandError(EXIT_FILE, f"no such file: {given!r}")


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float] | None, str | None]:
    """Read a mask file (`foambryo.segmentation_masks.read_segmentation_mask`): labels, recorded spacing and unit.

    Args:
        path: a `.npy`, `.npz` or `.tif`/`.tiff` file.

    Returns:
        tuple: the mask (3-D non-negative integer labels, axis 0 = z), the spacing `(dz, dy, dx)` an `.npz`
        records or `None`, and the unit it records or `None`.

    Raises:
        CommandError: (exit 3) on an unreadable file, an unsupported suffix, a missing optional
            reader, or an array that is not 3-D non-negative integer labels.
    """
    try:
        return read_segmentation_mask(path)
    except SegmentationMaskError as error:
        raise CommandError(EXIT_FILE, str(error)) from error


def _resolve_spacing(
    args: argparse.Namespace,
    mask_path: Path,
    recorded_spacing: tuple[float, float, float] | None,
    recorded_unit: str | None,
) -> tuple[tuple[float, float, float], str]:
    """The spacing and unit to use: the command line's, else the file's; never a default spacing."""
    spacing = tuple(args.spacing) if args.spacing is not None else recorded_spacing
    if spacing is None:
        message = (f"{mask_path.name} records no voxel spacing: give it with --spacing DZ DY DX "
                   "(for example --spacing 1 1 1 to keep voxel units)")
        raise CommandError(EXIT_USAGE, message)
    if args.spacing is not None and recorded_spacing is not None and tuple(args.spacing) != recorded_spacing:
        warnings.warn(f"--spacing {' '.join(f'{v:g}' for v in args.spacing)} replaces the spacing "
                      f"{mask_path.name} records, {' '.join(f'{v:g}' for v in recorded_spacing)}", UserWarning,
                      stacklevel=2)
    unit = args.spacing_unit or recorded_unit or "um"
    if args.spacing_unit and recorded_unit and args.spacing_unit != recorded_unit:
        warnings.warn(f"--spacing-unit {args.spacing_unit} replaces the unit {mask_path.name} records, "
                      f"{recorded_unit}", UserWarning, stacklevel=2)
    return spacing, unit


def _check_spacing(spacing: tuple[float, float, float], unit: str, from_file: str | None) -> None:
    """Refuse a spacing the resampling cannot handle; say what is done with an anisotropic one.

    Raises:
        CommandError: (exit 2, or 3 when the spacing comes from the mask file) when the two
            shorter spacings differ: only one axis may be longer than the other two.
    """
    from dw3d.mask_preprocessing import coarse_axis_and_factor, isotropic_spacing

    try:
        found = coarse_axis_and_factor(spacing)
    except ValueError as error:
        if from_file is None:
            raise CommandError(EXIT_USAGE, f"--spacing: {error}") from error
        raise CommandError(EXIT_FILE, f"{from_file}: {error}") from error
    if found is not None:
        axis, factor = found
        warnings.warn(
            f"the voxels are {factor:.3g} times longer along axis {axis} ({'zyx'[axis]}) than along the other two: "
            f"the mask is resampled to cubic voxels of {isotropic_spacing(spacing):g} {unit} by shape-based "
            f"interpolation before meshing ({factor:.3g} times the voxels; --min-distance counts cubic voxels)",
            AnisotropicVoxelSpacingWarning, stacklevel=3,
        )


def _run_mesh(args: argparse.Namespace) -> int:
    """`foambryo mesh MASK -o MESH.rec [--spacing DZ DY DX] [--spacing-unit UNIT] [--min-distance N] [--overwrite]`.

    Reads and checks the mask, resolves the spacing (the command line's, else the `.npz`'s; none is exit 2),
    prints the mask report, reconstructs (`foambryo.io.reconstruct_mask`) and writes the `.rec` file. Exit 2 for
    an output not ending in `.rec`, 3 for a missing or unusable file or an existing output without `--overwrite`,
    4 when the reconstruction refuses the mask.
    """
    from dw3d.reconstruction_algorithm import MeshReconstructionAlgorithm

    mask_path, output = Path(args.mask), Path(args.output)
    _require_file(mask_path, args.mask)
    if output.suffix != ".rec":
        raise CommandError(EXIT_USAGE, f"the mesh is written as .rec: -o must end in .rec, not {output.name!r}")
    if output.exists() and not args.overwrite:
        raise CommandError(EXIT_FILE, f"{args.output} exists: pass --overwrite to replace it")
    if not hasattr(MeshReconstructionAlgorithm, "construct_mesh_in_physical_units"):
        raise CommandError(EXIT_FILE, "foambryo mesh needs delaunay-watershed-3d 0.7 or later (installed: "
                                      f"{_versions()['delaunay-watershed-3d']}): pip install -U delaunay-watershed-3d")
    mask, recorded_spacing, recorded_unit = read_mask(mask_path)
    with _WarningRelay(f"{PROGRAM} mesh"):
        spacing, unit = _resolve_spacing(args, mask_path, recorded_spacing, recorded_unit)
        _check_spacing(spacing, unit, None if args.spacing is not None else mask_path.name)
        # What the reconstruction will meet, before the long step: a report only, the mask is not changed.
        print(segmentation_mask_report(mask).summary(), flush=True)
        try:
            reconstruction = reconstruct_mask(mask, spacing, spacing_unit=unit, min_distance=args.min_distance)
        except ValueError as error:
            raise CommandError(EXIT_REFUSED, f"the reconstruction refused {mask_path.name}: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    reconstruction.write_rec(output)
    print(f"wrote {args.output}: {reconstruction.summary()}")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# foambryo infer
# ---------------------------------------------------------------------------------------------
def _read_mesh(path: Path, media: Sequence[int] | None = None) -> DcelData:
    """The mesh of `path` (`dcel_mesh_from_file`), with `media` added to the media the file records (`--media`).

    Exit 3 for an unreadable file; exit 2 for a `--media` label that no triangle of the mesh carries.
    """
    try:
        mesh = dcel_mesh_from_file(path)
    except Exception as error:  # every reader error is the same failure for the user
        raise CommandError(EXIT_FILE, f"cannot read the mesh {path.name}: {type(error).__name__}: {error}") from error
    if media:
        try:
            mesh.set_media([*mesh.media, *media])
        except ValueError as error:
            raise CommandError(EXIT_USAGE, f"--media: {error} ({path.name})") from error
    return mesh


def _refuse_unused_pressure_options(args: argparse.Namespace) -> None:
    """A pressure option typed with `--no-pressures` would not be used: refuse it (exit 2) rather than ignore it."""
    given = [name for name in ("--pressure-method", "--base-pressure") if name in getattr(args, "given_options", ())]
    if args.no_pressures and given:
        raise CommandError(EXIT_USAGE, f"{given[0]} applies to the pressures: --no-pressures infers the tensions only")


def _infer_record_options(args: argparse.Namespace) -> dict[str, Any]:
    """`run.json`'s `options` of `foambryo infer`: every option, defaults included.

    With `--no-pressures`, the pressure options, which were not used, are `None`, and `no_pressures` is `True`.
    """
    options: dict[str, Any] = {
        "tension_method": args.tension_method,
        "pressure_method": args.pressure_method,
        "mean_tension": args.mean_tension,
        "base_pressure": args.base_pressure,
        "solver": args.solver,
        "diagnostics": args.diagnostics,
        "media": sorted(set(args.media)) if args.media else None,
    }
    if args.no_pressures:
        options.update(pressure_method=None, base_pressure=None, no_pressures=True)
    return options


def _run_infer(args: argparse.Namespace, arguments: Sequence[str]) -> int:
    """`foambryo infer MESH -o OUTDIR [...]`: `infer_tensions_then_pressures`, then the tables and `run.json`.

    Writes `tensions.csv`, `pressures.csv` (not with `--no-pressures`, nor when the pressures are refused),
    `solves.csv` (with `--diagnostics basic` or `spectrum`), `spectra.npz` (with `spectrum`) and `run.json`.
    Exit 2 for a pressure option with `--no-pressures`, 3 for a missing or unreadable mesh or earlier outputs
    without `--overwrite`, 4 when the library refuses the tensions or the pressures (after writing the files).
    """
    mesh_path, outdir = Path(args.mesh), Path(args.output)
    _refuse_unused_pressure_options(args)
    _require_file(mesh_path, args.mesh)
    if outdir.exists() and not outdir.is_dir():
        raise CommandError(EXIT_FILE, f"{args.output} exists and is not a directory")
    existing = [name for name in tables.INFER_OUTPUTS if (outdir / name).exists()]
    if existing and not args.overwrite:
        raise CommandError(EXIT_FILE, f"{args.output} already holds {', '.join(existing)}: pass --overwrite to "
                                      "replace them")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    clock = time.perf_counter()
    sha256 = tables.file_sha256(mesh_path)
    appendix = tables.read_rec_metadata(mesh_path)
    relay = _WarningRelay(f"{PROGRAM} infer")
    with relay:
        mesh = _read_mesh(mesh_path, args.media)
        outcome = infer_tensions_then_pressures(
            mesh, args.mean_tension, args.base_pressure, TENSION_METHODS[args.tension_method],
            PRESSURE_METHODS[args.pressure_method], solver=args.solver, diagnostics=DIAGNOSTICS[args.diagnostics],
            with_pressures=not args.no_pressures,
        )
    tensions, pressures, status, error = outcome.tensions, outcome.pressures, outcome.status, outcome.error

    outdir.mkdir(parents=True, exist_ok=True)
    for name in existing:
        (outdir / name).unlink()
    outputs = []
    if tensions is not None:
        tables.write_tension_table(outdir / tables.TENSION_TABLE, tensions)
        outputs.append(tables.TENSION_TABLE)
    if pressures is not None:
        tables.write_pressure_table(outdir / tables.PRESSURE_TABLE, pressures, media=mesh.media)
        outputs.append(tables.PRESSURE_TABLE)
    if args.diagnostics != "none":
        tables.write_solve_table(outdir / tables.SOLVE_TABLE, outcome.records)
        outputs.append(tables.SOLVE_TABLE)
    if args.diagnostics == "spectrum":
        tables.write_spectra(outdir / tables.SPECTRA, [record.singular_values for record in outcome.records])
        outputs.append(tables.SPECTRA)
    outputs.append(tables.RUN_RECORD)
    record = tables.infer_run_record(
        command=f"{PROGRAM} infer",
        arguments=arguments,
        date=started,
        elapsed_seconds=round(time.perf_counter() - clock, 3),
        input_record={"path": args.mesh, "sha256": sha256, "bytes": mesh_path.stat().st_size, "metadata": appendix},
        options=_infer_record_options(args),
        mesh=mesh,
        outcome=outcome,
        units=tables.infer_units(args.mean_tension, appendix),
        warnings_summary=relay.summary,
        outputs=outputs,
        versions=_versions(),
    )
    tables.write_run_record(outdir / tables.RUN_RECORD, record)
    for name in outputs:
        print(f"wrote {outdir / name}")
    if status != "complete":
        written = "the tensions were written" if tensions is not None else "nothing but run.json was written"
        raise CommandError(EXIT_REFUSED, f"{status}: {error} ({written})")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# foambryo infer-over-seeds
# ---------------------------------------------------------------------------------------------
def _run_infer_over_seeds(args: argparse.Namespace, arguments: Sequence[str]) -> int:
    """`foambryo infer-over-seeds MASK -o OUTDIR [...]`: `infer_tensions_over_seeds` over the seeds 0..N-1.

    Writes `tensions.csv` (with the seed spread, the seeds that found each interface and the flag),
    `tensions_per_seed.csv` and `run.json`; no pressures. Exit 2 for a missing spacing, 3 for a missing or unusable
    file or earlier outputs without `--overwrite`, 4 when the library refuses (only `run.json` is written).
    """
    from dw3d.reconstruction_algorithm import MeshReconstructionAlgorithm

    from foambryo.seed_ensembles import infer_tensions_over_seeds

    mask_path, outdir = Path(args.mask), Path(args.output)
    _require_file(mask_path, args.mask)
    if outdir.exists() and not outdir.is_dir():
        raise CommandError(EXIT_FILE, f"{args.output} exists and is not a directory")
    existing = [name for name in tables.ENSEMBLE_OUTPUTS if (outdir / name).exists()]
    if existing and not args.overwrite:
        raise CommandError(EXIT_FILE, f"{args.output} already holds {', '.join(existing)}: pass --overwrite to "
                                      "replace them")
    if not hasattr(MeshReconstructionAlgorithm, "construct_mesh_in_physical_units"):
        raise CommandError(EXIT_FILE, "foambryo infer-over-seeds needs delaunay-watershed-3d 0.7 or later (installed: "
                                      f"{_versions()['delaunay-watershed-3d']}): pip install -U delaunay-watershed-3d")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    clock = time.perf_counter()
    sha256 = tables.file_sha256(mask_path)
    mask, recorded_spacing, recorded_unit = read_mask(mask_path)
    seeds = list(range(args.seeds))
    result, status, error = None, "complete", None
    relay = _WarningRelay(f"{PROGRAM} infer-over-seeds")
    with relay:
        spacing, unit = _resolve_spacing(args, mask_path, recorded_spacing, recorded_unit)
        _check_spacing(spacing, unit, None if args.spacing is not None else mask_path.name)
        try:
            result = infer_tensions_over_seeds(
                mask, seeds, min_distance=args.min_distance, mode=TENSION_METHODS[args.tension_method],
                angle_reading=args.angle_reading, combine=args.combine, max_workers=args.max_workers, spacing=spacing,
            )
        except ValueError as refused:
            status, error = "failed", str(refused)

    outdir.mkdir(parents=True, exist_ok=True)
    for name in existing:
        (outdir / name).unlink()
    outputs = []
    if result is not None:
        tables.write_ensemble_tension_table(outdir / tables.TENSION_TABLE, result)
        tables.write_per_seed_tension_table(outdir / tables.PER_SEED_TENSION_TABLE, result.per_seed_tensions)
        outputs += [tables.TENSION_TABLE, tables.PER_SEED_TENSION_TABLE]
    outputs.append(tables.RUN_RECORD)
    record = tables.ensemble_run_record(
        command=f"{PROGRAM} infer-over-seeds",
        arguments=arguments,
        date=started,
        elapsed_seconds=round(time.perf_counter() - clock, 3),
        input_record={"path": args.mask, "sha256": sha256, "bytes": mask_path.stat().st_size,
                      "shape": list(mask.shape), "spacing": list(spacing), "spacing_unit": unit,
                      "spacing_from": "command line" if args.spacing is not None else mask_path.name},
        options={
            "seeds": args.seeds,
            "combine": args.combine,
            "angle_reading": args.angle_reading,
            "tension_method": args.tension_method,
            "min_distance": args.min_distance,
            "max_workers": args.max_workers,
        },
        seeds=seeds,
        library_call=tables.ensemble_library_call(
            n_seeds=args.seeds, min_distance=args.min_distance, mode=TENSION_METHODS[args.tension_method],
            angle_reading=args.angle_reading, combine=args.combine, max_workers=args.max_workers, spacing=spacing,
        ),
        result=result,
        length_unit=unit,
        status=status,
        error=error,
        warnings_summary=relay.summary,
        outputs=outputs,
        versions=_versions(),
    )
    tables.write_run_record(outdir / tables.RUN_RECORD, record)
    for name in outputs:
        print(f"wrote {outdir / name}")
    if status != "complete":
        raise CommandError(EXIT_REFUSED, f"{status}: {error} (nothing but run.json was written)")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# foambryo view
# ---------------------------------------------------------------------------------------------
def _read_results(outdir: Path, mesh_path: Path) -> tuple[dict, dict | None]:
    tension_path = outdir / tables.TENSION_TABLE
    pressure_path = outdir / tables.PRESSURE_TABLE
    record_path = outdir / tables.RUN_RECORD
    _require_file(tension_path, str(tension_path))
    if record_path.is_file():
        try:
            recorded = tables.read_run_record(record_path)["input"]["sha256"]
        except (ValueError, TypeError, KeyError) as error:
            raise CommandError(EXIT_FILE, f"cannot read {record_path}: {error}") from error
        actual = tables.file_sha256(mesh_path)
        if recorded != actual:
            raise CommandError(EXIT_FILE, f"{record_path} was computed from another mesh (SHA-256 {recorded[:12]}...; "
                                          f"{mesh_path.name} is {actual[:12]}...)")
    try:
        tensions = tables.read_tension_table(tension_path)
        pressures = tables.read_pressure_table(pressure_path) if pressure_path.is_file() else None
    except ValueError as error:
        raise CommandError(EXIT_FILE, str(error)) from error
    return tensions, pressures


#: `foambryo view`'s inference options: `infer`'s, without `--diagnostics` (the window's Export has its own
#: diagnostics choice).
_VIEW_INFERENCE_OPTIONS = ("tension_method", "pressure_method", "mean_tension", "base_pressure", "solver")


def _view_inference_args(args: argparse.Namespace) -> argparse.Namespace:
    """The view's inference options as `infer` takes them: those given, `infer`'s (the library's) defaults otherwise."""
    from foambryo.force_inference import infer_forces

    defaults = {
        "tension_method": option_name(_library_default(infer_forces, "mode_tension")),
        "pressure_method": option_name(_library_default(infer_forces, "mode_pressure")),
        "mean_tension": float(_library_default(infer_forces, "mean_tension")),
        "base_pressure": float(_library_default(infer_forces, "base_pressure")),
        "solver": _library_default(infer_forces, "solver"),
    }
    given = {name: getattr(args, name) for name in defaults}
    values = {name: default if given[name] is None else given[name] for name, default in defaults.items()}
    return argparse.Namespace(**values, diagnostics="none")


class _ViewValues:
    """What `foambryo view` shows: tensions on the mesh's interfaces, pressures or `None`, and where they come from.

    For a seed ensemble, `tensions` holds the ensemble tensions of the mesh's interfaces only, and `ensemble` the
    ensemble's maps as read, for the viewer.
    """

    def __init__(
        self,
        tensions: dict | None,
        pressures: dict | None,
        values_from: str,
        ensemble: Any = None,  # noqa: ANN401 - a namespace of the ensemble's maps
    ) -> None:
        self.tensions = tensions
        self.pressures = pressures
        self.values_from = values_from
        self.ensemble = ensemble


def _table_header(path: Path) -> str:
    with path.open(encoding="ascii", errors="replace") as file:
        return file.readline().strip()


def _folder_name(folder: Path) -> str:
    """A folder as a saved figure names it: its own name, never the path to it (a shared figure shows no local path)."""
    return folder.name or folder.resolve().name or str(folder)


def _read_view_results(outdir: Path, mesh_path: Path, mesh: DcelData) -> _ViewValues:
    """Read what `foambryo infer` or `foambryo infer-over-seeds` wrote in OUTDIR, for MESH.

    An `infer` run is refused when its `run.json` names another mesh. An ensemble's `run.json` names the mask, so
    the mesh cannot be checked against it: its interfaces are matched to the ensemble's by label pair, a warning
    says how many do not match, and none matching is refused.
    """
    from types import SimpleNamespace

    from foambryo._viewer_figures import ensemble_on_mesh

    tension_path = outdir / tables.TENSION_TABLE
    record_path = outdir / tables.RUN_RECORD
    _require_file(tension_path, str(tension_path))
    record: dict[str, Any] = {}
    if record_path.is_file():
        try:
            record = tables.read_run_record(record_path)
        except (ValueError, TypeError) as error:
            raise CommandError(EXIT_FILE, f"cannot read {record_path}: {error}") from error
    if _table_header(tension_path) != ",".join(tables.ENSEMBLE_TENSION_COLUMNS):
        tensions, pressures = _read_results(outdir, mesh_path)
        call = record.get("library_call")
        return _ViewValues(tensions, pressures,
                           f"read from {_folder_name(outdir)}" + (f" (foambryo infer: {call})" if call else ""))
    try:
        rows = tables.read_ensemble_tension_table(tension_path)
    except ValueError as error:
        raise CommandError(EXIT_FILE, str(error)) from error
    ensemble = SimpleNamespace(
        tensions={key: row[0] for key, row in rows.items()},
        tension_seed_sd={key: row[1] for key, row in rows.items()},
        found_by_fewer_than_half_the_seeds={key: row[3] for key, row in rows.items()},
    )
    matched = ensemble_on_mesh(mesh, ensemble.tensions, ensemble.tension_seed_sd,
                               ensemble.found_by_fewer_than_half_the_seeds)
    if not matched.tensions:
        raise CommandError(EXIT_FILE, f"no interface of {mesh_path.name} is in {tension_path}: the ensemble was "
                                      "computed from another mask")
    if matched.not_on_the_mesh or matched.missing_from_the_ensemble:
        warnings.warn(f"{matched.not_on_the_mesh} interfaces of the ensemble are not on {mesh_path.name} (not drawn); "
                      f"{matched.missing_from_the_ensemble} interfaces of {mesh_path.name} are not in the ensemble "
                      "(drawn undetermined)", UserWarning, stacklevel=2)
    call = record.get("library_call")
    values_from = f"tensions combined over seeds, read from {_folder_name(outdir)}" + (
        f" (foambryo infer-over-seeds: {call})" if call else "")
    return _ViewValues(matched.tensions, None, values_from, ensemble)


def _length_unit(appendix: dict[str, Any] | None) -> str | None:
    """The mesh's length unit, when its file records physical coordinates and their unit; else `None`."""
    if appendix and appendix.get("coordinate_frame") == "physical" and appendix.get("spacing_unit"):
        return str(appendix["spacing_unit"])
    return None


def _run_view(args: argparse.Namespace) -> int:
    """`foambryo view MESH|MASK [...]`: the colour bars (`--colour-bars`), the figure (`--figure`), the window.

    A mask (by its suffix) goes to `_run_view_mask`. With a mesh: the values are read (`--results`) or inferred
    as `foambryo infer` does; `--colour-bars` is written first (no display needed), then the window opens unless
    `--no-show` (with `--figure`, Polyscope renders the view once). Exit 2 for contradictory options, 3 for a
    missing or unusable file or a missing optional package, 4 when the library refuses the tensions or no tension
    is finite for the colour bars.
    """
    mesh_path = Path(args.mesh)
    if mesh_path.suffix.lower() in MASK_SUFFIXES:
        return _run_view_mask(args, mesh_path)
    given = [option for option, value in (("--spacing", args.spacing), ("--spacing-unit", args.spacing_unit),
                                          ("--min-distance", args.min_distance)) if value is not None]
    if given:
        raise CommandError(EXIT_USAGE, f"{given[0]} applies to a mask: {mesh_path.name} is a mesh")
    if args.no_pressures and (args.pressure_method is not None or args.base_pressure is not None):
        name = "--pressure-method" if args.pressure_method is not None else "--base-pressure"
        raise CommandError(EXIT_USAGE, f"{name} applies to the pressures: --no-pressures infers the tensions only")
    if args.no_pressures and args.results is not None:
        raise CommandError(EXIT_USAGE, "--no-pressures applies when 'foambryo view' infers: with --results the values "
                                       "are read from OUTDIR")
    if args.no_show and not args.colour_bars and not args.figure:
        raise CommandError(EXIT_USAGE, "--no-show needs --colour-bars: there would be nothing to do")
    given = [f"--{name.replace('_', '-')}" for name in _VIEW_INFERENCE_OPTIONS if getattr(args, name) is not None]
    if args.results is not None and given:
        raise CommandError(EXIT_USAGE, f"{given[0]} applies when 'foambryo view' infers: with --results the values "
                                       "are read from OUTDIR")
    _require_file(mesh_path, args.mesh)
    inference = _view_inference_args(args)
    with _WarningRelay(f"{PROGRAM} view"):
        mesh = _read_mesh(mesh_path, args.media)
        values = _view_values(args, mesh_path, mesh, inference)
    if args.colour_bars:
        _write_view_colour_bars(Path(args.colour_bars), mesh, values)
    if args.no_show and not args.figure:
        return EXIT_OK
    _open_the_viewer(args, mesh_path, mesh, values, inference)
    if args.figure:
        print(f"wrote {args.figure}")
    return EXIT_OK


def _view_values(
    args: argparse.Namespace,
    mesh_path: Path,
    mesh: DcelData,
    inference: argparse.Namespace,
) -> _ViewValues:
    """The values `foambryo view` shows: read with `--results`, else inferred as `foambryo infer` does."""
    if args.results is not None:
        return _read_view_results(Path(args.results), mesh_path, mesh)
    outcome = infer_tensions_then_pressures(
        mesh, inference.mean_tension, inference.base_pressure, TENSION_METHODS[inference.tension_method],
        PRESSURE_METHODS[inference.pressure_method], solver=inference.solver,
        with_pressures=not getattr(args, "no_pressures", False),
    )
    if outcome.status == "failed":
        raise CommandError(EXIT_REFUSED, f"failed: {outcome.error}")
    if outcome.status == "pressures refused":
        warnings.warn(f"the pressures are not shown: {outcome.error}", UserWarning, stacklevel=2)
    return _ViewValues(outcome.tensions, outcome.pressures, f"inferred by 'foambryo view': {outcome.library_call}")


def _write_view_colour_bars(path: Path, mesh: DcelData, values: _ViewValues) -> None:
    """`--colour-bars FILE`: `save_colour_bars` on what the viewer colours (for an ensemble, the mesh's interfaces)."""
    try:
        from foambryo.colour_scales import save_colour_bars
    except ImportError as error:
        raise CommandError(EXIT_FILE, f"the colour bars need matplotlib: pip install 'foambryo[viewing]' "
                                      f"({error})") from error
    try:
        ranges = save_colour_bars(mesh, values.tensions, values.pressures, path)
    except ValueError as error:
        raise CommandError(EXIT_REFUSED, f"no colour bar: {error}") from error
    pressure_text = "" if ranges.pressure is None else f", pressure 0 to {ranges.pressure[1]:.6g}"
    print(f"wrote {path}: tension {ranges.tension[0]:.6g} to {ranges.tension[1]:.6g}{pressure_text}")


def _open_the_viewer(
    args: argparse.Namespace,
    mesh_path: Path,
    mesh: DcelData,
    values: _ViewValues,
    inference: argparse.Namespace,
) -> None:
    """The Polyscope window (and `--figure`): the force viewer, or the tension viewer when there are no pressures."""
    try:
        from foambryo import viewing
    except ImportError as error:
        message = f"the viewer needs Polyscope: pip install 'foambryo[viewing]' ({error})"
        raise CommandError(EXIT_FILE, message) from error
    appendix = tables.read_rec_metadata(mesh_path)
    unit = _length_unit(appendix)
    if unit is None and args.scale_bar is not None:  # asked for: drawn in the mesh's coordinates, named as such
        unit = "voxel" if (appendix or {}).get("coordinate_frame") == "voxel_index" else "mesh units"
    common: dict[str, Any] = {
        "mean_tension": inference.mean_tension,
        "mode_tension": TENSION_METHODS[inference.tension_method],
        "solver": inference.solver,
        "length_unit": unit,
        "scale_bar": False if args.no_scale_bar else (True if args.scale_bar is None else args.scale_bar),
        "save_figure": args.figure,
        "show": not args.no_show,
        "values_from": values.values_from,
        "scattered": args.scattered is not None,
        "mesh_file": mesh_path.name,
    }
    if args.scattered is not None:
        common["scatter_coeff"] = args.scattered
    if values.ensemble is not None:
        viewing.plot_tension_inference(mesh, seed_ensemble=values.ensemble, **common)
    elif values.pressures is None:
        viewing.plot_tension_inference(mesh, values.tensions, **common)
    else:
        viewing.plot_force_inference(mesh, values.tensions, values.pressures, base_pressure=inference.base_pressure,
                                     mode_pressure=PRESSURE_METHODS[inference.pressure_method], **common)


def _run_view_mask(args: argparse.Namespace, mask_path: Path) -> int:
    """`foambryo view MASK --spacing ...`: the mask in the window, reconstructed and inferred there.

    The mask and its spacing go through `foambryo mesh`'s own checks (and its mask report is printed); the
    inference options are `infer`'s, resolved to the library's defaults; the window does the rest
    (`foambryo.viewing._view_mask_workflow`). What needs a mesh beforehand is refused.
    """
    from dw3d import get_default_mesh_reconstruction_algorithm

    for option, value in (("--results", args.results), ("--figure", args.figure), ("--colour-bars", args.colour_bars),
                          ("--media", args.media)):
        if value is not None:
            raise CommandError(EXIT_USAGE, f"{option} needs a mesh: {mask_path.name} is a mask, reconstructed in the "
                                           "window (run 'foambryo mesh' first to write it without a window)")
    if args.no_show:
        raise CommandError(EXIT_USAGE, f"--no-show: a mask is opened in the window to be reconstructed there "
                                       f"({mask_path.name})")
    if args.no_pressures and (args.pressure_method is not None or args.base_pressure is not None):
        name = "--pressure-method" if args.pressure_method is not None else "--base-pressure"
        raise CommandError(EXIT_USAGE, f"{name} applies to the pressures: --no-pressures infers the tensions only")
    _require_file(mask_path, args.mesh)
    mask, recorded_spacing, recorded_unit = read_mask(mask_path)
    with _WarningRelay(f"{PROGRAM} view"):
        spacing, unit = _resolve_spacing(args, mask_path, recorded_spacing, recorded_unit)
        _check_spacing(spacing, unit, None if args.spacing is not None else mask_path.name)
        print(segmentation_mask_report(mask).summary(), flush=True)
    try:
        from foambryo import viewing
        from foambryo._mask_workflow import MaskWorkflow, ViewerChoices
    except ImportError as error:
        message = f"the viewer needs Polyscope: pip install 'foambryo[viewing]' ({error})"
        raise CommandError(EXIT_FILE, message) from error
    inference = _view_inference_args(args)
    choices = ViewerChoices.library(
        tension_method=inference.tension_method, pressure_method=inference.pressure_method,
        mean_tension=inference.mean_tension, base_pressure=inference.base_pressure, solver=inference.solver,
        pressures=not args.no_pressures,
    )
    min_distance = (args.min_distance if args.min_distance is not None
                    else _library_default(get_default_mesh_reconstruction_algorithm, "min_distance"))
    workflow = MaskWorkflow(mask, tuple(spacing), unit, mask_path.name, int(min_distance), choices)
    scale_bar = False if args.no_scale_bar else (True if args.scale_bar is None else args.scale_bar)
    if args.scattered is not None:
        viewing._view_mask_workflow(workflow, scattered=True, scatter_coeff=args.scattered, scale_bar=scale_bar)
    else:
        viewing._view_mask_workflow(workflow, scale_bar=scale_bar)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Run the `foambryo` command (the console entry point).

    Args:
        argv: the arguments after the program name; `None` reads `sys.argv[1:]`.

    Returns:
        int: the exit code (0 success, also after `--help` and `--version`; 2 invalid command line,
        argparse's own errors included; 3 unusable file; 4 refused by the library). An unexpected error
        propagates, with its traceback (exit code 1 from Python).
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(arguments)
    except SystemExit as exit_request:  # --help, --version, or argparse's own usage error (2)
        return int(exit_request.code or 0)
    program = f"{PROGRAM} {args.command}"
    try:
        if args.command == "mesh":
            return _run_mesh(args)
        if args.command == "infer":
            return _run_infer(args, arguments)
        if args.command == "infer-over-seeds":
            return _run_infer_over_seeds(args, arguments)
        return _run_view(args)
    except CommandError as error:
        print(f"{program}: error: {error.message}", file=sys.stderr)
        return error.code


if __name__ == "__main__":
    sys.exit(main())
