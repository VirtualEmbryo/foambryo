"""The result files of the `foambryo` command, for every front end that writes or reads them.

`foambryo infer` and `foambryo infer-over-seeds` write CSV tables and a JSON run record, `run.json` (schema 1);
the napari plugin writes the same files. Both use this module, so a table written by either reads back as the
library's 64-bit numbers, and a run record has the same keys:

* the tables: `write_tension_table` / `read_tension_table`, `write_pressure_table` / `read_pressure_table`, the
  seed ensemble's `write_ensemble_tension_table` and `write_per_seed_tension_table` and their readers, the solve
  diagnostics' `write_solve_table` / `read_solve_table` (`solves.csv`, one row per linear solve) and
  `write_spectra` (`spectra.npz`), and `format_float`, the shortest decimal string that reads back as the same
  float. The file names are `TENSION_TABLE`, `PRESSURE_TABLE`, `SOLVE_TABLE`, `SPECTRA`, `RUN_RECORD`, ...;
  the readers and writers take a `pathlib.Path`;
* the run record: `infer_run_record` and `ensemble_run_record` build it, `write_run_record` writes it,
  `read_run_record` reads it; `RUN_RECORD_SCHEMA` is its schema number;
* what a run record names: the methods by their command-line spellings (`option_name`, `TENSION_METHODS`,
  `PRESSURE_METHODS`, `DIAGNOSTICS`), the versions (`library_versions`), and the warnings raised, each once with
  its count (`WarningCollector`).

The tables and the record's writers are implemented in `foambryo._result_tables`; this module is their public
interface. Changing a column, or a run-record key's name or meaning, makes a new schema.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from platform import python_version
from typing import TYPE_CHECKING, Any

import numpy as np

from foambryo._result_tables import (
    ENSEMBLE_OUTPUTS,
    ENSEMBLE_TENSION_COLUMNS,
    INFER_OUTPUTS,
    PER_SEED_TENSION_COLUMNS,
    PER_SEED_TENSION_TABLE,
    PRESSURE_COLUMNS,
    PRESSURE_COLUMNS_WITH_REGION_KIND,
    PRESSURE_TABLE,
    REGION_KINDS,
    RUN_RECORD,
    RUN_RECORD_SCHEMA,
    SOLVE_COLUMNS,
    SOLVE_TABLE,
    SPECTRA,
    TENSION_COLUMNS,
    TENSION_TABLE,
    file_sha256,
    format_float,
    json_safe,
    read_ensemble_tension_table,
    read_per_seed_tension_table,
    read_pressure_region_kinds,
    read_pressure_table,
    read_run_record,
    read_solve_table,
    read_tension_table,
    write_ensemble_tension_table,
    write_per_seed_tension_table,
    write_pressure_table,
    write_run_record,
    write_solve_table,
    write_spectra,
    write_tension_table,
)
from foambryo.pressure_inference import PressureComputationMethod
from foambryo.tension_inference import TensionComputationMethod

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from enum import Enum
    from pathlib import Path

    from foambryo.dcel import DcelData
    from foambryo.force_inference import InferenceOutcome
    from foambryo.seed_ensembles import SeedEnsembleResult

__all__ = [
    "DIAGNOSTICS",
    "ENSEMBLE_OUTPUTS",
    "ENSEMBLE_TENSION_COLUMNS",
    "INFER_OUTPUTS",
    "PER_SEED_TENSION_COLUMNS",
    "PER_SEED_TENSION_TABLE",
    "PRESSURE_COLUMNS",
    "PRESSURE_COLUMNS_WITH_REGION_KIND",
    "PRESSURE_METHODS",
    "PRESSURE_TABLE",
    "REGION_KINDS",
    "RUN_RECORD",
    "RUN_RECORD_SCHEMA",
    "SOLVE_COLUMNS",
    "SOLVE_TABLE",
    "SPECTRA",
    "TENSION_COLUMNS",
    "TENSION_METHODS",
    "TENSION_TABLE",
    "WarningCollector",
    "count_undetermined",
    "ensemble_library_call",
    "ensemble_run_record",
    "file_sha256",
    "format_float",
    "infer_run_record",
    "infer_units",
    "json_safe",
    "library_versions",
    "option_name",
    "read_ensemble_tension_table",
    "read_per_seed_tension_table",
    "read_pressure_region_kinds",
    "read_pressure_table",
    "read_rec_metadata",
    "read_run_record",
    "read_solve_table",
    "read_tension_table",
    "write_ensemble_tension_table",
    "write_per_seed_tension_table",
    "write_pressure_table",
    "write_run_record",
    "write_solve_table",
    "write_spectra",
    "write_tension_table",
]


# ---------------------------------------------------------------------------------------------
# the methods' spellings
# ---------------------------------------------------------------------------------------------
def option_name(member: Enum) -> str:
    """The command-line spelling of a method: its enumeration name in lower case, hyphenated.

    `YoungDupre` -> `young-dupre`, `ForceBalanceMAP` -> `force-balance-map`,
    `WeightedLaplace` -> `weighted-laplace`.
    """
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "-", member.name).lower()


#: Tension methods by their spelling, in the enumeration's order.
TENSION_METHODS = {option_name(member): member for member in TensionComputationMethod}
#: Pressure methods by their spelling, in the enumeration's order.
PRESSURE_METHODS = {option_name(member): member for member in PressureComputationMethod}
#: The diagnostics' spellings and the library's `diagnostics` argument each one passes.
DIAGNOSTICS = {"none": False, "basic": True, "spectrum": "spectrum"}


# ---------------------------------------------------------------------------------------------
# what a run record names
# ---------------------------------------------------------------------------------------------
def library_versions(extra: Iterable[str] = ()) -> dict[str, str]:
    """The versions a run record names: Python, foambryo, delaunay-watershed-3d, NumPy, SciPy, then `extra`.

    Args:
        extra: further distributions to name, for example a front end's own (`"napari"`).

    Returns:
        dict[str, str]: distribution -> version, `"not installed"` when it is not.
    """
    found = {"python": python_version()}
    for distribution in ("foambryo", "delaunay-watershed-3d", "numpy", "scipy", *extra):
        try:
            found[distribution] = version(distribution)
        except PackageNotFoundError:
            found[distribution] = "not installed"
    return found


class WarningCollector:
    """Catch the warnings raised inside a block, and count each distinct one.

    Use it as a context manager (`with WarningCollector() as collector: ...`). `DeprecationWarning`,
    `PendingDeprecationWarning`, `ImportWarning` and `ResourceWarning` are ignored inside the block (neither shown
    nor counted); every other warning is caught, wherever it is raised, rather than once per code location, so a
    count is the number of times it was raised. Caught warnings are not shown. After the block, `summary` holds one
    entry per distinct warning (same category and message), `{"category": ..., "message": ..., "count": ...}`
    with the category's class name, in the order they were first raised: a run record's `warnings`.
    """

    def __init__(self) -> None:
        """Start with an empty summary."""
        self.summary: list[dict[str, Any]] = []
        self._context = warnings.catch_warnings(record=True)

    def __enter__(self) -> WarningCollector:
        """Start catching."""
        self._caught = self._context.__enter__()
        warnings.simplefilter("always")
        for category in (DeprecationWarning, PendingDeprecationWarning, ImportWarning, ResourceWarning):
            warnings.simplefilter("ignore", category)
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop catching, and count."""
        self._context.__exit__(*exc_info)
        counts = Counter((caught.category.__name__, str(caught.message)) for caught in self._caught)
        self.summary = [{"category": category, "message": message, "count": count}
                        for (category, message), count in counts.items()]


def read_rec_metadata(path: Path) -> dict[str, Any] | None:
    """A `.rec` file's metadata appendix as delaunay-watershed-3d reads it, its `_version` key renamed.

    Args:
        path: the mesh file (a `pathlib.Path`); only the suffixes `.rec` and `.arec` are read.

    Returns:
        dict | None: the appendix with `_version` as `appendix_version`; `None` for another suffix, a file without
        an appendix, or a delaunay-watershed-3d that cannot read one.
    """
    if path.suffix not in (".rec", ".arec"):
        return None
    try:
        from dw3d.io import load_rec_appendix
    except ImportError:
        return None
    appendix = load_rec_appendix(path)
    if appendix is None:
        return None
    return {("appendix_version" if key == "_version" else key): value for key, value in appendix.items()}


def infer_units(mean_tension: float, metadata: dict[str, Any] | None) -> dict[str, str]:
    """The units a run record states: tensions relative to their mean, pressures in tension per mesh length.

    Args:
        mean_tension: the mean the tensions were gauged to.
        metadata: the mesh file's appendix (`read_rec_metadata`), which says whether the mesh is in physical units.

    Returns:
        dict[str, str]: `{"tension": "relative, gauged to a mean tension of <T>", "pressure": "tension per <unit>"}`,
        the unit being the appendix's `spacing_unit` for physical coordinates, `voxel` for voxel coordinates,
        else `mesh length unit (not recorded in the mesh file)`.
    """
    frame = (metadata or {}).get("coordinate_frame")
    unit = (metadata or {}).get("spacing_unit")
    if frame == "physical" and unit:
        length = unit
    elif frame == "voxel_index":
        length = "voxel"
    else:
        length = "mesh length unit (not recorded in the mesh file)"
    return {"tension": f"relative, gauged to a mean tension of {mean_tension!r}", "pressure": f"tension per {length}"}


def count_undetermined(values: dict | None) -> int | None:
    """How many values of a map (`key -> float`) are not finite; `None` when the map itself is `None`."""
    if values is None:
        return None
    return int(sum(not math.isfinite(value) for value in values.values()))


# ---------------------------------------------------------------------------------------------
# run records
# ---------------------------------------------------------------------------------------------
def infer_run_record(
    *,
    command: str,
    arguments: Sequence[str],
    date: str,
    elapsed_seconds: float,
    input_record: dict[str, Any],
    options: dict[str, Any],
    mesh: DcelData,
    outcome: InferenceOutcome,
    units: dict[str, str],
    warnings_summary: list[dict[str, Any]],
    outputs: list[str],
    versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The run record of an inference on one mesh, schema 1, as `foambryo infer` writes it.

    Args:
        command: the command that ran (`"foambryo infer"`, or a front end's name).
        arguments: its arguments.
        date: when it started, ISO 8601.
        elapsed_seconds: how long it took.
        input_record: the mesh file: `path`, `sha256`, `bytes`, `metadata`.
        options: the options by their command-line names (`tension_method`, `pressure_method`, `mean_tension`,
            `base_pressure`, `solver`, `diagnostics`), the methods spelled as `option_name` spells them;
            `diagnostics` is required (`"none"`, `"basic"` or `"spectrum"`).
        mesh: the mesh inferred on.
        outcome: what `foambryo.infer_tensions_then_pressures` returned.
        units: `infer_units`.
        warnings_summary: `WarningCollector.summary`.
        outputs: the files written beside the record, the record included.
        versions: `library_versions()` by default.

    Returns:
        dict: the record, for `write_run_record`: `schema`, `command`, `arguments`, `date`, `elapsed_seconds`,
        `versions`, `input`, `options`, `library_call`, `mesh` (counts, the media and the cut cells), `results`
        (counts of the values and of the undetermined ones), `units`, `status`, `error`, `warnings`, `solves`
        (`None` with `diagnostics` `"none"`) and `outputs`.

    Raises:
        KeyError: if `options` has no `diagnostics`.
    """
    return {
        "schema": RUN_RECORD_SCHEMA,
        "command": command,
        "arguments": list(arguments),
        "date": date,
        "elapsed_seconds": elapsed_seconds,
        "versions": library_versions() if versions is None else versions,
        "input": input_record,
        "options": options,
        "library_call": outcome.library_call,
        "mesh": {
            "points": len(mesh.v),
            "triangles": len(mesh.f),
            "materials": int(mesh.n_materials),
            "cells": int(mesh.n_cells),
            "media": [int(label) for label in mesh.media],
            "unresolved": [int(label) for label in mesh.unresolved],
            "crop_faces": int(np.count_nonzero(mesh.crop_faces)),
            "cut_cells": list(mesh.cut_cells),
        },
        "results": {
            "tensions": None if outcome.tensions is None else len(outcome.tensions),
            "undetermined_tensions": count_undetermined(outcome.tensions),
            "pressures": None if outcome.pressures is None else len(outcome.pressures),
            "undetermined_pressures": count_undetermined(outcome.pressures),
        },
        "units": units,
        "status": outcome.status,
        "error": outcome.error,
        "warnings": warnings_summary,
        "solves": None if options["diagnostics"] == "none" else [record.scalars() for record in outcome.records],
        "outputs": outputs,
    }


def ensemble_library_call(
    *,
    n_seeds: int,
    min_distance: int,
    mode: TensionComputationMethod,
    angle_reading: str,
    combine: str,
    max_workers: int,
    spacing: tuple[float, float, float],
) -> str:
    """The `infer_tensions_over_seeds` call of a seed ensemble over the seeds 0..N-1, as text (`run.json`'s).

    Every argument is keyword-only and is written into the call as given: `n_seeds` as `seeds=range(N)`, `mode`
    by its enumeration name, `spacing` as `(dz, dy, dx)`.
    """
    return (
        f"infer_tensions_over_seeds(mask, seeds=range({n_seeds}), min_distance={min_distance}, "
        f"mode=TensionComputationMethod.{mode.name}, "
        f"angle_reading={angle_reading!r}, combine={combine!r}, max_workers={max_workers}, "
        f"spacing={tuple(spacing)!r})"
    )


def ensemble_run_record(
    *,
    command: str,
    arguments: Sequence[str],
    date: str,
    elapsed_seconds: float,
    input_record: dict[str, Any],
    options: dict[str, Any],
    seeds: list[int],
    library_call: str,
    result: SeedEnsembleResult | None,
    length_unit: str,
    status: str,
    error: str | None,
    warnings_summary: list[dict[str, Any]],
    outputs: list[str],
    versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The run record of a seed ensemble, schema 1, as `foambryo infer-over-seeds` writes it.

    Args:
        command: the command that ran.
        arguments: its arguments.
        date: when it started, ISO 8601.
        elapsed_seconds: how long it took.
        input_record: the mask: `path`, `sha256`, `bytes`, `shape`, `spacing`, `spacing_unit`, `spacing_from`.
        options: `seeds` (N), `combine`, `angle_reading`, `tension_method`, `min_distance`, `max_workers`.
        seeds: the seeds, 0..N-1.
        library_call: `ensemble_library_call`.
        result: what `infer_tensions_over_seeds` returned, or `None` when it refused.
        length_unit: the unit of the spacing.
        status: `"complete"` or `"failed"`.
        error: the library's message when it refused, else `None`.
        warnings_summary: `WarningCollector.summary`.
        outputs: the files written beside the record, the record included.
        versions: `library_versions()` by default.

    Returns:
        dict: the record, for `write_run_record`: `schema`, `command`, `arguments`, `date`, `elapsed_seconds`,
        `versions`, `input`, `options`, `seeds`, `library_call`, `reconstructions` (one entry per seed: `seed`,
        `points`, `triangles`, `crop_faces`, `seconds`; `None` when refused), `results` (counts), `units`,
        `status`, `error`, `warnings` and `outputs`.
    """
    return {
        "schema": RUN_RECORD_SCHEMA,
        "command": command,
        "arguments": list(arguments),
        "date": date,
        "elapsed_seconds": elapsed_seconds,
        "versions": library_versions() if versions is None else versions,
        "input": input_record,
        "options": options,
        "seeds": seeds,
        "library_call": library_call,
        "reconstructions": None if result is None else [
            {"seed": r.seed, "points": r.n_points, "triangles": r.n_triangles, "crop_faces": r.n_crop_faces,
             "seconds": round(r.seconds, 3)} for r in result.reconstructions
        ],
        "results": {
            "tensions": None if result is None else len(result.tensions),
            "undetermined_tensions": None if result is None else count_undetermined(result.tensions),
            "found_by_fewer_than_half_the_seeds": None if result is None
            else int(sum(result.found_by_fewer_than_half_the_seeds.values())),
        },
        "units": {"tension": "relative: each seed's tensions are gauged to a mean tension of 1.0",
                  "length": length_unit},
        "status": status,
        "error": error,
        "warnings": warnings_summary,
        "outputs": outputs,
    }
