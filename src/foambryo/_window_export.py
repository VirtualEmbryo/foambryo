"""Export from the viewer's window: the `foambryo infer` command's files for the values the window shows.

The panel's *Export* writes, in a folder, the files `foambryo infer` writes for the same values, with the command's own
writers (`foambryo.results`), so that a folder exported from the window and one written by the command hold the same
tables byte for byte:

* `tensions.csv`, and `pressures.csv` when pressures were inferred;
* `solves.csv` with diagnostics (`basic` or `spectrum`), and `spectra.npz` with the spectrum: those of the window's
  last inference (values given to the window carry none);
* `run.json`: the command's run record (schema 1), its `command` the window's, its `input` the mesh file the
  window was opened on (or `mesh.rec`, written beside it, for a mesh reconstructed in the window);
* `colour_bars.png`: `save_colour_bars` on the values shown, as `foambryo view --colour-bars` writes it;
* `mesh.rec` for a mesh reconstructed in the window: `foambryo mesh`'s file, byte for byte.

Like the napari plugin's Export, it never replaces a file silently: `earlier_window_outputs` lists foambryo's files
already in the folder, and the panel asks *Replace* or *Cancel*; any other file in the folder is never touched.
No Polyscope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from foambryo import results as tables
from foambryo._colour_scales import save_colour_bars
from foambryo.force_inference import InferenceOutcome

if TYPE_CHECKING:
    from foambryo.dcel import DcelData

__all__ = ["COLOUR_BARS", "MESH_FILE", "WINDOW_EXPORT_OUTPUTS", "WindowInference", "earlier_window_outputs",
           "export_window_values"]

MESH_FILE = "mesh.rec"
COLOUR_BARS = "colour_bars.png"
#: Every file the window's Export may write, and so the only ones it replaces.
WINDOW_EXPORT_OUTPUTS = (MESH_FILE, *tables.INFER_OUTPUTS, COLOUR_BARS)


@dataclass
class WindowInference:
    """An inference the window made (or was given), as its run record needs it.

    Attributes:
        outcome: what `infer_tensions_then_pressures` returned (its solve records with diagnostics).
        options: `run.json`'s `options`, by the command's spellings (`foambryo infer`'s `_infer_record_options`).
        date: when it started, ISO 8601.
        seconds: how long it took.
        warnings: the warnings raised, each once with its count (`foambryo.results.WarningCollector.summary`).
        given: the values were given to the window, not inferred there (no solve record; `run.json` says so).
    """

    outcome: InferenceOutcome
    options: dict[str, Any]
    date: str = ""
    seconds: float = 0.0
    warnings: list = field(default_factory=list)
    given: bool = False


def earlier_window_outputs(folder: str | Path) -> list[str]:
    """The files of foambryo already in `folder` (`WINDOW_EXPORT_OUTPUTS`), which an Export would replace."""
    folder = Path(folder)
    return [name for name in WINDOW_EXPORT_OUTPUTS if (folder / name).is_file()]


def export_window_values(
    folder: str | Path,
    mesh: DcelData,
    inference: WindowInference,
    *,
    mesh_file: str | None = None,
    rec_bytes: bytes | None = None,
    command: str = "foambryo view",
    overwrite: bool = False,
) -> list[str]:
    """Write the window's values in `folder` (created if missing), as `foambryo infer` writes them; returns the names.

    Args:
        folder: the folder.
        mesh: the mesh the values belong to.
        inference: the values and how they were obtained.
        mesh_file: the mesh file the window was opened on, recorded as `run.json`'s `input` (its path as given, its
            SHA-256, its size, its metadata appendix). Keyword-only.
        rec_bytes: the `.rec` bytes of a mesh reconstructed in the window, written as `mesh.rec` and recorded as the
            input. Keyword-only. With neither, `input` records no file.
        command: `run.json`'s `command`. Keyword-only.
        overwrite: delete the earlier outputs `earlier_window_outputs` lists first (and no other file).
            Keyword-only.

    Returns:
        list[str]: the files written, in the order the command lists them.

    Raises:
        FileExistsError: if the folder holds earlier outputs and `overwrite` is false.
        ValueError: if there is no finite tension to write.
    """
    outcome = inference.outcome
    if outcome.tensions is None or not any(np.isfinite(v) for v in outcome.tensions.values()):
        message = "no finite tension to export"
        raise ValueError(message)
    folder = Path(folder)
    existing = earlier_window_outputs(folder)
    if existing and not overwrite:
        message = f"{folder} already holds {', '.join(existing)}"
        raise FileExistsError(message)
    folder.mkdir(parents=True, exist_ok=True)
    for name in existing:
        (folder / name).unlink()

    outputs: list[str] = []
    input_path = None if mesh_file is None else Path(mesh_file)
    if rec_bytes is not None:
        (folder / MESH_FILE).write_bytes(rec_bytes)
        outputs.append(MESH_FILE)
        input_path = folder / MESH_FILE
    tables.write_tension_table(folder / tables.TENSION_TABLE, outcome.tensions)
    outputs.append(tables.TENSION_TABLE)
    if outcome.pressures is not None:
        tables.write_pressure_table(folder / tables.PRESSURE_TABLE, outcome.pressures, media=mesh.media)
        outputs.append(tables.PRESSURE_TABLE)
    diagnostics = inference.options.get("diagnostics", "none")
    if diagnostics != "none" and outcome.records:
        tables.write_solve_table(folder / tables.SOLVE_TABLE, outcome.records)
        outputs.append(tables.SOLVE_TABLE)
        if diagnostics == "spectrum":
            tables.write_spectra(folder / tables.SPECTRA, [record.singular_values for record in outcome.records])
            outputs.append(tables.SPECTRA)
    save_colour_bars(mesh, outcome.tensions, outcome.pressures, folder / COLOUR_BARS)
    outputs.append(COLOUR_BARS)
    outputs.append(tables.RUN_RECORD)

    if input_path is not None and input_path.is_file():
        metadata = tables.read_rec_metadata(input_path)
        input_record = {"path": MESH_FILE if rec_bytes is not None else str(mesh_file),
                        "sha256": tables.file_sha256(input_path), "bytes": input_path.stat().st_size,
                        "metadata": metadata}
    else:
        metadata = None
        input_record = {"path": None, "sha256": None, "bytes": None, "metadata": None}
    record = tables.infer_run_record(
        command=command,
        arguments=[],
        date=inference.date,
        elapsed_seconds=inference.seconds,
        input_record=input_record,
        # `media` as `foambryo infer` records it: the labels given with --media, here none (the mesh's own media,
        # read from its file, are in the record's `mesh.media`).
        options={"media": None, **inference.options},
        mesh=mesh,
        outcome=outcome,
        units=tables.infer_units(inference.options.get("mean_tension", 1.0), metadata),
        warnings_summary=inference.warnings,
        outputs=outputs,
    )
    if inference.given:
        record["values_given_to_the_window"] = outcome.library_call
    tables.write_run_record(folder / tables.RUN_RECORD, record)
    return outputs
