"""The files `foambryo infer` writes and `foambryo view` reads: CSV tables and a JSON run record.

Implemented here and public through `foambryo.results`, which front ends import. The file formats (the columns
and the `run.json` keys, schema 1) are specified in the README's "Command line" section and in
`docs/command_line_specification.md` of the source repository. The readers and writers take a `pathlib.Path`.

**Numbers are written exactly.** A float is written as Python's `repr`, the shortest decimal string
that reads back as the same 64-bit float, so a table read back with `float()` (or numpy, or pandas)
holds the library's values bit for bit. `nan` and `inf` are written as `nan`, `inf`, `-inf`, which
those readers all accept.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

#: The run record's schema. Keys may be added within a schema; removing or renaming one, or changing
#: what one means, makes a new schema number.
RUN_RECORD_SCHEMA = 1

TENSION_TABLE = "tensions.csv"
PRESSURE_TABLE = "pressures.csv"
RUN_RECORD = "run.json"
SPECTRA = "spectra.npz"

#: The solve diagnostics as a table, one row per linear solve: written with `--diagnostics basic` or `spectrum`.
SOLVE_TABLE = "solves.csv"

#: Every file `foambryo infer` may write in its output directory, and so the only ones
#: `--overwrite` deletes.
INFER_OUTPUTS = (TENSION_TABLE, PRESSURE_TABLE, RUN_RECORD, SPECTRA, SOLVE_TABLE)

TENSION_COLUMNS = ("label_1", "label_2", "tension")
#: `foambryo infer-over-seeds`' tensions.csv: the ensemble tension, then what the seeds say about it.
ENSEMBLE_TENSION_COLUMNS = (*TENSION_COLUMNS, "tension_seed_sd", "n_seeds_found", "found_by_fewer_than_half_the_seeds")
#: `foambryo infer-over-seeds`' per-seed tensions: one row per seed and interface, in seed order.
PER_SEED_TENSION_TABLE = "tensions_per_seed.csv"
PER_SEED_TENSION_COLUMNS = ("seed", *TENSION_COLUMNS)
#: The files `foambryo infer-over-seeds` writes in OUTDIR, and only these (no pressures: no single mesh).
ENSEMBLE_OUTPUTS = (TENSION_TABLE, PER_SEED_TENSION_TABLE, RUN_RECORD)
PRESSURE_COLUMNS = ("label", "pressure")
#: `solves.csv`: `index` (the solve's position, as in `run.json`'s `solves` and `spectra.npz`'s
#: `singular_values_<index>`), then the scalars of the library's `SolveRecord`, in its order.
SOLVE_COLUMNS = (
    "index", "system", "solver", "n_rows", "n_unknowns", "rank", "residual_norm", "condition_number",
    "condition_number_source", "seconds", "lsqr_istop", "lsqr_iterations", "spectrum_seconds", "spectrum_note",
    "n_singular_values",
)
#: The integer columns of `solves.csv`, and its floats; the others are text.
_SOLVE_INTEGERS = ("index", "n_rows", "n_unknowns", "rank", "lsqr_istop", "lsqr_iterations", "n_singular_values")
_SOLVE_FLOATS = ("residual_norm", "condition_number", "seconds", "spectrum_seconds")


def format_float(value: float) -> str:
    """The shortest decimal string that reads back as the same float (`nan`, `inf`, `-inf` as such)."""
    return repr(float(value))


def write_tension_table(path: Path, dict_tensions: dict[tuple[int, int], float]) -> None:
    """Write `tensions.csv`: `label_1,label_2,tension`, one row per interface, sorted by the label pair.

    Args:
        path: the file (a `pathlib.Path`), replaced if it exists.
        dict_tensions: tension per interface, keyed `(label_1, label_2)` with `label_1 < label_2`; a `nan`
            tension is written `nan`. Floats are written exactly (`format_float`).
    """
    with path.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(TENSION_COLUMNS)
        for (label_1, label_2), tension in sorted(dict_tensions.items(), key=lambda item: (int(item[0][0]),
                                                                                            int(item[0][1]))):
            writer.writerow((int(label_1), int(label_2), format_float(tension)))


def write_ensemble_tension_table(path: Path, result: Any) -> None:  # noqa: ANN401 - a SeedEnsembleResult
    """Write `foambryo infer-over-seeds`' tensions.csv: one row per interface of the ensemble, sorted by label pair.

    Columns `ENSEMBLE_TENSION_COLUMNS`: the ensemble tension and its seed SD in the shortest exact float format,
    the number of seeds that found the interface, and `true`/`false` for the flag.

    Args:
        path: the file (a `pathlib.Path`), replaced if it exists.
        result: a `foambryo.seed_ensembles.SeedEnsembleResult` (its `tensions`, `tension_seed_sd`,
            `n_seeds_found` and `found_by_fewer_than_half_the_seeds` maps).
    """
    with path.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(ENSEMBLE_TENSION_COLUMNS)
        for key in sorted(result.tensions, key=lambda k: (int(k[0]), int(k[1]))):
            writer.writerow((
                int(key[0]), int(key[1]), format_float(result.tensions[key]), format_float(result.tension_seed_sd[key]),
                int(result.n_seeds_found[key]), "true" if result.found_by_fewer_than_half_the_seeds[key] else "false",
            ))


def read_ensemble_tension_table(path: Path) -> dict[tuple[int, int], tuple[float, float, int, bool]]:
    """Read `foambryo infer-over-seeds`' tensions.csv.

    Args:
        path: the file (a `pathlib.Path`).

    Returns:
        dict: `(label_1, label_2) -> (tension, tension_seed_sd, n_seeds_found, found_by_fewer_than_half_the_seeds)`.

    Raises:
        ValueError: if the header is not `ENSEMBLE_TENSION_COLUMNS` or a row cannot be parsed.
    """
    rows = _read_rows(path, ENSEMBLE_TENSION_COLUMNS)
    flags = {"true": True, "false": False}
    try:
        return {(int(r[0]), int(r[1])): (float(r[2]), float(r[3]), int(r[4]), flags[r[5]]) for r in rows}
    except KeyError as error:
        message = f"{path.name}: found_by_fewer_than_half_the_seeds must be true or false, not {error}"
        raise ValueError(message) from error


def write_per_seed_tension_table(
    path: Path,
    per_seed_tensions: dict[int, dict[tuple[int, int], float]],
) -> None:
    """Write `tensions_per_seed.csv`: `seed,label_1,label_2,tension`, each seed's own tensions.

    Args:
        path: the file (a `pathlib.Path`), replaced if it exists.
        per_seed_tensions: `seed -> {(label_1, label_2): tension}`; the seeds are written in the map's order, each
            seed's interfaces sorted by label pair.
    """
    with path.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(PER_SEED_TENSION_COLUMNS)
        for seed, tensions in per_seed_tensions.items():
            for (label_1, label_2), tension in sorted(tensions.items(), key=lambda item: (int(item[0][0]),
                                                                                          int(item[0][1]))):
                writer.writerow((int(seed), int(label_1), int(label_2), format_float(tension)))


def read_per_seed_tension_table(path: Path) -> dict[int, dict[tuple[int, int], float]]:
    """Read `tensions_per_seed.csv` back into `seed -> {(label_1, label_2): tension}`, in file order.

    Args:
        path: the file (a `pathlib.Path`).

    Returns:
        dict[int, dict[tuple[int, int], float]]: each seed's tensions.

    Raises:
        ValueError: if the header is not `seed,label_1,label_2,tension` or a row cannot be parsed.
    """
    out: dict[int, dict[tuple[int, int], float]] = {}
    for row in _read_rows(path, PER_SEED_TENSION_COLUMNS):
        out.setdefault(int(row[0]), {})[(int(row[1]), int(row[2]))] = float(row[3])
    return out


def write_pressure_table(path: Path, dict_pressures: dict[int, float]) -> None:
    """Write `pressures.csv`: `label,pressure`, one row per material (the medium, 0, included), sorted by label.

    Args:
        path: the file (a `pathlib.Path`), replaced if it exists.
        dict_pressures: pressure per label; a `nan` pressure is written `nan`. Floats are written exactly
            (`format_float`).
    """
    with path.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(PRESSURE_COLUMNS)
        for label, pressure in sorted(dict_pressures.items(), key=lambda item: int(item[0])):
            writer.writerow((int(label), format_float(pressure)))


def read_tension_table(path: Path) -> dict[tuple[int, int], float]:
    """Read a `tensions.csv` back into the library's map `(label_1, label_2) -> tension`.

    Args:
        path: the file (a `pathlib.Path`).

    Returns:
        dict[tuple[int, int], float]: the tensions, bitwise the numbers written.

    Raises:
        ValueError: if the header is not `label_1,label_2,tension` or a row cannot be parsed.
    """
    rows = _read_rows(path, TENSION_COLUMNS)
    return {(int(row[0]), int(row[1])): float(row[2]) for row in rows}


def read_pressure_table(path: Path) -> dict[int, float]:
    """Read a `pressures.csv` back into the library's map `label -> pressure`.

    Args:
        path: the file (a `pathlib.Path`).

    Returns:
        dict[int, float]: the pressures, bitwise the numbers written.

    Raises:
        ValueError: if the header is not `label,pressure` or a row cannot be parsed.
    """
    rows = _read_rows(path, PRESSURE_COLUMNS)
    return {int(row[0]): float(row[1]) for row in rows}


def _read_rows(path: Path, columns: tuple[str, ...]) -> list[list[str]]:
    with path.open(newline="", encoding="ascii") as file:
        rows = list(csv.reader(file))
    if not rows or tuple(rows[0]) != columns:
        found = ",".join(rows[0]) if rows else "an empty file"
        message = f"{path.name}: expected the header {','.join(columns)}, found {found}"
        raise ValueError(message)
    body = rows[1:]
    if any(len(row) != len(columns) for row in body):
        message = f"{path.name}: every row must have {len(columns)} values"
        raise ValueError(message)
    return body


def file_sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes (`path`, a `pathlib.Path`), as 64 hexadecimal digits."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:  # noqa: ANN401 - any JSON-like value, converted recursively
    """Convert numpy scalars and arrays, tuples and non-finite floats into plain JSON values.

    Non-finite floats become the strings `"nan"`, `"inf"`, `"-inf"`: JSON has no such numbers, and
    a reader that meets a bare `NaN` token (which Python's `json` would write) may refuse the file.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_run_record(path: Path, record: dict[str, Any]) -> None:
    """Write the run record to `path` (a `pathlib.Path`) as indented JSON (UTF-8), non-finite numbers as strings.

    Args:
        path: the file, replaced if it exists.
        record: the record (`infer_run_record` or `ensemble_run_record`); converted by `json_safe`.
    """
    path.write_text(json.dumps(json_safe(record), indent=1, allow_nan=False) + "\n", encoding="utf-8")


def read_run_record(path: Path) -> dict[str, Any]:
    """Read a run record (`path`, a `pathlib.Path`).

    Returns:
        dict: the record as written; non-finite numbers stay the strings `"nan"`, `"inf"`, `"-inf"`.

    Raises:
        ValueError: if the file is not valid JSON.
        TypeError: if it is not a JSON object.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        message = f"{path.name} is not a JSON object"
        raise TypeError(message)
    return record


def write_solve_table(path: Path, records: list[Any]) -> None:
    """Write `solves.csv`: one row per linear solve, in the order they ran (`SOLVE_COLUMNS`).

    `index` is the solve's position (as in `run.json`'s `solves` and `spectra.npz`'s names). A value the solve
    does not have (the rank of a sparse solve, `lsqr`'s counts on a dense one) is an empty field; floats are
    written exactly (`format_float`), a non-finite one as `nan` / `inf`. The run's warnings are not per solve:
    they are in `run.json`'s `warnings`.

    Args:
        path: the file (a `pathlib.Path`), replaced if it exists.
        records: the solves' `foambryo.linear_solves.SolveRecord`s (an `InferenceOutcome`'s `records`, or a
            `SolveDiagnostics`' `records`).
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(SOLVE_COLUMNS)
        for index, record in enumerate(records):
            scalars = {"index": index, **record.scalars()}
            row = []
            for column in SOLVE_COLUMNS:
                value = scalars.get(column)
                if value is None:
                    row.append("")
                elif column in _SOLVE_FLOATS and not isinstance(value, str):
                    row.append(format_float(value))
                else:
                    row.append(str(value))
            writer.writerow(row)


def read_solve_table(path: Path) -> list[dict[str, Any]]:
    """Read `solves.csv` back: one dict per solve, integers and floats as such, empty fields as `None`.

    Args:
        path: the file (a `pathlib.Path`).

    Returns:
        list[dict[str, Any]]: one dict per row, keyed by `SOLVE_COLUMNS`, in file order; equal field by field to
        the run's `solves` in `run.json`, the floats bitwise.

    Raises:
        ValueError: if the header is not `SOLVE_COLUMNS` or a row cannot be parsed.
    """
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if not rows or tuple(rows[0]) != SOLVE_COLUMNS:
        found = ",".join(rows[0]) if rows else "an empty file"
        message = f"{path.name}: expected the header {','.join(SOLVE_COLUMNS)}, found {found}"
        raise ValueError(message)
    out = []
    for row in rows[1:]:
        if len(row) != len(SOLVE_COLUMNS):
            message = f"{path.name}: every row must have {len(SOLVE_COLUMNS)} values"
            raise ValueError(message)
        solve: dict[str, Any] = {}
        for column, text in zip(SOLVE_COLUMNS, row, strict=True):
            if text == "":
                solve[column] = None
            elif column in _SOLVE_INTEGERS:
                solve[column] = int(text)
            elif column in _SOLVE_FLOATS:
                solve[column] = float(text)
            else:
                solve[column] = text
        out.append(solve)
    return out


def write_spectra(path: Path, singular_values: list[np.ndarray | None]) -> None:
    """Write the singular values of the solves that have them, as `foambryo.save_solve_diagnostics` does.

    Args:
        path: the `.npz` file, replaced if it exists.
        singular_values: per solve, in the order of the run record's `solves`, its singular values
            or `None`; solve `i`'s are stored under `singular_values_<i>`.
    """
    arrays = {f"singular_values_{index}": values for index, values in enumerate(singular_values) if values is not None}
    np.savez_compressed(path, **arrays)
