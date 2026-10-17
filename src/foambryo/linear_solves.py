r"""Solver choice and solve diagnostics for the inference's linear systems.

Every tension and pressure of this package is the solution of a linear least-squares system. Small
systems are solved densely (`scipy.linalg.lstsq`), exactly; large ones sparsely (`lsqr`, or a
sparse direct factorisation), because a dense system grows as the product of its dimensions in
memory and as their cube in time: the Young-Dupre system of a 1 866-cell tissue is 49 984 x 11 746,
4.7 GB dense, and at 10 000 cells it would be some 120 GB.

This module holds what the public entry points (`infer_tensions*`, `infer_pressures*`,
`infer_forces`) pass down to every solve:

* **`solver`**: `"auto"` (the default) solves a system densely below `sparse_threshold` unknowns
  and sparsely at or above it; `"dense"` and `"sparse"` force one path whatever the size. Below the
  threshold, `"auto"` is bitwise the dense solve. Forcing `"dense"` on a system whose dense form
  would not fit in the memory available emits a `DenseSolveMemoryWarning` with the estimate, before
  anything is allocated. The tension methods `Equilibrium`, `Variational` and `ForceBalance*` have
  no sparse path: they are solved densely under `"auto"`, and `"sparse"` raises `ValueError` for
  them. A sparse `lsqr` solve that does not converge falls back to the dense one with a
  `SparseSolveFallbackWarning` (or, when the dense one cannot fit in memory, returns `lsqr`'s last
  iterate with that warning).
* **`sparse_threshold`**: the number of unknowns at which `"auto"` switches (default
  `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS`, 2 000). What counts as "unknowns" is the
  number of columns of the system the dense path would assemble: the interfaces of one
  junction-graph component for a tension solve, the materials for a Laplace pressure, the
  interfaces plus the cells for the variational pressure. One exception: the variational
  pressure of a cropped mesh is solved densely up to
  `foambryo.pressure_inference.DENSE_CROP_PRESSURE_MAX_ENTRIES` matrix entries under `"auto"`,
  whatever `sparse_threshold` is.
* **`diagnostics`**: `False` (the default: nothing is computed and each entry point returns its
  usual values), `True` (one `SolveRecord` per solve: solver, shape, rank, residual norm,
  condition number, time) or `"spectrum"` (also the extreme singular values of the large, sparse
  systems, by `scipy.sparse.linalg.svds`, at a measured extra cost). With diagnostics on, each
  entry point returns one more value, a `SolveDiagnostics`, last; `save_solve_diagnostics` writes it
  to a JSON file (the scalars) and an NPZ file (the singular values).

They are keyword-only on every entry point that takes them, and an invalid value raises
`ValueError` (`validate_solve_options`). A lower-level function called directly (for example
`infer_tensions_young_dupre_local`) uses the options in force, the defaults outside any entry point;
`solve_options` puts others in force around such a call.

**What the condition numbers are.** On a dense least-squares solve, the 2-norm condition number
`s_max / s_min` from the singular values `scipy.linalg.lstsq` computes anyway (free; `inf` when the
smallest is zero). On an `lsqr` solve, `lsqr`'s own running estimate of the Frobenius-norm
condition number `||A||_F ||A^+||_F` (free; an estimate, and an over-estimate of the 2-norm one by
up to a factor `n`). On a sparse direct solve, the 1-norm condition number estimated from the
factorisation (Hager-Higham, `scipy.sparse.linalg.onenormest`; cheap). The source is recorded
beside every number: the three are not the same quantity.

**How the options travel.** A public entry point sets them in a context variable for the duration
of its call, and the solve helpers read it, so the options reach every solve without threading an
argument through each builder. The context is per thread and per asynchronous task
(`contextvars`); outside any entry point the defaults apply.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import ArpackNoConvergence, eigsh, svds

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scipy import sparse

#: The values `solver` accepts.
SOLVERS = ("auto", "dense", "sparse")

#: The values `diagnostics` accepts.
DIAGNOSTICS = (False, True, "spectrum")

#: How many of the largest and of the smallest singular values `diagnostics="spectrum"` asks
#: `svds` for on a sparse system.
SPECTRUM_EXTREME_VALUES = 6

#: ARPACK's restart limit for each end of the spectrum. Its default (ten times the number of
#: unknowns) leaves the smallest values of an ill-conditioned system free to run for hours; on a
#: 110-cell tissue's tension system the smallest values converge within 1 000.
SPECTRUM_MAX_ITERATIONS = 3000

#: Bytes per entry of a dense float64 matrix, and how many copies of it a dense least-squares
#: solve holds at once: the matrix and the copy `scipy.linalg.lstsq` factorises (the LAPACK
#: workspace is small beside them).
_BYTES_PER_ENTRY = 8
_DENSE_COPIES = 2


class DenseSolveMemoryWarning(UserWarning):
    """A dense solve was forced on a system whose dense form exceeds the memory available.

    Emitted before anything is allocated, with the estimate, so that the caller can stop the run
    (turn it into an error with `warnings.simplefilter("error", DenseSolveMemoryWarning)`). Only
    `solver="dense"` emits it: `"auto"` never chooses the dense path for a large system.
    """


@dataclass(frozen=True)
class SolveOptions:
    """The solver options in force for the solves of one entry-point call.

    Attributes:
        solver: `"auto"`, `"dense"` or `"sparse"`.
        sparse_threshold: unknowns at which `"auto"` goes sparse; `None` means
            `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS`, read when the solve runs.
        diagnostics: `False`, `True` or `"spectrum"`.
    """

    solver: str = "auto"
    sparse_threshold: int | None = None
    diagnostics: bool | str = False


@dataclass
class SolveRecord:
    """What one linear solve did. Scalars are JSON-ready; the singular values go to NPZ.

    Attributes:
        system: which system, e.g. `"tensions YoungDupre, junction-graph component 0"`.
        solver: the solver that produced the solution (`"dense lstsq"`, `"sparse lsqr"`,
            `"dense lstsq after lsqr did not converge"`, `"sparse direct (SuperLU)"`, `"dense inv"`,
            ...).
        n_rows: rows of the system solved (after any reduction, e.g. undetermined cells left out).
        n_unknowns: columns of the system solved.
        rank: numerical rank, where the solver computes it (dense `lstsq`: its `rank`); `None` on the
            sparse paths, which do not.
        residual_norm: `||b - A x||_2` of the system solved.
        condition_number: see the module docstring for what each source means; `None` when the
            path has none.
        condition_number_source: how `condition_number` was obtained.
        seconds: wall time of the solve, including the diagnostics' own cost.
        lsqr_istop, lsqr_iterations: `lsqr`'s stop code and iteration count (sparse path only).
        spectrum_seconds: time spent on `svds` for `diagnostics="spectrum"` (sparse path only).
        spectrum_note: why a requested spectrum is missing or partial, if it is.
        singular_values: the singular values in decreasing order: all of them on a dense
            `lstsq` path; on a sparse path with `diagnostics="spectrum"`, the largest and the
            smallest `SPECTRUM_EXTREME_VALUES`, concatenated (a gap, not a full spectrum).
    """

    system: str
    solver: str
    n_rows: int
    n_unknowns: int
    rank: int | None = None
    residual_norm: float | None = None
    condition_number: float | None = None
    condition_number_source: str | None = None
    seconds: float | None = None
    lsqr_istop: int | None = None
    lsqr_iterations: int | None = None
    spectrum_seconds: float | None = None
    spectrum_note: str | None = None
    singular_values: NDArray[np.float64] | None = field(default=None, repr=False)

    def scalars(self) -> dict[str, Any]:
        """The record without its singular values, with non-finite floats as strings (JSON-safe)."""
        out = asdict(self)
        out.pop("singular_values")
        out["n_singular_values"] = 0 if self.singular_values is None else len(self.singular_values)
        for key, value in out.items():
            if isinstance(value, float) and not np.isfinite(value):
                out[key] = str(value)
        return out


@dataclass
class SolveDiagnostics:
    """The records of every solve one entry-point call ran, in the order they ran.

    Attributes:
        options: the options in force.
        records: one `SolveRecord` per solve.
    """

    options: SolveOptions
    records: list[SolveRecord] = field(default_factory=list)

    def __len__(self) -> int:
        """Number of solves recorded."""
        return len(self.records)

    def table(self) -> list[dict[str, Any]]:
        """The records' scalars, one dict per solve (what the JSON file holds)."""
        return [record.scalars() for record in self.records]


def save_solve_diagnostics(diagnostics: SolveDiagnostics, path_stem: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Write solve diagnostics to `<path_stem>.json` (scalars) and `<path_stem>.npz` (spectra).

    The JSON file holds the options and one object per solve (`SolveRecord.scalars`, non-finite
    numbers as the strings `"inf"`/`"nan"`). The NPZ file holds, for every solve that has them, its
    singular values under the key `singular_values_<index>`, `<index>` being the solve's position in
    the JSON list.

    Args:
        diagnostics: what an entry point returned with `diagnostics` on.
        path_stem: path without extension; parent directories must exist.

    Returns:
        tuple[Path, Path]: the JSON path and the NPZ path.
    """
    stem = Path(path_stem)
    json_path = stem.with_name(stem.name + ".json")
    npz_path = stem.with_name(stem.name + ".npz")
    payload = {"options": asdict(diagnostics.options), "solves": diagnostics.table()}
    json_path.write_text(json.dumps(payload, indent=1) + "\n")
    arrays = {
        f"singular_values_{index}": record.singular_values
        for index, record in enumerate(diagnostics.records)
        if record.singular_values is not None
    }
    np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path


_OPTIONS: ContextVar[SolveOptions] = ContextVar("foambryo_solve_options", default=SolveOptions())  # noqa: B039
_RECORDS: ContextVar[list[SolveRecord] | None] = ContextVar("foambryo_solve_records", default=None)


def validate_solve_options(solver: str, sparse_threshold: int | None, diagnostics: bool | str) -> SolveOptions:
    """Check the three options and return them as a `SolveOptions`.

    Raises:
        ValueError: on a `solver` outside `SOLVERS`, a `diagnostics` outside `DIAGNOSTICS`, or a
            negative `sparse_threshold`.
    """
    if solver not in SOLVERS:
        message = f"solver must be one of {SOLVERS}, not {solver!r}"
        raise ValueError(message)
    if not any(diagnostics is allowed or (isinstance(allowed, str) and diagnostics == allowed)
               for allowed in DIAGNOSTICS):
        message = f"diagnostics must be one of {DIAGNOSTICS}, not {diagnostics!r}"
        raise ValueError(message)
    if sparse_threshold is not None and (not isinstance(sparse_threshold, (int, np.integer)) or sparse_threshold < 0):
        message = f"sparse_threshold must be a non-negative integer or None, not {sparse_threshold!r}"
        raise ValueError(message)
    return SolveOptions(solver=solver, sparse_threshold=None if sparse_threshold is None else int(sparse_threshold),
                        diagnostics=diagnostics)


@contextmanager
def solve_options(
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> Iterator[SolveDiagnostics | None]:
    """Put solver options in force for the solves run inside the block.

    Used by the public entry points; usable directly around any lower-level call. Yields the
    `SolveDiagnostics` the solves inside fill, or `None` when `diagnostics` is `False`.

    Raises:
        ValueError: see `validate_solve_options`.
    """
    options = validate_solve_options(solver, sparse_threshold, diagnostics)
    collected = SolveDiagnostics(options) if diagnostics is not False else None
    options_token = _OPTIONS.set(options)
    records_token = _RECORDS.set(None if collected is None else collected.records)
    try:
        yield collected
    finally:
        _RECORDS.reset(records_token)
        _OPTIONS.reset(options_token)


def current_solve_options() -> SolveOptions:
    """The options in force (the defaults outside any entry point)."""
    return _OPTIONS.get()


def collecting_diagnostics() -> bool:
    """Whether the solves now running should build `SolveRecord`s."""
    return _RECORDS.get() is not None


def spectrum_requested() -> bool:
    """Whether `diagnostics="spectrum"` is in force."""
    return collecting_diagnostics() and _OPTIONS.get().diagnostics == "spectrum"


def record_solve(record: SolveRecord) -> None:
    """Append `record` to the diagnostics being collected; nothing when none are."""
    records = _RECORDS.get()
    if records is not None:
        records.append(record)


def sparse_threshold_in_force() -> int:
    """The `sparse_threshold` in force, the package default when the caller gave none."""
    threshold = _OPTIONS.get().sparse_threshold
    if threshold is not None:
        return threshold
    # Imported here, not at the top: `junction_graph` imports this module. Read at call time, so
    # that a test can patch the constant.
    from foambryo import junction_graph

    return junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS


def use_sparse_solver(n_unknowns: int) -> bool:
    """Whether the system of `n_unknowns` unknowns takes the sparse path under the options in force."""
    solver = _OPTIONS.get().solver
    if solver == "dense":
        return False
    if solver == "sparse":
        return True
    return n_unknowns >= sparse_threshold_in_force()


def require_dense_only_method_is_not_forced_sparse(method: str) -> None:
    """Refuse `solver="sparse"` for a method that has no sparse path, rather than run it densely.

    Raises:
        ValueError: when `solver="sparse"` is in force.
    """
    if _OPTIONS.get().solver == "sparse":
        message = (
            f"{method} has no sparse solver: its system is solved densely whatever the size. Use "
            "solver='auto' or 'dense', or a method with a sparse path (YoungDupre and the other "
            "junction-line methods for tensions; Laplace, WeightedLaplace or Variational for pressures)."
        )
        raise ValueError(message)


def available_memory_bytes() -> int | None:
    """The memory a new allocation can use without swapping, as well as this platform tells.

    `psutil.virtual_memory().available` when `psutil` is installed (it is not a dependency); else
    `MemAvailable` from `/proc/meminfo` on Linux; else the total physical memory (`os.sysconf`),
    an upper bound, on platforms that expose no available-memory figure without `psutil` (macOS).
    `None` when nothing is readable.
    """
    try:
        import psutil  # optional, not a dependency

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def dense_solve_bytes(n_rows: int, n_cols: int, copies: int = _DENSE_COPIES) -> int:
    """Estimated bytes of a dense float64 least-squares solve of an `n_rows x n_cols` system."""
    return copies * _BYTES_PER_ENTRY * int(n_rows) * int(n_cols)


def warn_if_forced_dense_exceeds_memory(n_rows: int, n_cols: int, system: str, *,
                                        copies: int = _DENSE_COPIES, stacklevel: int = 3) -> None:
    """Emit `DenseSolveMemoryWarning` when `solver="dense"` is forced on a system too large for memory.

    Call it before allocating the dense matrix. Does nothing under `"auto"` or `"sparse"`.

    Args:
        n_rows: rows of the dense system.
        n_cols: columns of the dense system.
        system: what the system is, for the message.
        copies: how many dense copies of that shape the solve holds at once.
        stacklevel: passed to `warnings.warn`.
    """
    if _OPTIONS.get().solver != "dense":
        return
    needed = dense_solve_bytes(n_rows, n_cols, copies)
    available = available_memory_bytes()
    if available is None or needed <= available:
        return
    message = (
        f"solver='dense' on the {system} ({n_rows} x {n_cols}): the dense solve needs about "
        f"{needed / 2**30:.1f} GiB ({copies} copies of the matrix), more than the "
        f"{available / 2**30:.1f} GiB available. It will likely fail or swap; solver='auto' or "
        "'sparse' solves it sparsely."
    )
    warnings.warn(message, DenseSolveMemoryWarning, stacklevel=stacklevel)


def condition_from_singular_values(singular_values: NDArray[np.float64]) -> float | None:
    """2-norm condition number `s_max / s_min` (`inf` if `s_min` is 0, `None` if there are none)."""
    if singular_values is None or len(singular_values) == 0:
        return None
    smallest = float(np.min(singular_values))
    largest = float(np.max(singular_values))
    return float("inf") if smallest == 0.0 else largest / smallest


def extreme_singular_values(
    operator: sparse.spmatrix | sparse.sparray,
    *,
    symmetric_positive: bool = False,
) -> tuple[NDArray[np.float64] | None, float, str | None]:
    """The largest and smallest `SPECTRUM_EXTREME_VALUES` singular values of a sparse operator.

    The largest by `scipy.sparse.linalg.svds` (ARPACK, `which="LM"`). The smallest first by `svds`
    with `which="SM"`; if ARPACK does not converge within `SPECTRUM_MAX_ITERATIONS`, by shift-invert
    at zero (`eigsh(..., sigma=0)`, a sparse LU factorisation) of the operator itself when it is
    symmetric positive (`symmetric_positive`: its singular values are its eigenvalues) or of the
    normal matrix `A^T A` otherwise. The normal matrix squares the condition number, so those values
    are accurate only down to about `1e-8` of the largest; the note says which route gave them.
    Never raises on a failure: what is missing is said in the note.

    Returns:
        tuple: the values in decreasing order (`None` if none were obtained), the seconds spent, and
        a note (`None` when both ends came from `svds`).
    """
    start = time.perf_counter()
    k = min(SPECTRUM_EXTREME_VALUES, min(operator.shape) - 1)
    if k < 1:
        return None, time.perf_counter() - start, "system too small for svds"
    pieces = []
    notes = []
    for which in ("LM", "SM"):
        try:
            values = svds(operator, k=k, which=which, return_singular_vectors=False, random_state=0,
                          solver="arpack", maxiter=SPECTRUM_MAX_ITERATIONS)
            pieces.append(np.asarray(values, dtype=np.float64))
        except (ArpackNoConvergence, RuntimeError, ValueError) as error:
            notes.append(f"svds which={which} failed: {type(error).__name__}: {error}")
            if which == "SM":
                smallest, note = _smallest_by_shift_invert(operator, k, symmetric_positive=symmetric_positive)
                notes.append(note)
                if smallest is not None:
                    pieces.append(smallest)
    seconds = time.perf_counter() - start
    if not pieces:
        return None, seconds, "; ".join(notes)
    values = np.unique(np.concatenate(pieces))[::-1]
    return values, seconds, ("; ".join(notes) or None)


def _smallest_by_shift_invert(
    operator: sparse.spmatrix | sparse.sparray,
    k: int,
    *,
    symmetric_positive: bool,
) -> tuple[NDArray[np.float64] | None, str]:
    """The `k` smallest singular values by shift-invert at zero; see `extreme_singular_values`."""
    matrix = operator if symmetric_positive else (operator.T @ operator)
    try:
        # Shift-invert converges in a few restarts at the end it targets: ARPACK's default bound.
        values = eigsh(matrix.tocsc(), k=k, sigma=0, which="LM", return_eigenvectors=False)
    except (ArpackNoConvergence, RuntimeError, ValueError) as error:
        return None, f"shift-invert also failed: {type(error).__name__}: {error}"
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    if symmetric_positive:
        return values, "smallest values by shift-invert of the (symmetric positive) operator"
    return np.sqrt(values), "smallest values by shift-invert of A^T A (accurate to about 1e-8 of the largest)"


def record_dense_lstsq(
    system: str,
    solver: str,
    dense: NDArray[np.float64],
    vector: NDArray[np.float64],
    x: NDArray[np.float64],
    residuals: NDArray[np.float64],
    rank: int,
    singular_values: NDArray[np.float64],
    start: float,
    *,
    lsqr_istop: int | None = None,
    lsqr_iterations: int | None = None,
) -> dict[str, Any]:
    """Append the `SolveRecord` of a dense `scipy.linalg.lstsq` solve; return its scalar summary.

    Reads the solve's own outputs and never alters `x`. The residual norm is the square root of the
    residual `lstsq` returned (a 0-d value for a 1-D right-hand side, shape `(1,)` from the sparse
    path) when it is a finite number, else (rank-deficient system) computed as
    `||vector - dense @ x||`.

    Args:
        system: which system, see `SolveRecord`.
        solver: which solver, see `SolveRecord`.
        dense: the matrix solved.
        vector: the right-hand side.
        x: `lstsq`'s solution.
        residuals: `lstsq`'s residuals.
        rank: `lstsq`'s rank.
        singular_values: `lstsq`'s singular values.
        start: `time.perf_counter()` taken before the solve.
        lsqr_istop: `lsqr`'s stop code, when this dense solve is a fallback after `lsqr`.
        lsqr_iterations: `lsqr`'s iterations, likewise.

    Returns:
        dict: `n_rows`, `rank`, `residual_norm`, `condition_number`, `condition_number_source`,
        `seconds`.
    """
    # `lstsq` returns a 0-d value for a 1-D right-hand side, and `nan` (or nothing) when the system
    # is rank-deficient: then the norm is computed here.
    returned_residuals = np.ravel(residuals)
    if returned_residuals.size and np.isfinite(returned_residuals[0]):
        residual_norm = float(np.sqrt(returned_residuals[0]))
    else:
        difference = vector - dense @ x
        residual_norm = float(np.sqrt(difference @ difference))
    condition = condition_from_singular_values(singular_values)
    seconds = time.perf_counter() - start
    source = "singular values of dense lstsq (2-norm)"
    record_solve(SolveRecord(
        system=system,
        solver=solver,
        n_rows=int(dense.shape[0]),
        n_unknowns=int(dense.shape[1]),
        rank=int(rank),
        residual_norm=residual_norm,
        condition_number=condition,
        condition_number_source=source,
        seconds=seconds,
        lsqr_istop=lsqr_istop,
        lsqr_iterations=lsqr_iterations,
        singular_values=np.asarray(singular_values, dtype=np.float64).copy(),
    ))
    return {
        "n_rows": int(dense.shape[0]),
        "rank": int(rank),
        "residual_norm": residual_norm,
        "condition_number": condition,
        "condition_number_source": source,
        "seconds": seconds,
    }
