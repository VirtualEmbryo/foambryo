"""Tensions and pressures in one call: `infer_forces`, and `infer_tensions_then_pressures` for front ends.

`infer_forces` runs `infer_tensions` then `infer_pressures` and raises when either refuses;
`infer_tensions_then_pressures` makes the same two calls, keeps the tensions when the pressure solve
refuses them, and returns the refusal as an `InferenceOutcome` instead of raising it.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from foambryo.linear_solves import SolveDiagnostics, validate_solve_options
from foambryo.pressure_inference import PressureComputationMethod, infer_pressures
from foambryo.tension_inference import TensionComputationMethod, infer_tensions

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


def infer_forces(
    mesh: "DcelData",
    mean_tension: float = 1,
    base_pressure: float = 0,
    mode_tension: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    mode_pressure: PressureComputationMethod = PressureComputationMethod.Variational,
    *,
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> tuple:
    """Infer the relative tensions, then the pressures, of a 3D mesh.

    `infer_tensions(mesh, mean_tension, mode_tension, ...)` then `infer_pressures(mesh, tensions,
    mode_pressure, base_pressure, ...)`, with the same solver options; see those two functions for
    every convention, warning and failure.

    Args:
        mesh (DcelData): Mesh to analyze, in any length unit.
        mean_tension (float, optional): The normalisation: the tensions are normalised so that
            their mean is `mean_tension`, exactly, in each connected piece of the junction graph
            (see `foambryo.infer_tensions`). Defaults to 1.
        base_pressure (float, optional): Pressure of the exterior medium (label 0). Defaults to 0.
        mode_tension (TensionComputationMethod, optional): Method for tension computation.
            Defaults to `TensionComputationMethod.YoungDupre`.
        mode_pressure (PressureComputationMethod, optional): Method for pressure computation.
            Defaults to `PressureComputationMethod.Variational`.
        solver (str, optional): `"auto"` (the default), `"dense"` or `"sparse"`, passed to both the
            tension and the pressure solves (`foambryo.linear_solves`). `"auto"` is dense below
            `sparse_threshold` unknowns and sparse at or above, and bitwise the dense result below
            it. `"sparse"` raises `ValueError` for a tension method with no sparse path
            (`Equilibrium`, `Variational`, `ForceBalance*`). Keyword-only.
        sparse_threshold (int | None, optional): unknowns at which `"auto"` goes sparse; `None`
            (the default) is `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS` (2 000).
            Keyword-only.
        diagnostics (bool | str, optional): `False` (the default), `True` or `"spectrum"`: also
            return a `foambryo.linear_solves.SolveDiagnostics` holding the records of the tension
            solves then of the pressure solve. Keyword-only.

    Returns:
        tuple[dict[tuple[int, int], float], dict[int, float]]:
            - map of interface `(label_1, label_2)`, `label_1 < label_2` -> relative tension on
              this interface. An undetermined interface (for example one with no triple line at
              all) is `float("nan")`; a triple-junction graph in several pieces gives tensions
              normalised separately in each piece, which are not on the same scale (see
              `foambryo.tension_inference.infer_tensions`'s return value, and
              `foambryo.tension_inference.infer_tensions_by_junction_component` for the
              per-piece alternative).
            - map of cell label (0 = exterior, at `base_pressure`) -> pressure, in the unit of the
              tensions divided by the mesh's length unit; `nan` for a cell whose pressure is
              undetermined or that the crop cuts.
            - with `diagnostics` on, a third value: the `SolveDiagnostics`.

    Raises:
        UndeterminedTensionError: on an uncropped mesh, if any inferred tension is not finite,
            which is the case on a mesh with an interface that has no triple line. The pressure
            solve couples every cell to every tension, so such a tension would make nearly every
            pressure non-finite; pressures are refused instead. Call `infer_tensions` for the
            tensions alone (or `infer_tensions_then_pressures`, which keeps them), and
            `infer_tensions_by_junction_component` to see which interfaces are undetermined.
        ValueError: on an invalid solver option; as `infer_tensions` for the tension method; on a
            cropped mesh with a Laplace pressure method.
        MalformedTripleLineError: as `infer_tensions`.
        numpy.linalg.LinAlgError: as `infer_pressures`.

    Warns:
        JunctionGraphGaugeWarning: as `infer_tensions`.
        DegenerateWedgeWarning: as `infer_tensions`.
        UndeterminedPressureWarning: as `infer_pressures`.
        CutCellPressureWarning: as `infer_pressures`, on a cropped mesh.
        SparseSolveFallbackWarning: as `infer_tensions` and `infer_pressures`.
        DenseSolveMemoryWarning: as `infer_tensions` and `infer_pressures`.
    """
    options = validate_solve_options(solver, sparse_threshold, diagnostics)
    if diagnostics is False:
        dict_tensions = infer_tensions(
            mesh, mean_tension=mean_tension, mode=mode_tension, solver=solver, sparse_threshold=sparse_threshold,
        )
        dict_pressures = infer_pressures(
            mesh,
            dict_tensions,
            mode=mode_pressure,
            base_pressure=base_pressure,
            solver=solver,
            sparse_threshold=sparse_threshold,
        )
        return (dict_tensions, dict_pressures)
    dict_tensions, tension_diagnostics = infer_tensions(
        mesh, mean_tension=mean_tension, mode=mode_tension,
        solver=solver, sparse_threshold=sparse_threshold, diagnostics=diagnostics,
    )
    dict_pressures, pressure_diagnostics = infer_pressures(
        mesh, dict_tensions, mode=mode_pressure, base_pressure=base_pressure,
        solver=solver, sparse_threshold=sparse_threshold, diagnostics=diagnostics,
    )
    combined = SolveDiagnostics(options, [*tension_diagnostics.records, *pressure_diagnostics.records])
    return (dict_tensions, dict_pressures, combined)


#: `InferenceOutcome.status` values.
INFERENCE_COMPLETE = "complete"
INFERENCE_PRESSURES_REFUSED = "pressures refused"
INFERENCE_FAILED = "failed"


@dataclass(frozen=True)
class InferenceOutcome:
    """What `infer_tensions_then_pressures` computed, and what the library refused.

    Attributes:
        tensions: interface -> tension, or `None` when the tension solve was refused.
        pressures: material -> pressure, or `None` when the pressure solve was refused, never ran, or was not asked
            for (`with_pressures=False`).
        records: the solve records (`foambryo.linear_solves.SolveRecord`), tensions first; empty without
            diagnostics.
        status: `"complete"`, `"pressures refused"` (the tensions hold an undetermined value, so the pressure solve
            refused them) or `"failed"` (the library refused the options or the mesh).
        error: the library's message when the status is not `"complete"`, else `None`.
        library_call: the `infer_forces` call that computes the same numbers, as text.
    """

    tensions: dict[tuple[int, int], float] | None
    pressures: dict[int, float] | None
    records: list = field(default_factory=list)
    status: str = INFERENCE_COMPLETE
    error: str | None = None
    library_call: str = ""


def infer_tensions_then_pressures(
    mesh: "DcelData",
    mean_tension: float = 1,
    base_pressure: float = 0,
    mode_tension: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    mode_pressure: PressureComputationMethod = PressureComputationMethod.Variational,
    *,
    solver: str = "auto",
    diagnostics: bool | str = False,
    with_pressures: bool = True,
) -> InferenceOutcome:
    """The tensions then the pressures, by the two calls `infer_forces` makes, keeping what succeeds.

    `infer_forces` runs `infer_tensions` then `infer_pressures` with these arguments and raises when either is
    refused. This function makes the same two calls itself, so that the tensions survive a pressure solve that
    refuses them (`UndeterminedTensionError`), and returns the refusal instead of raising it: what the `foambryo
    infer` command and the napari plugin need to report a partial result. The numbers are bitwise those of
    `infer_forces` with the same arguments. Every default is `infer_forces`'s.

    With `with_pressures=False` only the first call is made (`infer_tensions` with the same arguments, so the
    tensions are bitwise those of `infer_forces`), `base_pressure` and `mode_pressure` are not used, the outcome's
    `pressures` is `None` with the status `"complete"`, and `library_call` is that `infer_tensions` call.

    `sparse_threshold` is not an argument: the default threshold (2 000 unknowns) always applies.

    Args:
        mesh (DcelData): the mesh, in any length unit.
        mean_tension (float, optional): the normalisation: the tensions are normalised so that their mean is
            `mean_tension`, exactly, in each connected piece of the junction graph. Defaults to 1.
        base_pressure (float, optional): the pressure of the exterior medium, label 0. Defaults to 0.
        mode_tension (TensionComputationMethod, optional): the tension method. Defaults to `YoungDupre`.
        mode_pressure (PressureComputationMethod, optional): the pressure method. Defaults to `Variational`.
        solver (str, optional): `"auto"` (the default), `"dense"` or `"sparse"`, as for `infer_forces`.
            Keyword-only.
        diagnostics (bool | str, optional): `False` (the default), `True` or `"spectrum"`, as for `infer_forces`;
            the solve records are returned in `InferenceOutcome.records`. Keyword-only.
        with_pressures (bool, optional): `False` infers the tensions only (no pressure solve). Defaults to `True`.
            Keyword-only.

    Returns:
        InferenceOutcome: the tensions and pressures computed, the solve records, the status and the library call.
            A `ValueError` or `MalformedTripleLineError` from either call is returned as the status `"failed"`
            with its message (the tensions kept when they were computed), an `UndeterminedTensionError` from the
            pressure call as `"pressures refused"`; nothing of these is raised.

    Raises:
        numpy.linalg.LinAlgError: if the variational pressure system is singular (not caught).

    Warns:
        UserWarning: every warning `infer_tensions` and `infer_pressures` emit, unchanged.
    """
    from foambryo.junction_graph import MalformedTripleLineError
    from foambryo.pressure_inference import UndeterminedTensionError

    if with_pressures:
        library_call = (
            f"infer_forces(mesh, mean_tension={mean_tension!r}, base_pressure={base_pressure!r}, "
            f"mode_tension=TensionComputationMethod.{mode_tension.name}, "
            f"mode_pressure=PressureComputationMethod.{mode_pressure.name}, "
            f"solver={solver!r}, diagnostics={diagnostics!r})"
        )
    else:
        library_call = (
            f"infer_tensions(mesh, mean_tension={mean_tension!r}, mode=TensionComputationMethod.{mode_tension.name}, "
            f"solver={solver!r}, diagnostics={diagnostics!r})"
        )
    solve: dict[str, Any] = {"solver": solver, "sparse_threshold": None}
    if diagnostics is not False:
        solve["diagnostics"] = diagnostics
    tensions: dict[tuple[int, int], float] | None = None
    pressures: dict[int, float] | None = None
    records: list = []
    try:
        validate_solve_options(solver, None, diagnostics)
        result = infer_tensions(mesh, mean_tension=mean_tension, mode=mode_tension, **solve)
        if diagnostics is False:
            tensions = cast("dict[tuple[int, int], float]", result)
        else:
            tensions, tension_records = cast("tuple[dict[tuple[int, int], float], SolveDiagnostics]", result)
            records += tension_records.records
        if not with_pressures:
            return InferenceOutcome(tensions, None, records, INFERENCE_COMPLETE, None, library_call)
        try:
            result = infer_pressures(mesh, tensions, mode=mode_pressure, base_pressure=base_pressure, **solve)
        except UndeterminedTensionError as error:
            return InferenceOutcome(tensions, None, records, INFERENCE_PRESSURES_REFUSED, str(error), library_call)
        if diagnostics is False:
            pressures = cast("dict[int, float]", result)
        else:
            pressures, pressure_records = cast("tuple[dict[int, float], SolveDiagnostics]", result)
            records += pressure_records.records
    except (ValueError, MalformedTripleLineError) as error:
        return InferenceOutcome(tensions, pressures, records, INFERENCE_FAILED, str(error), library_call)
    return InferenceOutcome(tensions, pressures, records, INFERENCE_COMPLETE, None, library_call)
