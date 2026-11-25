"""The 2-D branch of foambryo's package-level inference functions, and the checks of the options that belong to one
dimension only.

`foambryo.infer_tensions`, `infer_tensions_by_junction_component`, `infer_pressures`, `infer_forces` and
`infer_tensions_then_pressures` keep one signature for both dimensions: the 3-D positional order, per-dimension
defaults written `None` (`angle_reading`: `"chord"` in 3D, `"arc"` in 2D; the pressure `mode`: `Variational` in 3D,
`Laplace` in 2D), and keywords that exist in one dimension only (`refusal` in 3D; `exclude_image_border` and
`excluded_vertices` in 2D), refused with `ValueError` in the other when given a value other than their default. Each
function asks `foambryo.planar_mesh.dimension_of(mesh)` first and, for a 2-D mesh, calls the function of this module,
which maps the arguments onto the 2-D implementation (`foambryo._two_dimensional`).
"""

from __future__ import annotations

import warnings
from collections.abc import Collection
from typing import Any

from foambryo.linear_solves import SolveDiagnostics, validate_solve_options
from foambryo.pressure_inference import PressureComputationMethod
from foambryo.tension_inference import TensionComputationMethod
from foambryo.wedge_refusal import NO_REFUSAL


def refuse_two_dimensional_options(function: str, exclude_image_border: bool, excluded_vertices: Collection[int]) -> None:
    """Refuse, on a 3-D mesh, a non-default value of the keywords that exist in 2D only."""
    given = []
    if exclude_image_border is not True:
        given.append(f"exclude_image_border={exclude_image_border!r}")
    if len(tuple(excluded_vertices)):
        given.append("excluded_vertices")
    if given:
        error = f"{function}: {', '.join(given)} applies to 2-D meshes only (the mesh is 3-D)"
        raise ValueError(error)


def _refuse_three_dimensional_options(function: str, refusal: object) -> None:
    if refusal is not NO_REFUSAL:
        error = f"{function}: refusal applies to 3-D meshes only (the mesh is 2-D)"
        raise ValueError(error)


def _two_dimensional_angle_reading(angle_reading: str | None) -> str:
    from foambryo._two_dimensional import DEFAULT_ANGLE_READING

    return DEFAULT_ANGLE_READING if angle_reading is None else angle_reading


def two_dimensional_pressure_mode(mode: PressureComputationMethod | None) -> PressureComputationMethod:
    """`Laplace`, the one pressure method offered in 2D (Laplace's law on the circle read on each interface)."""
    if mode not in (None, PressureComputationMethod.Laplace):
        error = (
            f"pressure mode {mode!r} is not offered in 2D: the 2-D pressures are Laplace's law on the circle read on "
            "each interface (PressureComputationMethod.Laplace)"
        )
        raise ValueError(error)
    return PressureComputationMethod.Laplace


def tensions(
    function: str,
    mesh: object,
    mean_tension: float,
    mode: TensionComputationMethod,
    *,
    angle_reading: str | None,
    refusal: object,
    exclude_image_border: bool,
    excluded_vertices: Collection[int],
    solver: str,
    sparse_threshold: int | None,
    diagnostics: bool | str,
) -> Any:  # noqa: ANN401 -- what the 2-D function returns
    """`infer_tensions` or `infer_tensions_by_junction_component` (`function`) on a 2-D mesh."""
    from foambryo import _two_dimensional

    _refuse_three_dimensional_options(function, refusal)
    return getattr(_two_dimensional, function)(
        mesh,
        mean_tension,
        mode,
        angle_reading=_two_dimensional_angle_reading(angle_reading),
        exclude_image_border=exclude_image_border,
        excluded_vertices=excluded_vertices,
        solver=solver,
        sparse_threshold=sparse_threshold,
        diagnostics=diagnostics,
    )


def pressures(
    mesh: object,
    dict_tensions: object,
    mode: PressureComputationMethod | None,
    base_pressure: float,
    *,
    exclude_image_border: bool,
    excluded_vertices: Collection[int],
    solver: str,
    sparse_threshold: int | None,
    diagnostics: bool | str,
) -> Any:  # noqa: ANN401 -- what the 2-D function returns
    """`infer_pressures` on a 2-D mesh."""
    from foambryo import _two_dimensional

    two_dimensional_pressure_mode(mode)
    return _two_dimensional.infer_pressures(
        mesh,
        dict_tensions,
        base_pressure,
        exclude_image_border=exclude_image_border,
        excluded_vertices=excluded_vertices,
        solver=solver,
        sparse_threshold=sparse_threshold,
        diagnostics=diagnostics,
    )


def forces(
    mesh: object,
    mean_tension: float,
    base_pressure: float,
    mode_tension: TensionComputationMethod,
    mode_pressure: PressureComputationMethod | None,
    *,
    solver: str,
    sparse_threshold: int | None,
    diagnostics: bool | str,
    angle_reading: str | None = None,
) -> tuple:
    """`infer_forces` on a 2-D mesh: the tensions of each junction-graph component, gauged alone, then the pressures
    of each component on its own tensions' scale (`infer_pressures` given the per-component solution).

    Returns `(tensions, pressures)` as in 3D (the flat map of `ComponentTensionSolution.dict_tensions`), and with
    `diagnostics` a third value, one `SolveDiagnostics` holding the records of the tension then the pressure solves.
    Warns `JunctionGraphGaugeWarning`, as 3D's `infer_forces` does, when the components are several or an interface
    is undetermined.
    """
    from foambryo import _two_dimensional
    from foambryo.junction_graph import JunctionGraphGaugeWarning

    options = validate_solve_options(solver, sparse_threshold, diagnostics)
    two_dimensional_pressure_mode(mode_pressure)
    solve = {"solver": solver, "sparse_threshold": sparse_threshold, "diagnostics": diagnostics}
    result = _two_dimensional.infer_tensions_by_junction_component(
        mesh, mean_tension, mode_tension, angle_reading=_two_dimensional_angle_reading(angle_reading), **solve,
    )
    solution, tension_report = (result, None) if diagnostics is False else result
    if solution.fragmented or solution.undetermined:
        warnings.warn(
            f"the junction graph has {solution.n_components} component(s) and {len(solution.undetermined)} "
            "undetermined interface(s): each component's tensions are gauged alone (mean_tension each), and each "
            "component's pressures are solved on its own tensions' scale; never compare values across components "
            "(infer_tensions_by_junction_component gives each interface's component).",
            JunctionGraphGaugeWarning,
            stacklevel=3,
        )
    result = _two_dimensional.infer_pressures(mesh, solution, base_pressure, **solve)
    if diagnostics is False:
        return solution.dict_tensions, result
    dict_pressures, pressure_report = result
    combined = SolveDiagnostics(options, [*tension_report.solves.records, *pressure_report.solves.records])
    return solution.dict_tensions, dict_pressures, combined
