"""Main module to easily compute both tensions and pressures.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from typing import TYPE_CHECKING

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
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Infer tensions and pressures from a 3D mesh.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        base_pressure (float, optional): Expected base pressure for exterior. Defaults to 0.
        mode_tension (TensionComputationMethod, optional): Method for tension computation. Defaults to "Young-Dupré".
        mode_pressure (PressureComputationMethod, optional): Method for pressure computation. Defaults to "Variational".

    Returns:
        tuple[dict[tuple[int, int], float], dict[int, float]]:
            - map of interface id (label 1, label 2) -> tensions on this interface.
            - map of cell id (0 = exterior) -> pressure in this cell.
    """
    dict_tensions = infer_tensions(mesh, mean_tension=mean_tension, mode=mode_tension)
    dict_pressures = infer_pressures(
        mesh,
        dict_tensions,
        mode=mode_pressure,
        base_pressure=base_pressure,
    )
    return (dict_tensions, dict_pressures)
