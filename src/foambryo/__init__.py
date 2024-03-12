"""Foambryo entry point.

Matthieu Perez 2024
"""

from foambryo.dcel import DcelData
from foambryo.force_inference import infer_forces
from foambryo.force_viewer import (
    plot_force_inference,
    plot_residual_junctions,
    plot_tension_inference,
    plot_valid_junctions,
)
from foambryo.io import dcel_mesh_from_file, dcel_mesh_from_multimaterial_mesh, dcel_mesh_from_segmentation_mask
from foambryo.pressure_inference import PressureComputationMethod, infer_pressures, infer_pressures_and_residuals
from foambryo.tension_inference import (
    TensionComputationMethod,
    infer_tensions,
    infer_tensions_and_residuals,
)

__all__ = (
    "DcelData",
    "plot_force_inference",
    "plot_tension_inference",
    "plot_residual_junctions",
    "plot_valid_junctions",
    "dcel_mesh_from_file",
    "dcel_mesh_from_multimaterial_mesh",
    "dcel_mesh_from_segmentation_mask",
    "infer_forces",
    "infer_tensions",
    "infer_tensions_and_residuals",
    "TensionComputationMethod",
    "infer_pressures",
    "infer_pressures_and_residuals",
    "PressureComputationMethod",
)
