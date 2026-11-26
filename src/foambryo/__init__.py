"""Foambryo entry point: relative tensions and pressures of 3D and 2D cell assemblies.

The package-level functions take a 3-D mesh (`DcelData`, triangles) or a 2-D one (`PlanarMesh`, segments; also dw2d's
`DCEL_Data`) and dispatch on it (`foambryo.planar_mesh.dimension_of`); the mask functions dispatch on the mask's
`ndim`. 2-D support is provisional: its names and signatures may change until declared stable. Please cite, for the
3D method, Ichbiah S. et al., Nature Methods 20(12), 1989-1999 (2023), doi:10.1038/s41592-023-02084-7, and for the 2D
inference, Yamamoto K., Ichbiah S., Perez M., Borrego-Pinto J., Delbary F., Goehring N. W., Northrop P., Turlier H.,
Charras G., Developmental Cell 61(6), 1273-1288.e4 (2026), doi:10.1016/j.devcel.2026.04.013.

Matthieu Perez 2024
"""

from foambryo._two_dimensional import ANGLE_READINGS as ANGLE_READINGS_2D
from foambryo._two_dimensional import DEFAULT_ANGLE_READING as DEFAULT_ANGLE_READING_2D
from foambryo._two_dimensional import (
    EQUILIBRIUM_TILING_MAX_EVALUATIONS,
    MAX_LENS_CONDITION,
    EquilibriumTilingReport,
    IllConditionedLensWarning,
    InterfaceArc,
    NoJunctionError,
    TwoDimensionalPressureReport,
    TwoDimensionalReport,
    image_border_vertices,
    infer_forces_by_equilibrium_tiling,
    interface_arcs,
    junction_table,
)
from foambryo.crop_faces import CutCellPressureWarning, UnrecordedCropFacesWarning
from foambryo.force_balance_inference import (
    ForceBalanceNormalisation,
    ForceBalanceResult,
    infer_force_balance,
)
from foambryo.force_inference import InferenceOutcome, infer_forces, infer_tensions_then_pressures
from foambryo.io import (
    MaskReconstruction,
    dcel_mesh_from_file,
    dcel_mesh_from_multimaterial_mesh,
    dcel_mesh_from_segmentation_mask,
    reconstruct_mask,
)
from foambryo.junction_graph import (
    ComponentTensionSolution,
    JunctionGraphGaugeWarning,
    JunctionGraphPartition,
    MalformedTripleLineError,
    SparseSolveFallbackWarning,
    partition_by_junction_graph,
    require_single_component_solution,
)
from foambryo.linear_solves import (
    DenseSolveMemoryWarning,
    SolveDiagnostics,
    SolveRecord,
    save_solve_diagnostics,
)
from foambryo.per_edge_tension_inference import (
    PerEdgeResidualField,
    PerEdgeWeighting,
    YoungDupreLocalResult,
    infer_tensions_young_dupre_local,
)
from foambryo.planar_mesh import PlanarMesh, dimension_of
from foambryo.pressure_inference import (
    PressureComputationMethod,
    UndeterminedInferenceError,
    UndeterminedPressureWarning,
    UndeterminedTensionError,
    infer_pressures,
    infer_pressures_and_residuals,
)
from foambryo.second_order_angle_reading import (
    ANGLE_READINGS,
    CHORD_READING,
    ORIENTED_CHORD_READING,
    ORIENTED_QUADRIC_READING,
    QUADRIC_READING,
    AngleReadingCoverageWarning,
    AngleReadingReport,
    quadric_angle_rotations,
)
from foambryo.seed_ensembles import (
    DEFAULT_ENSEMBLE_SEEDS,
    SeedEnsembleExperimentalWarning,
    SeedEnsembleResult,
    SeedReconstruction,
    infer_tensions_over_seeds,
)
from foambryo.tension_inference import (
    ANGLE_READING_METHODS,
    DegenerateWedgeWarning,
    TensionComputationMethod,
    VariationalRoute,
    infer_tensions,
    infer_tensions_and_residuals,
    infer_tensions_and_residuals_with_report,
    infer_tensions_by_junction_component,
)
from foambryo.uncertainty import (
    InferenceDiagnostics,
    TensionUncertaintyResult,
    UncertaintyModel,
    compute_inference_diagnostics,
    infer_tensions_with_uncertainty,
)
from foambryo.wedge_refusal import (
    NO_REFUSAL,
    REFLEX_TOLERANCE_DEG,
    RefusalReport,
    WedgeRefusal,
    WedgeRefusalPolicy,
    reflex_trijunction_edges,
    refusal_report,
)

__all__ = (
    "ANGLE_READINGS",
    "ANGLE_READINGS_2D",
    "ANGLE_READING_METHODS",
    "CHORD_READING",
    "DEFAULT_ANGLE_READING_2D",
    "DEFAULT_ENSEMBLE_SEEDS",
    "EQUILIBRIUM_TILING_MAX_EVALUATIONS",
    "MAX_LENS_CONDITION",
    "NO_REFUSAL",
    "ORIENTED_CHORD_READING",
    "ORIENTED_QUADRIC_READING",
    "QUADRIC_READING",
    "REFLEX_TOLERANCE_DEG",
    "AngleReadingCoverageWarning",
    "AngleReadingReport",
    "ComponentTensionSolution",
    "CutCellPressureWarning",
    "DegenerateWedgeWarning",
    "DenseSolveMemoryWarning",
    "EquilibriumTilingReport",
    "ForceBalanceNormalisation",
    "ForceBalanceResult",
    "IllConditionedLensWarning",
    "InferenceDiagnostics",
    "InferenceOutcome",
    "InterfaceArc",
    "JunctionGraphGaugeWarning",
    "JunctionGraphPartition",
    "MalformedTripleLineError",
    "MaskReconstruction",
    "NoJunctionError",
    "PerEdgeResidualField",
    "PerEdgeWeighting",
    "PlanarMesh",
    "PressureComputationMethod",
    "RefusalReport",
    "SeedEnsembleExperimentalWarning",
    "SeedEnsembleResult",
    "SeedReconstruction",
    "SolveDiagnostics",
    "SolveRecord",
    "SparseSolveFallbackWarning",
    "TensionComputationMethod",
    "TensionUncertaintyResult",
    "TwoDimensionalPressureReport",
    "TwoDimensionalReport",
    "UncertaintyModel",
    "UndeterminedInferenceError",
    "UndeterminedPressureWarning",
    "UndeterminedTensionError",
    "UnrecordedCropFacesWarning",
    "VariationalRoute",
    "WedgeRefusal",
    "WedgeRefusalPolicy",
    "YoungDupreLocalResult",
    "compute_inference_diagnostics",
    "dcel_mesh_from_file",
    "dcel_mesh_from_multimaterial_mesh",
    "dcel_mesh_from_segmentation_mask",
    "dimension_of",
    "image_border_vertices",
    "infer_force_balance",
    "infer_forces",
    "infer_forces_by_equilibrium_tiling",
    "infer_pressures",
    "infer_pressures_and_residuals",
    "infer_tensions",
    "infer_tensions_and_residuals",
    "infer_tensions_and_residuals_with_report",
    "infer_tensions_by_junction_component",
    "infer_tensions_over_seeds",
    "infer_tensions_then_pressures",
    "infer_tensions_with_uncertainty",
    "infer_tensions_young_dupre_local",
    "interface_arcs",
    "junction_table",
    "partition_by_junction_graph",
    "quadric_angle_rotations",
    "reconstruct_mask",
    "reflex_trijunction_edges",
    "refusal_report",
    "require_single_component_solution",
    "save_solve_diagnostics",
)
