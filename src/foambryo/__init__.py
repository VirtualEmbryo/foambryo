"""Foambryo entry point.

Matthieu Perez 2024
"""

from foambryo.force_balance_inference import (
    ForceBalanceNormalisation,
    ForceBalanceResult,
    infer_force_balance,
)
from foambryo.force_inference import infer_forces
from foambryo.io import dcel_mesh_from_file, dcel_mesh_from_multimaterial_mesh, dcel_mesh_from_segmentation_mask
from foambryo.junction_graph import (
    ComponentTensionSolution,
    JunctionGraphGaugeWarning,
    JunctionGraphPartition,
    MalformedTripleLineError,
    partition_by_junction_graph,
    require_single_component_solution,
)
from foambryo.per_edge_tension_inference import (
    PerEdgeResidualField,
    PerEdgeWeighting,
    YoungDupreLocalResult,
    infer_tensions_young_dupre_local,
)
from foambryo.pressure_inference import (
    PressureComputationMethod,
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
from foambryo.tension_inference import (
    ANGLE_READING_METHODS,
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
    "ANGLE_READING_METHODS",
    "CHORD_READING",
    "NO_REFUSAL",
    "ORIENTED_CHORD_READING",
    "ORIENTED_QUADRIC_READING",
    "QUADRIC_READING",
    "REFLEX_TOLERANCE_DEG",
    "AngleReadingCoverageWarning",
    "AngleReadingReport",
    "ComponentTensionSolution",
    "ForceBalanceNormalisation",
    "ForceBalanceResult",
    "InferenceDiagnostics",
    "JunctionGraphGaugeWarning",
    "JunctionGraphPartition",
    "MalformedTripleLineError",
    "PerEdgeResidualField",
    "PerEdgeWeighting",
    "PressureComputationMethod",
    "RefusalReport",
    "TensionComputationMethod",
    "TensionUncertaintyResult",
    "UncertaintyModel",
    "UndeterminedTensionError",
    "VariationalRoute",
    "WedgeRefusal",
    "WedgeRefusalPolicy",
    "YoungDupreLocalResult",
    "compute_inference_diagnostics",
    "dcel_mesh_from_file",
    "dcel_mesh_from_multimaterial_mesh",
    "dcel_mesh_from_segmentation_mask",
    "infer_force_balance",
    "infer_forces",
    "infer_pressures",
    "infer_pressures_and_residuals",
    "infer_tensions",
    "infer_tensions_and_residuals",
    "infer_tensions_and_residuals_with_report",
    "infer_tensions_by_junction_component",
    "infer_tensions_with_uncertainty",
    "infer_tensions_young_dupre_local",
    "partition_by_junction_graph",
    "quadric_angle_rotations",
    "reflex_trijunction_edges",
    "refusal_report",
    "require_single_component_solution",
)
