"""Validation and benchmarking infrastructure for foambryo.

Ported (algorithm unchanged) from the reference implementation that accompanies the
technical note on tension inference (H. Turlier, in preparation; not distributed), so the
unified force-balance operator can be tested and benchmarked from within foambryo instead of
as a standalone script.
This subpackage introduces no new inference method and changes no existing public API.
"""

from foambryo.validation.force_balance import (
    block_anchors,
    constrained_covariance,
    effective_surface_tensions,
    effective_tensions,
    force_balance_operator,
    map_estimate,
    marginalise_pressures,
    residual_at,
    residual_force_field,
    residual_noise_scale,
    singular_values,
    soft_mode,
    solve_constrained,
    split,
)
from foambryo.validation.mesh_io import (
    cell_labels,
    check_label_orientation_convention,
    is_intact,
    list_interfaces,
    list_junctions,
    mean_edge_length,
    mesh_integrity,
    read_arec,
    read_mesh,
    read_rec,
    require_label_orientation_convention,
    signed_cell_volumes,
)

__all__ = (
    "block_anchors",
    "cell_labels",
    "check_label_orientation_convention",
    "constrained_covariance",
    "effective_surface_tensions",
    "effective_tensions",
    "force_balance_operator",
    "is_intact",
    "list_interfaces",
    "list_junctions",
    "map_estimate",
    "marginalise_pressures",
    "mean_edge_length",
    "mesh_integrity",
    "read_arec",
    "read_mesh",
    "read_rec",
    "require_label_orientation_convention",
    "residual_at",
    "residual_force_field",
    "residual_noise_scale",
    "signed_cell_volumes",
    "singular_values",
    "soft_mode",
    "solve_constrained",
    "split",
)
