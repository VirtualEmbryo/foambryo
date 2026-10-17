"""Validation and benchmarking tools: the unified force-balance operator and mesh checks, on raw mesh arrays.

The unified force-balance operator (the gradient of the capillary energy with respect to the mesh vertices, one
column per unknown), its solvers and diagnostics, so that it can be tested and benchmarked within foambryo; the
`ForceBalance*` tension methods are built on it. The test suite compares these functions with an independent
reference implementation of the same algorithms (`tests/reference/force_balance_reference_implementation.py`).
These functions check and benchmark an inference; they are not an inference method of their own.

A mesh here is three arrays, `(V, F, regions)`: vertices `(nv, 3)`, triangles `(nf, 3)` (vertex indices) and the
two regions each triangle separates, `(nf, 2)`, the medium being region 0 by default
(`foambryo.validation.mesh_io`); `foambryo.validation.force_balance` builds and solves the operator on them.
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
