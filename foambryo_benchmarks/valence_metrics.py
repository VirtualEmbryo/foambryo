"""Edge-valence measurements on a multimaterial mesh (the converged-mesh-cohort work).

"Valence" here is **edge valence**: the number of triangles incident to an (undirected) mesh
edge -- the quantity the `dw3d` benchmarks report as "valence-≥4 edges" on `dw3d` output. This is not the same axis as
`foambryo.validation.mesh_io.mesh_integrity`'s `n_quadjunction_edges`, which counts edges
touching **≥4 distinct region labels** -- a plain interface edge with 4 incident
triangles that all carry the *same* label pair (a film pinched against itself: found among
`dw3d`'s residual defects, and on none of the 72 converged simulator meshes) has `triangles_per_edge == 4` but
`regions_per_edge == 2`, so it is a valence-4 edge that `mesh_integrity` would not flag as a
quadjunction. Both axes are reported here, independently, precisely to tell the two apart.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from foambryo.validation.mesh_io import _count_distinct_per_edge, _undirected_edges


def edge_valence_report(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> dict:
    """Count valence-≥4 edges and 2-material valence-4 edges.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        dict: `{"n_edges", "valence_histogram", "n_valence_ge4", "n_two_material_valence4",
            "two_material_valence4_region_pairs", "total_length_valence_ge4"}`.
    """
    edges, occ_to_edge = _undirected_edges(f)
    n_edges = len(edges)
    triangles_per_edge = np.bincount(occ_to_edge, minlength=n_edges)

    occ_face = np.tile(np.arange(len(f)), 3)
    regions_per_edge = _count_distinct_per_edge(
        np.concatenate([occ_to_edge, occ_to_edge]),
        np.concatenate([regions[occ_face, 0], regions[occ_face, 1]]),
        n_edges,
    )

    valence_ge4 = triangles_per_edge >= 4
    two_material_valence4 = valence_ge4 & (triangles_per_edge == 4) & (regions_per_edge == 2)

    edge_length = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)

    # For each 2-material valence-4 edge, the (single) region pair its four triangles share.
    pairs: list[tuple[int, int]] = []
    if two_material_valence4.any():
        flagged_edges = np.nonzero(two_material_valence4)[0]
        edge_of_occ = occ_to_edge
        for edge_idx in flagged_edges:
            occs = np.nonzero(edge_of_occ == edge_idx)[0]
            faces = occs % len(f)
            labels = set(map(int, regions[faces].flatten().tolist()))
            pairs.append(tuple(sorted(labels)))

    return {
        "n_edges": int(n_edges),
        "valence_histogram": {
            int(k): int(c)
            for k, c in zip(*np.unique(triangles_per_edge, return_counts=True), strict=True)
        },
        "n_valence_ge4": int(valence_ge4.sum()),
        "n_two_material_valence4": int(two_material_valence4.sum()),
        "two_material_valence4_region_pairs": pairs,
        "total_length_valence_ge4": float(edge_length[valence_ge4].sum()),
    }
