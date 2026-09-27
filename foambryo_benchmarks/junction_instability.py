"""Spatial junction-instability analysis — the paper's robustness proxy (the robustness work).

The foambryo paper's own robustness proxy is the fraction of junctions flagged unstable
(`tension_ij > tension_jk + tension_ki`). A better mesh should reduce it in the embryo
periphery while leaving genuinely unstable central junctions flagged, so it is reported
spatially, not just as a count. A triple junction's three interfacial
tensions describe a
Neumann force triangle; the configuration cannot exist in mechanical equilibrium once one
tension exceeds the sum of the other two (the triangle degenerates — one film would recede
or be engulfed). This module computes that flag **per junction**, alongside its position and
length, so the claimed centre-vs-periphery attribution can be checked rather than assumed.
"""

from __future__ import annotations

import numpy as np

from foambryo import geometry


def is_unstable(gamma_ab: float, gamma_bc: float, gamma_ca: float) -> bool:
    """Whether three tensions meeting at a junction violate the Neumann triangle inequality.

    Args:
        gamma_ab (float): Tension of one of the three interfaces at the junction.
        gamma_bc (float): Tension of the second interface.
        gamma_ca (float): Tension of the third interface.

    Returns:
        bool: `True` iff the largest of the three exceeds the sum of the other two
            (equivalently `2 * max(...) > sum(...)`) — symmetric in the three arguments.
    """
    values = (gamma_ab, gamma_bc, gamma_ca)
    return 2 * max(values) > sum(values)


def junction_instability_report(mesh: object, tensions: dict[tuple[int, int], float]) -> dict:
    """Per-junction instability flag, position and length.

    Args:
        mesh (DcelData): A reconstructed or ground-truth mesh (`foambryo.dcel.DcelData`).
        tensions (dict[tuple[int, int], float]): Tensions keyed by sorted interface label
            pair, any consistent gauge (the flag is scale-invariant: multiplying every
            tension by the same positive factor cannot change which one is the largest
            relative to the sum of the other two).

    Returns:
        dict: `{"n_candidate_triples", "n_non_triple", "n_present", "n_absent", "n_unstable",
            "fraction_unstable", "effective_radius", "junctions": [...]}`.
            `n_non_triple` counts trijunction-detector keys spanning more than 3 regions
            (an abnormal non-manifold edge on a `dw3d` reconstruction)
            — structurally excluded, reported rather than silently coerced.
            `n_absent` counts genuine triple lines missing at least one of their three
            interfaces from `tensions` — **counted, never read as tension 0**. Each
            junction record in `"junctions"` carries `"triple"`, `"unstable"`, `"length"`,
            `"position"`, `"radial_distance"` and `"normalized_radial_distance"` (radial
            distance divided by the mesh's mean vertex-to-centroid distance, so `<1` reads
            as "inner", `>1` as "outer" — a per-mesh-shape-normalised proxy for centre vs.
            periphery, since the dataset has no other embryo-centre definition).

    Raises:
        ValueError: If no junction has all three interfaces present (a fraction over zero
            junctions is undefined, not zero — the same "absent is not zero" rule applied
            to the aggregate).
    """
    edges_trijunctions = geometry.extract_edges_trijunctions(mesh)
    centroid = mesh.v.mean(axis=0)
    effective_radius = float(np.linalg.norm(mesh.v - centroid, axis=1).mean())

    records = []
    n_non_triple = 0
    n_absent = 0
    for triple, edges in edges_trijunctions.items():
        if len(triple) != 3:
            n_non_triple += 1
            continue
        a, b, c = sorted(int(x) for x in triple)
        interfaces = [(a, b), (b, c), (a, c)]
        if any(pair not in tensions for pair in interfaces):
            n_absent += 1
            continue
        gamma_ab, gamma_bc, gamma_ac = (tensions[pair] for pair in interfaces)
        length = float(geometry.compute_length_edges_trijunctions(mesh.v, edges).sum())
        position = mesh.v[edges].reshape(-1, 3).mean(axis=0)
        radial_distance = float(np.linalg.norm(position - centroid))
        records.append(
            {
                "triple": (a, b, c),
                "unstable": is_unstable(gamma_ab, gamma_bc, gamma_ac),
                "length": length,
                "position": position.tolist(),
                "radial_distance": radial_distance,
                "normalized_radial_distance": (
                    radial_distance / effective_radius if effective_radius > 0 else float("nan")
                ),
            },
        )

    if not records:
        message = (
            f"0 of {len(edges_trijunctions)} candidate triple lines had all three interfaces "
            "present in `tensions` - a fraction over zero junctions is undefined, not 0."
        )
        raise ValueError(message)

    n_unstable = sum(1 for r in records if r["unstable"])
    return {
        "n_candidate_triples": len(edges_trijunctions),
        "n_non_triple": n_non_triple,
        "n_present": len(records),
        "n_absent": n_absent,
        "n_unstable": n_unstable,
        "fraction_unstable": n_unstable / len(records),
        "effective_radius": effective_radius,
        "junctions": records,
    }


def spatial_summary(report: dict, n_bins: int = 3) -> list[dict]:
    """Bin a `junction_instability_report`'s junctions by normalised radial distance.

    Equal-count (quantile) bins, labelled innermost-first, so "the embryo centre" in the
    paper's claim maps to bin 0 regardless of how many junctions a given mesh has.

    Args:
        report (dict): Output of `junction_instability_report`.
        n_bins (int, optional): Number of radial bins. Defaults to 3 (inner/mid/outer).

    Returns:
        list[dict]: One entry per bin, innermost first: `{"bin", "n", "n_unstable",
            "fraction_unstable", "radial_distance_range", "median_length"}`. A bin with
            `n == 0` reports `fraction_unstable` as `None` (absent, not 0) rather than
            dividing by zero.
    """
    junctions = sorted(report["junctions"], key=lambda r: r["normalized_radial_distance"])
    n = len(junctions)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    summary = []
    for i in range(n_bins):
        chunk = junctions[edges[i] : edges[i + 1]]
        if not chunk:
            summary.append(
                {
                    "bin": i,
                    "n": 0,
                    "n_unstable": 0,
                    "fraction_unstable": None,
                    "radial_distance_range": None,
                    "median_length": None,
                },
            )
            continue
        n_unstable = sum(1 for r in chunk if r["unstable"])
        radii = [r["normalized_radial_distance"] for r in chunk]
        summary.append(
            {
                "bin": i,
                "n": len(chunk),
                "n_unstable": n_unstable,
                "fraction_unstable": n_unstable / len(chunk),
                "radial_distance_range": (min(radii), max(radii)),
                "median_length": float(np.median([r["length"] for r in chunk])),
            },
        )
    return summary
