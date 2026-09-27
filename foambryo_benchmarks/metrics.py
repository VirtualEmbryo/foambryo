"""Pure measurement functions for the foambryo tension/pressure inference harness.

Used by `foambryo_benchmarks/run_case.py`. Every function here is a measurement, not an
inference method: it compares an already-inferred result to a ground-truth value, or computes
a diagnostic (`rho`, singular spectrum) that is intrinsic to the mesh and ground-truth
tensions, independent of any inference method.

`mesh_diagnostics` uses the harness-default force-balance operator, `beta = (gamma, p)`
with **no line-tension block** (by design) — it calls `force_balance_operator` with empty
`junctions`/`edge_to_triple`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from foambryo.validation import force_balance as fb
from foambryo.validation import mesh_io


def _relative_errors(
    inferred: dict,
    truth: dict,
    *,
    exclude_keys: set = frozenset(),
) -> dict[str, float]:
    """Per-key relative error `|inferred - truth| / max(|truth|, eps)`, plus aggregate stats.

    Args:
        inferred (dict): Map key -> inferred value.
        truth (dict): Map key -> ground-truth value.
        exclude_keys (set, optional): Keys to skip (e.g. the exterior pressure reference,
            which is fixed by convention rather than inferred). Defaults to no exclusions.

    Returns:
        dict[str, float]: `{"per_key": {...}, "mean": ..., "median": ..., "p95": ..., "n": ...}`.
    """
    eps = 1e-12
    keys = sorted(k for k in truth if k in inferred and k not in exclude_keys)
    per_key = {}
    for key in keys:
        rel = abs(inferred[key] - truth[key]) / max(abs(truth[key]), eps)
        per_key[key] = float(rel)
    values = np.array(list(per_key.values())) if per_key else np.array([np.nan])
    return {
        "per_key": per_key,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "n": len(per_key),
    }


def tension_relative_errors(
    dict_inferred: dict[tuple[int, int], float],
    dict_truth: dict[tuple[int, int], float],
) -> dict:
    """Relative tension error per interface, plus aggregate (mean, median, p95).

    Both dictionaries are expected on the same normalisation (`mean_tension = 1`); the
    `benchmarking-dataset` ground-truth tensions already have this convention (their
    values sum to the number of interfaces), matching `infer_tensions`' default.

    Args:
        dict_inferred (dict[tuple[int, int], float]): Inferred tension per interface.
        dict_truth (dict[tuple[int, int], float]): Ground-truth tension per interface.

    Returns:
        dict: See `_relative_errors`.
    """
    return _relative_errors(dict_inferred, dict_truth)


def gauge_matching_scale(
    dict_inferred: dict[tuple[int, int], float],
    dict_truth: dict[tuple[int, int], float],
) -> float:
    """The single factor that puts inferred tensions in the same gauge as the truth.

    **Why this is needed even when both are nominally "mean 1"** (the rule every accuracy
    comparison in this package follows: errors are reported in both gauges, as published and
    after matching the overall scale). `foambryo.infer_tensions` imposes `mean(gamma) = 1` as
    a *soft least-squares row*, not a constraint: the normalisation competes with the physics
    rows, so the returned mean is only approximately 1. Any error reported without matching
    gauges therefore contains a component that is pure convention, and a single-case check
    cannot tell the two apart: at the package's initial validation stage, on the doublet
    phantom (ground truth of mean `(2 + ratio) / 3`), the as-published tension errors at
    `ratio` 0.5 and 1.5 (17.5 %, 13.9 %) fell to 6.2 % and 2.3 % once gauge-matched, and
    `ratio = 1.0` looked healthy only because its gauge factor is exactly 1.

    The factor is `mean(truth) / mean(inferred)` over the interfaces the two dictionaries
    share — the gauge transformation that makes the two means agree, which is the same
    convention both sides claim to use. It must be applied to **pressures as well**, and by
    the *same* factor: `Delta_P = 2 * gamma / R` is linear in tension, so one factor keeps
    the tension/pressure pair a consistent equilibrium.

    Args:
        dict_inferred (dict[tuple[int, int], float]): Inferred tension per interface.
        dict_truth (dict[tuple[int, int], float]): Ground-truth tension per interface.

    Returns:
        float: The scale factor, or 1.0 if it cannot be formed (no shared keys, or the
            inferred mean is 0).
    """
    shared = [key for key in dict_truth if key in dict_inferred]
    if not shared:
        return 1.0
    mean_inferred = float(np.mean([dict_inferred[key] for key in shared]))
    mean_truth = float(np.mean([dict_truth[key] for key in shared]))
    return mean_truth / mean_inferred if mean_inferred != 0 else 1.0


def pressure_relative_errors(
    dict_inferred: dict[int, float],
    dict_truth: dict[int, float],
    exterior: int = 0,
) -> dict:
    """Relative pressure error per cell, plus aggregate (mean, median, p95).

    The exterior cell's pressure is fixed by convention (`base_pressure`, ground truth 0),
    not inferred, so it is excluded from the error statistics (a relative error against a
    true value of 0 is undefined).

    Args:
        dict_inferred (dict[int, float]): Inferred pressure per cell.
        dict_truth (dict[int, float]): Ground-truth pressure per cell.
        exterior (int, optional): Label of the external medium. Defaults to 0.

    Returns:
        dict: See `_relative_errors`.
    """
    return _relative_errors(dict_inferred, dict_truth, exclude_keys={exterior})


def interface_areas(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> dict[tuple[int, int], float]:
    """Total triangle area per interface, keyed by sorted region pair.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        dict[tuple[int, int], float]: Interface key -> area.
    """
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    keys = list(map(tuple, np.sort(regions, axis=1).tolist()))
    totals: dict[tuple[int, int], float] = {}
    for key, a in zip(keys, area.tolist(), strict=True):
        totals[key] = totals.get(key, 0.0) + a
    return totals


def triple_line_lengths(
    v: NDArray[np.float64],
    edge_to_triple: dict[tuple[int, int], tuple[int, int, int]],
) -> dict[tuple[int, int, int], float]:
    """Total edge length per triple line, keyed by sorted region triple.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        edge_to_triple (dict[tuple[int, int], tuple[int, int, int]]): Map mesh edge -> its
            region triple, as returned by `foambryo.validation.mesh_io.list_junctions`.

    Returns:
        dict[tuple[int, int, int], float]: Junction key -> total length.
    """
    totals: dict[tuple[int, int, int], float] = {}
    for (a, b), triple in edge_to_triple.items():
        totals[triple] = totals.get(triple, 0.0) + float(np.linalg.norm(v[a] - v[b]))
    return totals


def residual_localisation(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
    m: NDArray[np.float64],
    ni: int,
    gamma: NDArray[np.float64],
    mean_edge_length: float,
) -> dict:
    """Where on the mesh the unbalanced force lives — the discriminator behind a large `rho`.

    `rho` says *whether* a mesh is at equilibrium; this says *where* it fails, which is what
    separates two physically different situations that `rho` alone conflates:

    - **Residual concentrated on triple lines** (vertices touching 3 regions, typically
      1-2% of the mesh). Young-Dupré is a pointwise balance that a discrete triangulation
      cannot satisfy exactly, so *every* mesh has a residual here. This is discretisation
      error and it is expected.
    - **Residual spread over interface interiors** (vertices touching 2 regions). There the
      balance is Young-Laplace, `gamma * 2H = Delta_P`, i.e. constant mean curvature. A
      converged interface satisfies it pointwise; a *non-converged* surface relaxation does
      not. A large interior share therefore means the mesh was never relaxed to equilibrium
      — a data defect, not a discretisation limit, and not fixable by any choice of
      tensions, line tensions or pressures.

    All fractions are of `||residual||^2`, so they sum to 1 over the vertex classes.

    **Gauge:** every quantity here is a ratio of like quantities and so is invariant under
    rescaling `gamma`, exactly as `rho` is.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles, shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        m (NDArray[np.float64]): The force-balance operator (no line-tension block).
        ni (int): Number of interfaces (surface-tension columns).
        gamma (NDArray[np.float64]): Tensions at which to evaluate the residual.
        mean_edge_length (float): Mesh mean edge length, used to make the per-vertex
            residual amplitude dimensionless.

    Returns:
        dict: `{"fraction_interface_interior", "fraction_triple_line",
            "fraction_quadjunction", "n_interface_interior_vertices",
            "n_triple_line_vertices", "n_quadjunction_vertices",
            "interior_residual_per_vertex", "triple_line_residual_per_vertex",
            "interior_over_triple_line", "top_1pct_vertex_share"}`. The two
            `*_per_vertex` figures are mean `|residual|` over that vertex class divided by
            `mean(gamma) * mean_edge_length` — dimensionless, since a discrete area
            gradient scales as one edge length.
    """
    residual = fb.residual_force_field(m, ni, 0, gamma, np.zeros(0))
    norm = np.linalg.norm(residual, axis=1)
    total = float((norm**2).sum())
    counts = mesh_io.vertex_region_counts(v, f, regions)
    scale = float(np.mean(gamma)) * mean_edge_length

    interior, triple, quad = counts == 2, counts == 3, counts >= 4
    share = {name: float((norm[sel] ** 2).sum() / total) for name, sel in
             (("interface_interior", interior), ("triple_line", triple), ("quadjunction", quad))}
    per_vertex = {
        name: (float(norm[sel].mean() / scale) if sel.any() else 0.0)
        for name, sel in (("interface_interior", interior), ("triple_line", triple))
    }
    n_top = max(1, len(v) // 100)
    top_share = float((np.sort(norm)[-n_top:] ** 2).sum() / total)

    return {
        "fraction_interface_interior": share["interface_interior"],
        "fraction_triple_line": share["triple_line"],
        "fraction_quadjunction": share["quadjunction"],
        "n_interface_interior_vertices": int(interior.sum()),
        "n_triple_line_vertices": int(triple.sum()),
        "n_quadjunction_vertices": int(quad.sum()),
        "interior_residual_per_vertex": per_vertex["interface_interior"],
        "triple_line_residual_per_vertex": per_vertex["triple_line"],
        "interior_over_triple_line": (
            per_vertex["interface_interior"] / per_vertex["triple_line"]
            if per_vertex["triple_line"]
            else float("inf")
        ),
        "top_1pct_vertex_share": top_share,
    }


def effective_tension_ratios(
    gamma_eff: NDArray[np.float64],
    gamma_nominal: NDArray[np.float64],
    interfaces: list[tuple[int, int]],
) -> dict:
    """Per-interface `gamma_eff / gamma_nominal`, in two explicitly-stated gauges.

    **Why the gauge must be stated** (the rule every accuracy comparison in this package
    follows: errors are reported in both gauges, as published and after matching the overall
    scale). Only tension *ratios* are physical; the overall scale is a gauge. So a *uniform*
    factor across every interface is not a finding, it is a gauge choice, and conversely a
    factor on a *subset* of interfaces is a real defect that survives any gauge. The
    factor-2 signature this exists to look for — a factor on a subset of interfaces (for
    example, some line tensions encoded at half their value by the program that generated
    the mesh) — is of the second kind: a spike at 0.5 or 2.0 *against a bulk at 1.0*.
    Reporting the ratio in a single unstated gauge cannot tell those two cases apart (see
    `gauge_matching_scale` for the case where an unstated gauge dominated the error).

    Three gauges are therefore reported:

    - `mean_one`: both vectors rescaled to `mean = 1`. This is `foambryo.infer_tensions`'
      own convention and the `benchmarking-dataset` ground truth's; it is the gauge in
      which "1.0 everywhere" is the correct answer.
    - `least_squares`: `gamma_eff` rescaled by the single factor `s*` minimising
      `||s * gamma_eff - gamma_nominal||`. This is the most generous possible gauge — it
      gives the systematic factor every chance to be absorbed — so a spike that *survives*
      it is not a scale artifact.
    - `median_one`: the `mean_one` ratio divided by *its own median*. **This is the gauge
      the factor-2 search needs**, and the reason is not obvious: in the `mean_one` gauge a
      factor `k` applied to a fraction `phi` of the interfaces is *diluted* by the gauge
      itself, because the affected interfaces also move the mean. With 2 of 6 interfaces at
      half nominal the `mean_one` ratios come out `0.6` and `1.2`, not `0.5` and `1.0` — so
      a literal "is anything near 0.5" test on `mean_one` **misses a real factor-2 bug**.
      Dividing by the median puts the (majority) unaffected bulk at exactly 1 and the
      affected subset at exactly `k`. Its limitation, stated rather than hidden: it assumes
      the affected interfaces are a *minority*; a factor applied to more than half of them
      would become the bulk, and the unaffected ones would show up shifted instead.

    The gauge-independent statement underneath all three is that the signature is
    **bimodality with a cluster ratio of 2**, not a spike at any particular absolute value.

    Args:
        gamma_eff (NDArray[np.float64]): Tensions that balance the mesh (see
            `foambryo.validation.force_balance.effective_surface_tensions`).
        gamma_nominal (NDArray[np.float64]): Ground-truth tensions, same interface order.
        interfaces (list[tuple[int, int]]): Interface keys, same order as both arrays.

    Returns:
        dict: `{"gauges": {name: {"per_key", "min", "median", "max", "iqr", "spread",
            "abs_deviation_median", "abs_deviation_p95", "abs_deviation_max",
            "n_near_half", "n_near_two", "n_near_one"}}, "gauge_factors": {...}}`, where
            `spread` is `max - min`, `near` means within 5% of 0.5 / 2.0 / 1.0, and the
            `abs_deviation_*` figures are `|gamma_eff - gamma_nominal|` in that gauge, in
            units of the mean tension.
    """
    eff = np.asarray(gamma_eff, dtype=float)
    nom = np.asarray(gamma_nominal, dtype=float)
    scale_mean_one = float(eff.mean()) / float(nom.mean())
    scale_least_squares = float(eff @ nom) / float(eff @ eff)

    mean_one_ratio = (eff / eff.mean()) / (nom / nom.mean())
    median_scale = float(np.median(mean_one_ratio))

    gauges = {}
    for name, gauged_eff, gauged_nom in (
        ("mean_one", eff / eff.mean(), nom / nom.mean()),
        ("least_squares", scale_least_squares * eff, nom),
        ("median_one", (eff / eff.mean()) / median_scale, nom / nom.mean()),
    ):
        ratio = gauged_eff / gauged_nom
        # The *absolute* deviation, in units of the mean tension. Reported alongside the
        # ratio because the two discriminate between the two things that make a ratio
        # deviate: a multiplicative defect (constant ratio, absolute deviation growing with
        # gamma) and a uniform additive error (constant absolute deviation, ratio blowing up
        # wherever gamma_nominal is small). Only the first is the signature of a
        # mis-encoded tension.
        deviation = np.abs(gauged_eff - gauged_nom)
        gauges[name] = {
            "per_key": dict(zip(interfaces, ratio.tolist(), strict=True)),
            "abs_deviation_median": float(np.median(deviation)),
            "abs_deviation_p95": float(np.percentile(deviation, 95)),
            "abs_deviation_max": float(deviation.max()),
            "min": float(ratio.min()),
            "median": float(np.median(ratio)),
            "max": float(ratio.max()),
            "iqr": float(np.percentile(ratio, 75) - np.percentile(ratio, 25)),
            "spread": float(ratio.max() - ratio.min()),
            "n_near_half": int((np.abs(ratio - 0.5) < 0.05 * 0.5).sum()),
            "n_near_one": int((np.abs(ratio - 1.0) < 0.05).sum()),
            "n_near_two": int((np.abs(ratio - 2.0) < 0.05 * 2.0).sum()),
            "n": int(ratio.size),
        }
    return {
        "gauges": gauges,
        "gauge_factors": {
            "mean_gamma_nominal": float(nom.mean()),
            "mean_gamma_eff": float(eff.mean()),
            "mean_one_scale": scale_mean_one,
            "least_squares_scale": scale_least_squares,
            "median_one_scale": median_scale,
        },
    }


def mesh_diagnostics(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
    dict_truth_tensions: dict[tuple[int, int], float],
) -> dict:
    """Equilibrium and identifiability diagnostics for a mesh, at its ground-truth tensions.

    Uses the harness-default operator `M = [M_gamma | M_p]` (no line-tension block).
    `rho` is `residual_at` at the mesh's own ground-truth tensions with pressures
    marginalised — an honest check of whether the mesh is at equilibrium there. This
    function computes it for one mesh and draws no conclusion; the audit over the full
    47-case `benchmarking-dataset` is `foambryo_benchmarks.run_dataset_audit`, and the
    equilibrium split it led to is `foambryo_benchmarks.case_groups`.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles, shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        dict_truth_tensions (dict[tuple[int, int], float]): Ground-truth tension per interface.

    Returns:
        dict: `{"n_interfaces", "n_cells", "rho", "singular_values", "sigma_soft",
            "soft_mode_fractions": [surface, line, pressure]}`. `singular_values` is the
            full ascending spectrum of the column-normalised operator; the line fraction
            of `soft_mode_fractions` is always 0 here since there is no line-tension block.
    """
    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    m = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)
    ni, nj = len(interfaces), 0

    gamma = np.array([dict_truth_tensions[key] for key in interfaces])
    rho = fb.residual_at(m, ni, nj, gamma, np.zeros(0))
    spectrum = fb.singular_values(m)
    sigma_soft, fractions = fb.soft_mode(m, ni, nj)

    return {
        "n_interfaces": ni,
        "n_cells": len(cells),
        "rho": rho,
        "singular_values": spectrum.tolist(),
        "sigma_soft": sigma_soft,
        "soft_mode_fractions": fractions.tolist(),
    }


def dataset_audit(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
    dict_truth_tensions: dict[tuple[int, int], float],
) -> dict:
    """The full per-case dataset audit: integrity, equilibrium, identifiability, gauge.

    Runs the three checks (integrity, equilibrium, identifiability) plus the effective-tension
    comparison, on one ground-truth mesh, using the harness-default operator
    `M = [M_gamma | M_p]` (no line-tension block).

    **Gauge conventions, stated explicitly** (every accuracy comparison in this package
    reports errors in both gauges, as published and after matching the overall scale).

    - `rho` is **gauge-invariant**: it is `||P_perp M_gamma gamma|| / ||M_gamma gamma||`,
      and both norms scale linearly in `gamma`, so `rho(c * gamma) = rho(gamma)` for any
      `c != 0`. No gauge matching is needed or possible for it. (This is *why* `rho` is the
      right primary equilibrium statistic, and also why the gauge-matching rule applies to
      the *tension-error* tables rather than to `rho`.)
    - `rho_free` is likewise gauge-invariant, by the same argument.
    - `sigma_soft` and the spectrum are computed on the **column-normalised** operator, so
      they are invariant under rescaling of the *columns* of `M`, i.e. under a change of
      units of any individual unknown — a stronger invariance than the tension gauge.
    - `effective_tension_ratios` is the only gauge-dependent quantity here, and it is
      reported in two explicit gauges. See that function.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles, shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        dict_truth_tensions (dict[tuple[int, int], float]): Nominal (ground-truth) tension
            per interface.

    Returns:
        dict: `{"integrity", "residual_localisation", "n_cells", "n_interfaces",
            "n_junctions", "rho", "rho_free", "sigma_soft", "sigma_gauge",
            "singular_values", "soft_mode_fractions", "effective_tensions", "geometry"}`.
    """
    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    junctions, edge_to_triple = mesh_io.list_junctions(f, regions)
    integrity = mesh_io.mesh_integrity(v, f, regions)

    m = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)
    ni, nj = len(interfaces), 0

    gamma_nominal = np.array([dict_truth_tensions[key] for key in interfaces])
    rho = fb.residual_at(m, ni, nj, gamma_nominal, np.zeros(0))
    gamma_eff, _p_eff, rho_free = fb.effective_surface_tensions(m, ni, nj, mean_tension=1.0)
    spectrum = fb.singular_values(m)
    sigma_soft, fractions = fb.soft_mode(m, ni, nj)

    areas = interface_areas(v, f, regions)
    lengths = triple_line_lengths(v, edge_to_triple)
    volumes = mesh_io.signed_cell_volumes(v, f, regions)
    total_area = float(sum(areas.values()))
    edge_length = mesh_io.mean_edge_length(v, f)
    localisation = residual_localisation(v, f, regions, m, ni, gamma_nominal, edge_length)

    return {
        "integrity": integrity,
        "residual_localisation": localisation,
        "n_cells": len(cells),
        "n_interfaces": ni,
        "n_junctions": len(junctions),
        "rho": rho,
        "rho_free": rho_free,
        "sigma_gauge": float(spectrum[0]),
        "sigma_soft": sigma_soft,
        "singular_values": spectrum.tolist(),
        "soft_mode_fractions": fractions.tolist(),
        "effective_tensions": effective_tension_ratios(gamma_eff, gamma_nominal, interfaces),
        "geometry": {
            "interface_areas": areas,
            "triple_line_lengths": lengths,
            "signed_cell_volumes": volumes,
            "total_interface_area": total_area,
            "min_interface_area": float(min(areas.values())),
            "min_interface_area_fraction": float(min(areas.values()) / total_area),
            "mean_edge_length": edge_length,
            "bounding_box_min": v.min(axis=0).tolist(),
            "bounding_box_max": v.max(axis=0).tolist(),
        },
    }


def interface_area_fractions(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> dict[tuple[int, int], float]:
    """Each interface's area as a fraction of the mesh's own total interfacial area.

    **Scale-invariant, so no registration is needed.** A candidate mesh reconstructed from a
    voxel mask and the reference mesh it is compared against generally live in different frames
    and at different overall scales; dividing by the mesh's own total removes both, leaving a
    quantity that says *how the area is distributed among the interfaces* rather than how big
    the tissue is. Comparing raw areas instead would require a registration and would conflate a
    scale error with a shape error.

    Interface-area error had **no predicate** in the benchmark suite; this is the predicate.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        dict[tuple[int, int], float]: Interface key -> its share of the total area, summing to
            1.0 over the mesh's interfaces. Empty if the mesh has no area at all.
    """
    areas = interface_areas(v, f, regions)
    total = float(sum(areas.values()))
    if total <= 0:
        return {}
    return {key: value / total for key, value in areas.items()}


def interface_area_fraction_errors(
    candidate: dict[tuple[int, int], float],
    reference: dict[tuple[int, int], float],
) -> dict:
    """Signed and absolute error of a candidate's interface-area fractions against a reference.

    Only interfaces present in **both** meshes are scored: an interface one mesh has and the
    other does not is an identity defect, not an area error, and is reported as a count rather
    than folded into an average: invented, missed, merged and split interfaces are reported as
    counts, never as a score.

    Args:
        candidate (dict[tuple[int, int], float]): `interface_area_fractions` of the candidate.
        reference (dict[tuple[int, int], float]): `interface_area_fractions` of the reference.

    Returns:
        dict: `per_interface` (interface key -> signed error, candidate minus reference),
            `n_matched`, `n_invented` (in the candidate only), `n_missed` (in the reference
            only), `mean_abs_error`, `median_abs_error`, `max_abs_error` and `rms_error` over
            the matched interfaces. The error statistics are `None` when nothing matched.
    """
    matched = sorted(set(candidate) & set(reference))
    per_interface = {key: float(candidate[key] - reference[key]) for key in matched}
    errors = np.array(list(per_interface.values()), dtype=float)
    return {
        "per_interface": per_interface,
        "n_matched": len(matched),
        "n_invented": len(set(candidate) - set(reference)),
        "n_missed": len(set(reference) - set(candidate)),
        "mean_abs_error": float(np.mean(np.abs(errors))) if len(errors) else None,
        "median_abs_error": float(np.median(np.abs(errors))) if len(errors) else None,
        "max_abs_error": float(np.max(np.abs(errors))) if len(errors) else None,
        "rms_error": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
    }


def closer_interface_area_fractions(
    first: dict[tuple[int, int], float],
    second: dict[tuple[int, int], float],
    reference: dict[tuple[int, int], float],
) -> dict:
    """Per interface, which of two candidates lands closer to the reference's area fraction.

    A **paired** predicate: it counts interfaces, not cases, and only interfaces all three
    dictionaries share, so the comparison is like for like. Ties (exactly equal absolute error,
    which is what two readings of the *same* mesh produce when neither moves a vertex) are
    counted separately rather than split, because a tie is the expected and informative answer
    when the thing being varied cannot move the geometry.

    The second-order-angle-reading work uses it to check that changing the angle *reading* leaves the interface areas
    alone — the reading moves no vertex, so `n_tied` should be every matched interface, and
    anything else means a reading has silently changed the mesh.

    Args:
        first (dict[tuple[int, int], float]): first candidate's area fractions.
        second (dict[tuple[int, int], float]): second candidate's area fractions.
        reference (dict[tuple[int, int], float]): the reference's area fractions.

    Returns:
        dict: `n_matched`, `n_first_closer`, `n_second_closer`, `n_tied`, and the mean absolute
            error of each candidate over the matched interfaces.
    """
    matched = sorted(set(first) & set(second) & set(reference))
    first_error = np.array([abs(first[key] - reference[key]) for key in matched], dtype=float)
    second_error = np.array([abs(second[key] - reference[key]) for key in matched], dtype=float)
    return {
        "n_matched": len(matched),
        "n_first_closer": int(np.sum(first_error < second_error)),
        "n_second_closer": int(np.sum(second_error < first_error)),
        "n_tied": int(np.sum(first_error == second_error)),
        "first_mean_abs_error": float(np.mean(first_error)) if len(first_error) else None,
        "second_mean_abs_error": float(np.mean(second_error)) if len(second_error) else None,
    }


def median_of_determined(
    values: list[float],
    *,
    allow_partial: bool = False,
) -> dict:
    """Median over the *determined* (finite) values, refusing to silently hide the rest.

    A value can be legitimately undetermined -- `nan` -- when it comes from a mesh with a
    fully closed, triple-line-free interface (`foambryo.junction_graph`'s
    `ComponentTensionSolution.undetermined`): that is not a failure to be dropped
    (`np.nanmedian`) or a signal to be swallowed (a bare `np.median`, which propagates one
    `nan` into the whole aggregate). Both of those silently turn "one of N cases is
    structurally undetermined" into either an invisible smaller N or an unreadable `nan`
    result that looks like every case failed.

    This function does neither: called plainly, it refuses (mirrors
    `foambryo_benchmarks.case_groups.require_single_group`'s precedent of raising rather than
    guessing) whenever any value is non-finite, so a caller cannot accidentally publish a
    partial median as if it were complete. `allow_partial=True` is the explicit opt-in a
    caller must make to accept a median over fewer than `n_total` values -- and the returned
    counts must travel with the median everywhere it is reported or rendered, so a reader
    can never mistake it for a complete one.

    Args:
        values (list[float]): Raw values; may contain `nan` for an undetermined case.
        allow_partial (bool, optional): Must be `True` to accept `n_determined < n_total`.
            Defaults to `False`.

    Returns:
        dict: `{"median": float | None, "n_determined": int, "n_total": int}`. `median` is
            `None` (not `nan`) when there are zero determined values -- still distinguishable
            from a real number in JSON, unlike a propagated `nan`.

    Raises:
        ValueError: if any value is non-finite and `allow_partial` is not `True`.
    """
    n_total = len(values)
    determined = [v for v in values if np.isfinite(v)]
    n_determined = len(determined)
    if n_determined < n_total and not allow_partial:
        message = (
            f"{n_total - n_determined} of {n_total} value(s) are undetermined (non-finite) -- "
            "refusing to silently produce a propagated nan (a bare median) or to silently drop "
            "them (nanmedian) for what would otherwise look like a complete aggregate. Pass "
            "allow_partial=True to accept a median over the n_determined comparable values, and "
            "carry n_determined/n_total alongside the median wherever it is reported or "
            "rendered -- never publish it as a bare number."
        )
        raise ValueError(message)
    return {
        "median": float(np.median(determined)) if determined else None,
        "n_determined": n_determined,
        "n_total": n_total,
    }
