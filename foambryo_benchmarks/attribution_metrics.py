"""Pure measurement functions for the attribution harness (meshing error versus inference error).

Every function here takes already-inferred
tensions or already-loaded mesh arrays; nothing here runs `dw3d` or `infer_tensions` itself
(that is `foambryo_benchmarks/run_attribution.py`'s job) so these functions can be unit-tested with
synthetic dictionaries and without a mesh at all.

**Gauge (the rule every accuracy comparison in this package follows).** Every error below is reported in both gauges:
as-published, and after `foambryo_benchmarks.metrics.gauge_matching_scale` rescales the inferred
tensions so their mean matches the ground truth's. The gauge-matched number is primary.

**The non-additivity caveat, restated at the one place it is computed.** `E_meshing =
E_total - E_inference` is *not* a clean separation: P1 (the ground truth mesh) is itself
under-converged (`rho_free` 0.0007-0.036 on the 42), so `E_inference` is already
contaminated by the simulator's under-convergence, not purely by the inference method.
`E_meshing`'s sign can therefore mislead - it is reported because the attribution is defined
by it, not because it is trusted as a pure quantity. `rho_free` is carried on both populations
specifically so a reader can judge how much of `E_inference` to discount.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from foambryo_benchmarks import metrics as m
from foambryo.validation import force_balance as fb
from foambryo.validation import mesh_io


def rho_free_of_mesh(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> dict:
    """`rho_free`: the residual once the tensions themselves are free to balance the mesh.

    Intrinsic to the mesh and the harness-default operator `M = [M_gamma | M_p]` (no
    line-tension block, by design) - it needs no ground truth. This is what
    lets it be computed identically on P1 and P2, which makes it the covariate the attribution needs: a
    method whose error tracks `rho_free` is being limited by the mesh, not by itself.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles, shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        dict: `{"rho_free", "n_interfaces", "n_cells"}`.
    """
    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    matrix_m = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)
    _gamma_eff, _p_eff, rho_free = fb.effective_surface_tensions(matrix_m, len(interfaces), 0, mean_tension=1.0)
    return {"rho_free": rho_free, "n_interfaces": len(interfaces), "n_cells": len(cells)}


def attribution_errors(
    dict_tensions_p1: dict[tuple[int, int], float],
    dict_tensions_p2: dict[tuple[int, int], float],
    dict_truth_tensions: dict[tuple[int, int], float],
) -> dict:
    """`E_total`, `E_inference`, `E_meshing` for one (case, method), both gauges.

    Keyed entirely by interface label pairs (never by vertex index or nearest-neighbour
    matching), which is automatic here because `dict_tensions_p1`/`p2` are already
    `foambryo.infer_tensions`'s label-keyed output - P1 and P2 have no vertex
    correspondence, but they share the same label alphabet (both come from the same
    underlying segmentation), so the label-keyed comparison is well posed on the
    intersection of keys (`foambryo_benchmarks.metrics._relative_errors` already restricts to it).

    Non-vacuity is asserted, not assumed: this raises if either inferred dict shares
    no key with the ground truth, rather than silently returning `nan` that a caller might
    mistake for "no error".

    Args:
        dict_tensions_p1 (dict[tuple[int, int], float]): Tensions inferred on the P1
            (simulator ground-truth) mesh.
        dict_tensions_p2 (dict[tuple[int, int], float]): Tensions inferred on the P2
            (`dw3d` reconstruction) mesh.
        dict_truth_tensions (dict[tuple[int, int], float]): Ground-truth tensions, `gamma*`.

    Returns:
        dict: `{"e_total", "e_inference", "e_meshing"}` (gauge-matched medians, the
            primary numbers) plus `"e_total_detail"`, `"e_inference_detail"` (the full
            `_relative_errors` dicts, both gauges) and `"n_common_interfaces": {"p1", "p2"}`.

    Raises:
        ValueError: If `dict_tensions_p1` or `dict_tensions_p2` shares zero keys with
            `dict_truth_tensions` - a harness defect (mismatched label alphabets, or an
            empty inference result), not a real measurement of "infinite error".
    """
    n_common_p1 = len(set(dict_tensions_p1) & set(dict_truth_tensions))
    n_common_p2 = len(set(dict_tensions_p2) & set(dict_truth_tensions))
    if n_common_p1 == 0 or n_common_p2 == 0:
        message = (
            f"zero common interfaces between an inferred population and ground truth "
            f"(n_common_p1={n_common_p1}, n_common_p2={n_common_p2}) - this is a harness "
            "defect (mismatched labels or an empty inference), not a measurement"
        )
        raise ValueError(message)

    scale_p1 = m.gauge_matching_scale(dict_tensions_p1, dict_truth_tensions)
    scale_p2 = m.gauge_matching_scale(dict_tensions_p2, dict_truth_tensions)

    e_inference_detail = {
        "as_published": m.tension_relative_errors(dict_tensions_p1, dict_truth_tensions),
        "gauge_matched": m.tension_relative_errors(
            {k: scale_p1 * v for k, v in dict_tensions_p1.items()},
            dict_truth_tensions,
        ),
    }
    e_total_detail = {
        "as_published": m.tension_relative_errors(dict_tensions_p2, dict_truth_tensions),
        "gauge_matched": m.tension_relative_errors(
            {k: scale_p2 * v for k, v in dict_tensions_p2.items()},
            dict_truth_tensions,
        ),
    }

    e_inference = e_inference_detail["gauge_matched"]["median"]
    e_total = e_total_detail["gauge_matched"]["median"]

    return {
        "e_total": e_total,
        "e_inference": e_inference,
        # Not a clean separation - see this module's docstring. All three are reported
        # explicitly, never the difference alone.
        "e_meshing": e_total - e_inference,
        "e_total_detail": e_total_detail,
        "e_inference_detail": e_inference_detail,
        "gauge_matching_scale_p1": scale_p1,
        "gauge_matching_scale_p2": scale_p2,
        "n_common_interfaces": {"p1": n_common_p1, "p2": n_common_p2},
    }
