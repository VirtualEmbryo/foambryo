"""F6 — mesh-realisation instability: the free error bar.

`ForceBalanceKKT`/`Variational`'s gauge-matched tension error has coefficient of variation
~0.89-0.95 across independently remeshed realisations of *identical* ground-truth physics;
`YoungDupre`'s is ~0.02-0.03 on the same targets (the two hold-last-value groups of the
converged cohort that junction-graph fragmentation never affected; on the other two groups,
after the fragmentation fix, `ForceBalanceKKT` reads 0.585 and 0.715 and `YoungDupre` 0.036
and 0.193). This is the single strongest argument in the project
for never reporting a `ForceBalanceKKT` number on one mesh realisation as a point estimate.

Data: `foambryo_benchmarks/baseline/converged_cohort.json` (the baseline after the junction-graph
fragmentation fix). Each mesh
carries its own `folder` and `hold_last_value_group` -- a group of stages holding the *same*
target tensions while only the triangulation/remeshing changes (`converged_mesh_parser.py`'s
`stage_groups`, computed once and stored per mesh, not recomputed here). Groups with >= 2
stages are the mesh-to-mesh-variability samples this figure plots; groups of size 1 carry no
variability information and are excluded.

Run with: `uv run python -m foambryo_benchmarks.figures.f6_mesh_realisation_instability` from `foambryo/`.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from foambryo_benchmarks.figures._style import (
    COLOR_FORCEBALANCE,
    COLOR_YOUNGDUPRE,
    FOAMBRYO_BASELINE,
    SINGLE_COL_IN,
    apply_style,
    save_fig,
    strip_spines,
)

METHODS = ["ForceBalanceKKT", "YoungDupre"]
METHOD_COLOR = {"ForceBalanceKKT": COLOR_FORCEBALANCE, "YoungDupre": COLOR_YOUNGDUPRE}


def load_groups() -> dict[tuple[str, int], dict[str, list[float]]]:
    records = json.loads((FOAMBRYO_BASELINE / "converged_cohort.json").read_text())
    groups: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: {m: [] for m in METHODS})
    for r in records:
        key = (r["folder"], r["hold_last_value_group"])
        for method in METHODS:
            median_err = r["methods"][method]["tension_relative_error_matched_gauge"]["median"]
            groups[key][method].append(median_err)
    return {k: v for k, v in groups.items() if len(v[METHODS[0]]) >= 2}


def cv(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    return float(arr.std(ddof=0) / mean) if mean != 0 else float("nan")


SHORT_NAME = {
    "165LumenA_AllEqu_Base1B_pTEiup_ICMup_mTo3_pTEmTEsplit15_pTEoup_V2_Y27": "165Lumen-Y27",
    "165LumenA_AllEqu_Base1B_pTEiup_ICMup_mTo3_pTEmTEsplit15_pTEoup_V2_mTEo_inversion": "165Lumen-\nmTEo_inv.",
    "90LumenA_AllEqu_Base1B_pTEidown_mToup": "90Lumen-A",
    "90LumenC_AllEqu_Base1B_pTEidown_ICMdown_mTo08_pTEmTEsplit14_pTEICMup_CN03+ICM+ap": "90Lumen-C",
}


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    groups = load_groups()
    # One (folder, group) per re-convergence group; stable order by folder then group id.
    keys = sorted(groups.keys())

    fig, axes = plt.subplots(1, 2, figsize=(2 * SINGLE_COL_IN, SINGLE_COL_IN * 0.95),
                              gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    x = np.arange(len(keys))
    width = 0.36
    cvs = {m: [cv(groups[k][m]) for k in keys] for m in METHODS}
    for i, method in enumerate(METHODS):
        offset = (i - 0.5) * width
        ax.bar(x + offset, cvs[method], width=width, color=METHOD_COLOR[method],
               label=method, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{SHORT_NAME.get(k[0], k[0][:12])}\n(n={len(groups[k][METHODS[0]])})" for k in keys],
        fontsize=6.3,
    )
    ax.set_ylabel("coefficient of variation\nacross remeshings, fixed physics")
    ax.set_title("(a) CV per re-convergence group", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=6.3)
    strip_spines(ax)

    # Panel (b): normalised trajectories within the two "clean" groups (never touched by the
    # junction-graph-fragmentation defect), to show the spread directly rather than only its
    # summary statistic.
    ax2 = axes[1]
    clean_keys = [
        ("165LumenA_AllEqu_Base1B_pTEiup_ICMup_mTo3_pTEmTEsplit15_pTEoup_V2_Y27", 6),
        ("90LumenA_AllEqu_Base1B_pTEidown_mToup", 10),
    ]
    clean_keys = [k for k in clean_keys if k in groups]
    offset = 0
    for j, key in enumerate(clean_keys):
        n_stages = len(groups[key][METHODS[0]])
        for method in METHODS:
            vals = np.array(groups[key][method])
            norm = vals / vals.mean()
            stage_idx = np.arange(n_stages) + offset
            ax2.plot(
                stage_idx, norm,
                marker="o", markersize=2.6, color=METHOD_COLOR[method],
                label=method if j == 0 else None, linewidth=1.0,
            )
        if j < len(clean_keys) - 1:
            ax2.axvline(offset + n_stages - 0.5, color="#cfcfc9", linewidth=0.7, linestyle=":")
        offset += n_stages + 1
    ax2.axhline(1.0, color="#cfcfc9", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("error / group mean")
    ax2.set_xlabel("stage within group\n(two clean groups, concatenated)")
    ax2.set_title("(b) trajectories, clean groups", fontsize=8.5)
    ax2.legend(loc="upper right", fontsize=6.3)
    strip_spines(ax2)

    fig.suptitle(
        "Mesh-realisation instability: same physics, different remeshing",
        fontsize=9.5, y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, "F6_mesh_realisation_instability")


if __name__ == "__main__":
    main()
