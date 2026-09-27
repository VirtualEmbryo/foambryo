#!/usr/bin/env python
r"""CLI: triple-junction-graph connectivity per mesh (backs the converged-mesh-cohort work's fragmentation finding).

`foambryo_benchmarks/run_converged_cohort.py` runs the full 7-method tension grid on all 72 meshes
(~35 minutes); the cause of `YoungDupre`'s error pinning at exactly 1.0 on 26 of those meshes does
not need any of that -- it needs only the mesh's own triple-junction graph, computed in seconds.
(Fragmentation is not the cause of the cohort's pooled-correlation anomaly: with it fixed, the
pooled Spearman r of `ForceBalanceKKT` against `rho_free` is unchanged at 0.116.) This script
exists so that claim is backed by committed, reusable code rather than by the one-off interactive
analysis that first found it.

The triple-junction graph: one node per interface (`foambryo.validation.mesh_io.list_interfaces`),
one edge between two interfaces that share a triple line (`list_junctions` gives each
triple-line's three constituent interfaces; all `C(3,2)=3` pairs among them become graph
edges). `YoungDupre`'s per-triple-line rows are homogeneous within each connected component of
this graph, so a mesh whose graph has more than one component has more than one free overall
scale, and the single global mean-tension row can pin only one linear combination of them.
On the two affected folders of the cohort, every stage whose graph has more than one
component is exactly a stage where `YoungDupre`'s gauge-matched median error pins at 1.0,
including one stage that briefly reconnects (one component) and drops back to 0.036.

Usage:
    uv run python -m foambryo_benchmarks.diagnose_junction_fragmentation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from foambryo_benchmarks.run_converged_cohort import CONVERGED_MESHES_DIR, discover_folders
from foambryo.validation import mesh_io

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "foambryo_benchmarks" / "baseline" / "junction_fragmentation.json"


def triple_junction_component_sizes(
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> list[int]:
    """Sizes of the connected components of the triple-junction graph, descending.

    Args:
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        list[int]: Component sizes (number of interfaces in each), largest first. A fully
            connected triple-junction graph returns a single-element list equal to the total
            interface count.
    """
    interfaces = mesh_io.list_interfaces(regions)
    junctions, _edge_to_triple = mesh_io.list_junctions(f, regions)
    graph = nx.Graph()
    graph.add_nodes_from(interfaces)
    for a, b, c in junctions:
        pairs = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))]
        for i in range(3):
            for j in range(i + 1, 3):
                graph.add_edge(pairs[i], pairs[j])
    return sorted((len(component) for component in nx.connected_components(graph)), reverse=True)


def main() -> None:
    """Run the connectivity check over every mesh in every converged-mesh folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=CONVERGED_MESHES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report: dict[str, list[dict]] = {}
    for folder in discover_folders(args.root):
        per_stage = []
        for stage_path in sorted((folder / "converged").glob("mesh*.rec")):
            stage = int(stage_path.stem.removeprefix("mesh"))
            _v, f, regions = mesh_io.read_rec(stage_path)
            sizes = triple_junction_component_sizes(f, regions)
            per_stage.append({"stage": stage, "n_interfaces": sum(sizes), "n_components": len(sizes), "sizes": sizes})
        report[folder.name] = per_stage
        n_components_by_stage = [s["n_components"] for s in per_stage]
        print(f"{folder.name}: n_components per stage = {n_components_by_stage}")

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
