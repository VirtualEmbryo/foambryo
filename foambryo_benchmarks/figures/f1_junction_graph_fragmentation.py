"""F1 -- the junction graph and its fragmentation.

Every `foambryo` tension method except `ForceBalance*`/`Variational`
solves a system coupled through the *triple-junction graph* (interfaces as nodes, an edge
wherever two interfaces share a triple line). A single soft gauge row can pin only one linear
combination of scales; if that graph has `k > 1` connected components -- or carries an
isolated, triple-line-free interface (a closed surface) -- the solver silently collapses the
extra degrees of freedom toward zero. This figure shows, schematically, what "the graph" means
and how it fragments, then shows the real component-size distribution of the worst-fragmenting
mesh in the 72-mesh converged cohort.

Schematic (panel a-c): hand-constructed, not data-driven.
A healthy 3-cell cluster's interface-coupling graph is the octahedron (6 interfaces as nodes,
12 edges -- two interfaces are adjacent iff they share a labelled region; the three
"antipodal" pairs (01,23), (02,13), (03,12) share no region and are never adjacent, which is
exactly what makes the graph an octahedron rather than the complete graph K6).

Real panel (d): `foambryo_benchmarks/baseline/converged_cohort.json`, the worst-fragmenting
case in the cohort (`165LumenA_..._mTEo_inversion`, stage 9: 13 proper components + 130
isolated interfaces, of 498 total) -- one of the 26 of 72 cohort meshes that fragment or carry
an isolated interface, and so one of those the junction-graph fragmentation fix applies to.

Run with: `uv run python -m foambryo_benchmarks.figures.f1_junction_graph_fragmentation` from `foambryo/`.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np

from foambryo_benchmarks.figures._style import (
    BLUE,
    FOAMBRYO_BASELINE,
    ORANGE,
    RED,
    SINGLE_COL_IN,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_style,
    save_fig,
    strip_spines,
)

FRAGMENTED_FOLDER = "165LumenA_AllEqu_Base1B_pTEiup_ICMup_mTo3_pTEmTEsplit15_pTEoup_V2_mTEo_inversion"
FRAGMENTED_STAGE = 9


def octahedron_graph() -> nx.Graph:
    """3-cell (+ exterior) interface-coupling graph: nodes = interfaces, edges = shared region."""
    interfaces = ["01", "02", "03", "12", "13", "23"]
    g = nx.Graph()
    g.add_nodes_from(interfaces)
    for i, a in enumerate(interfaces):
        for b in interfaces[i + 1 :]:
            if set(a) & set(b):
                g.add_edge(a, b)
    return g


def doublet_graph(suffix: str) -> nx.Graph:
    """2-cell (+ exterior) interface-coupling graph: a triangle (complete graph on 3 nodes)."""
    interfaces = [f"0{suffix}", f"1{suffix}", f"01{suffix}"]
    g = nx.Graph()
    g.add_nodes_from(interfaces)
    g.add_edges_from([(interfaces[0], interfaces[1]), (interfaces[0], interfaces[2]),
                       (interfaces[1], interfaces[2])])
    return g


def load_fragmented_mesh_entry() -> dict:
    records = json.loads((FOAMBRYO_BASELINE / "converged_cohort.json").read_text())
    for r in records:
        if r["folder"] == FRAGMENTED_FOLDER and r["stage"] == FRAGMENTED_STAGE:
            return r
    raise RuntimeError("fragmented example mesh not found in converged_cohort.json")


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2 * SINGLE_COL_IN, 2 * SINGLE_COL_IN * 0.62))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.55, wspace=0.35)

    # --- (a) healthy octahedron ------------------------------------------------------------
    # Hexagon layout ordered so antipodal (non-adjacent) pairs sit opposite each other -- the
    # standard way to draw K_{2,2,2}: every vertex connects to all but its antipode.
    ax_a = fig.add_subplot(gs[0, 0])
    g = octahedron_graph()
    hex_order = ["01", "02", "03", "23", "13", "12"]
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 2
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(hex_order, angles)}
    nx.draw_networkx_edges(g, pos, ax=ax_a, edge_color=TEXT_MUTED, width=1.1)
    nx.draw_networkx_nodes(g, pos, ax=ax_a, node_color=BLUE, node_size=320)
    nx.draw_networkx_labels(g, pos, ax=ax_a, font_size=7, font_color="white")
    ax_a.set_title("(a) healthy: 1 component", fontsize=7.6)
    ax_a.set_xlim(-1.4, 1.4)
    ax_a.set_ylim(-1.4, 1.4)
    ax_a.axis("off")

    # --- (b) two separated sub-clusters -----------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    g1, g2 = doublet_graph("a"), doublet_graph("b")
    g = nx.union(g1, g2)
    pos1 = nx.circular_layout(g1, center=(-1.1, 0))
    pos2 = nx.circular_layout(g2, center=(1.1, 0))
    pos = {**pos1, **pos2}
    for comp, color in zip(nx.connected_components(g), [BLUE, ORANGE]):
        sub = g.subgraph(comp)
        nx.draw_networkx_edges(sub, pos, ax=ax_b, edge_color=TEXT_MUTED, width=1.1)
        nx.draw_networkx_nodes(sub, pos, ax=ax_b, node_color=color, node_size=220)
    nx.draw_networkx_labels(g, pos, ax=ax_b, font_size=6, font_color="white")
    ax_b.set_title("(b) split cluster: 2 components", fontsize=7.6)
    ax_b.axis("off")

    # --- (c) a closed interface: an isolated node ------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    g = doublet_graph("")
    g.add_node("closed")
    pos = nx.circular_layout(doublet_graph(""), center=(-0.5, 0))
    pos["closed"] = np.array([1.3, 0.0])
    nx.draw_networkx_edges(g, pos, ax=ax_c, edge_color=TEXT_MUTED, width=1.1)
    nx.draw_networkx_nodes(g, pos, ax=ax_c, nodelist=list(doublet_graph("").nodes),
                            node_color=BLUE, node_size=220)
    nx.draw_networkx_nodes(g, pos, ax=ax_c, nodelist=["closed"], node_color=RED, node_size=320)
    labels_c = {n: n for n in doublet_graph("").nodes}
    nx.draw_networkx_labels(g, pos, ax=ax_c, labels=labels_c, font_size=6, font_color="white")
    ax_c.set_title("(c) closed interface: isolated node", fontsize=7.6)
    ax_c.annotate("no triple line at all\n(e.g. a sealed lumen)", xy=pos["closed"],
                  xytext=(1.3, -0.75), fontsize=5.8, ha="center", color=TEXT_SECONDARY,
                  arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "linewidth": 0.6})
    ax_c.set_xlim(-1.1, 1.9)
    ax_c.axis("off")

    # --- (d) the real, worst-fragmenting cohort mesh ----------------------------------------
    ax_d = fig.add_subplot(gs[1, :])
    entry = load_fragmented_mesh_entry()
    jg = entry["junction_graph"]
    sizes = sorted(jg["component_sizes"], reverse=True)
    labels = [f"C{i+1}" for i in range(len(sizes))] + ["isolated\n(no triple line)"]
    values = sizes + [jg["n_isolated"]]
    colors = [BLUE] * len(sizes) + [RED]
    bars = ax_d.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(values):
        ax_d.text(i, v + 3, str(v), ha="center", fontsize=6.4, color="#33322f")
    ax_d.set_xticks(range(len(values)))
    ax_d.set_xticklabels(labels, fontsize=6.2)
    ax_d.set_ylabel("interfaces")
    ax_d.set_title(
        f"(d) real cohort mesh, worst case: {jg['n_components']} proper components + "
        f"{jg['n_isolated']} isolated, of {entry['n_interfaces']} interfaces total\n"
        f"({FRAGMENTED_FOLDER[:24]}…, stage {FRAGMENTED_STAGE})",
        fontsize=7.2,
    )
    strip_spines(ax_d)

    fig.suptitle("The junction-coupling graph, and how it fragments", fontsize=9.6, y=1.01)
    save_fig(fig, "F1_junction_graph_fragmentation")


if __name__ == "__main__":
    main()
