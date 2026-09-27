"""F7 -- before/after mesh quality: triangle min-angle distribution and a visual panel.

`dithered` (the published algorithm, and `dw3d`'s current shipped default) vs. the opt-in
`offset_excluded_linkcheck` ("link checked") variant: the case-level summary (triangle
min-angle p10 rises 25.47deg -> 31.77deg on this case) is measured elsewhere; this figure
goes one level below that summary and plots the actual per-triangle minimum-angle
distribution of one real reconstructed mesh, before and after.

**Input this figure needs, and where to get it.** It reads two already-reconstructed `.rec`
meshes for case `011` from `foambryo_benchmarks/outputs/dw3d_variant_comparison/011/`
(`DW3D_VARIANT_COMPARISON_OUTPUTS` in `figures/_style.py`) rather than reconstructing them
itself. That directory is not committed (it is large, regenerable output, gitignored) and
will not exist until you run the comparison script that produces it:

    PYTHONPATH="../delaunay-watershed-3d/src" uv run python -m foambryo_benchmarks.dw3d_variant_comparison \
        --cases 011

Case `011` (10 cells) is used because it has a registered ground truth alongside both
variants' reconstructions. No dw3d run happens inside *this* script -- it only reads and
measures meshes that must already exist on disk from the command above.

Run with: `uv run python -m foambryo_benchmarks.figures.f7_mesh_quality_before_after` from `foambryo/`.
"""

from __future__ import annotations

import sys

import numpy as np

from foambryo_benchmarks.figures._style import (
    BLUE,
    DW3D_VARIANT_COMPARISON_OUTPUTS,
    DW3D_REPO,
    ORANGE,
    SINGLE_COL_IN,
    apply_style,
    save_fig,
    strip_spines,
)

sys.path.insert(0, str(DW3D_REPO / "src"))

CASE = "011"
BEFORE = ("v0_3", "dithered (published, shipped default)", BLUE)
AFTER = ("offset_excluded_linkcheck", "opt-in link-checked variant", ORANGE)
MESH_DIR = DW3D_VARIANT_COMPARISON_OUTPUTS / CASE


def triangle_min_angles_deg(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Minimum interior angle of every triangle, in degrees, fully vectorised."""
    p_a = points[triangles[:, 0]]
    p_b = points[triangles[:, 1]]
    p_c = points[triangles[:, 2]]

    len_a = np.linalg.norm(p_b - p_c, axis=1)  # opposite vertex A
    len_b = np.linalg.norm(p_c - p_a, axis=1)  # opposite vertex B
    len_c = np.linalg.norm(p_a - p_b, axis=1)  # opposite vertex C

    def angle_from_sides(opp, s1, s2):
        cosang = np.clip((s1**2 + s2**2 - opp**2) / (2 * s1 * s2 + 1e-300), -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    angle_a = angle_from_sides(len_a, len_b, len_c)
    angle_b = angle_from_sides(len_b, len_a, len_c)
    angle_c = angle_from_sides(len_c, len_a, len_b)
    return np.minimum(np.minimum(angle_a, angle_b), angle_c)


def load_mesh(variant: str):
    from dw3d.io import load_rec

    path = MESH_DIR / f"{CASE}_{variant}.rec"
    if not path.exists():
        raise RuntimeError(f"expected dw3d_variant_comparison output mesh not found: {path}")
    points, triangles, _labels = load_rec(path)
    return points, triangles


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    meshes = {}
    min_angles = {}
    for variant, _label, _color in (BEFORE, AFTER):
        points, triangles = load_mesh(variant)
        meshes[variant] = (points, triangles)
        min_angles[variant] = triangle_min_angles_deg(points, triangles)

    fig = plt.figure(figsize=(2.3 * SINGLE_COL_IN, SINGLE_COL_IN * 1.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.05)

    # --- Panel (a): the distribution ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 90, 46)
    for variant, label, color in (BEFORE, AFTER):
        vals = min_angles[variant]
        ax.hist(vals, bins=bins, color=color, alpha=0.55, density=True, label=label,
                edgecolor="none")
        ax.axvline(np.median(vals), color=color, linewidth=1.3, linestyle="--")
    ax.set_xlabel("triangle minimum angle (deg)")
    ax.set_ylabel("density")
    ax.set_title(f"(a) case {CASE}, {min_angles[BEFORE[0]].size} vs. "
                 f"{min_angles[AFTER[0]].size} triangles", fontsize=7.6)
    ax.legend(loc="upper left", fontsize=6.2)
    strip_spines(ax)

    # --- Panels (b), (c): the visual panel, real mesh geometry coloured by min-angle --------
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=60)
    for i, (variant, label, _color) in enumerate((BEFORE, AFTER)):
        ax3d = fig.add_subplot(gs[0, i + 1], projection="3d")
        points, triangles = meshes[variant]
        angles = min_angles[variant]
        # Subsample triangles for a legible, non-overplotted render if the mesh is large.
        rng = np.random.default_rng(0)
        max_tris = 6000
        if triangles.shape[0] > max_tris:
            keep = rng.choice(triangles.shape[0], size=max_tris, replace=False)
        else:
            keep = np.arange(triangles.shape[0])
        verts = points[triangles[keep]]
        face_colors = cmap(norm(angles[keep]))
        collection = Poly3DCollection(verts, facecolor=face_colors, edgecolor="#1a1a19",
                                       linewidths=0.12)
        ax3d.add_collection3d(collection)
        mins, maxs = points.min(axis=0), points.max(axis=0)
        centre = (mins + maxs) / 2
        radius = (maxs - mins).max() / 2
        ax3d.set_xlim(centre[0] - radius, centre[0] + radius)
        ax3d.set_ylim(centre[1] - radius, centre[1] + radius)
        ax3d.set_zlim(centre[2] - radius, centre[2] + radius)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_axis_off()
        ax3d.view_init(elev=20, azim=35)
        panel_letter = "b" if i == 0 else "c"
        ax3d.set_title(f"({panel_letter}) {label}", fontsize=7.6, y=0.95)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.68, 0.06, 0.22, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("triangle min angle (deg)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Mesh quality before/after: v0_3 vs. the current default", fontsize=9.4, y=1.02)
    save_fig(fig, "F7_mesh_quality_before_after")


if __name__ == "__main__":
    main()
