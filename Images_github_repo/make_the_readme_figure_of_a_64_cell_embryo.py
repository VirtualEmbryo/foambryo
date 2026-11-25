"""Make the README's first figure: the surface tensions of a 64-cell ascidian embryo, seen from both poles.

    python make_the_readme_figure_of_a_64_cell_embryo.py SEGMENTATION PROPERTIES WORK_DIR OUTPUT.jpg

Data: the public Astec-Pm10 deposit of a *Phallusia mammillata* embryo (Lemaire 2020,
doi:10.6084/m9.figshare.8235485, CC BY 4.0; Guignard et al., Science 2020): SEGMENTATION is its time point 10,
`Astec-pm10_post_t010.inr.gz` (in `Astec-pm10_post.tar.gz`, folder `Astec-pm10/POST/POST_RELEASE/`), PROPERTIES its
`Astec-pm10_properties.xml` (the cells' names). WORK_DIR keeps the mask, the names and the mesh between runs.

Steps, each with foambryo's public API unless said otherwise:

1. The segmentation (INRIMAGE-4, read here) becomes foambryo's mask: its background label (1, on the image's
   faces) becomes 0, the medium; time point 10 has no enclosed pocket of medium (checked). Voxel size: the header's
   (0.3 µm, cubic).
2. The mesh at the segmentation's full resolution: `foambryo.io.reconstruct_mask(mask, spacing, spacing_unit="um")`,
   as `foambryo mesh` does (delaunay-watershed's default reconstruction); written as WORK_DIR/mesh.rec.
3. The relative surface tensions: `infer_tensions(mesh)` (Young-Dupré, mean 1).
4. The bilateral symmetry plane, from the cells' names: in the ascidian nomenclature a trailing `_` marks a left
   cell and `*` its right partner. Normal: the leading eigenvector of sum_k w_k d_k d_k^T over the 32 pairs, d_k the
   vector from the left to the right cell's centroid, w_k the pair's mean volume. The animal-vegetal axis from the
   index of the names (Astec-Pm10 writes them in lower case: index 1-8 are the vegetal A7/B7 cells, 9-16 the animal
   a7/b7 cells), the anterior-posterior one from the lineages (a/A anterior, b/B posterior); both are projected
   into the plane and made orthogonal.
5. Two views with the plane edge-on and vertical in the image, anterior up: from the animal pole (top row) and from
   the vegetal pole (bottom row). Seen from opposite sides, the embryo's left and right swap between the rows. The
   mesh is turned by a rotation (determinant +1, chirality kept) so that each view is the viewer's home view.
6. Each panel is drawn by `foambryo.viewing.plot_tension_inference(..., show=False)`: the whole embryo (left) and
   `scattered=True, scatter_coeff=0.8` (right). One camera for the four panels: Polyscope's scene box is fixed
   (`set_automatically_compute_scene_extents(False)`, `set_bounding_box`) to the union of the two rows' boxes at
   scatter_coeff 0.8, so every panel has the same zoom and scale. The viewer's graph of the cells ("Graph Edges",
   "Graph Nodes") is switched off with Polyscope's `set_enabled(False)`, as unticking these structures in the
   window's "Structures" panel does: the figure is what a user gets in the window. The image is Polyscope's
   rendering without its interface (`screenshot_to_buffer(include_UI=False)`), as the viewer's saved figures are.
7. One scale bar, the viewer's own, drawn under the whole embryo of the top row; its length is written under it,
   at its ends as projected by the camera (`foambryo._viewer_figures`: the viewer's own placement and projection).
8. One colour bar, written by `foambryo.viewing.save_colour_bars` for these tensions (the viewer's range and
   colours) at 600 dpi, its gradient made a third as thick, placed under the four panels.

The colour scale is the same in every panel (the same tensions); no midline is drawn. The figure is 1 800 pixels
wide, for 900 on the page. Needs foambryo 0.8.0 with its viewing extra (Polyscope) and a display.
"""

from __future__ import annotations

import argparse
import gzip
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
from scipy import ndimage

from foambryo import dcel_mesh_from_file, infer_tensions
from foambryo._display_frame import drawn_points
from foambryo._viewer_figures import project_to_pixels, scale_bar_for_points
from foambryo.dcel import DcelData
from foambryo.io import reconstruct_mask
from foambryo.viewing import plot_tension_inference, save_colour_bars

TIME_POINT = 10
SCATTER_COEFF = 0.8
WINDOW_PIXELS = 1400
FIGURE_WIDTH_INCHES, FIGURE_DPI = 9.0, 200  # 1 800 pixels, shown at 900
ROWS = ("animal pole", "vegetal pole")
PANELS = ("whole", "exploded")
#: (x, y, z) as drawn = DRAWN @ (z, y, x), the mesh's columns; DRAWN is its own inverse.
DRAWN = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)


def read_inr(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """An INRIMAGE-4 file (one unsigned value per voxel) as a (z, y, x) array, and its voxel size (VZ, VY, VX)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        raw = handle.read()
    end = raw.index(b"##}\n") + 4
    fields = dict(line.split("=", 1) for line in raw[:end].decode("ascii").splitlines() if "=" in line)
    x, y, z = (int(fields[k]) for k in ("XDIM", "YDIM", "ZDIM"))
    if fields.get("VDIM", "1") != "1" or fields.get("TYPE") != "unsigned fixed":
        message = f"{path.name}: only one unsigned value per voxel is read here"
        raise ValueError(message)
    order = "<" if fields.get("CPU", "decm").lower() in ("decm", "pc", "alpha") else ">"
    dtype = np.dtype(f"{order}u{int(fields['PIXSIZE'].split()[0]) // 8}")
    data = np.frombuffer(raw, dtype=dtype, count=x * y * z, offset=-(-end // 256) * 256)
    return data.reshape(z, y, x).astype(dtype.newbyteorder("=")), tuple(float(fields[k]) for k in ("VZ", "VY", "VX"))


def mask_of_the_segmentation(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Step 1: the background label becomes 0; stops if the medium has an enclosed pocket."""
    volume, spacing = read_inr(path)
    background = int(np.bincount(volume[0].ravel()).argmax())
    if background != 0 and (volume == 0).any():
        message = "label 0 is already used by a cell"
        raise ValueError(message)
    mask = np.where(volume == background, 0, volume).astype(np.uint16)
    if ndimage.label(mask == 0)[1] != 1:
        message = "the medium has an enclosed pocket: this script does not handle one"
        raise ValueError(message)
    return mask, spacing


def names_of_the_cells(path: Path) -> dict[int, str]:
    """Each label's name at TIME_POINT, from the deposit's properties (cell id = time x 10 000 + label)."""
    low, high, names, inside = TIME_POINT * 10_000, (TIME_POINT + 1) * 10_000, {}, False
    for event, element in ElementTree.iterparse(path, events=("start", "end")):  # noqa: S314 - a local file
        if element.tag == "cell_name":
            inside = event == "start"
        elif event == "end" and element.tag == "cell" and inside and low <= int(element.get("cell-id")) < high:
            names[int(element.get("cell-id")) - low] = (element.text or "").strip().strip("'")
        if event == "end":
            element.clear()
    names.pop(1, None)  # the background
    return names


def inputs(args: argparse.Namespace) -> tuple[np.ndarray, tuple[float, ...], dict[int, str], Path]:
    """Steps 1 and 2, kept in the work folder: the mask, its voxel size, the cells' names and the mesh file."""
    args.work.mkdir(parents=True, exist_ok=True)
    mask_file, names_file, mesh_file = (args.work / n for n in ("mask.npz", "names.json", "mesh.rec"))
    if not mask_file.exists():
        mask, spacing = mask_of_the_segmentation(args.segmentation)
        np.savez_compressed(mask_file, mask=mask, spacing=np.asarray(spacing))
    with np.load(mask_file) as data:
        mask, spacing = data["mask"], tuple(float(s) for s in data["spacing"])
    if not names_file.exists():
        names_file.write_text(json.dumps(names_of_the_cells(args.properties)))
    names = {int(k): v for k, v in json.loads(names_file.read_text()).items()}
    if not mesh_file.exists():
        mesh_file.write_bytes(reconstruct_mask(mask, spacing, spacing_unit="um").rec_bytes)
    return mask, spacing, names, mesh_file


def animal_and_anterior_in_the_symmetry_plane(
    mask: np.ndarray, spacing: tuple[float, ...], names: dict[int, str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Step 4: the unit animal and anterior directions in the symmetry plane (mesh columns z, y, x), and the pairs."""
    labels = sorted(names)
    ones = np.ones(mask.shape, np.uint8)
    volumes = ndimage.sum_labels(ones, mask, labels)
    centroids = np.asarray(ndimage.center_of_mass(ones, mask, labels)) * np.asarray(spacing)
    cell = {label: (centroids[i], volumes[i]) for i, label in enumerate(labels)}
    by_name = {name: label for label, name in names.items()}
    pairs = [
        (by_name[name], by_name[name[:-1] + "*"])
        for name in names.values()
        if name.endswith("_") and name[:-1] + "*" in by_name
    ]
    offsets = np.array([cell[right][0] - cell[left][0] for left, right in pairs])
    weights = np.array([(cell[left][1] + cell[right][1]) / 2 for left, right in pairs])
    normal = np.linalg.eigh(np.einsum("k,ki,kj->ij", weights, offsets, offsets))[1][:, -1]
    index = {label: int(names[label].split(".")[1].rstrip("_*")) for label in labels}

    def mean(chosen: list[int]) -> np.ndarray:
        return np.mean([cell[label][0] for label in chosen], axis=0)

    def in_plane(vector: np.ndarray) -> np.ndarray:
        return vector - (vector @ normal) * normal

    animal = in_plane(mean([c for c in labels if index[c] >= 9]) - mean([c for c in labels if index[c] <= 8]))
    animal /= np.linalg.norm(animal)
    lineage = {label: names[label][0].lower() for label in labels}
    anterior = in_plane(mean([c for c in labels if lineage[c] == "a"]) - mean([c for c in labels if lineage[c] == "b"]))
    anterior -= (anterior @ animal) * animal
    anterior /= np.linalg.norm(anterior)
    return animal, anterior, len(pairs)


def exploded_box(
    points: np.ndarray, triangles: np.ndarray, labels: np.ndarray, coeff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The box of the mesh with each cell moved by coeff x (its centroid - the tissue's), as the viewer explodes."""
    low, high = points.min(axis=0), points.max(axis=0)
    for c in set(np.unique(labels).tolist()) - {0}:
        vertices = np.unique(triangles[(labels == c).any(axis=1)])
        moved = points[vertices] + coeff * (points[vertices].mean(axis=0) - points.mean(axis=0))
        low, high = np.minimum(low, moved.min(axis=0)), np.maximum(high, moved.max(axis=0))
    return low, high


def turned_meshes(points: np.ndarray, animal: np.ndarray, anterior: np.ndarray) -> dict[str, np.ndarray]:
    """Step 5: the points turned so that each row's view is Polyscope's home view (look along -z, y up)."""
    camera = ps.get_view_camera_parameters()
    look = DRAWN @ np.asarray(camera.get_look_dir(), float)
    look /= np.linalg.norm(look)
    up = DRAWN @ np.asarray(camera.get_up_dir(), float)
    up -= (up @ look) * look
    up /= np.linalg.norm(up)
    home = np.column_stack([look, up, np.cross(look, up)])
    views = {"animal pole": (-animal, anterior), "vegetal pole": (animal, anterior)}  # (direction looked along, up)
    centre = points.mean(axis=0)
    turned = {}
    for row, (direction, upward) in views.items():
        rotation = home @ np.column_stack([direction, upward, np.cross(direction, upward)]).T
        assert abs(np.linalg.det(rotation) - 1) < 1e-9
        turned[row] = (points - centre) @ rotation.T + centre
    return turned


def render_the_panels(
    mesh: DcelData, tensions: dict, turned: dict[str, np.ndarray],
) -> tuple[dict[tuple[str, str], np.ndarray], tuple[np.ndarray, str]]:
    """Step 6 and the scale bar of step 7: the four screenshots with one camera, and the bar's pixel ends and label."""
    triangles, labels = np.asarray(mesh.f)[:, :3], np.asarray(mesh.f)[:, 3:5]
    boxes = [exploded_box(points, triangles, labels, SCATTER_COEFF) for points in turned.values()]
    low, high = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
    margin = 0.02 * (high - low)
    ps.set_bounding_box(drawn_points(low - margin), drawn_points(high + margin))
    ps.set_length_scale(float(np.linalg.norm(high - low)))
    images, bar = {}, None
    for row, points in turned.items():
        row_mesh = DcelData(points, triangles, labels, media=getattr(mesh, "media", None))
        for panel in PANELS:
            with_bar = (row, panel) == ("animal pole", "whole")
            plot_tension_inference(
                row_mesh,
                tensions,
                scattered=panel == "exploded",
                scatter_coeff=SCATTER_COEFF,
                length_unit="um",
                scale_bar=with_bar,
                show=False,
            )
            ps.get_curve_network("Graph Edges").set_enabled(False)
            ps.get_point_cloud("Graph Nodes").set_enabled(False)
            ps.reset_camera_to_home_view()
            ps.show(forFrames=1)
            images[row, panel] = ps.screenshot_to_buffer(transparent_bg=False, include_UI=False)[..., :3]
            if with_bar:
                scale_bar = scale_bar_for_points(points, "um")
                view = ps.get_view_camera_parameters()
                height, width = images[row, panel].shape[:2]
                ends = project_to_pixels(
                    drawn_points(np.array([scale_bar.start, scale_bar.end])),
                    view.get_view_mat(),
                    view.get_fov_vertical_deg(),
                    width,
                    height,
                )
                bar = (ends, scale_bar.label)
    return images, bar


def colour_bar_image(mesh: DcelData, tensions: dict, path: Path) -> np.ndarray:
    """Step 8: `save_colour_bars`' image at 600 dpi, its margins trimmed and its gradient a third as thick.

    Rows of the gradient are removed: the colours along it are unchanged, and its text stays sharp once scaled.
    """
    with mpl.rc_context({"savefig.dpi": 600}):
        save_colour_bars(mesh, tensions, None, path)
    image = plt.imread(path)[..., :3]
    ink = (image < 0.97).any(axis=2)
    rows, columns = np.nonzero(ink.any(axis=1))[0], np.nonzero(ink.any(axis=0))[0]
    image = image[rows.min() - 8 : rows.max() + 9, columns.min() - 8 : columns.max() + 9]
    probe = image[:, image.shape[1] // 4]  # a column through the gradient's blue half
    gradient = np.nonzero((probe.max(axis=1) - probe.min(axis=1)) > 0.15)[0]
    first, last = gradient.min(), gradient.max()
    return np.concatenate([image[: first + (last - first + 1) // 3], image[last + 1 :]])


def compose(images: dict, bar: tuple[np.ndarray, str], colour_bar: np.ndarray, output: Path) -> None:
    """The 2 x 2 grid (rows: poles; columns: whole, exploded), one crop for all panels, the colour bar under them."""
    background = next(iter(images.values()))[0, 0].astype(int)
    used = np.any([np.abs(image.astype(int) - background).max(axis=2) > 6 for image in images.values()], axis=0)
    ys, xs = np.nonzero(used)
    y0, y1, x0, x1 = max(ys.min() - 12, 0), ys.max() + 12, max(xs.min() - 12, 0), xs.max() + 12
    width, left_label, gap, bar_width = FIGURE_WIDTH_INCHES, 0.035, 0.01, 0.9
    panel_width = (1 - left_label - gap) / 2
    panel_height = panel_width * width * (y1 - y0) / (x1 - x0)
    bar_height = bar_width * width * colour_bar.shape[0] / colour_bar.shape[1]
    height = 0.05 * width + 2 * panel_height + 0.25 + bar_height + 0.1
    figure = plt.figure(figsize=(width, height), dpi=FIGURE_DPI, facecolor=background / 255)
    for i, row in enumerate(ROWS):
        bottom = (bar_height + 0.35 + (1 - i) * panel_height) / height
        figure.text(
            left_label * 0.55,
            bottom + panel_height / height / 2,
            f"{row}, anterior up",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
        )
        for j, panel in enumerate(PANELS):
            axes = figure.add_axes((left_label + j * (panel_width + gap), bottom, panel_width, panel_height / height))
            axes.imshow(images[row, panel][y0:y1, x0:x1], interpolation="lanczos")
            axes.set_axis_off()
            if i == 0:
                title = "whole embryo" if panel == "whole" else f"cells moved apart (scatter_coeff {SCATTER_COEFF:g})"
                axes.set_title(title, pad=4, fontsize=12)
            if (row, panel) == ("animal pole", "whole"):
                ends, label = bar
                axes.text(
                    ends[:, 0].mean() - x0,
                    ends[:, 1].mean() - y0 + 0.045 * (y1 - y0),
                    label,
                    ha="center",
                    va="top",
                    fontsize=11,
                )
    axes = figure.add_axes(((1 - bar_width) / 2, 0.1 / height, bar_width, bar_height / height))
    axes.imshow(colour_bar, interpolation="lanczos")
    axes.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI, facecolor=figure.get_facecolor(), pil_kwargs={"quality": 90})


def main() -> None:
    """Make the figure from the deposit's files (see the module's docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("segmentation", "properties", "work", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    mask, spacing, names, mesh_file = inputs(args)
    animal, anterior, pairs = animal_and_anterior_in_the_symmetry_plane(mask, spacing, names)
    del mask
    mesh = dcel_mesh_from_file(str(mesh_file))
    tensions = infer_tensions(mesh)
    ps.init()
    ps.set_window_size(WINDOW_PIXELS, WINDOW_PIXELS)
    ps.set_automatically_compute_scene_extents(False)
    turned = turned_meshes(np.asarray(mesh.v, float), animal, anterior)
    images, bar = render_the_panels(mesh, tensions, turned)
    colour_bar = colour_bar_image(mesh, tensions, args.work / "colour_bar.png")
    compose(images, bar, colour_bar, args.output)
    shape = plt.imread(args.output).shape[:2]
    print(json.dumps({"pairs": pairs, "cells": len(names), "output": str(args.output), "pixels": list(shape)}))


if __name__ == "__main__":
    main()
