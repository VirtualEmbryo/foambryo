#!/usr/bin/env python
r"""The old-vs-new `dw3d` comparison, as artefacts rather than a table.

**What this is for.** Changing `dw3d`'s configuration moves inferred tensions by a factor
that depends on which inference method reads the mesh: on the 42 equilibrium cases, going from
the published `dithered` configuration to the former shipped default `offset_included` raised
the gauge-matched median total tension error by +26 % for the angle-reading `YoungDupre`
family and by +450 % for the `M`-inverting `ForceBalanceKKT` family (0.130 -> 0.715). Those
are summary statistics over 42 cases. This script produces the objects behind them, so the
comparison can be *looked at*: for one
segmentation mask it writes the ground-truth mesh and every `dw3d` variant's reconstruction,
each painted with tensions **in one matched gauge**, plus a per-interface
`inferred - ground truth` difference map.

**The matched gauge, and why it is the whole point.** Two gauges are matched, and both are
recorded numerically in the per-case manifest so a reader can check them rather than trust
them:

1. *The tension gauge.* `foambryo.infer_tensions` imposes `mean(gamma) = 1` as a soft
   least-squares row, not a constraint, so each variant's inferred
   mean is only approximately 1 and differs between variants. Every inferred dictionary is
   therefore rescaled by `foambryo_benchmarks.metrics.gauge_matching_scale` -- `mean(truth)/mean(inferred)`
   over shared interfaces -- exactly as every error number in this package is. Without this, part of
   what a reader would see as a difference between variants is pure convention.
2. *The colour gauge.* One `(vmin, vmax)` pair, computed over the union of the ground truth
   and every variant's gauge-matched tensions for that case, is used for every panel of that
   case. The difference maps share one symmetric `(-D, +D)`. **An unmatched colour map would
   make the comparison meaningless**, so the shared range is computed once, applied to all
   panels, and written to the manifest.

Both a *full* gauge (the union's true min/max) and a *robust* gauge (its 5th-95th percentile)
are emitted as separate quantities. The full one is honest about outliers; the robust one stays
readable when one variant produces one extreme interface. Neither is per-panel.

**Two frames, one registration.** `benchmarking-dataset`'s `.rec` ground-truth meshes live in a
normalised frame whose axes run in the opposite order to the `.tif` mask, while `dw3d` emits
voxel coordinates -- so the panels cannot be laid side by side, let alone compared, without the
fit. This script reuses `delaunay-watershed-3d/dw3d_benchmarks/metrics.similarity_to_mask_frame`
(Umeyama, label-keyed correspondences, fitted **mask-to-ground-truth only** so it never sees a
reconstruction and no variant can improve its score by moving the fit) and records the fitted
scale and its residual in the manifest. The ground-truth artefacts this script writes are in
the **registered (voxel) frame**, i.e. they overlay the reconstructions; the inverse transform
is in the manifest.

**Rendering is `foambryo`'s, not new.** Every painted panel goes through
`foambryo.plotting_utilities.view_faces_values_on_embryo` -- the same function
`foambryo.viewing.plot_tension_inference` paints tensions with -- with `adapt_values=False` so
that this module, not the viewer, owns the shared gauge. The interactive path (`--interactive`)
is the deliverable; the PNGs are for a report.

**Nothing is claimed to exist without being checked.** `_assert_mesh_sane` runs on every mesh
before it is written (non-empty, finite, in-range indices, degenerate-triangle count reported),
every written mesh file is **read back** and its counts compared, and every PNG is reopened and
checked to be non-empty *and* non-uniform. A zero-byte file or a blank render fails the run
rather than being reported as a success.

Usage (from `foambryo/`, both packages needed)::

    PYTHONPATH="../delaunay-watershed-3d/src" uv run python -m foambryo_benchmarks.dw3d_variant_comparison \
        --cases 002 023 011  # writes to foambryo_benchmarks/outputs/dw3d_variant_comparison by default

    # ... and to look at one case interactively:
    PYTHONPATH="../delaunay-watershed-3d/src" uv run python -m foambryo_benchmarks.dw3d_variant_comparison \
        --cases 023 --interactive --no-png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import skimage.io
from numpy.typing import NDArray

from foambryo_benchmarks import case_groups
from foambryo_benchmarks import dw3d_variants as dv
from foambryo_benchmarks import metrics as bm
from foambryo import (
    TensionComputationMethod,
    dcel_mesh_from_multimaterial_mesh,
    infer_tensions,
)

#: Panel key for the ground-truth mesh painted with the ground-truth tensions.
TRUTH = "truth"
#: Panel key for the ground-truth mesh painted with tensions *inferred on it* -- the
#: `E_inference` panel. Same geometry as `TRUTH`; a separate panel so the two sit side by side.
TRUTH_INFERRED = "truth_inferred"

#: Tension methods painted. Two, deliberately, and not one: the two families respond to the
#: same variant switch by +26 % and +450 % (see the module docstring), so a
#: single-method figure would misrepresent the comparison whichever family it picked.
#:
#: Read every `ForceBalanceKKT` panel and difference map with the measured floor: on `dw3d`
#: reconstructions `rho_free` is about 50x the reference meshes' own (median 0.578 against
#: 0.0118 on the 40-case equilibrium cohort, all above the 0.2 alarm threshold), so the
#: force-balance panels compare two fits to geometries that no tension assignment balances,
#: and a difference between variants through this family is a difference between two such
#: floors. See `foambryo.force_balance_inference`'s module docstring.
METHOD_REGISTRY = {
    "YoungDupre": lambda mesh: infer_tensions(mesh, 1.0, TensionComputationMethod.YoungDupre),
    "ForceBalanceKKT": lambda mesh: infer_tensions(mesh, 1.0, TensionComputationMethod.ForceBalance),
}

#: Fixed camera directions (unit vectors) for the static PNGs. Fixed rather than auto-framed so
#: that two runs, and two variants, are photographed identically. All three are kept close to
#: the `+z` axis on purpose: the panels are laid out along `x`, so a camera with a large `x`
#: component looks *down the row* and the panels occlude each other. None is collinear with the
#: up direction (`neg_y_up`), which is what polyscope's `lookAt()` rejects.
VIEWS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 0.0, 1.0),
    "oblique_up": (0.22, 0.42, 0.88),
    "oblique_down": (-0.22, -0.42, 0.88),
}


class ArtefactError(RuntimeError):
    """Raised when an artefact this script claims to have written is empty or degenerate."""


# ---------------------------------------------------------------------------
# dw3d-side registration, imported by path rather than by package import
# ---------------------------------------------------------------------------


def load_dw3d_benchmark_metrics(dw3d_repo_root: Path) -> object:
    """Load `delaunay-watershed-3d/dw3d_benchmarks/metrics.py` under a non-colliding module name.

    **Why by path and not by import.** This predates both repositories' benchmark packages
    being renamed to their current, distinct names (`foambryo_benchmarks`, `dw3d_benchmarks`)
    -- at the time this was written, both were a top-level package called `benchmarks`, and a
    plain `import benchmarks.metrics` from `foambryo/` with the sibling repo on `sys.path`
    resolved to whichever directory came first, silently. That specific collision cannot
    recur under the current names; loading by path is kept anyway as the smallest fix, not
    because it is still strictly required. `dw3d`'s `dw3d_benchmarks/metrics.py` imports only
    from `dw3d` itself (checked), so loading it by file path under the name
    `dw3d_benchmark_metrics` is safe and keeps the registration this script needs as the *one*
    fitted definition the whole project uses, rather than a second copy of Umeyama's formula.

    Args:
        dw3d_repo_root (Path): The `delaunay-watershed-3d` checkout root.

    Returns:
        object: The loaded module, exposing `similarity_to_mask_frame`.

    Raises:
        ArtefactError: If the file is absent, or loads without the function this needs.
    """
    path = dw3d_repo_root / "dw3d_benchmarks" / "metrics.py"
    if not path.exists():
        message = f"{path} not found; pass --dw3d-repo-root pointing at the delaunay-watershed-3d checkout"
        raise ArtefactError(message)
    spec = importlib.util.spec_from_file_location("dw3d_benchmark_metrics", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        message = f"could not build an import spec for {path}"
        raise ArtefactError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dw3d_benchmark_metrics"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "similarity_to_mask_frame"):
        message = f"{path} loaded but has no similarity_to_mask_frame"
        raise ArtefactError(message)
    return module


# ---------------------------------------------------------------------------
# Discipline: nothing is claimed to exist without being checked
# ---------------------------------------------------------------------------


def mesh_sanity(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> dict:
    """Describe a multimaterial mesh's non-emptiness and non-degeneracy, as numbers.

    Args:
        points (NDArray[np.float64]): `(nv, 3)` vertices.
        triangles (NDArray[np.int64]): `(nf, 3)` triangles.
        labels (NDArray[np.int64]): `(nf, 2)` material pair per triangle.

    Returns:
        dict: Counts and extrema, including `n_zero_area_triangles` and
            `min_triangle_area`, so a caller can record *how* non-degenerate a mesh is
            rather than only that an assertion passed.
    """
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return {
        "n_points": len(points),
        "n_triangles": len(triangles),
        "n_interfaces": len({tuple(sorted(pair)) for pair in labels.tolist()}),
        "n_zero_area_triangles": int((areas == 0.0).sum()),
        "min_triangle_area": float(areas.min()) if len(areas) else float("nan"),
        "total_area": float(areas.sum()),
        "bbox_min": [float(x) for x in points.min(axis=0)],
        "bbox_max": [float(x) for x in points.max(axis=0)],
        "n_degenerate_index_triangles": int((triangles[:, 0] == triangles[:, 1]).sum()
                                            + (triangles[:, 1] == triangles[:, 2]).sum()
                                            + (triangles[:, 0] == triangles[:, 2]).sum()),
    }


def assert_mesh_sane(name: str, sanity: dict, points: NDArray, triangles: NDArray) -> None:
    """Fail loudly if a mesh about to be written is empty, non-finite or index-degenerate.

    Zero-area triangles are *reported* rather than fatal (a primal Delaunay extraction can
    legitimately produce a sliver), but a mesh with no points, no triangles, a non-finite
    coordinate, an out-of-range index or a repeated vertex within one triangle is not an
    artefact and must not be written and announced.

    Args:
        name (str): Artefact name, for the error message.
        sanity (dict): Output of `mesh_sanity`.
        points (NDArray): The vertices.
        triangles (NDArray): The triangles.

    Raises:
        ArtefactError: On any of the fatal conditions above.
    """
    problems = []
    if sanity["n_points"] == 0:
        problems.append("0 points")
    if sanity["n_triangles"] == 0:
        problems.append("0 triangles")
    if not np.isfinite(points).all():
        problems.append("non-finite coordinates")
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(points)):
        problems.append(f"triangle index out of range [0, {len(points)})")
    if sanity["n_degenerate_index_triangles"]:
        problems.append(f"{sanity['n_degenerate_index_triangles']} triangles with a repeated vertex")
    if problems:
        message = f"{name} is not a usable mesh: {', '.join(problems)}"
        raise ArtefactError(message)


def assert_png_sane(path: Path, min_bytes: int = 4096) -> dict:
    """Reopen a written PNG and check it is neither empty nor a single flat colour.

    A renderer that silently failed to bind a framebuffer writes a valid PNG of uniform
    background, which passes every file-size check. So the pixels are read back and their
    standard deviation asserted non-zero -- the cheapest test that distinguishes "an image of
    the mesh" from "an image of nothing".

    Args:
        path (Path): The PNG just written.
        min_bytes (int, optional): Lower bound on file size. Defaults to 4096.

    Returns:
        dict: `{"bytes", "shape", "pixel_std", "n_distinct_colours_sampled"}`.

    Raises:
        ArtefactError: If the file is missing, too small, unreadable, or flat.
    """
    import matplotlib.image as mpimg

    if not path.exists():
        message = f"{path} was not written"
        raise ArtefactError(message)
    size = path.stat().st_size
    if size < min_bytes:
        message = f"{path} is {size} bytes, below the {min_bytes}-byte floor - almost certainly a blank render"
        raise ArtefactError(message)
    image = mpimg.imread(path)
    std = float(image[..., :3].std())
    flat = image[..., :3].reshape(-1, 3)
    stride = max(1, len(flat) // 20000)
    n_distinct = len(np.unique(np.round(flat[::stride] * 255).astype(np.uint8), axis=0))
    if std == 0.0 or n_distinct < 3:
        message = f"{path} is a flat image (pixel std {std}, {n_distinct} distinct sampled colours) - nothing rendered"
        raise ArtefactError(message)
    return {"bytes": int(size), "shape": list(image.shape), "pixel_std": std, "n_distinct_colours_sampled": n_distinct}


# ---------------------------------------------------------------------------
# Writing the artefacts
# ---------------------------------------------------------------------------


def write_rec(path: Path, points: NDArray, triangles: NDArray, labels: NDArray) -> dict:
    """Write a binary `.rec` (the format Simba-Viewer and the dataset itself use) and verify it.

    Binary rather than text to match `benchmarking-dataset/*_mesh.rec` exactly (checked: the
    dataset's own files are binary), since that is the variant known to open. The file is read
    back with `dw3d.io.load_rec` and its arrays compared to what was handed in, so "wrote a
    .rec" is a verified statement.

    Args:
        path (Path): Destination.
        points (NDArray): `(nv, 3)` vertices.
        triangles (NDArray): `(nf, 3)` triangles.
        labels (NDArray): `(nf, 2)` material pairs.

    Returns:
        dict: `{"path", "bytes", "roundtrip_n_points", "roundtrip_n_triangles"}`.

    Raises:
        ArtefactError: If the round-trip does not reproduce the arrays.
    """
    from dw3d.io import load_rec, save_rec

    save_rec(path, np.ascontiguousarray(points, dtype=np.float64), triangles, labels, binary_mode=True)
    back_points, back_triangles, back_labels = load_rec(path)
    if not (
        np.array_equal(back_triangles, triangles)
        and np.array_equal(back_labels, labels)
        and np.allclose(back_points, points, rtol=0, atol=0)
    ):
        message = f"{path} did not round-trip through load_rec"
        raise ArtefactError(message)
    if path.stat().st_size == 0:
        message = f"{path} is 0 bytes"
        raise ArtefactError(message)
    return {
        "path": path.name,
        "bytes": int(path.stat().st_size),
        "roundtrip_n_points": len(back_points),
        "roundtrip_n_triangles": len(back_triangles),
    }


def write_vtk(
    path: Path,
    points: NDArray,
    triangles: NDArray,
    labels: NDArray,
    face_scalars: dict[str, NDArray[np.float64]],
) -> dict:
    """Write a legacy `.vtk` carrying the material labels **and** every per-face scalar.

    `dw3d.io.save_vtk` writes only `label1`/`label2`, and this deliverable's point is the
    painted scalar, so the `meshio.Mesh` is assembled here with the tension and difference
    fields as additional cell data. ParaView and MeshLab both read this; the file is read back
    and its cell-data arrays compared, again so the claim is verified.

    NaN is used for "this interface has no ground-truth counterpart" (a reconstruction can
    invent an interface the truth does not have) rather than 0, which would read as a real
    tension of zero.

    Args:
        path (Path): Destination, `.vtk`.
        points (NDArray): `(nv, 3)` vertices.
        triangles (NDArray): `(nf, 3)` triangles.
        labels (NDArray): `(nf, 2)` material pairs.
        face_scalars (dict[str, NDArray[np.float64]]): Name -> `(nf,)` per-face field.

    Returns:
        dict: `{"path", "bytes", "cell_data_fields", "roundtrip_n_triangles"}`.

    Raises:
        ArtefactError: If a scalar has the wrong length, or the round-trip disagrees.
    """
    import meshio

    for name, values in face_scalars.items():
        if len(values) != len(triangles):
            message = f"{path}: face scalar {name!r} has {len(values)} values for {len(triangles)} triangles"
            raise ArtefactError(message)

    cell_data = {
        "label1": [labels[:, 0].astype(np.int32)],
        "label2": [labels[:, 1].astype(np.int32)],
        **{name: [np.asarray(values, dtype=np.float64)] for name, values in face_scalars.items()},
    }
    mesh = meshio.Mesh(np.asarray(points, dtype=np.float64), [("triangle", np.asarray(triangles))], cell_data=cell_data)
    mesh.write(path, binary=False)

    back = meshio.read(path)
    back_triangles = back.get_cells_type("triangle")
    if len(back_triangles) != len(triangles):
        message = f"{path} round-tripped {len(back_triangles)} triangles, expected {len(triangles)}"
        raise ArtefactError(message)
    for name in face_scalars:
        if len(back.get_cell_data(name, "triangle")) != len(triangles):
            message = f"{path} lost cell-data field {name!r} on round-trip"
            raise ArtefactError(message)
    if path.stat().st_size == 0:
        message = f"{path} is 0 bytes"
        raise ArtefactError(message)
    return {
        "path": path.name,
        "bytes": int(path.stat().st_size),
        "cell_data_fields": sorted(["label1", "label2", *face_scalars]),
        "roundtrip_n_triangles": len(back_triangles),
    }


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


def per_face_values(
    label_pairs: NDArray[np.int64],
    dict_values: dict[tuple[int, int], float],
) -> NDArray[np.float64]:
    """Broadcast an interface-keyed dictionary onto one value per triangle.

    Missing keys become `NaN`, never 0: a reconstruction can carry an interface the ground
    truth has no entry for, and painting that grey ("no reference") rather than dark blue
    ("tension zero") is the difference between a readable figure and a misleading one.

    Args:
        label_pairs (NDArray[np.int64]): `(nf, 2)` material pair per triangle, sorted ascending
            (which is what `DcelData.f[:, 3:5]` gives).
        dict_values (dict[tuple[int, int], float]): Interface -> value.

    Returns:
        NDArray[np.float64]: `(nf,)` values, `NaN` where the interface is not in the dict.
    """
    normalised = {(int(a), int(b)): float(v) for (a, b), v in dict_values.items()}
    out = np.full(len(label_pairs), np.nan)
    for i, (a, b) in enumerate(label_pairs.tolist()):
        key = (int(a), int(b)) if a <= b else (int(b), int(a))
        value = normalised.get(key)
        if value is not None:
            out[i] = value
    return out


def gauge_over(values: list[NDArray[np.float64]], robust_percentile: float = 5.0) -> dict:
    """One shared `(vmin, vmax)` over every panel's values -- the matched colour gauge.

    Args:
        values (list[NDArray[np.float64]]): One array per panel/method; `NaN` ignored.
        robust_percentile (float, optional): Tail percentage trimmed for the robust gauge.
            Defaults to 5.0.

    Returns:
        dict: `{"vmin", "vmax", "robust_vmin", "robust_vmax", "n_finite"}`. Degenerate input
            (all-NaN, or a constant field) still yields a usable, non-zero-width range so the
            colour map does not divide by zero.
    """
    pooled = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in values]) if values else np.array([])
    finite = pooled[np.isfinite(pooled)]
    if len(finite) == 0:
        return {"vmin": 0.0, "vmax": 1.0, "robust_vmin": 0.0, "robust_vmax": 1.0, "n_finite": 0}
    vmin, vmax = float(finite.min()), float(finite.max())
    rmin = float(np.percentile(finite, robust_percentile))
    rmax = float(np.percentile(finite, 100.0 - robust_percentile))
    if vmax == vmin:
        vmax = vmin + 1.0
    if rmax == rmin:
        rmin, rmax = vmin, vmax
    return {"vmin": vmin, "vmax": vmax, "robust_vmin": rmin, "robust_vmax": rmax, "n_finite": len(finite)}


def symmetric_gauge(
    values: list[NDArray[np.float64]],
    robust_percentile: float = 95.0,
    robust_over: list[NDArray[np.float64]] | None = None,
) -> dict:
    """A shared symmetric `(-D, +D)` gauge for the difference maps, so 0 is the colour centre.

    Args:
        values (list[NDArray[np.float64]]): One difference array per panel/method; sets the full
            gauge.
        robust_percentile (float, optional): Percentile of `|diff|` used for the robust
            half-width. Defaults to 95.0.
        robust_over (list[NDArray[np.float64]] | None, optional): Restrict the robust gauge's
            sample to these arrays. Used to exclude reference panels whose difference field is
            zero *by construction* and would otherwise halve the percentile's sample with
            zeros. Defaults to `values`.

    Returns:
        dict: `{"vmin", "vmax", "robust_vmin", "robust_vmax", "n_finite", "n_finite_robust",
            "max_abs"}`.
    """
    def _finite(arrays: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        pooled = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in arrays]) if arrays else np.array([])
        return pooled[np.isfinite(pooled)]

    finite = _finite(values)
    if len(finite) == 0:
        return {"vmin": -1.0, "vmax": 1.0, "robust_vmin": -1.0, "robust_vmax": 1.0,
                "n_finite": 0, "n_finite_robust": 0, "max_abs": 0.0}
    finite_robust = _finite(robust_over) if robust_over else finite
    if len(finite_robust) == 0:
        finite_robust = finite
    d = float(np.abs(finite).max()) or 1.0
    r = float(np.percentile(np.abs(finite_robust), robust_percentile)) or d
    return {
        "vmin": -d,
        "vmax": d,
        "robust_vmin": -r,
        "robust_vmax": r,
        "n_finite": len(finite),
        "n_finite_robust": len(finite_robust),
        "max_abs": float(np.abs(finite).max()),
    }


def build_panel(
    key: str,
    dcel: object,
    dict_truth: dict[tuple[int, int], float],
    painted_by_method: dict[str, dict[tuple[int, int], float]],
    explode_coeff: float = 0.30,
) -> dict:
    """Assemble one panel: its mesh, its gauge-matched tensions and its difference fields.

    Also builds the **exploded** copy of the same mesh, via `DcelData.compute_scattered_arrays`
    — the same displacement `foambryo.viewing`'s own `scattered=True` path uses. This is not
    decoration: on a whole mesh only the outermost cell-medium films are visible, so the
    cell-cell interfaces, which are where the tension differences between `dw3d` variants
    actually live, would be hidden inside the object. Every scalar field is carried onto the
    exploded copy through `idx_scattered`, so the two views show the same numbers.

    Args:
        key (str): Panel key, e.g. `"dithered"`.
        dcel (object): The `DcelData` for this panel. Its own `v` / `f` are used for both the
            painting and the written artefact, so the two cannot drift apart.
        dict_truth (dict[tuple[int, int], float]): Ground-truth tensions.
        painted_by_method (dict[str, dict]): Method -> the *unscaled* tension dict to paint.
            For the `TRUTH` panel this is `{method: dict_truth}` for every method, so its
            difference map is identically zero and serves as the visual zero reference.
        explode_coeff (float, optional): Displacement coefficient for the exploded copy.
            Defaults to 0.30.

    Returns:
        dict: `{"key", "points", "triangles", "labels", "sanity", "fields", "scales",
            "interface_table", "exploded"}`, where `fields` maps a field name
            (`tension_<method>`, `diff_<method>`) to a `(nf,)` array.
    """
    points = np.asarray(dcel.v)
    triangles = np.asarray(dcel.f[:, :3])
    labels = np.asarray(dcel.f[:, 3:5])

    fields: dict[str, NDArray[np.float64]] = {}
    scales: dict[str, float] = {}
    matched_by_method: dict[str, dict[tuple[int, int], float]] = {}

    fields["ground_truth_tension"] = per_face_values(labels, dict_truth)

    for method, dict_inferred in painted_by_method.items():
        scale = bm.gauge_matching_scale(dict_inferred, dict_truth)
        matched = {k: scale * v for k, v in dict_inferred.items()}
        scales[method] = float(scale)
        matched_by_method[method] = matched
        fields[f"tension_{method}"] = per_face_values(labels, matched)
        fields[f"diff_{method}"] = fields[f"tension_{method}"] - fields["ground_truth_tension"]

    interfaces = sorted({(int(min(a, b)), int(max(a, b))) for a, b in labels.tolist()})
    interface_table = []
    for a, b in interfaces:
        row: dict = {"interface": f"{a}-{b}", "ground_truth": dict_truth.get((a, b))}
        for method, matched in matched_by_method.items():
            value = matched.get((a, b))
            row[f"tension_{method}"] = value
            truth_value = dict_truth.get((a, b))
            row[f"diff_{method}"] = None if (value is None or truth_value is None) else value - truth_value
        row["has_ground_truth"] = (a, b) in dict_truth
        interface_table.append(row)

    sanity = mesh_sanity(points, triangles, labels)
    assert_mesh_sane(key, sanity, points, triangles)

    dcel.compute_scattered_arrays(explode_coeff)
    exploded_points = np.asarray(dcel.v_scattered)
    exploded_triangles = np.asarray(dcel.f_scattered)[:, :3].astype(np.int64)
    exploded_idx = np.asarray(dcel.idx_scattered).astype(np.int64)
    if len(exploded_idx) != len(exploded_triangles) or exploded_idx.max(initial=-1) >= len(triangles):
        message = f"{key}: the exploded copy's face index map does not address the panel's faces"
        raise ArtefactError(message)
    exploded_sanity = mesh_sanity(exploded_points, exploded_triangles, labels[exploded_idx])
    assert_mesh_sane(f"{key} (exploded)", exploded_sanity, exploded_points, exploded_triangles)

    return {
        "key": key,
        "points": points,
        "triangles": triangles,
        "labels": labels,
        "sanity": sanity,
        "fields": fields,
        "scales": scales,
        "interface_table": interface_table,
        "exploded": {
            "points": exploded_points,
            "triangles": exploded_triangles,
            "face_index": exploded_idx,
            "coeff": float(explode_coeff),
            "sanity": exploded_sanity,
        },
    }


def register_truth_to_voxel_frame(
    mask: NDArray[np.integer],
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    dw3d_metrics: object,
) -> tuple[NDArray[np.float64], dict]:
    """Map ground-truth `.rec` vertices into the mask's voxel frame, and report the fit.

    Args:
        mask (NDArray[np.integer]): The segmentation mask (defines the destination frame).
        points (NDArray[np.float64]): Ground-truth vertices, normalised frame.
        triangles (NDArray[np.int64]): Ground-truth triangles.
        labels (NDArray[np.int64]): Ground-truth material pairs.
        dw3d_metrics (object): Module from `load_dw3d_benchmark_metrics`.

    Returns:
        tuple[NDArray[np.float64], dict]: Registered vertices, and the fit record (scale,
            rotation, translation, correspondence count and residual in voxels).

    Raises:
        ArtefactError: If the fit cannot be formed, since every downstream comparison and the
            side-by-side layout depend on it and a silent fallback to the unregistered frame
            would put the reference panel at 1/200 of the reconstructions' size.
    """
    fit = dw3d_metrics.similarity_to_mask_frame(mask, points, triangles, labels)
    if fit.get("scale") is None:
        message = (
            f"the ground-truth-to-mask registration could not be fitted "
            f"({fit.get('n_correspondences')} correspondences); the reference panel would be "
            "in a different frame from the reconstructions, so this case is not comparable"
        )
        raise ArtefactError(message)
    scale, rotation, translation = float(fit["scale"]), np.asarray(fit["rotation"]), np.asarray(fit["translation"])
    registered = (scale * (rotation @ points.T)).T + translation
    record = {
        "scale": scale,
        "rotation": [[float(x) for x in row] for row in rotation],
        "translation": [float(x) for x in translation],
        "determinant": float(np.linalg.det(rotation)),
        "n_correspondences": int(fit["n_correspondences"]),
        "residual_mean_voxels": float(fit["residual_mean_voxels"]),
        "residual_max_voxels": float(fit["residual_max_voxels"]),
        "note": (
            "ground-truth artefacts are written in this registered (voxel) frame so they "
            "overlay the reconstructions; invert with p_native = rotation^T (p - translation) / scale"
        ),
    }
    return registered, record


# ---------------------------------------------------------------------------
# Polyscope: the interactive deliverable, and the PNGs
# ---------------------------------------------------------------------------


def _bad_colormap(name: str) -> object:
    """A matplotlib colormap whose NaN colour is an explicit neutral grey.

    `NaN` marks "no ground-truth counterpart for this interface". Left at matplotlib's default
    the bad colour is fully transparent, which polyscope renders as black -- indistinguishable
    from the low end of `jet`. Grey is unambiguous against both `jet` and `coolwarm`.
    """
    import matplotlib as mpl

    cmap = mpl.colormaps[name].with_extremes(bad=(0.58, 0.58, 0.58, 1.0))
    return cmap


def _quantity_specs(methods: list[str], gauges: dict) -> list[dict]:
    """Enumerate the painted quantities: `(field, gauge flavour)` with its shared range.

    Args:
        methods (list[str]): Tension method names.
        gauges (dict): `{"tension": ..., "diff": ...}`.

    Returns:
        list[dict]: One spec per quantity, each carrying the *shared* `lo`/`hi` so that no
            caller can accidentally normalise a panel against its own extrema.
    """
    specs = []
    for method in methods:
        for field, gauge_key, cmap_name in (
            (f"tension_{method}", "tension", "jet"),
            (f"diff_{method}", "diff", "coolwarm"),
        ):
            for flavour, lo_key, hi_key in (("full", "vmin", "vmax"), ("robust", "robust_vmin", "robust_vmax")):
                lo, hi = float(gauges[gauge_key][lo_key]), float(gauges[gauge_key][hi_key])
                specs.append(
                    {
                        "field": field,
                        "gauge_key": gauge_key,
                        "flavour": flavour,
                        "cmap": cmap_name,
                        "lo": lo,
                        "hi": hi,
                        "name": f"{field} [{flavour} gauge {lo:.4g}..{hi:.4g}]",
                    },
                )
    return specs


def _normalise_to_gauge(values: NDArray[np.float64], spec: dict) -> NDArray[np.float64]:
    """Map values into `[0, 1]` against the **shared** gauge, keeping `NaN` as `NaN`."""
    normalised = np.clip((values - spec["lo"]) / (spec["hi"] - spec["lo"]), 0.0, 1.0)
    normalised[~np.isfinite(values)] = np.nan
    return normalised


def _paint(panels: list[dict], ps_meshes: dict, spec: dict) -> None:
    """Paint one quantity onto every panel (whole and exploded), through `foambryo`'s viewer.

    `polyscope`'s Python binding returns no handle from `add_color_quantity` and has no
    "activate this quantity" call, so selecting which field a screenshot shows is done by
    re-adding it under the same name with `enabled=True` (which replaces it and demotes the
    others). Re-adding is a colormap evaluation and a buffer upload, i.e. cheap.

    Args:
        panels (list[dict]): Panels from `build_panel`.
        ps_meshes (dict): Structure name -> registered `ps.SurfaceMesh`.
        spec (dict): One entry of `_quantity_specs`.
    """
    from foambryo.plotting_utilities import view_faces_values_on_embryo

    for panel in panels:
        values = panel["fields"][spec["field"]]
        for structure_key, face_values, shim in (
            (panel["key"], values, _DcelShim(panel["points"], panel["triangles"], panel["labels"])),
            (
                f"{panel['key']}::exploded",
                values[panel["exploded"]["face_index"]],
                _DcelShim(
                    panel["exploded"]["points"],
                    panel["exploded"]["triangles"],
                    panel["labels"][panel["exploded"]["face_index"]],
                ),
            ),
        ):
            view_faces_values_on_embryo(
                shim,
                _normalise_to_gauge(face_values, spec),
                name_values=spec["name"],
                ps_mesh=ps_meshes[structure_key],
                colormap=_bad_colormap(spec["cmap"]),
                adapt_values=False,
                clean_before=False,
                clean_after=False,
                show=False,
            )


def _scene_aspect(
    points: NDArray[np.float64],
    direction: tuple[float, float, float],
    up: NDArray[np.float64],
) -> float:
    """The scene's width/height as it projects onto the screen for one view direction.

    Args:
        points (NDArray[np.float64]): Every point in the scene.
        direction (tuple[float, float, float]): Camera direction.
        up (NDArray[np.float64]): Viewer up vector.

    Returns:
        float: Projected width divided by projected height (1.0 if the height degenerates).
    """
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    screen_up = np.cross(forward, right)
    centred = points - 0.5 * (points.min(axis=0) + points.max(axis=0))
    half_h = float(np.abs(centred @ screen_up).max())
    half_w = float(np.abs(centred @ right).max())
    return (half_w / half_h) if half_h > 0 else 1.0


def _camera_for(
    points: NDArray[np.float64],
    direction: tuple[float, float, float],
    up: NDArray[np.float64],
    fov_vertical_deg: float,
    aspect: float,
    margin: float = 1.10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Place the camera so the whole scene fits the frame, for a *fixed* view direction.

    The distance is solved rather than guessed: the scene is projected onto the camera's own
    screen basis, and the distance is the larger of the two requirements (vertical fov, and
    horizontal fov derived from it and the aspect ratio), plus the scene's half-depth. A guessed
    multiple of the bounding-sphere radius leaves a wide, panel-row-shaped scene floating in the
    middle of a mostly empty image, which is what makes a comparison figure unreadable.

    Args:
        points (NDArray[np.float64]): Every point in the scene.
        direction (tuple[float, float, float]): Camera direction (from target towards camera).
        up (NDArray[np.float64]): The viewer's up vector.
        fov_vertical_deg (float): Polyscope's vertical field of view.
        aspect (float): Window width / height.
        margin (float, optional): Framing slack. Defaults to 1.10.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: `(camera_position, target)`.
    """
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    screen_up = np.cross(forward, right)

    target = 0.5 * (points.min(axis=0) + points.max(axis=0))
    centred = points - target
    tan_v = np.tan(np.radians(fov_vertical_deg) / 2.0)
    tan_h = aspect * tan_v

    # Solved per point, not from the bounding box: a point at screen offset `w` and depth `d`
    # (measured towards the camera) needs `distance >= |w| / tan_h + d`. Taking the max of the
    # bounding box's half-width and its half-depth *separately* is conservative by the depth of
    # the whole scene and leaves the content floating in the middle of the frame.
    offset_w = np.abs(centred @ right)
    offset_h = np.abs(centred @ screen_up)
    depth = centred @ forward
    distance = float(np.max(np.maximum(offset_w / tan_h, offset_h / tan_v) * margin + depth))
    return target + forward * distance, target


def render_case(  # noqa: C901 - one linear scene-assembly procedure; splitting it would hide the order
    case_id: str,
    panels: list[dict],
    gauges: dict,
    png_dir: Path,
    methods: list[str],
    interactive: bool,
    write_png: bool,
    resolution: tuple[int, int] = (1800, 900),
) -> dict:
    """Register every panel in one polyscope scene, laid out in a row, and photograph it.

    The painting goes through `foambryo.plotting_utilities.view_faces_values_on_embryo` (the
    function `foambryo.viewing` itself uses) with `adapt_values=False`, so the normalisation to
    `[0, 1]` -- and therefore the gauge -- is done here, once, from the shared range, and never
    per panel by the viewer. A raw scalar quantity carrying the *unnormalised* values with an
    explicit shared `vminmax` is added alongside, so polyscope's own colour bar reports the
    tension numbers rather than a 0-1 axis.

    Args:
        case_id (str): Case id, used in structure and file names.
        panels (list[dict]): Output of `build_panel`, in display order.
        gauges (dict): `{"tension": gauge, "diff": gauge}` from `gauge_over`/`symmetric_gauge`.
        png_dir (Path): Directory for the static renders.
        methods (list[str]): Method names painted.
        interactive (bool): Open the polyscope window and block.
        write_png (bool): Emit the static renders.
        resolution (tuple[int, int], optional): PNG size. Defaults to (1800, 900).

    Returns:
        dict: `{"panel_offsets", "panel_pitch", "views", "quantities", "png"}`.

    Raises:
        ArtefactError: If two different quantities photograph byte-identically from the same
            camera, which would mean the quantity selection had silently not taken effect and
            the PNG set was a set of copies rather than a comparison.
    """
    import polyscope as ps

    ps.init()
    ps.remove_all_structures()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("neg_y_up")
    ps.set_SSAA_factor(2)
    up = np.array([0.0, -1.0, 0.0])

    spans = np.array([np.ptp(np.asarray(p["points"]), axis=0) for p in panels])
    pitch = float(spans[:, 0].max()) * 1.20
    row_pitch = float(spans[:, 1].max()) * 1.35
    offsets: dict[str, list[float]] = {}
    ps_meshes: dict[str, object] = {}

    for i, panel in enumerate(panels):
        centre_panel = np.asarray(panel["points"]).mean(axis=0)
        # Laid out along **-x**: with `neg_y_up` and a camera on the `+z` side, increasing `x`
        # renders to the *left*, so `+x` would print the panel list backwards. Measured with a
        # two-object probe rather than assumed.
        for structure_key, geometry_key, row in (
            (panel["key"], None, 0.0),
            (f"{panel['key']}::exploded", "exploded", row_pitch),
        ):
            source = panel if geometry_key is None else panel["exploded"]
            offset = np.array([-i * pitch, row, 0.0]) - centre_panel
            offsets[structure_key] = [float(x) for x in offset]
            ps_meshes[structure_key] = ps.register_surface_mesh(
                f"{case_id}::{structure_key}",
                np.asarray(source["points"]) + offset,
                np.asarray(source["triangles"]),
                smooth_shade=False,
            )

    all_points = np.concatenate(
        [
            np.asarray(panel["points"] if key == panel["key"] else panel["exploded"]["points"])
            + np.asarray(offsets[key])
            for panel in panels
            for key in (panel["key"], f"{panel['key']}::exploded")
        ],
    )
    specs = _quantity_specs(methods, gauges)

    result: dict = {
        "structure_offsets": offsets,
        "panel_pitch": pitch,
        "exploded_row_offset_y": row_pitch,
        "layout": (
            "top row: whole meshes; bottom row: the same meshes exploded per cell so the "
            "cell-cell interfaces are visible. Left to right in both rows: "
            + ", ".join(p["key"] for p in panels)
        ),
        "views": {},
        "quantities": [{k: v for k, v in s.items() if k != "cmap"} for s in specs],
        "png": [],
    }

    if write_png:
        png_dir.mkdir(parents=True, exist_ok=True)
        # Size the window to the scene rather than the reverse. The panel row's aspect ratio
        # depends on the case (a 2-cell mask is nearly square, a 10-cell one is wide), and a
        # fixed 2:1 window prints a near-square scene as a small island of content with half the
        # frame empty -- the framing is then correct and the figure still unreadable.
        height = resolution[1]
        width = int(np.clip(round(height * _scene_aspect(all_points, VIEWS["front"], up)), 900, 3200))
        resolution = (width, height)
        ps.set_window_size(*resolution)
        aspect = resolution[0] / resolution[1]
        result["png_resolution"] = list(resolution)
        cameras = {
            name: _camera_for(all_points, direction, up, ps.get_vertical_fov_degrees(), aspect)
            for name, direction in VIEWS.items()
        }
        result["views"] = {
            name: {"camera": [float(x) for x in cam], "target": [float(x) for x in target],
                   "direction": list(VIEWS[name])}
            for name, (cam, target) in cameras.items()
        }
        digests: dict[tuple[str, str], tuple[str, str]] = {}
        for spec in specs:
            _paint(panels, ps_meshes, spec)
            for view_name, (camera, target) in cameras.items():
                ps.look_at(tuple(camera), tuple(target))
                path = png_dir / f"{case_id}_{spec['field']}_{spec['flavour']}gauge_{view_name}.png"
                ps.screenshot(str(path), transparent_bg=False)
                record = assert_png_sane(path)
                digest = _file_digest(path)
                clash = digests.get((view_name, digest))
                identical_to = None
                if clash is not None:
                    clash_path, clash_name = clash
                    # Two quantities that are genuinely different (different field, or the
                    # same field on a different numeric range) photographing identically means
                    # the selection silently did not take effect and this PNG set is a set of
                    # copies. Two specs whose *identity* coincides are the same picture
                    # legitimately - the full and robust gauges collapse onto each other whenever
                    # p95 of |diff| equals max |diff|, which happens on a case with few
                    # interfaces - and that is recorded rather than treated as a failure.
                    if clash_name != _quantity_identity(spec):
                        message = (
                            f"{path.name} is byte-identical to {clash_path} from the same camera, "
                            f"but they are different quantities ({spec['name']!r} vs {clash_name!r}) - "
                            "the quantity selection did not take effect, so these PNGs are copies, "
                            "not a comparison"
                        )
                        raise ArtefactError(message)
                    identical_to = clash_path
                else:
                    digests[(view_name, digest)] = (path.name, _quantity_identity(spec))
                record.update(
                    {
                        "path": path.name,
                        "quantity": spec["name"],
                        "view": view_name,
                        "sha256": digest,
                        "identical_to": identical_to,
                    },
                )
                result["png"].append(record)

    # Leave the interactive scene carrying every quantity, plus the raw-valued scalars whose
    # colour bar shows the tension numbers on the shared gauge.
    for spec in specs:
        _paint(panels, ps_meshes, spec)
    for panel in panels:
        for method in methods:
            for field, gauge_key in ((f"tension_{method}", "tension"), (f"diff_{method}", "diff")):
                values = panel["fields"][field]
                index = panel["exploded"]["face_index"]
                for structure_key, face_values in (
                    (panel["key"], values),
                    (f"{panel['key']}::exploded", values[index]),
                ):
                    ps_meshes[structure_key].add_scalar_quantity(
                        f"{field} (raw values, shared gauge)",
                        np.nan_to_num(face_values, nan=float(gauges[gauge_key]["vmin"])),
                        defined_on="faces",
                        vminmax=(float(gauges[gauge_key]["vmin"]), float(gauges[gauge_key]["vmax"])),
                        cmap="jet" if gauge_key == "tension" else "coolwarm",
                        enabled=False,
                    )

    if interactive:
        print(
            f"    opening polyscope for case {case_id}. Top row: whole meshes; bottom row: the "
            f"same meshes exploded. Left to right: {', '.join(p['key'] for p in panels)}. "
            "Close the window to continue.",
        )
        ps.show()
    return result


def _quantity_identity(spec: dict) -> tuple[str, float, float]:
    """What makes two painted quantities genuinely different: the field and its numeric range.

    Not the display name: the display name distinguishes the `full` and `robust` gauges even when
    the two resolve to the *same* interval, which they do whenever the 95th percentile of `|diff|`
    equals its maximum. Two renders that differ only by that label are the same picture, and the
    non-vacuity check must not call that a failure.
    """
    return (spec["field"], spec["lo"], spec["hi"])


def _file_digest(path: Path) -> str:
    """SHA-256 of a file, used to prove two renders are genuinely different images."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class _DcelShim:
    """The two attributes `view_faces_values_on_embryo` reads, without rebuilding a `DcelData`.

    That function needs `mesh.v` and `mesh.f` only to size the value array and, when no
    `ps_mesh` is passed, to register one. A panel here already *is* a registered polyscope
    mesh with translated vertices, and rebuilding a full `DcelData` per panel per quantity per
    view would rebuild the half-edge structure dozens of times per case for nothing.
    Deliberately minimal, and only used with `ps_mesh` supplied and `scattered=False` (the
    exploded geometry is passed in explicitly rather than via `DcelData.v_scattered`, so the
    same code path serves both rows).
    """

    def __init__(self, points: NDArray[np.float64], triangles: NDArray[np.int64], labels: NDArray[np.int64]) -> None:
        self.v = np.asarray(points)
        self.f = np.column_stack([np.asarray(triangles), np.asarray(labels)])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_case(
    case_id: str,
    dataset_dir: Path,
    out_dir: Path,
    dw3d_module: object,
    dw3d_metrics: object,
    variant_names: list[str],
    method_names: list[str],
    min_distance: int,
    interactive: bool,
    write_png: bool,
) -> dict:
    """Produce every side-by-side comparison artefact for one case and return its manifest.

    Args:
        case_id (str): Dataset case id, e.g. `"023"`.
        dataset_dir (Path): `benchmarking-dataset`.
        out_dir (Path): Root of the artefact tree (outside both repositories).
        dw3d_module (object): Imported `dw3d`.
        dw3d_metrics (object): `delaunay-watershed-3d/dw3d_benchmarks/metrics.py`, loaded by path.
        variant_names (list[str]): `dw3d` variants to compare.
        method_names (list[str]): Keys of `METHOD_REGISTRY`.
        min_distance (int): `dw3d` seeding parameter.
        interactive (bool): Open polyscope per case.
        write_png (bool): Emit static renders.

    Returns:
        dict: The manifest, also written to `{case}/{case}_manifest.json`.
    """
    from dw3d.io import load_rec

    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    dict_truth = {
        (int(a), int(b)): float(v)
        for (a, b), v in np.load(dataset_dir / f"{case_id}_dict_tensions.npy", allow_pickle=True).item().items()
    }
    mask = skimage.io.imread(dataset_dir / f"{case_id}_labels_filled.tif")
    truth_points_native, truth_triangles_raw, truth_labels_raw = load_rec(dataset_dir / f"{case_id}_mesh.rec")
    truth_points, registration = register_truth_to_voxel_frame(
        mask, truth_points_native, truth_triangles_raw, truth_labels_raw, dw3d_metrics,
    )

    # One DcelData per mesh, and its arrays are what gets both painted and written, so the
    # per-face scalars and the artefact are aligned by construction rather than by a
    # convention two code paths have to agree on. DcelData re-winds triangles and sorts each
    # label pair, so this is checked to be the same mesh, not assumed.
    dcel_truth = dcel_mesh_from_multimaterial_mesh(truth_points, truth_triangles_raw, truth_labels_raw)
    _assert_same_triangle_set("truth", truth_triangles_raw, dcel_truth.f[:, :3])
    # A second, independent DcelData for the second truth panel: `compute_scattered_arrays`
    # mutates the object it is called on, and the two truth panels would otherwise share it.
    dcel_truth_inferred = dcel_mesh_from_multimaterial_mesh(truth_points, truth_triangles_raw, truth_labels_raw)

    inferred_on_truth = {m: METHOD_REGISTRY[m](dcel_truth) for m in method_names}

    panels: list[dict] = [
        build_panel(TRUTH, dcel_truth, dict_truth, dict.fromkeys(method_names, dict_truth)),
        build_panel(TRUTH_INFERRED, dcel_truth_inferred, dict_truth, inferred_on_truth),
    ]
    variant_meta: dict[str, dict] = {}

    for variant in variant_names:
        algorithm = dv.resolve_variant(dw3d_module, variant, min_distance=min_distance)
        t0 = time.perf_counter()
        points, triangles, labels = dv.reconstruct(algorithm, mask)
        wall = time.perf_counter() - t0
        dcel = dcel_mesh_from_multimaterial_mesh(points, triangles, labels)
        _assert_same_triangle_set(variant, triangles, dcel.f[:, :3])
        inferred = {m: METHOD_REGISTRY[m](dcel) for m in method_names}
        panels.append(build_panel(variant, dcel, dict_truth, inferred))
        variant_meta[variant] = {"reconstruction_wall_time_s": wall}

    # The matched colour gauges: one per field family, shared by every panel of this case.
    #
    # The *robust* difference gauge is a percentile of |diff| over the **reconstruction** panels
    # only. `TRUTH`'s difference field is identically zero by construction and `TRUTH_INFERRED`'s
    # is ~1e-5, so pooling all four panels puts half the sample at zero and drags p95 down to a
    # value the reconstructions saturate everywhere - which paints them uniformly red and blue
    # and hides exactly the spatial structure the map exists to show. The *full* gauge stays
    # pooled over everything (its extremes come from the reconstructions anyway), and both
    # gauges are still one shared range across variants, which is what "matched" requires.
    reconstruction_panels = [p for p in panels if p["key"] not in (TRUTH, TRUTH_INFERRED)]
    gauges = {
        "tension": gauge_over(
            [p["fields"]["ground_truth_tension"] for p in panels]
            + [p["fields"][f"tension_{m}"] for p in panels for m in method_names],
        ),
        "diff": symmetric_gauge(
            [p["fields"][f"diff_{m}"] for p in panels for m in method_names],
            robust_over=[p["fields"][f"diff_{m}"] for p in reconstruction_panels for m in method_names],
        ),
    }

    artefacts = []
    written_meshes: set[str] = set()
    for panel in panels:
        # TRUTH and TRUTH_INFERRED are the same geometry; write the mesh once, with every
        # panel's scalars on the one .vtk.
        stem = "truth_registered" if panel["key"] in (TRUTH, TRUTH_INFERRED) else panel["key"]
        if stem in written_meshes:
            continue
        written_meshes.add(stem)
        merged_fields: dict[str, NDArray[np.float64]] = {}
        for other in panels:
            if ("truth_registered" if other["key"] in (TRUTH, TRUTH_INFERRED) else other["key"]) != stem:
                continue
            for name, values in other["fields"].items():
                merged_fields[name if other["key"] != TRUTH_INFERRED else f"{name}_inferred_on_truth_mesh"] = values
        artefacts.append(
            {
                "panel": stem,
                "rec": write_rec(
                    case_dir / f"{case_id}_{stem}.rec", panel["points"], panel["triangles"], panel["labels"],
                ),
                "vtk": write_vtk(
                    case_dir / f"{case_id}_{stem}.vtk",
                    panel["points"],
                    panel["triangles"],
                    panel["labels"],
                    merged_fields,
                ),
                "sanity": panel["sanity"],
            },
        )

    render = render_case(
        case_id, panels, gauges, case_dir / "png", method_names, interactive=interactive, write_png=write_png,
    )

    manifest = {
        "artefact": "dw3d_variant_side_by_side",
        "case": case_id,
        "equilibrium_group": case_groups.group_of(case_id),
        "min_distance": min_distance,
        "mask": {"path": f"{case_id}_labels_filled.tif", "shape": list(mask.shape), "dtype": str(mask.dtype),
                 "n_cells": len([v for v in np.unique(mask) if v != 0])},
        "methods": method_names,
        "panel_order_left_to_right": [p["key"] for p in panels],
        "registration_ground_truth_to_mask_frame": registration,
        "matched_colour_gauges": gauges,
        "gauge_matching_scales_per_panel": {p["key"]: p["scales"] for p in panels},
        "ground_truth_tensions": {f"{a}-{b}": v for (a, b), v in sorted(dict_truth.items())},
        "per_panel": {
            p["key"]: {
                "sanity": p["sanity"],
                "exploded_sanity": p["exploded"]["sanity"],
                "explode_coeff": p["exploded"]["coeff"],
                "interfaces_without_ground_truth": [
                    r["interface"] for r in p["interface_table"] if not r["has_ground_truth"]
                ],
                "interface_table": p["interface_table"],
                **variant_meta.get(p["key"], {}),
            }
            for p in panels
        },
        "artefacts": artefacts,
        "render": render,
    }
    (case_dir / f"{case_id}_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    return manifest


def _assert_same_triangle_set(name: str, raw: NDArray, dcel: NDArray) -> None:
    """Check `DcelData` kept the same triangles (it re-winds them and sorts their labels).

    Args:
        name (str): Mesh name, for the message.
        raw (NDArray): Triangles as `dw3d`/`load_rec` produced them.
        dcel (NDArray): `DcelData.f[:, :3]`.

    Raises:
        ArtefactError: If the two are not the same set of vertex triples.
    """
    if len(raw) != len(dcel):
        message = f"{name}: DcelData has {len(dcel)} triangles, the source mesh has {len(raw)}"
        raise ArtefactError(message)
    a = np.unique(np.sort(np.asarray(raw), axis=1), axis=0)
    b = np.unique(np.sort(np.asarray(dcel), axis=1), axis=0)
    if not np.array_equal(a, b):
        message = f"{name}: DcelData's triangles are not the source mesh's triangles up to winding"
        raise ArtefactError(message)


def _json_safe(obj: object) -> object:
    """Recursively convert numpy scalars, arrays and tuple keys for `json.dumps`."""
    if isinstance(obj, dict):
        return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parent.parent
    workspace = repo_root.parent
    parser.add_argument("--dataset-dir", type=Path, default=workspace / "benchmarking-dataset")
    parser.add_argument("--dw3d-repo-root", type=Path, default=workspace / "delaunay-watershed-3d")
    parser.add_argument(
        "--out", type=Path, default=repo_root / "foambryo_benchmarks" / "outputs" / "dw3d_variant_comparison",
    )
    parser.add_argument("--cases", nargs="*", default=["002", "023", "011"],
                        help="Case ids; the default spans the dataset's cell-count range (2, 5, 10 cells).")
    parser.add_argument("--variants", nargs="*", default=["v0_3", "offset_excluded_linkcheck"])
    parser.add_argument("--methods", nargs="*", default=list(METHOD_REGISTRY))
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--interactive", action="store_true", help="Open the polyscope window per case.")
    parser.add_argument("--no-png", action="store_true", help="Skip the static renders.")
    args = parser.parse_args()

    unknown = [m for m in args.methods if m not in METHOD_REGISTRY]
    if unknown:
        message = f"unknown method(s) {unknown}; available: {sorted(METHOD_REGISTRY)}"
        raise SystemExit(message)

    dw3d_module = dv.import_dw3d()
    dw3d_metrics = load_dw3d_benchmark_metrics(args.dw3d_repo_root)
    print(f"dw3d:            {getattr(dw3d_module, '__file__', '?')}")
    print(f"dw3d benchmarks: {args.dw3d_repo_root / 'benchmarks' / 'metrics.py'}")
    print(f"dataset:         {args.dataset_dir} ({len(sorted(args.dataset_dir.glob('*_mesh.rec')))} cases present)")
    print(f"out:             {args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for case_id in args.cases:
        t0 = time.perf_counter()
        manifest = run_case(
            case_id,
            args.dataset_dir,
            args.out,
            dw3d_module,
            dw3d_metrics,
            args.variants,
            args.methods,
            args.min_distance,
            interactive=args.interactive,
            write_png=not args.no_png,
        )
        summary.append(
            {
                "case": case_id,
                "n_cells": manifest["mask"]["n_cells"],
                "panels": {k: v["sanity"]["n_triangles"] for k, v in manifest["per_panel"].items()},
                "n_png": len(manifest["render"]["png"]),
                "tension_gauge": [manifest["matched_colour_gauges"]["tension"]["vmin"],
                                  manifest["matched_colour_gauges"]["tension"]["vmax"]],
                "diff_gauge_max_abs": manifest["matched_colour_gauges"]["diff"]["max_abs"],
                "wall_time_s": time.perf_counter() - t0,
            },
        )
        s = summary[-1]
        print(
            f"  case {case_id}: {s['n_cells']} cells, panels {s['panels']}, {s['n_png']} PNGs checked, "
            f"tension gauge [{s['tension_gauge'][0]:.4g}, {s['tension_gauge'][1]:.4g}], "
            f"max|diff| {s['diff_gauge_max_abs']:.4g}  ({s['wall_time_s']:.1f}s)",
        )

    (args.out / "d2_1_summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
    print(f"\nwrote {len(summary)} case manifests under {args.out}")


if __name__ == "__main__":
    main()
