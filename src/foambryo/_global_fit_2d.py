r"""Global fit of a 2D tissue by an equilibrium tiling of circular arcs (VMSI-lite, research port; private).

**Model** (a Möbius tiling, Noll N. et al., *Variational method for image-based inference of internal stress in
epithelial tissues*, Phys. Rev. X 10, 011072, 2020, doi:10.1103/PhysRevX.10.011072). Every material
`alpha` (0 the medium) carries a lifted vector `v_alpha = (A, B_r, B_c, C)` and the function
`f_alpha(x) = A |x|^2 + B . x + C` of the position `x = (row, column)`; a point belongs to the material whose
`f` is smallest. With `d = v_a - v_b` and `g = f_a - f_b = d . phi(x)`, `phi(x) = (|x|^2, row, column, 1)`, the
interface `a|b` is the circle (or straight line) `g = 0`. Its tension is `T_ab = sqrt(|d_B|^2 / 4 - d_A d_C)`
(`|grad g| = 2 T` on the circle) and the pressure difference is `p_a - p_b = d_A`, on the same scale: Laplace's
law `T kappa = d_A` and Young's force balance at every junction hold exactly for any `v`. Only the ratios of
the tensions, and the pressures on their scale, are physical.

**Data.** Points `x_n` on the interfaces, each with the sorted material pair `(a, b)` of its interface and a
weight `w_n`: the vertices of a polyline mesh, points spaced along its edges, or the crack points of a label
image (between two 4-adjacent pixels of different labels).

**Start** (Noll et al., Appendix B). `min sum_n w_n [(v_a - v_b) . phi(x_n)]^2` over `|v| = 1` with the reference
row fixed at 0: the right singular vector of the smallest singular value of the column-scaled design matrix,
found by inverse iteration on its sparse normal matrix.

**Fit** (Noll et al., Eq. 5, in lifted variables, as in the deltamic `arcs2d` prototype). Least squares of the
signed geometric distance of each point to its pair's circle,
`rho_n = g(x_n) / (|grad g(x_n)| / 2 + T_ab)`, which is exactly `sign(d_A) (|x_n - centre| - radius)` for a
circle and the signed distance for a line, by Levenberg-Marquardt with the analytic sparse Jacobian (each
residual depends on the eight parameters of two materials).

**Gauge.** The reference row (the medium, or the reference cell when no medium is in the data) is held at 0;
the overall scale, the one direction the residuals do not see, is fixed by one linear row; the overall sign,
which the residuals do not see either, is chosen so that every material lies on its own side of its
interfaces.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import ConvexHull, QhullError

MEDIUM = 0
#: An interface (material pair) whose data hold fewer points than this is given no tension (`nan`).
MIN_POINTS_PER_INTERFACE = 3
#: Levenberg-Marquardt stopping rules: relative cost decrease, relative step and gradient tolerances, and the
#: largest number of residual evaluations (as the arcs2d prototype's `scipy.optimize.least_squares` call).
FIT_FTOL = 1e-12
FIT_XTOL = 1e-12
FIT_GTOL = 1e-12
FIT_MAX_EVALUATIONS = 200
#: Width (px) of the band along the window border inside which a junction found in one of data and fit but
#: not in the other is not counted as a disagreement.
BORDER_BAND_PX = 2.0
_TIE_TOLERANCE = 1e-9
_ALL_TRIPLES_UP_TO = 16

Interface = tuple[int, int]


# ==================================================================================================== data
@dataclass(frozen=True)
class FitData:
    """Points on the interfaces of a 2D tissue, with what the fit and its checks need.

    Attributes:
        points: `(N, 2)` positions `(row, column)` in pixels.
        pairs: `(N, 2)` sorted material pairs `(a, b)`, `a < b`, of the interface each point lies on.
        weights: `(N,)` positive weights of the squared residuals.
        normals: `(N, 2)` unit vectors from material `a`'s side towards material `b`'s side at each point (sets the
            overall sign of the fit).
        window: `(row_min, row_max, column_min, column_max)` of the image, in pixels.
        junctions: the data's junctions: one `(materials, (row, column))` per point where exactly three materials
            meet, `materials` sorted.
        source: `"vertices"`, `"edge_samples"` or `"cracks"`.
        excluded_vertices: for a mesh, the vertices excluded (the image border by default).
        labels: the label image the data come from (crack points), or None; the second validity check
            (`validity="regions"`) compares each cell's fitted region with it.
    """

    points: NDArray[np.float64]
    pairs: NDArray[np.int64]
    weights: NDArray[np.float64]
    normals: NDArray[np.float64]
    window: tuple[float, float, float, float]
    junctions: list[tuple[tuple[int, int, int], tuple[float, float]]]
    source: str
    excluded_vertices: frozenset[int] = frozenset()
    labels: NDArray[np.integer] | None = None


def _edge_normals(vertices: NDArray[np.float64], edges: NDArray[np.int64]) -> NDArray[np.float64]:
    """Unit normal of each edge from its material `m1` to `m2`, `m1` lying on the left of `v1 -> v2` in `(row, column)`.

    The orientation rule puts it there. With `t = (dr, dc)`, the left of `t` in `(row, column)` is `(-dc, dr)`, so the
    normal from `m1` towards `m2` is `(dc, -dr)`: the edge's normal of the orientation rule, as a triangle's right-hand
    normal in 3D. (Computed from the tangent `v2 -> v1` before the rule was unified, it is bitwise the same: each
    component is only negated.)
    """
    t = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    n = np.column_stack([t[:, 1], -t[:, 0]])
    norm = np.linalg.norm(n, axis=1)
    norm[norm == 0] = 1.0
    n = n / norm[:, None]
    flip = edges[:, 2] > edges[:, 3]  # a mesh that breaks the m1 < m2 half of the contract: the normal still a -> b
    n[flip] *= -1.0
    return n


def _mesh_junctions(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int64],
    excluded: set[int],
) -> list[tuple[tuple[int, int, int], tuple[float, float]]]:
    materials: dict[int, set[int]] = {}
    for v1, v2, m1, m2 in edges.tolist():
        for v in (v1, v2):
            materials.setdefault(v, set()).update((m1, m2))
    out = []
    for v, mats in sorted(materials.items()):
        if len(mats) == 3 and v not in excluded:
            out.append((tuple(sorted(mats)), (float(vertices[v, 0]), float(vertices[v, 1]))))
    return out


def _window_of(vertices: NDArray[np.float64]) -> tuple[float, float, float, float]:
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def data_from_mesh_vertices(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int64],
    excluded: set[int],
) -> FitData:
    """Every mesh vertex not excluded, once per material pair of the edges that meet there (a junction thrice)."""
    vertices = np.asarray(vertices, np.float64)
    edges = np.asarray(edges, np.int64)
    keep = ~(np.isin(edges[:, 0], list(excluded)) & np.isin(edges[:, 1], list(excluded)))
    edges = edges[keep]
    normals = _edge_normals(vertices, edges)
    pair = np.sort(edges[:, 2:4], axis=1)
    acc: dict[tuple[int, int, int], NDArray[np.float64]] = {}
    for k in range(len(edges)):
        a, b = int(pair[k, 0]), int(pair[k, 1])
        for v in (int(edges[k, 0]), int(edges[k, 1])):
            if v in excluded:
                continue
            key = (v, a, b)
            acc[key] = acc.get(key, np.zeros(2)) + normals[k]
    keys = sorted(acc)
    points = np.array([vertices[v] for v, _, _ in keys], np.float64).reshape(-1, 2)
    pairs = np.array([(a, b) for _, a, b in keys], np.int64).reshape(-1, 2)
    nrm = np.array([acc[k] for k in keys], np.float64).reshape(-1, 2)
    length = np.linalg.norm(nrm, axis=1)
    length[length == 0] = 1.0
    return FitData(
        points=points,
        pairs=pairs,
        weights=np.ones(len(points)),
        normals=nrm / length[:, None],
        window=_window_of(vertices),
        junctions=_mesh_junctions(vertices, edges, excluded),
        source="vertices",
        excluded_vertices=frozenset(excluded),
    )


def data_from_mesh_edges(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int64],
    excluded: set[int],
    spacing: float = 1.0,
) -> FitData:
    """Points spaced along the mesh edges, each weighted by the length it stands for.

    Each edge is cut into `ceil(length / spacing)` equal pieces, one point at the middle of each piece, weighted by the
    piece's length (the fit's sum is then the length integral along the edges).
    """
    vertices = np.asarray(vertices, np.float64)
    edges = np.asarray(edges, np.int64)
    keep = ~(np.isin(edges[:, 0], list(excluded)) & np.isin(edges[:, 1], list(excluded)))
    edges = edges[keep]
    normals = _edge_normals(vertices, edges)
    # Each edge cut from its second vertex: the points of the walk foambryo took before the orientation rule was
    # unified (dw2d's edges were stored the other way round), bit for bit.
    p, q = vertices[edges[:, 1]], vertices[edges[:, 0]]
    length = np.linalg.norm(q - p, axis=1)
    pieces = np.maximum(1, np.ceil(length / spacing).astype(np.int64))
    owner = np.repeat(np.arange(len(edges)), pieces)
    first = np.concatenate([[0], np.cumsum(pieces)[:-1]])
    k = np.arange(len(owner)) - np.repeat(first, pieces)
    u = (k + 0.5) / pieces[owner]
    points = p[owner] + u[:, None] * (q - p)[owner]
    keep_points = length[owner] > 0
    pair = np.sort(edges[:, 2:4], axis=1)
    return FitData(
        points=points[keep_points],
        pairs=pair[owner][keep_points],
        weights=(length / pieces)[owner][keep_points],
        normals=normals[owner][keep_points],
        window=_window_of(vertices),
        junctions=_mesh_junctions(vertices, edges, excluded),
        source="edge_samples",
        excluded_vertices=frozenset(excluded),
    )


def data_from_label_image(labels: NDArray[np.integer]) -> FitData:
    """The crack points of a label image: the middle of every side shared by two 4-adjacent pixels of different labels.

    Pixel `(i, j)` is centred at `(row, column) = (i, j)`. Junctions: the pixel corners whose four pixels hold exactly
    three labels.
    """
    lab = np.asarray(labels)
    pts, prs, nrm = [], [], []
    a, b = lab[:, :-1], lab[:, 1:]
    i, j = np.nonzero(a != b)
    pts.append(np.column_stack([i.astype(float), j + 0.5]))
    lo, hi = np.minimum(a[i, j], b[i, j]), np.maximum(a[i, j], b[i, j])
    prs.append(np.column_stack([lo, hi]))
    sign = np.where(a[i, j] == lo, 1.0, -1.0)  # lower label on the left pixel: normal towards +column
    nrm.append(np.column_stack([np.zeros_like(sign), sign]))
    a, b = lab[:-1, :], lab[1:, :]
    i, j = np.nonzero(a != b)
    pts.append(np.column_stack([i + 0.5, j.astype(float)]))
    lo, hi = np.minimum(a[i, j], b[i, j]), np.maximum(a[i, j], b[i, j])
    prs.append(np.column_stack([lo, hi]))
    sign = np.where(a[i, j] == lo, 1.0, -1.0)
    nrm.append(np.column_stack([sign, np.zeros_like(sign)]))
    w = np.stack([lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]], axis=-1)
    srt = np.sort(w, axis=-1)
    distinct = 1 + (srt[..., 1] != srt[..., 0]) + (srt[..., 2] != srt[..., 1]) + (srt[..., 3] != srt[..., 2])
    junctions = []
    for r, c in zip(*np.nonzero(distinct == 3), strict=True):
        junctions.append((tuple(sorted({int(x) for x in w[r, c]})), (r + 0.5, c + 0.5)))
    points = np.vstack(pts)
    return FitData(
        points=points,
        pairs=np.vstack(prs).astype(np.int64),
        weights=np.ones(len(points)),
        normals=np.vstack(nrm),
        window=(0.0, float(lab.shape[0] - 1), 0.0, float(lab.shape[1] - 1)),
        junctions=junctions,
        source="cracks",
        labels=lab,
    )


# ==================================================================================================== model
def lifted(points: NDArray) -> NDArray:
    """`phi(x) = (|x|^2, row, column, 1)` for `(N, 2)` points."""
    return np.column_stack([np.sum(points * points, axis=1), points[:, 0], points[:, 1], np.ones(len(points))])


def tension_squared(d: NDArray) -> NDArray:
    """`T^2 = |d_B|^2 / 4 - d_A d_C` of lifted differences `d` (`(..., 4)`)."""
    return (d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]) / 4.0 - d[..., 0] * d[..., 3]


_T2_FLOOR = 1e-300


def distances_and_derivatives(
    d: NDArray,
    phi: NDArray,
    points: NDArray,
    *,
    with_derivatives: bool = True,
) -> tuple[NDArray, NDArray | None]:
    """Signed geometric distance `rho = g / (|grad g| / 2 + T)` of each point to its circle, and `d rho / d d`.

    `d` `(N, 4)` the lifted difference of the point's two materials, `phi` `(N, 4)` its lifted position. Works for
    complex arguments (the complex-step check): it uses only `+ - * /` and `sqrt`. Where `T^2 < 0` (no real circle),
    `T` is floored at `sqrt(1e-300)`, as the prototype clamps it, and its derivative is 0.
    """
    g = np.sum(d * phi, axis=1)
    gr = 2.0 * d[:, 0] * points[:, 0] + d[:, 1]
    gc = 2.0 * d[:, 0] * points[:, 1] + d[:, 2]
    n = np.sqrt(gr * gr + gc * gc)
    t2 = tension_squared(d)
    real = np.real(t2) > _T2_FLOOR
    t = np.sqrt(np.where(real, t2, _T2_FLOOR))
    h = n / 2.0 + t
    rho = g / h
    if not with_derivatives:
        return rho, None
    dn = np.column_stack([2.0 * (gr * points[:, 0] + gc * points[:, 1]), gr, gc, np.zeros_like(gr)]) / n[:, None]
    dt = np.column_stack([-d[:, 3], d[:, 1] / 2.0, d[:, 2] / 2.0, -d[:, 0]]) / (2.0 * t)[:, None]
    dt = np.where(real[:, None], dt, 0.0)
    dh = dn / 2.0 + dt
    drho = (phi - rho[:, None] * dh) / h[:, None]
    return rho, drho


# ==================================================================================================== problem
@dataclass
class _Problem:
    """The data in the model frame, indexed by rows: row 0 is the reference (fixed at 0), rows 1.. are free."""

    labels: list[int]  # material of each row
    reference: int  # the material of row 0
    medium_present: bool
    phi: NDArray[np.float64]
    points: NDArray[np.float64]  # model frame
    row_a: NDArray[np.int64]
    row_b: NDArray[np.int64]
    sqrt_w: NDArray[np.float64]
    centre: NDArray[np.float64]
    scale: float
    component: NDArray[np.int64]  # component of each row (-1 for row 0): rows linked by interfaces between free rows
    basis: sparse.csr_matrix | None = None  # theta = basis @ w: each free row in its own frame (`_local_basis`)
    basis_inverse: sparse.csr_matrix | None = None
    anchors: tuple[sparse.csr_matrix, NDArray] | None = None  # linear rows `M w - b` kept in the fit (`anchor_rows`)

    @property
    def n_free(self) -> int:
        return 4 * (len(self.labels) - 1)


def _problem(data: FitData, reference: int | None = None) -> _Problem:
    mats = sorted({int(x) for x in np.unique(data.pairs)})
    medium_present = MEDIUM in mats
    if reference is None and medium_present:
        reference = MEDIUM
    elif reference is None:  # the cell with the most data points (ties: the smallest label) plays the medium's part
        counts = {m: int(np.sum(data.pairs == m)) for m in mats}
        reference = min(mats, key=lambda m: (-counts[m], m))
    labels = [reference] + [m for m in mats if m != reference]
    row = {m: k for k, m in enumerate(labels)}
    centre = data.points.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((data.points - centre) ** 2, axis=1))))
    if not scale > 0:
        scale = 1.0
    points = (data.points - centre) / scale
    to_row = np.vectorize(row.__getitem__, otypes=[np.int64])
    row_a, row_b = to_row(data.pairs[:, 0]), to_row(data.pairs[:, 1])
    free = (row_a > 0) & (row_b > 0)
    n_rows = len(labels)
    graph = sparse.coo_matrix((np.ones(int(free.sum())), (row_a[free], row_b[free])), shape=(n_rows, n_rows))
    _, comp = csgraph.connected_components(graph, directed=False)
    # renumber the components of rows 1.. by their first row; row 0 is -1
    order, component = {}, np.full(n_rows, -1, np.int64)
    for k in range(1, n_rows):
        component[k] = order.setdefault(int(comp[k]), len(order))
    return _Problem(
        labels=labels,
        reference=reference,
        medium_present=medium_present,
        phi=lifted(points),
        points=points,
        row_a=row_a,
        row_b=row_b,
        sqrt_w=np.sqrt(np.asarray(data.weights, np.float64)),
        centre=centre,
        scale=scale,
        component=component,
    )


def _full(theta: NDArray, n_rows: int) -> NDArray:
    return np.vstack([np.zeros((1, 4), dtype=theta.dtype), theta.reshape(n_rows - 1, 4)])


def _block_matrix(problem: _Problem, values: NDArray) -> sparse.csr_matrix:
    """Sparse `(N, 4 (rows - 1))` matrix with `+values` in row a's block and `-values` in row b's (row 0 dropped)."""
    n = len(values)
    cols, vals, rows = [], [], []
    for sign, which in ((1.0, problem.row_a), (-1.0, problem.row_b)):
        free = which > 0
        r = np.repeat(np.flatnonzero(free), 4)
        c = (4 * (which[free] - 1))[:, None] + np.arange(4)[None, :]
        rows.append(r)
        cols.append(c.ravel())
        vals.append(sign * values[free].ravel())
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, problem.n_free),
    )


def appendix_b_start(problem: _Problem, *, per_interface: bool = False) -> NDArray[np.float64]:
    """Noll et al.'s Appendix B: the smallest right singular vector of the column-scaled design matrix.

    Rows `sqrt(w_n) (v_a - v_b) . phi(x_n)`. The matrix is block-diagonal over the components (`_Problem.component`):
    a row involves two rows of one component, or one row and the reference; each component's own null direction is
    independent, so each block is solved alone: inverse iteration on `M_s^T M_s + delta I` (`M_s` the block with unit
    columns, `delta = 1e-10`) from a start of ones, until the iterate moves by less than 1e-13 (at most 100
    iterations). Returns the free parameters `theta`, unit norm in each block's scaled columns. With `per_interface`,
    each interface's rows are weighted so that every interface counts once (a variant start, not Noll's).
    """
    weight = problem.sqrt_w
    if per_interface:  # every interface counts once: each point's weight divided by its interface's total weight
        key = problem.row_a * len(problem.labels) + problem.row_b
        _, inverse = np.unique(key, return_inverse=True)
        total_w = np.bincount(inverse, weights=problem.sqrt_w**2)
        weight = problem.sqrt_w / np.sqrt(total_w[inverse])
    m = _block_matrix(problem, weight[:, None] * problem.phi).tocsc()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    ms = (m @ sparse.diags(1.0 / norms)).tocsc()
    theta = np.zeros(problem.n_free)
    col_component = np.repeat(problem.component[1:], 4)
    for k in range(int(problem.component.max()) + 1):
        cols = np.flatnonzero(col_component == k)
        block = ms[:, cols]
        kk = (block.T @ block).tocsc() + 1e-10 * sparse.identity(len(cols), format="csc")
        lu = sparse_linalg.splu(kk)
        x = np.ones(len(cols)) / math.sqrt(len(cols))
        for _ in range(100):
            y = lu.solve(x)
            y /= np.linalg.norm(y)
            if y @ x < 0:
                y = -y
            moved = np.linalg.norm(y - x)
            x = y
            if moved < 1e-13:
                break
        theta[cols] = x
    return theta / norms


def _local_basis(problem: _Problem) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Each free row's parameters in a frame centred on its data points `c` and scaled by their RMS spread `s`.

    `f(x) = a' |y|^2 + b' . y + c'` with `x = c + s y`, so `A = a'/s^2`, `B = b'/s - 2 A c`,
    `C = c' + a' |c|^2 / s^2 - b' . c / s`: a block-diagonal linear change of variables (the minimum is the same),
    under which a cell's four parameters are of one size and weakly correlated, whereas in the tissue's frame a small
    cell's circle is a nearly degenerate combination of them.
    """
    n_rows = len(problem.labels)
    blocks, inverses = [], []
    for k in range(1, n_rows):
        mine = (problem.row_a == k) | (problem.row_b == k)
        x = problem.points[mine]
        c = x.mean(axis=0)
        s = float(np.sqrt(np.mean(np.sum((x - c) ** 2, axis=1))))
        if not s > 0:
            s = 1.0
        cr, cc = c
        cc2 = float(c @ c)
        blocks.append(
            np.array(
                [
                    [1 / s**2, 0, 0, 0],
                    [-2 * cr / s**2, 1 / s, 0, 0],
                    [-2 * cc / s**2, 0, 1 / s, 0],
                    [cc2 / s**2, -cr / s, -cc / s, 1],
                ],
            ),
        )
        inverses.append(np.array([[s**2, 0, 0, 0], [2 * s * cr, s, 0, 0], [2 * s * cc, 0, s, 0], [cc2, cr, cc, 1]]))
    as_sparse = [sparse.csr_matrix(b) for b in blocks]
    as_sparse_inverse = [sparse.csr_matrix(b) for b in inverses]
    return sparse.block_diag(as_sparse, format="csr"), sparse.block_diag(as_sparse_inverse, format="csr")


def _gauge_rows(problem: _Problem, theta0: NDArray) -> sparse.csr_matrix:
    """One sparse scale row per component, on one interface's lifted difference.

    The row is `u . (v_a - v_b) / |d0|` (in `theta`) for the component's interface with the most data points, `u` the
    unit direction of its start difference `d0`. It fixes the component's scale (the one
    direction its residuals do not see), cannot vanish while that interface's tension is finite, and keeps the
    normal matrix sparse (eight entries).
    """
    v0 = _full(theta0, len(problem.labels))
    n_comp = int(problem.component.max()) + 1
    comp_of_point = np.where(problem.row_a > 0, problem.component[problem.row_a], problem.component[problem.row_b])
    rows, cols, vals = [], [], []
    for k in range(n_comp):
        mine = comp_of_point == k
        pair_key = problem.row_a[mine] * len(problem.labels) + problem.row_b[mine]
        keys, counts = np.unique(pair_key, return_counts=True)
        best = int(keys[np.argmax(counts)])
        a, b = divmod(best, len(problem.labels))
        d0 = v0[a] - v0[b]
        u = d0 / float(d0 @ d0)
        for sign, rr in ((1.0, a), (-1.0, b)):
            if rr > 0:
                rows += [k] * 4
                cols += list(range(4 * (rr - 1), 4 * rr))
                vals += list(sign * u)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_comp, problem.n_free))


def _residuals(problem: _Problem, w: NDArray, gauge: tuple[NDArray, float, float]) -> NDArray:
    theta = w if problem.basis is None else problem.basis @ w
    v = _full(theta, len(problem.labels))
    d = v[problem.row_a] - v[problem.row_b]
    rho, _ = distances_and_derivatives(d, problem.phi, problem.points, with_derivatives=False)
    u, target, weight = gauge
    parts = [problem.sqrt_w * rho, weight * (u @ w - target)]
    if problem.anchors is not None:
        parts.append(problem.anchors[0] @ w - problem.anchors[1])
    return np.concatenate(parts)


def _residuals_and_jacobian(
    problem: _Problem,
    w: NDArray,
    gauge: tuple[NDArray, float, float],
) -> tuple[NDArray, sparse.csr_matrix]:
    theta = w if problem.basis is None else problem.basis @ w
    v = _full(theta, len(problem.labels))
    d = v[problem.row_a] - v[problem.row_b]
    rho, drho = distances_and_derivatives(d, problem.phi, problem.points)
    jac = _block_matrix(problem, problem.sqrt_w[:, None] * drho)
    if problem.basis is not None:
        jac = jac @ problem.basis
    u, target, weight = gauge
    r = np.concatenate([problem.sqrt_w * rho, weight * (u @ w - target)])
    jac = sparse.vstack([jac, weight * u], format="csr")
    if problem.anchors is not None:
        m, b = problem.anchors
        r = np.concatenate([r, m @ w - b])
        jac = sparse.vstack([jac, m], format="csr")
    return r, jac


@dataclass(frozen=True)
class FitStatus:
    """How the Levenberg-Marquardt fit stopped."""

    converged: bool
    reason: str
    n_evaluations: int
    n_iterations: int
    initial_cost: float
    final_cost: float


def _scaled_norm(x: NDArray, diag: NDArray) -> float:
    return float(np.linalg.norm(np.sqrt(diag) * x))


def levenberg_marquardt(  # noqa: C901 -- one damped loop with its stopping rules
    problem: _Problem,
    theta0: NDArray,
    gauge: tuple[NDArray, float, float],
    *,
    ftol: float = FIT_FTOL,
    xtol: float = FIT_XTOL,
    gtol: float = FIT_GTOL,
    max_evaluations: int = FIT_MAX_EVALUATIONS,
    trace: list | None = None,
) -> tuple[NDArray, FitStatus]:
    """Levenberg-Marquardt with Marquardt's diagonal scaling and Nielsen's damping update, sparse direct solves.

    Stops when the relative cost decrease of an accepted step is below `ftol`, the step below
    `xtol (xtol + |theta|)` in the norm scaled by the normal matrix's diagonal (MINPACK's rule: the test does not depend
    on the parameters' units), or the largest scaled gradient component below `gtol`, or after `max_evaluations`
    residual evaluations (not converged).
    """
    theta = np.asarray(theta0, np.float64).copy()
    r, jac = _residuals_and_jacobian(problem, theta, gauge)
    cost = 0.5 * float(r @ r)
    initial = cost
    evaluations, iterations = 1, 0
    a = (jac.T @ jac).tocsc()
    grad = jac.T @ r
    diag = np.maximum(a.diagonal(), 1e-12 * max(float(a.diagonal().max()), 1e-300))
    mu = 1e-3
    nu = 2.0
    reason = "max evaluations"
    converged = False
    if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(a.data))):  # a non-finite start or scale row
        return theta, FitStatus(False, "non-finite", evaluations, iterations, initial, cost)
    while evaluations < max_evaluations:
        current = np.maximum(a.diagonal(), 1e-300)
        if np.max(np.abs(grad) / np.sqrt(current)) <= gtol * max(math.sqrt(2.0 * cost), 1e-300):
            reason, converged = "gradient", True
            break
        iterations += 1
        try:
            step = -sparse_linalg.splu(a + mu * sparse.diags(diag, format="csc")).solve(grad)
        except RuntimeError:  # singular factor: damp harder
            mu *= nu
            nu *= 2.0
            if not math.isfinite(mu):  # no damping makes it factorisable (non-finite entries): stop
                reason = "singular"
                break
            continue
        trial = theta + step
        r_trial = _residuals(problem, trial, gauge)
        evaluations += 1
        cost_trial = 0.5 * float(r_trial @ r_trial) if np.all(np.isfinite(r_trial)) else math.inf
        if trace is not None:
            trace.append((cost, cost_trial, mu, float(np.linalg.norm(step))))
        predicted = -(grad @ step) - 0.5 * float(step @ (a @ step))
        if cost_trial < cost:
            ratio = (cost - cost_trial) / predicted if predicted > 0 else 1.0
            small_step = _scaled_norm(step, diag) <= xtol * (xtol + _scaled_norm(theta, diag))
            small_decrease = (cost - cost_trial) <= ftol * cost
            theta, cost = trial, cost_trial
            r, jac = _residuals_and_jacobian(problem, theta, gauge)
            a = (jac.T @ jac).tocsc()
            grad = jac.T @ r
            diag = np.maximum(diag, a.diagonal())
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
            nu = 2.0
            if small_step:
                reason, converged = "step", True
                break
            if small_decrease:
                reason, converged = "cost", True
                break
        else:
            if _scaled_norm(step, diag) <= xtol * (xtol + _scaled_norm(theta, diag)):
                reason, converged = "step", True
                break
            mu *= nu
            nu *= 2.0
    return theta, FitStatus(converged, reason, evaluations, iterations, initial, cost)


def _orient(problem: _Problem, v: NDArray, data: FitData) -> NDArray:
    """`v` or `-v`: the sign for which `grad (f_a - f_b)` points from `a`'s side to `b`'s at most of the weight."""
    d = v[problem.row_a] - v[problem.row_b]
    gr = 2.0 * d[:, 0] * problem.points[:, 0] + d[:, 1]
    gc = 2.0 * d[:, 0] * problem.points[:, 1] + d[:, 2]
    score = float(np.sum(problem.sqrt_w**2 * (gr * data.normals[:, 0] + gc * data.normals[:, 1])))
    return v if score >= 0 else -v


def _to_pixels(problem: _Problem, v_model: NDArray) -> NDArray:
    """Lifted vectors in the model frame `x' = (x - centre) / scale` -> the pixel frame (same `f` up to a factor)."""
    s, c = problem.scale, problem.centre
    a = v_model[:, 0] / (s * s)
    b = v_model[:, 1:3] / s - 2.0 * a[:, None] * c[None, :]
    cc = v_model[:, 3] - (v_model[:, 1:3] @ c) / s + a * (c @ c)
    return np.column_stack([a, b, cc])


# ==================================================================================================== the fitted tiling
def evaluate(v: NDArray, points: NDArray) -> NDArray:
    """`f_alpha(x)` of every row at every point, `(N, rows)`."""
    return lifted(np.atleast_2d(points)) @ v.T


def _candidate_triples(v: NDArray) -> NDArray[np.int64]:
    n = len(v)
    if n > _ALL_TRIPLES_UP_TO:
        try:
            hull = ConvexHull(v, qhull_options="Qt")
            lower = hull.equations[:, 3] < 1e-12 * np.abs(hull.equations[:, :4]).max(axis=1)
            tri = {t for s in hull.simplices[lower] for t in itertools.combinations(sorted(s.tolist()), 3)}
            return np.array(sorted(tri), dtype=np.int64).reshape(-1, 3)
        except QhullError:
            if n > 200:
                raise
    return np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64).reshape(-1, 3)


def fitted_junctions(v: NDArray, window: tuple[float, float, float, float], margin: float) -> list[dict]:
    """Every point of the window (grown by `margin`) where exactly three rows tie at the smallest `f`.

    Each three-way tie lies on the line `f_a = f_b = f_c` of the lifted space `(row, column, |x|^2)`, which meets the
    paraboloid in at most two points; the candidate triples are the triangles of the lower convex hull of the rows
    (all triples up to 16 rows). Returns dicts `rows` (sorted), `position` (row, column).
    """
    tri = _candidate_triples(v)
    if len(tri) == 0:
        return []
    va, vb, vc = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    d1, d2 = va - vb, va - vc
    n1 = np.column_stack([d1[:, 1], d1[:, 2], d1[:, 0]])
    n2 = np.column_stack([d2[:, 1], d2[:, 2], d2[:, 0]])
    h1, h2 = -d1[:, 3], -d2[:, 3]
    direction = np.cross(n1, n2)
    dd = np.sum(direction * direction, axis=1)
    ok = dd > 1e-30 * (np.abs(v).max() ** 4 + 1e-300)
    tri, n1, n2, h1, h2, direction, dd = tri[ok], n1[ok], n2[ok], h1[ok], h2[ok], direction[ok], dd[ok]
    p0 = (h1[:, None] * np.cross(n2, direction) + h2[:, None] * np.cross(direction, n1)) / dd[:, None]
    qa = direction[:, 0] ** 2 + direction[:, 1] ** 2
    qb = 2.0 * (p0[:, 0] * direction[:, 0] + p0[:, 1] * direction[:, 1]) - direction[:, 2]
    qc = p0[:, 0] ** 2 + p0[:, 1] ** 2 - p0[:, 2]
    disc = qb * qb - 4.0 * qa * qc
    real = disc >= 0
    sq = np.sqrt(np.where(real, disc, 0.0))
    q = -(qb + np.where(qb >= 0, 1.0, -1.0) * sq) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = [q / qa, qc / q]
    r0, r1, c0, c1 = window
    cand_xy, cand_tri = [], []
    for t in roots:
        xy = p0[:, :2] + t[:, None] * direction[:, :2]
        good = real & np.all(np.isfinite(xy), axis=1)
        good &= (xy[:, 0] >= r0 - margin) & (xy[:, 0] <= r1 + margin)
        good &= (xy[:, 1] >= c0 - margin) & (xy[:, 1] <= c1 + margin)
        cand_xy.append(xy[good])
        cand_tri.append(tri[good])
    xy = np.vstack(cand_xy)
    tri = np.vstack(cand_tri)
    if len(xy) == 0:
        return []
    t2 = tension_squared(va - vb)
    typical = float(np.median(np.sqrt(t2[t2 > 0]))) if np.any(t2 > 0) else 1.0
    tol = 2.0 * typical * _TIE_TOLERANCE * (1.0 + np.abs(xy).max())
    out = []
    step = max(1, (1 << 22) // len(v))
    for start in range(0, len(xy), step):
        chunk = slice(start, start + step)
        f = evaluate(v, xy[chunk])
        own = np.take_along_axis(f, tri[chunk], axis=1)
        level = own.max(axis=1)
        np.put_along_axis(f, tri[chunk], np.inf, axis=1)
        others = f.min(axis=1)
        valid = others > level + tol
        out.extend(
            {"rows": tuple(int(x) for x in tri[chunk][k]), "position": (float(xy[chunk][k, 0]), float(xy[chunk][k, 1]))}
            for k in np.flatnonzero(valid)
        )
    # one root found twice (from two triangles sharing the triple) is one junction
    seen, unique = set(), []
    for j in out:
        key = (j["rows"], round(j["position"][0], 6), round(j["position"][1], 6))
        if key not in seen:
            seen.add(key)
            unique.append(j)
    return unique


def raster_rows(v: NDArray, window: tuple[float, float, float, float]) -> NDArray[np.int64]:
    """The row of smallest `f` at every pixel centre `(i, j)` of the window (integers from its corner)."""
    r0, r1, c0, c1 = window
    rows = np.arange(math.ceil(r0), math.floor(r1) + 1, dtype=np.float64)
    cols = np.arange(math.ceil(c0), math.floor(c1) + 1, dtype=np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xy = np.column_stack([rr.ravel(), cc.ravel()])
    out = np.empty(len(xy), np.int64)
    vt = v.T.copy()
    chunk = max(1, (1 << 22) // len(v))  # about 32 MB of f values per chunk
    for start in range(0, len(xy), chunk):
        part = xy[start : start + chunk]
        out[start : start + chunk] = np.argmin(lifted(part) @ vt, axis=1)
    return out.reshape(rr.shape)


def _adjacent_pairs(raster: NDArray[np.int64]) -> set[Interface]:
    pairs: set[Interface] = set()
    for a, b in ((raster[:, :-1], raster[:, 1:]), (raster[:-1, :], raster[1:, :])):
        diff = a != b
        lo = np.minimum(a[diff], b[diff])
        hi = np.maximum(a[diff], b[diff])
        pairs |= set(zip(lo.tolist(), hi.tolist(), strict=True))
    return pairs


def _inside(position: tuple[float, float], window: tuple[float, float, float, float], band: float) -> bool:
    r0, r1, c0, c1 = window
    return r0 + band <= position[0] <= r1 - band and c0 + band <= position[1] <= c1 - band


# ==================================================================================================== the fit
@dataclass(frozen=True)
class GlobalFitResult:
    """The fitted tiling and what was checked on it.

    Attributes:
        tensions: every interface (sorted material pair) of the data -> its tension, `nan` where refused. Only ratios
            within one component are physical (`report["component_of"]`): the interfaces between cells link their
            scales, an interface with the reference (medium) does not; each component's finite tensions have mean 1.
        pressures: every material -> its pressure relative to the reference (the medium when present), in the unit of
            the tensions per pixel; `nan` for a cell empty in the fit.
        reference: the material held at pressure 0 (the medium, or the reference cell when no medium is in the data).
        junctions: the fitted junctions inside the window: `materials` (sorted), `position` (row, column),
            `wedges_rad` (material -> the angle inside it between its two interfaces' tangents).
        lifted: material -> its lifted vector `(A, B_row, B_column, C)` in pixels, on the scale of the tensions.
        refused: interface -> the reason it was given no tension.
        report: counts and the checks of validity (see `fit_tiling`).
    """

    tensions: dict[Interface, float]
    pressures: dict[int, float]
    reference: int
    junctions: list[dict]
    lifted: dict[int, tuple[float, float, float, float]]
    refused: dict[Interface, str]
    report: dict = field(default_factory=dict)


def _wedges(v: NDArray, rows: tuple[int, int, int], position: tuple[float, float]) -> dict[int, float]:
    """Angle inside each of the three rows between the tangents of its two interfaces at the junction.

    The tangent of interface `a|b` is perpendicular to `grad (f_a - f_b)`, oriented so that the third row's `f`
    grows along it (the interface leaves the junction away from the third material).
    """
    x = np.asarray(position, np.float64)
    grad = {m: 2.0 * v[m, 0] * x + v[m, 1:3] for m in rows}
    tangents = {}
    for a, b in itertools.combinations(rows, 2):
        (c,) = set(rows) - {a, b}
        gab = grad[a] - grad[b]
        t = np.array([-gab[1], gab[0]])
        t /= np.linalg.norm(t)
        if t @ (grad[c] - grad[a]) < 0:
            t = -t
        tangents[(a, b)] = t
    out = {}
    for m in rows:
        p, q = (tangents[k] for k in tangents if m in k)
        out[m] = float(np.arccos(np.clip(p @ q, -1.0, 1.0)))
    return out


def biconnected_blocks(pairs: list[Interface]) -> list[list[Interface]]:  # noqa: C901 -- one iterative DFS
    """The blocks (biconnected components) of the graph whose vertices are materials and whose edges are `pairs`.

    Iterative Hopcroft-Tarjan: each block is a list of edges; blocks share at most one vertex (an articulation).
    """
    adj: dict[int, list[tuple[int, int]]] = {}
    for k, (a, b) in enumerate(pairs):
        adj.setdefault(a, []).append((b, k))
        adj.setdefault(b, []).append((a, k))
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    blocks: list[list[Interface]] = []
    stack: list[int] = []
    counter = 0
    for root in sorted(adj):
        if root in disc:
            continue
        disc[root] = low[root] = counter
        counter += 1
        work = [(root, -1, iter(adj[root]))]
        while work:
            node, via, it = work[-1]
            advanced = False
            for nxt, k in it:
                if k == via:
                    continue
                if nxt not in disc:
                    stack.append(k)
                    disc[nxt] = low[nxt] = counter
                    counter += 1
                    work.append((nxt, k, iter(adj[nxt])))
                    advanced = True
                    break
                if disc[nxt] < disc[node]:
                    stack.append(k)
                    low[node] = min(low[node], disc[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
                if low[node] >= disc[parent]:
                    block = []
                    while True:
                        k = stack.pop()
                        block.append(pairs[k])
                        if k == via:
                            break
                    blocks.append(sorted(block))
    return blocks


def _subset(data: FitData, keep: NDArray[np.bool_]) -> FitData:
    return FitData(
        points=data.points[keep],
        pairs=data.pairs[keep],
        weights=data.weights[keep],
        normals=data.normals[keep],
        window=data.window,
        junctions=data.junctions,
        source=data.source,
        excluded_vertices=data.excluded_vertices,
        labels=data.labels,
    )


STARTS = ("appendix_b", "appendix_b_per_interface", "local_readings", "anchored")
#: Weight of the anchored start's anchors: a relative tension error of 1 costs as much as moving every point of the
#: interface by this many pixels (`anchor_rows`).
ANCHORED_START_KAPPA_PX = 1.0


@dataclass(frozen=True)
class LocalReading:
    """One interface as a local reading gives it: its circle (or line) and its tension, for the `local_readings` start.

    Attributes:
        left: the material on the side of positive curvature's centre (`curvature > 0`: the centre is on `left`'s side).
        right: the other material.
        curvature: signed curvature (1/px), as `foambryo._two_dimensional.InterfaceArc.curvature`; 0 for a line.
        centre: the circle's centre (row, column), None for a line.
        point: a point of the line (row, column), for a line.
        normal: the line's unit normal from `left` towards `right` (row, column), for a line.
        tension: the interface's tension (any positive scale; `nan` is read as 1).
    """

    left: int
    right: int
    curvature: float
    centre: tuple[float, float] | None
    point: tuple[float, float] | None
    normal: tuple[float, float] | None
    tension: float


def _difference_of_a_reading(reading: LocalReading) -> NDArray:
    """`d = v_left - v_right` (pixels) of an interface read as a circle or line with a tension.

    Circle: `g = f_left - f_right = d_A (|x - c|^2 - R^2)` with `d_A = T kappa` (Laplace's law; the centre on `left`'s
    side when `kappa > 0`, so `g < 0` inside, on `left`'s side), hence `T_model = |d_A| R = T`.
    Line: `g = 2 T n . (x - x0)`, `n` from `left` to `right`.
    """
    t = reading.tension if math.isfinite(reading.tension) and reading.tension > 0 else 1.0
    if reading.centre is not None and math.isfinite(reading.curvature) and reading.curvature != 0:
        c = np.asarray(reading.centre, np.float64)
        da = t * reading.curvature
        r2 = 1.0 / reading.curvature**2
        return np.array([da, -2.0 * da * c[0], -2.0 * da * c[1], da * (c @ c - r2)])
    n = np.asarray(reading.normal, np.float64)
    x0 = np.asarray(reading.point, np.float64)
    return np.array([0.0, 2.0 * t * n[0], 2.0 * t * n[1], -2.0 * t * float(n @ x0)])


def local_readings_start(
    problem: _Problem,
    readings: dict[Interface, list[LocalReading]],
) -> NDArray[np.float64] | None:
    """The lifted vectors closest, in least squares, to every read interface's difference (reference row at 0).

    Each reading's difference (`_difference_of_a_reading`) is moved to the model frame and every interface of the data
    gives the rows `v_a - v_b = d_ab` (one per reading of that pair, the material order taken from the reading); the
    normal equations are solved with a ridge of 1e-12 of their largest diagonal entry. Returns None when a free row has
    no reading (that start is then not available for this block).
    """
    n_rows = len(problem.labels)
    row = {m: k for k, m in enumerate(problem.labels)}
    s, c = problem.scale, problem.centre
    rr, cc, vv, rhs = [], [], [], []
    k = 0
    pairs = {(int(a), int(b)) for a, b in zip(problem.row_a.tolist(), problem.row_b.tolist(), strict=True)}
    covered = set()
    for ra, rb in sorted(pairs):
        pair_labels = tuple(sorted((problem.labels[ra], problem.labels[rb])))
        for reading in readings.get(pair_labels, []):
            d = _difference_of_a_reading(reading)
            # pixel -> model frame (inverse of `_to_pixels`)
            dm = np.array(
                [
                    d[0] * s * s,
                    s * (d[1] + 2.0 * d[0] * c[0]),
                    s * (d[2] + 2.0 * d[0] * c[1]),
                    d[3] + d[1] * c[0] + d[2] * c[1] + d[0] * (c @ c),
                ],
            )
            for sign, m in ((1.0, reading.left), (-1.0, reading.right)):
                q = row.get(m, -1)
                if q > 0:
                    for j in range(4):
                        rr.append(k + j)
                        cc.append(4 * (q - 1) + j)
                        vv.append(sign)
                    covered.add(q)
            rhs.append(dm)
            k += 4
    if k == 0 or len(covered) < n_rows - 1:
        return None
    a = sparse.csr_matrix((vv, (rr, cc)), shape=(k, problem.n_free))
    b = np.concatenate(rhs)
    normal = (a.T @ a).tocsc()
    ridge = 1e-12 * float(normal.diagonal().max())
    return sparse_linalg.splu(normal + ridge * sparse.identity(problem.n_free, format="csc")).solve(a.T @ b)


def _reading_tensions(problem: _Problem, readings: dict[Interface, list[LocalReading]]) -> NDArray[np.float64]:
    """Each data point's interface tension from the readings (their mean); the median where an interface has none."""
    n_rows = len(problem.labels)
    out: dict[Interface, float] = {}
    for p, rs in readings.items():
        t = [r.tension for r in rs if math.isfinite(r.tension) and r.tension > 0]
        if t:
            out[p] = float(np.mean(t))
    fallback = float(np.median(list(out.values()))) if out else 1.0
    key = problem.row_a * n_rows + problem.row_b
    uniq, inverse = np.unique(key, return_inverse=True)
    t_of = np.array(
        [
            out.get(tuple(sorted((problem.labels[int(k) // n_rows], problem.labels[int(k) % n_rows]))), fallback)
            for k in uniq
        ],
    )
    return t_of[inverse]


def anchor_rows(
    problem: _Problem,
    normals: NDArray[np.float64],
    tension_of_point: NDArray[np.float64],
    kappa_px: float,
) -> tuple[sparse.csr_matrix, NDArray[np.float64]]:
    r"""One linear row per interface that pulls its tension towards a reading's: `M theta - b` (model frame).

    The row is `sqrt(beta W) [mean_n grad g(x_n) . n_n / (2 T_read) - 1]`, with `g = f_a - f_b`, the mean over the
    interface's points weighted by `w_n`, `n_n` their data normals (from `a` to `b`), `W` the interface's total weight
    and `T_read` the reading's tension in the model frame (`scale` times its pixel value: in the model frame
    `x' = (x - c) / s`, `g` keeps its values and `grad' g = s grad g`). `grad g(x) . n = (v_a - v_b) . (2 x . n, n, 0)`
    is linear in `v`, and on the circle it is `2 T` at every point (also on a closed loop, whose mean normal vanishes).
    `beta = (kappa_px / scale)^2`: a relative tension error of 1 costs as much as moving every point of the interface by
    `kappa_px` pixels (the data rows are distances in model units, pixels / scale).
    """
    n_rows = len(problem.labels)
    key = problem.row_a * n_rows + problem.row_b
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    w = problem.sqrt_w**2
    total = np.bincount(inverse, weights=w)
    x_dot_n = 2.0 * np.sum(problem.points * normals, axis=1)
    psi = (
        np.column_stack(
            [np.bincount(inverse, weights=w * q) for q in (x_dot_n, normals[:, 0], normals[:, 1])]
            + [np.zeros(len(total))],
        )
        / total[:, None]
    )
    beta = (kappa_px / problem.scale) ** 2
    per_interface = _Problem(**{**problem.__dict__})
    per_interface.row_a, per_interface.row_b = problem.row_a[first], problem.row_b[first]
    t_model = tension_of_point[first] * problem.scale
    weight = np.sqrt(beta * total) / (2.0 * t_model)
    return _block_matrix(per_interface, weight[:, None] * psi), np.sqrt(beta * total)


def anchored_start(
    problem: _Problem,
    normals: NDArray[np.float64],
    readings: dict[Interface, list[LocalReading]],
    kappa_px: float = ANCHORED_START_KAPPA_PX,
) -> NDArray[np.float64]:
    r"""Noll's algebraic problem made inhomogeneous by a reading's tensions: one sparse linear least-squares solve.

    `min_v sum_n w_n [(v_a - v_b) . phi(x_n) / (2 T_read)]^2 + |M theta - b|^2` (`anchor_rows` with `kappa_px`): the
    first term is, to first order, the squared distance of each point to its circle (on the circle `|grad g| = 2 T`), so
    the misfit is measured in pixels, not in raw lifted coefficients as `local_readings_start` measures it; the anchors
    fix each interface's scale near the reading's tension. The reference row is 0; a ridge of 1e-12 of the largest
    diagonal entry of the normal matrix.
    """
    t_point = _reading_tensions(problem, readings)
    rows = _block_matrix(problem, (problem.sqrt_w / (2.0 * t_point * problem.scale))[:, None] * problem.phi)
    m, b = anchor_rows(problem, normals, t_point, kappa_px)
    a = sparse.vstack([rows, m], format="csr")
    rhs = np.concatenate([np.zeros(rows.shape[0]), b])
    normal = (a.T @ a).tocsc()
    ridge = 1e-12 * float(normal.diagonal().max())
    return sparse_linalg.splu(normal + ridge * sparse.identity(problem.n_free, format="csc")).solve(a.T @ rhs)


def _fit_block(
    data: FitData,
    reference: int,
    starts: tuple[str, ...] = ("appendix_b",),
    max_evaluations: int = FIT_MAX_EVALUATIONS,
    readings: dict[Interface, list[LocalReading]] | None = None,
    anchor_kappa_px: float | None = None,
) -> tuple[dict[int, NDArray], FitStatus, _Problem, str]:
    """Fit one block alone, `reference` held at 0: a start, local frames, one scale row, Levenberg-Marquardt, sign.

    With several `starts` (`STARTS`), the fit is run from each and the one of lowest final cost is kept (the first on a
    tie). With `anchor_kappa_px`, the readings' anchor rows (`anchor_rows`, that weight) are kept in the fit; they need
    `readings`. Returns each material's lifted vector in pixels (the reference's is 0), the kept fit's status, the
    problem and the name of the kept start.
    """
    problem = _problem(data, reference)
    problem.basis, problem.basis_inverse = _local_basis(problem)
    if anchor_kappa_px is not None:
        m, b = anchor_rows(problem, data.normals, _reading_tensions(problem, readings or {}), anchor_kappa_px)
        problem.anchors = ((m @ problem.basis).tocsr(), b)
    best = None
    for name in starts:
        if name not in STARTS:
            message = f"unknown start {name!r}; expected one of {STARTS}"
            raise ValueError(message)
        if name == "local_readings":
            theta0 = local_readings_start(problem, readings or {})
            if theta0 is None:
                continue
        elif name == "anchored":
            theta0 = anchored_start(problem, data.normals, readings or {})
        else:
            theta0 = appendix_b_start(problem, per_interface=name == "appendix_b_per_interface")
        w0 = problem.basis_inverse @ theta0
        u = (_gauge_rows(problem, theta0) @ problem.basis).tocsr()
        gauge = (u, u @ w0, math.sqrt(float(np.sum(problem.sqrt_w**2))))
        w, status = levenberg_marquardt(problem, w0, gauge, max_evaluations=max_evaluations)
        # costs of different starts compare on the data rows only (each start has its own scale row, 0 at its optimum)
        r = _residuals(problem, w, gauge)[: len(problem.sqrt_w)]
        data_cost = 0.5 * float(r @ r)
        if best is None or data_cost < best[0]:
            best = (data_cost, w, status, name)
    if best is None:  # only `local_readings` was asked and it is not available here: Noll's start
        return _fit_block(data, reference, ("appendix_b",), max_evaluations)
    _, w, status, name = best
    theta = problem.basis @ w
    v = _to_pixels(problem, _orient(problem, _full(theta, len(problem.labels)), data))
    return {m: v[k] for k, m in enumerate(problem.labels)}, status, problem, name


#: The second validity check (`validity="regions"`): a cell whose fitted region holds more than this fraction of its
#: data area (and at least `REGION_MIN_PIXELS` pixels) farther than `REGION_BAND_PX` from its own data pixels is
#: excluded from the tiling and its interfaces refused; at most `REGION_ROUNDS` rounds of exclusion.
REGION_FRACTION = 0.1
REGION_MIN_PIXELS = 4
REGION_BAND_PX = 2
REGION_ROUNDS = 10
VALIDITY_CHECKS = ("pixels", "regions")


def _regions_beyond_their_data(raster_materials: NDArray, labels: NDArray) -> dict[int, int]:
    """Materials whose fitted region spreads beyond their data: material -> its pixels beyond the band.

    A pixel counts when its fitted material differs from the label image's and no pixel of that material lies within
    `REGION_BAND_PX` (Chebyshev distance) of it in the label image.
    """
    labels = np.asarray(labels, np.int64)
    raster_materials = np.asarray(raster_materials, np.int64)
    wrong = raster_materials != labels
    near = np.zeros_like(wrong)
    b = REGION_BAND_PX
    padded = np.pad(labels, b, mode="constant", constant_values=np.iinfo(np.int64).min)
    h, w = labels.shape
    for dy in range(-b, b + 1):
        for dx in range(-b, b + 1):
            near |= padded[b + dy : b + dy + h, b + dx : b + dx + w] == raster_materials
    beyond = wrong & ~near
    if not beyond.any():
        return {}
    mats, counts = np.unique(raster_materials[beyond], return_counts=True)
    lab_mats, lab_counts = np.unique(labels, return_counts=True)
    area = dict(zip(lab_mats.tolist(), lab_counts.tolist(), strict=True))
    return {
        int(m): int(c)
        for m, c in zip(mats.tolist(), counts.tolist(), strict=True)
        if m != MEDIUM and c > max(REGION_MIN_PIXELS, REGION_FRACTION * area.get(m, 0))
    }


def _n_disconnected(raster: NDArray[np.int64], rows: list[int], skip: int | None) -> list[int]:
    """The rows (of `rows`) whose pixels form more than one 8-connected piece (each labelled in its bounding box)."""
    out = []
    structure = np.ones((3, 3), bool)
    boxes = ndimage.find_objects(raster + 1)
    for k in rows:
        if k == skip or k >= len(boxes) or boxes[k] is None:
            continue
        _, n_pieces = ndimage.label(raster[boxes[k]] == k, structure=structure)
        if n_pieces > 1:
            out.append(k)
    return out


def fit_tiling(  # noqa: C901 -- one pipeline, written in order
    data: FitData,
    *,
    starts: tuple[str, ...] = ("appendix_b",),
    max_evaluations: int = FIT_MAX_EVALUATIONS,
    readings: dict[Interface, list[LocalReading]] | None = None,
    anchor_kappa_px: float | None = None,
    validity: str = "pixels",
) -> GlobalFitResult:
    """Fit an equilibrium tiling of circular arcs to `data`; check it; read tensions and pressures.

    **Blocks.** The residuals of an interface depend only on the direction of its lifted difference, so the tensions
    share a scale only within a block (biconnected component) of the graph of materials linked by the data's
    interfaces: around any cycle the differences sum to zero, which ties their scales, and a block can be scaled about
    an articulation material without changing a residual. Each block is fitted alone (exactly the same minimum as the
    joint fit: blocks share no residual), its root material held at 0: the medium for the block that holds the most
    interfaces among those that contain it (else the material with most data points of the largest block), and for
    every other block the articulation it shares with its parent in the block tree. An interface that is a block by
    itself is refused: its tension shares a scale with no other.

    Each block: the model frame (centred on its points, scaled by their RMS distance to the centre); Appendix B's start;
    each material in its own frame (`_local_basis`); one scale row; Levenberg-Marquardt; the sign; back to pixels; its
    tensions scaled to mean 1. The blocks are then placed in one tiling (each shifted so that its root matches its
    parent's) for the checks.

    Checks (`report["validity"]`):

    * `junctions`: the materials of the fitted junctions inside the window agree with the data's junctions, ignoring
      those within `BORDER_BAND_PX` of the window border: `missing` (in the data, not in the fit) and `extra` (in the
      fit, not in the data) list the disagreeing triples;
    * `cells`: every material of the data occupies at least one pixel centre of the fitted tiling in the window and,
      for a cell, those pixels form one 8-connected piece: `empty` and `disconnected` list the others.

    Refusals (`refused`): an interface with fewer than `MIN_POINTS_PER_INTERFACE` data points (left out of the fit); an
    interface that is a block by itself; an interface of the data that is not an interface of the fitted tiling in the
    window (its two materials share no pixel side and no fitted junction there); an interface of a cell empty in the
    fit. Pressures: relative to the medium, for the cells of the blocks that contain the medium, on that block's scale;
    `nan` for the others.

    `anchor_kappa_px`: keep the readings' anchor rows in the fit (`anchor_rows`, that weight in pixels; needs
    `readings`). `validity="regions"`: the junctions are searched among the same rows as the pixel test (the default
    check, `"pixels"`, searches them among every row, so that a cell left out of the pixel test as unbounded can hide
    every junction); and, when the data carry their label image, a cell whose fitted region spreads beyond its own data
    pixels (`_regions_beyond_their_data`) is left out too, round after round, and its interfaces are refused (`spread`).
    """
    if validity not in VALIDITY_CHECKS:
        message = f"unknown validity check {validity!r}; expected one of {VALIDITY_CHECKS}"
        raise ValueError(message)
    seconds = {}
    t0 = time.perf_counter()
    data_pairs = sorted({(int(a), int(b)) for a, b in data.pairs.tolist()})
    key = data.pairs[:, 0] * (int(data.pairs.max()) + 1) + data.pairs[:, 1]
    uniq, counts_arr = np.unique(key, return_counts=True)
    count_of = dict(zip(uniq.tolist(), counts_arr.tolist(), strict=True))
    counts = {p: count_of[p[0] * (int(data.pairs.max()) + 1) + p[1]] for p in data_pairs}
    refused: dict[Interface, str] = {}
    for p in data_pairs:
        if counts[p] < MIN_POINTS_PER_INTERFACE:
            refused[p] = f"fewer than {MIN_POINTS_PER_INTERFACE} data points ({counts[p]})"
    fitted_pairs = [p for p in data_pairs if p not in refused]
    blocks = biconnected_blocks(fitted_pairs)
    blocks.sort(key=lambda b: (-len(b), b[0]))
    materials_of = [sorted({m for p in b for m in p}) for b in blocks]
    with_medium = [k for k, mats in enumerate(materials_of) if MEDIUM in mats]
    order: list[int] = []
    root_of: dict[int, int] = {}
    placed: set[int] = set()
    # block tree: start from the largest block holding the medium (else the largest block); breadth first
    remaining = set(range(len(blocks)))
    while remaining:
        start = min((with_medium and [k for k in with_medium if k in remaining]) or remaining)
        if start in with_medium:
            root_of[start] = MEDIUM
        else:
            mats = materials_of[start]
            npts = {m: sum(counts[p] for p in blocks[start] if m in p) for m in mats}
            root_of[start] = min(mats, key=lambda m: (-npts[m], m))
        queue = [start]
        remaining.discard(start)
        while queue:
            k = queue.pop(0)
            order.append(k)
            placed |= set(materials_of[k])
            for j in sorted(remaining):
                shared = set(materials_of[j]) & set(materials_of[k])
                if shared:
                    root_of[j] = min(shared)
                    remaining.discard(j)
                    queue.append(j)
    seconds["blocks"] = time.perf_counter() - t0
    t = time.perf_counter()
    lifted_v: dict[int, NDArray] = {}
    block_of: dict[Interface, int] = {}
    statuses = []
    tensions: dict[Interface, float] = {}
    pressures: dict[int, float] = dict.fromkeys(sorted({m for p in data_pairs for m in p}), math.nan)
    unbounded: set[int] = set()
    pair_index = {p: k for k, p in enumerate(data_pairs)}
    point_pair = np.array([pair_index[(int(a), int(b))] for a, b in data.pairs.tolist()], np.int64)
    for k in order:
        in_block = np.isin(point_pair, [pair_index[p] for p in blocks[k]])
        sub = _subset(data, in_block)
        v_block, status, _, start_used = _fit_block(sub, root_of[k], starts, max_evaluations, readings, anchor_kappa_px)
        statuses.append(
            dict(block=k, n_interfaces=len(blocks[k]), root=root_of[k], start=start_used, **status.__dict__),
        )
        raw = {}
        for a, b in blocks[k]:
            t2 = float(tension_squared(v_block[a] - v_block[b]))
            raw[(a, b)] = math.sqrt(t2) if t2 > 0 else math.nan
            block_of[(a, b)] = k
        finite = [x for x in raw.values() if math.isfinite(x)]
        mean = float(np.mean(finite)) if finite and len(blocks[k]) > 1 else (finite[0] if finite else 1.0)
        shift = lifted_v.get(root_of[k], np.zeros(4))
        for m, vec in v_block.items():
            if m not in lifted_v:
                lifted_v[m] = shift + vec / mean
        for p, x in raw.items():
            if len(blocks[k]) == 1:
                refused[p] = "a block by itself: its tension shares a scale with no other interface"
            elif not math.isfinite(x):
                refused[p] = "no real circle (T^2 <= 0)"
            tensions[p] = x / mean
        if root_of[k] == MEDIUM and len(blocks[k]) > 1:
            for m in materials_of[k]:
                if m != MEDIUM and math.isnan(pressures[m]):
                    pressures[m] = float(v_block[m][0] / mean)
                    if v_block[m][0] <= 0:
                        unbounded.add(m)
    if MEDIUM in pressures:
        pressures[MEDIUM] = 0.0
    seconds["fit"] = time.perf_counter() - t
    # one tiling for the checks; materials of refused-only interfaces get no vector (cannot be placed)
    t = time.perf_counter()
    labels = sorted(lifted_v)
    v = np.array([lifted_v[m] for m in labels]).reshape(-1, 4)
    if MEDIUM in lifted_v:
        v = v - lifted_v[MEDIUM][None, :]
    row_of = {m: k for k, m in enumerate(labels)}
    excluded = set(unbounded)
    spread: dict[int, int] = {}

    def raster_of(left_out: set[int]) -> NDArray[np.int64]:
        keep = np.array([m not in left_out for m in labels])
        return raster_rows(np.where(keep[:, None], v, np.array([0.0, 0.0, 0.0, np.inf])), data.window)

    raster = raster_of(excluded)
    if validity == "regions" and data.labels is not None and data.labels.shape == raster.shape:
        for _ in range(REGION_ROUNDS):
            beyond = _regions_beyond_their_data(np.asarray(labels)[raster], np.asarray(data.labels))
            new = {m: c for m, c in beyond.items() if m in row_of and m not in excluded}
            if not new:
                break
            spread.update(new)
            excluded |= set(new)
            raster = raster_of(excluded)
    searched = [k for k, m in enumerate(labels) if validity == "pixels" or m not in excluded]
    junctions = fitted_junctions(_to_model(v[searched], data), _window_to_model(data), BORDER_BAND_PX / _scale_of(data))
    for j in junctions:
        j["rows"] = tuple(sorted(searched[k] for k in j["rows"]))
        j["position"] = _position_to_pixels(j["position"], data)
    fitted_triples_in = {
        tuple(sorted(labels[k] for k in j["rows"]))
        for j in junctions
        if _inside(j["position"], data.window, BORDER_BAND_PX)
    }
    fitted_triples_all = {tuple(sorted(labels[k] for k in j["rows"])) for j in junctions}
    data_triples_in = {m for m, p in data.junctions if _inside(p, data.window, BORDER_BAND_PX)}
    data_triples_all = {m for m, _ in data.junctions}
    missing = sorted(data_triples_in - fitted_triples_all)
    extra = sorted(fitted_triples_in - data_triples_all)
    present = set(np.unique(raster).tolist())
    all_materials = sorted({m for p in data_pairs for m in p})
    empty = sorted(m for m in all_materials if m not in lifted_v or row_of[m] not in present)
    disconnected = [labels[k] for k in _n_disconnected(raster, sorted(present), row_of.get(MEDIUM))]
    adjacent = {tuple(sorted((labels[a], labels[b]))) for a, b in _adjacent_pairs(raster)}
    for j in junctions:
        m = sorted(labels[k] for k in j["rows"])
        adjacent |= {(m[0], m[1]), (m[0], m[2]), (m[1], m[2])}
    seconds["checks"] = time.perf_counter() - t
    for p in data_pairs:
        if p in refused:
            continue
        if p[0] in unbounded or p[1] in unbounded:
            refused[p] = "a cell of this interface is unbounded in the fit (pressure not above the medium's)"
        elif p[0] in spread or p[1] in spread:
            refused[p] = "a cell of this interface has a fitted region beyond its data (left out of the tiling)"
        elif p[0] in empty or p[1] in empty:
            refused[p] = "a cell of this interface is empty in the fitted tiling"
        elif p not in adjacent:
            refused[p] = "not an interface of the fitted tiling in the window"
    for m in set(empty) | unbounded | set(spread):
        if m in pressures and m != MEDIUM:
            pressures[m] = math.nan
    out_tensions = {p: (math.nan if p in refused else float(tensions[p])) for p in data_pairs}
    # each block's finite, kept tensions back to mean 1 after the refusals
    for k in set(block_of.values()):
        kept = [out_tensions[p] for p in blocks[k] if math.isfinite(out_tensions[p])]
        if kept:
            f = float(np.mean(kept))
            for p in blocks[k]:
                out_tensions[p] /= f
            for m in materials_of[k]:
                if root_of[k] == MEDIUM and m != MEDIUM and math.isfinite(pressures.get(m, math.nan)):
                    pressures[m] /= f
    out_junctions = []
    for j in junctions:
        if not _inside(j["position"], data.window, 0.0):
            continue
        rows3 = j["rows"]
        w = _wedges(v, rows3, j["position"])
        out_junctions.append(
            {
                "materials": tuple(sorted(labels[k] for k in rows3)),
                "position": j["position"],
                "wedges_rad": {labels[k]: w[k] for k in rows3},
            },
        )
    sizes = {k: sum(1 for p in blocks[k] if math.isfinite(out_tensions[p])) for k in set(block_of.values())}
    largest = max(sizes, key=lambda k: (sizes[k], -k)) if sizes else None
    seconds["total"] = time.perf_counter() - t0
    report = {
        "source": data.source,
        "validity_check": validity,
        "anchor_kappa_px": anchor_kappa_px,
        "n_points": len(data.points),
        "n_materials": len(all_materials),
        "medium_present": MEDIUM in all_materials,
        "n_blocks": len(blocks),
        "n_single_interface_blocks": sum(1 for b in blocks if len(b) == 1),
        "blocks": statuses,
        "converged": all(s["converged"] for s in statuses),
        "validity": {
            "junctions": {"missing": missing, "extra": extra, "agree": not missing and not extra},
            "cells": {
                "empty": empty,
                "disconnected": sorted(disconnected),
                "unbounded": sorted(unbounded),
                "spread": sorted(spread),
                "agree": not empty and not disconnected and not unbounded and not spread,
            },
        },
        "n_refused": len(refused),
        "component_of": {p: block_of.get(p) for p in data_pairs},
        "largest_component": largest,
        "seconds": seconds,
    }
    report["validity"]["valid"] = report["validity"]["junctions"]["agree"] and report["validity"]["cells"]["agree"]
    return GlobalFitResult(
        tensions=out_tensions,
        pressures=pressures,
        reference=MEDIUM if MEDIUM in all_materials else (root_of[order[0]] if order else MEDIUM),
        junctions=out_junctions,
        lifted={m: tuple(float(x) for x in lifted_v[m]) for m in labels},
        refused=refused,
        report=report,
    )


def _scale_of(data: FitData) -> float:
    c = data.points.mean(axis=0)
    s = float(np.sqrt(np.mean(np.sum((data.points - c) ** 2, axis=1))))
    return s if s > 0 else 1.0


def _to_model(v: NDArray, data: FitData) -> NDArray:
    """Pixel-frame lifted vectors -> the frame centred on the data and scaled by their RMS spread.

    The inverse of `_to_pixels`: well scaled for the convex hull.
    """
    c = data.points.mean(axis=0)
    s = _scale_of(data)
    a = v[:, 0] * s * s
    b = s * (v[:, 1:3] + 2.0 * v[:, :1] * c[None, :])
    cc = v[:, 3] + v[:, 1:3] @ c + v[:, 0] * (c @ c)
    return np.column_stack([a, b, cc])


def _window_to_model(data: FitData) -> tuple[float, float, float, float]:
    c = data.points.mean(axis=0)
    s = _scale_of(data)
    r0, r1, c0, c1 = data.window
    return (r0 - c[0]) / s, (r1 - c[0]) / s, (c0 - c[1]) / s, (c1 - c[1]) / s


def _position_to_pixels(position: tuple[float, float], data: FitData) -> tuple[float, float]:
    c = data.points.mean(axis=0)
    s = _scale_of(data)
    return float(c[0] + s * position[0]), float(c[1] + s * position[1])
