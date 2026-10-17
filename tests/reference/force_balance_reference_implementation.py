"""A standalone numpy/scipy implementation of the force-balance operator, kept as a test oracle.

`foambryo.validation.force_balance` and `foambryo.validation.mesh_io` implement the same functions, under the same
names and with the same algorithms. `tests/test_validation_force_balance.py` loads this file unmodified
(`tests.conftest.load_reference_implementation`) and compares foambryo's functions with it, so the comparison is
against an independent copy's own outputs rather than against a second hand-written version inside the test.
It is test code only: foambryo never imports it.

The mesh is assumed to be a mechanical equilibrium of the capillary energy

    E(v) = sum_interfaces gamma*area + sum_junctions t*length - sum_cells p*volume.

At equilibrium grad_v E = 0 at every vertex. Because E is linear in the unknowns, this
reads  M @ beta = 0, with M built from the discrete gradients of area, length and volume.
The pressure columns of M hold +d(volume)/dv, so beta = (gamma, t, -p): the pressure
entries of any solution are minus the physical pressures.

This file provides the generic pieces only: reading a mesh, building M, and a few solvers
and diagnostics. It contains no ground-truth values and no assumption about a particular
dataset.

Conventions
-----------
Regions are integers; one of them (0 by default) is the external medium and is used as the
pressure reference, so it carries no pressure unknown. A mesh is given by V (nv,3 float),
F (nf,3 int vertex indices), regions (nf,2 int, the two regions each triangle separates).
The right-hand normal of each triangle points out of its first region into its second.

`list_junctions` admits an edge as a triple line when its triangles carry three regions.
foambryo's `mesh_io.list_junctions` additionally requires exactly three incident triangles
spanning three distinct interfaces; the two agree on well-formed meshes.

Two mesh files are understood:
  .arec : text  -> nv / nv lines "x y z" / nf / nf lines "v0 v1 v2 ra rb"
  .rec  : binary-> int64 nv / nv*3 float64 / int64 nf / nf*(3 int64 verts + 2 int32 regions)
"""

# ruff: noqa
# Kept as written (its own style) on purpose: it is an independent copy to compare against, not foambryo code.

import numpy as np
import struct
from collections import defaultdict
from scipy.linalg import svd, lstsq


# =====================================================================
# reading meshes
# =====================================================================
def read_arec(path):
    """Read a text multimaterial mesh (.arec)."""
    with open(path) as f:
        nv = int(f.readline())
        V = np.array([list(map(float, f.readline().split())) for _ in range(nv)])
        nf = int(f.readline())
        rows = np.array([list(map(float, f.readline().split())) for _ in range(nf)])
    return V, rows[:, :3].astype(int), rows[:, 3:5].astype(int)


def read_rec(path):
    """Read a binary multimaterial mesh (.rec)."""
    raw = open(path, "rb").read()
    pos = 0
    nv = struct.unpack_from("<q", raw, pos)[0]; pos += 8
    V = np.frombuffer(raw, "<f8", count=3 * nv, offset=pos).reshape(nv, 3).copy()
    pos += 3 * nv * 8
    nf = struct.unpack_from("<q", raw, pos)[0]; pos += 8
    rec = np.frombuffer(raw, np.dtype([("v", "<3i8"), ("r", "<2i4")]), count=nf, offset=pos)
    return V, rec["v"].astype(int), rec["r"].astype(int)


def read_mesh(path):
    """Read a mesh, choosing the reader from the extension (falls back to a
    binary/text sniff). Returns V, F, regions."""
    if path.endswith(".rec"):
        return read_rec(path)
    if path.endswith(".arec"):
        return read_arec(path)
    head = open(path, "rb").read(16)          # sniff: text meshes start with an integer
    try:
        int(head.split()[0]); return read_arec(path)
    except ValueError:
        return read_rec(path)


def is_intact(V, F):
    """True if every face points to an existing vertex (no dangling reference)."""
    return int(F.max()) < len(V)


# =====================================================================
# mesh combinatorics
# =====================================================================
def cell_labels(regions, exterior=0):
    """Sorted list of cell regions (everything except the external medium)."""
    return sorted(r for r in set(regions.flatten()) if r != exterior)


def list_interfaces(regions):
    """Region pairs present in the mesh; one surface-tension unknown each."""
    return sorted(set(map(tuple, np.sort(regions, axis=1))))


def list_junctions(F, regions):
    """Triple lines: edges whose surrounding triangles span three regions.
    Returns the sorted region-triples and a dict edge -> triple."""
    edge_regs = defaultdict(set)
    for f in range(len(F)):
        a, b, c = F[f]; ra, rb = regions[f]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_regs[(u, v) if u < v else (v, u)] |= {ra, rb}
    junctions, edge_to_triple = set(), {}
    for e, rs in edge_regs.items():
        if len(rs) == 3:
            tri = tuple(sorted(rs))
            junctions.add(tri); edge_to_triple[e] = tri
    return sorted(junctions), edge_to_triple


def mean_edge_length(V, F):
    """Average triangle-edge length; handy to express noise or tolerances."""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1).mean()


# =====================================================================
# the force-balance operator  M = [ d(area) | d(length) | d(volume) ]
# =====================================================================
def force_balance_operator(V, F, regions, interfaces, junctions, edge_to_triple, cells):
    """Assemble M of shape (3*nv, n_interfaces + n_junctions + n_cells).
    Column order: surface tensions, line tensions, pressures.
    Row i (three components) is the energy gradient at vertex i."""
    nv = len(V)
    ni, nj, nc = len(interfaces), len(junctions), len(cells)
    ncol = ni + nj + nc
    icol = {k: i for i, k in enumerate(interfaces)}
    jcol = {k: i for i, k in enumerate(junctions)}
    ccol = {c: i for i, c in enumerate(cells)}
    G = np.zeros((nv, 3, ncol))

    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    normal = cross / np.linalg.norm(cross, axis=1)[:, None]

    # surface tension: dA/dv_a = 1/2 n x (opposite edge)
    dA = [0.5 * np.cross(normal, p2 - p1),
          0.5 * np.cross(normal, p0 - p2),
          0.5 * np.cross(normal, p1 - p0)]
    face_col = np.array([icol[tuple(pr)] for pr in np.sort(regions, axis=1)])
    onehot = np.eye(ncol)[face_col]
    for corner in range(3):
        contrib = (onehot[:, None, :] * dA[corner][:, :, None]).reshape(-1, 3 * ncol)
        np.add.at(G.reshape(nv, 3 * ncol), F[:, corner], contrib)

    # pressure: dV/dv_a = 1/6 v_b x v_c, oriented outward for each cell
    dV = [np.cross(p1, p2) / 6, np.cross(p2, p0) / 6, np.cross(p0, p1) / 6]
    for c in cells:
        sign = np.where(regions[:, 0] == c, 1.0, np.where(regions[:, 1] == c, -1.0, 0.0))
        for corner in range(3):
            np.add.at(G[:, :, ni + nj + ccol[c]], F[:, corner], sign[:, None] * dV[corner])

    # line tension: dL/dv = unit tangent, +/- on the two endpoints
    for (u, v), tri in edge_to_triple.items():
        d = V[u] - V[v]
        tangent = d / np.linalg.norm(d)
        G[u, :, ni + jcol[tri]] += tangent
        G[v, :, ni + jcol[tri]] -= tangent

    return G.reshape(3 * nv, ncol)


def split(M, ni, nj):
    """Split M into its surface, line and pressure column blocks."""
    return M[:, :ni], M[:, ni:ni + nj], M[:, ni + nj:]


# =====================================================================
# diagnostics
# =====================================================================
def residual_at(M, ni, nj, gamma, t):
    """Fraction of the surface-tension force left unbalanced by (gamma, t) once
    the best pressures are fitted. ~0 means the mesh is at equilibrium there."""
    Mg, Ml, Mp = split(M, ni, nj)
    load = Mg @ gamma + Ml @ t
    p, *_ = lstsq(Mp, -load)
    return np.linalg.norm(load + Mp @ p) / np.linalg.norm(Mg @ gamma)


def effective_tensions(M, ni, nj, gamma):
    """Given the surface tensions, the line tensions (and pressures) that best
    balance the mesh, by least squares."""
    Mg, Ml, Mp = split(M, ni, nj)
    x, *_ = lstsq(np.hstack([Ml, Mp]), -(Mg @ gamma))
    return x[:nj]


def singular_values(M):
    """Sorted singular values of the column-normalised operator (each column scaled
    to unit norm). The smallest belongs to the overall-scale (gauge) direction."""
    scale = np.linalg.norm(M, axis=0); scale[scale == 0] = 1
    return np.sort(svd(M / scale, compute_uv=False))


def soft_mode(M, ni, nj):
    """Return (sigma_soft, block_fractions) for the worst-determined non-gauge
    direction: sigma_soft is its singular value, block_fractions = how much of it
    sits in the (surface, line, pressure) blocks (squared norms of its three parts)."""
    scale = np.linalg.norm(M, axis=0); scale[scale == 0] = 1
    U, s, Wt = svd(M / scale, full_matrices=False)
    order = np.argsort(s)
    w = Wt[order[1]]                      # second-smallest = softest non-gauge
    frac = np.array([(w[:ni] ** 2).sum(),
                     (w[ni:ni + nj] ** 2).sum(),
                     (w[ni + nj:] ** 2).sum()])
    return s[order[1]], frac


# =====================================================================
# solvers  (the caller chooses which one, and in which order, to use)
# =====================================================================
def solve_constrained(M, anchors, targets):
    """Minimise |M beta|^2 subject to  anchors @ beta = targets, via the KKT
    system. `anchors` is a list of constraint vectors (e.g. the mean of a block;
    block_anchors gives one for each tension block)."""
    A = np.atleast_2d(anchors)
    n, k = M.shape[1], len(A)
    K = np.zeros((n + k, n + k))
    K[:n, :n] = 2 * M.T @ M; K[:n, n:] = A.T; K[n:, :n] = A
    rhs = np.zeros(n + k); rhs[n:] = targets
    return np.linalg.solve(K, rhs)[:n]


def block_anchors(ni, nj, ncols):
    """Two constraint vectors: the mean surface tension and the mean line tension.
    Passing both to solve_constrained fixes the scale of each block."""
    a_surface = np.zeros(ncols); a_surface[:ni] = 1.0 / ni
    a_line = np.zeros(ncols); a_line[ni:ni + nj] = 1.0 / nj
    return [a_surface, a_line]


def map_estimate(M, prior_mean, prior_cov, sigma_f=1.0, nonneg_upto=None):
    """Gaussian MAP estimate: minimise |M beta|^2/(2 sigma_f^2)
    + 1/2 (beta-prior_mean)^T prior_cov^{-1} (beta-prior_mean).
    Returns (beta, posterior_cov), the posterior covariance being the inverse of
    M^T M / sigma_f^2 + prior_cov^{-1}. If nonneg_upto is given (e.g. n_surface+n_line), those first entries are
    kept >= 0 by a few projected-gradient steps (pressures left free)."""
    Pinv = np.linalg.inv(prior_cov)
    A = M.T @ M / sigma_f ** 2 + Pinv
    rhs = Pinv @ prior_mean
    post_cov = np.linalg.inv(A)
    beta = post_cov @ rhs
    if nonneg_upto is not None:
        step = 1.0 / np.linalg.eigvalsh(A).max()
        for _ in range(500):
            beta = beta - step * (A @ beta - rhs)
            beta[:nonneg_upto] = np.maximum(beta[:nonneg_upto], 0.0)
    return beta, post_cov
