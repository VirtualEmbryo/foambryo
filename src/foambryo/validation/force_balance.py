"""The unified force-balance operator, ported from the reference implementation.

Ported (algorithm unchanged) from the reference implementation that accompanies the
technical note on tension inference (H. Turlier, in preparation; not distributed). It assumes
the mesh is a mechanical equilibrium of the capillary energy

    E(v) = sum_interfaces gamma * area + sum_junctions t * length - sum_cells p * volume.

The minus sign on the pressure term is what makes a cell's pressure come out positive
when it is at higher pressure than its surroundings (exterior medium at p = 0); see
`foambryo.force_balance_inference`'s module docstring for how the sign is carried
through the solved operator. At equilibrium `grad_v E = 0` at every vertex. Because `E` is linear in the unknowns,
this reads `M @ beta = 0`, with `beta = (gamma, t, p)` and `M` built from the discrete
gradients of area, length and volume; every classical balance (Young-Dupre, its
projection, Young-Laplace) is a row or combination of rows of `M`.

The operator is kept general here (it can build the line-tension block `M_t`), but by the
package author's design choice the foambryo benchmark harness (`foambryo_benchmarks/run_case.py`) calls it
with no junctions (`beta = (gamma, p)`, `M = [M_gamma | M_p]`) — pass empty `junctions`
and `edge_to_triple` to get that operator directly; the assembly loop below is a no-op
for the line-tension block in that case, so no separate code path is needed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.linalg import lstsq, svd


def force_balance_operator(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
    interfaces: list[tuple[int, int]],
    junctions: list[tuple[int, int, int]],
    edge_to_triple: dict[tuple[int, int], tuple[int, int, int]],
    cells: list[int],
) -> NDArray[np.float64]:
    """Assemble `M` of shape `(3*nv, n_interfaces + n_junctions + n_cells)`.

    Column order: surface tensions, line tensions, pressures. Row `i` (three
    components) is the energy gradient at vertex `i`.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        interfaces (list[tuple[int, int]]): Sorted interface keys (see `list_interfaces`).
        junctions (list[tuple[int, int, int]]): Sorted junction keys (see `list_junctions`).
            Pass `[]` for the no-line-tension operator `M = [M_gamma | M_p]`.
        edge_to_triple (dict[tuple[int, int], tuple[int, int, int]]): Map mesh edge -> its
            region-triple (see `list_junctions`). Pass `{}` together with `junctions=[]`.
        cells (list[int]): Sorted cell labels (see `cell_labels`).

    Returns:
        NDArray[np.float64]: The operator `M`, shape `(3*nv, n_interfaces + n_junctions + n_cells)`.
    """
    nv = len(v)
    ni, nj, nc = len(interfaces), len(junctions), len(cells)
    ncol = ni + nj + nc
    icol = {k: i for i, k in enumerate(interfaces)}
    jcol = {k: i for i, k in enumerate(junctions)}
    ccol = {c: i for i, c in enumerate(cells)}
    g = np.zeros((nv, 3, ncol))

    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    normal = cross / np.linalg.norm(cross, axis=1)[:, None]

    # surface tension: dA/dv_a = 1/2 n x (opposite edge)
    d_area = [
        0.5 * np.cross(normal, p2 - p1),
        0.5 * np.cross(normal, p0 - p2),
        0.5 * np.cross(normal, p1 - p0),
    ]
    face_col = np.array([icol[tuple(pr)] for pr in np.sort(regions, axis=1)])
    onehot = np.eye(ncol)[face_col]
    for corner in range(3):
        contrib = (onehot[:, None, :] * d_area[corner][:, :, None]).reshape(-1, 3 * ncol)
        np.add.at(g.reshape(nv, 3 * ncol), f[:, corner], contrib)

    # pressure: dV/dv_a = 1/6 v_b x v_c, oriented outward for each cell
    d_volume = [np.cross(p1, p2) / 6, np.cross(p2, p0) / 6, np.cross(p0, p1) / 6]
    for c in cells:
        sign = np.where(regions[:, 0] == c, 1.0, np.where(regions[:, 1] == c, -1.0, 0.0))
        for corner in range(3):
            np.add.at(g[:, :, ni + nj + ccol[c]], f[:, corner], sign[:, None] * d_volume[corner])

    # line tension: dL/dv = unit tangent, +/- on the two endpoints
    for (u, v_idx), tri in edge_to_triple.items():
        d = v[u] - v[v_idx]
        tangent = d / np.linalg.norm(d)
        g[u, :, ni + jcol[tri]] += tangent
        g[v_idx, :, ni + jcol[tri]] -= tangent

    return g.reshape(3 * nv, ncol)


def split(
    m: NDArray[np.float64],
    ni: int,
    nj: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split `M` into its surface, line and pressure column blocks.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        ni (int): Number of interfaces (surface-tension columns).
        nj (int): Number of junctions (line-tension columns).

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: `(M_gamma, M_t, M_p)`.
    """
    return m[:, :ni], m[:, ni : ni + nj], m[:, ni + nj :]


def residual_at(
    m: NDArray[np.float64],
    ni: int,
    nj: int,
    gamma: NDArray[np.float64],
    t: NDArray[np.float64],
) -> float:
    """Fraction of the surface-tension force left unbalanced once pressures are fitted.

    This is the equilibrium residual `rho`: the norm of the force left over at the given
    tensions once the best pressures are fitted, relative to the norm of the surface-tension
    force. With `b = M_gamma @ gamma + M_t @ t`, the best pressures are
    `p* = -M_p^+ @ b` and `rho = ||b + M_p @ p*|| / ||M_gamma @ gamma|| = ||P_perp @ b|| /
    ||M_gamma @ gamma||`, with `P_perp = I - M_p M_p^+` the projector orthogonal to the
    pressure column space, computed here via an explicit least-squares fit of the best
    pressures rather than forming `P_perp` directly (equivalent, cheaper). `rho ~ 0` means
    the mesh is at equilibrium at the given `(gamma, t)`.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        ni (int): Number of interfaces (surface-tension columns).
        nj (int): Number of junctions (line-tension columns).
        gamma (NDArray[np.float64]): Surface tensions, shape `(ni,)`.
        t (NDArray[np.float64]): Line tensions, shape `(nj,)`. Pass `np.zeros(0)` when
            `nj == 0` (the no-line-tension operator).

    Returns:
        float: `rho`.
    """
    m_gamma, m_t, m_p = split(m, ni, nj)
    load = m_gamma @ gamma + m_t @ t
    p, *_ = lstsq(m_p, -load)
    return float(np.linalg.norm(load + m_p @ p) / np.linalg.norm(m_gamma @ gamma))


def residual_force_field(
    m: NDArray[np.float64],
    ni: int,
    nj: int,
    gamma: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The per-vertex residual force whose norm `residual_at` reduces to `rho`.

    Same quantity as inside `residual_at` — `P_perp @ b`, computed via an explicit
    least-squares fit of the best pressures — but returned as a force *field* on the
    vertices instead of a single norm. `residual_at` answers "is this mesh at equilibrium";
    this answers "and if not, **where**", which is what distinguishes a mesh whose triple
    lines are merely discretised from one whose interfaces have not converged.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        ni (int): Number of interfaces (surface-tension columns).
        nj (int): Number of junctions (line-tension columns).
        gamma (NDArray[np.float64]): Surface tensions, shape `(ni,)`.
        t (NDArray[np.float64]): Line tensions, shape `(nj,)`. Pass `np.zeros(0)` when
            `nj == 0`.

    Returns:
        NDArray[np.float64]: Residual force per vertex, shape `(nv, 3)`, and the load
            `M_gamma @ gamma` it should be compared against is
            `(m[:, :ni] @ gamma).reshape(-1, 3)`.
    """
    m_gamma, m_t, m_p = split(m, ni, nj)
    load = m_gamma @ gamma + m_t @ t
    p, *_ = lstsq(m_p, -load)
    return (load + m_p @ p).reshape(-1, 3)


def marginalise_pressures(
    m: NDArray[np.float64],
    ni: int,
    nj: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eliminate the pressure block in closed form, keeping the tension columns.

    For any fixed tensions the pressures that minimise `||M beta||^2` are unique (when
    `M_p` has full column rank) and given by `p* = -M_p^+ (M_gamma gamma + M_t t)`. Substituting
    them back leaves

        min ||P_perp [M_gamma | M_t] (gamma, t)||^2,     P_perp = I - M_p M_p^+,

    an *exactly* equivalent problem in the tensions alone — this is block elimination of a
    quadratic form, not an approximation, so any constraint acting only on the tension
    columns (the mean-tension gauge) gives the same tensions from the reduced and the joint
    problem. The test `test_marginalised_and_joint_solves_agree` in
    `tests/test_force_balance_inference.py` checks this numerically.

    `P_perp` is applied through a thin QR factorisation of `M_p` rather than by forming the
    `3nv x 3nv` projector, which would be a ~9 GB dense array on a 20 000-vertex mesh.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        ni (int): Number of interfaces (surface-tension columns).
        nj (int, optional): Number of junctions (line-tension columns). Defaults to 0, the
            harness-default operator `M = [M_gamma | M_p]`.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: `(reduced, pressure_map)`.
            `reduced` is `P_perp [M_gamma | M_t]`, shape `(3nv, ni + nj)`; `pressure_map`
            recovers the eliminated pressures as `p = pressure_map @ (gamma, t)`, shape
            `(n_cells, ni + nj)`.
    """
    m_tension, m_p = m[:, : ni + nj], m[:, ni + nj :]
    q, r = np.linalg.qr(m_p)
    projected = q.T @ m_tension
    reduced = m_tension - q @ projected
    # p* = -M_p^+ M_tension = -R^-1 Q^T M_tension.
    pressure_map = -linalg.solve_triangular(r, projected)
    return reduced, pressure_map


def effective_tensions(
    m: NDArray[np.float64],
    ni: int,
    nj: int,
    gamma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Given the surface tensions, find the line tensions (and pressures) that best balance the mesh.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        ni (int): Number of interfaces (surface-tension columns).
        nj (int): Number of junctions (line-tension columns).
        gamma (NDArray[np.float64]): Surface tensions, shape `(ni,)`.

    Returns:
        NDArray[np.float64]: Best-fit line tensions, shape `(nj,)`.
    """
    m_gamma, m_t, m_p = split(m, ni, nj)
    x, *_ = lstsq(np.hstack([m_t, m_p]), -(m_gamma @ gamma))
    return x[:nj]


def effective_surface_tensions(
    m: NDArray[np.float64],
    ni: int,
    nj: int,
    mean_tension: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Ask which surface tensions balance the mesh, fixing nothing but the overall gauge.

    This is the surface-tension analogue of the effective-line-tension check (keep the true
    surface tensions and ask which *line* tensions balance the mesh), which detects a factor
    on a subset of interfaces (for example, some line tensions encoded at half their value
    by the program that generated the mesh) — a defect in the data, not in the inference.
    Here the harness operator carries no
    line-tension block (`beta = (gamma, p)`, by the package author's design choice), so the same question
    is asked of the surface tensions directly: solve

        min ||M beta||^2   subject to   mean(gamma) = mean_tension

    and compare `gamma` to nominal. Only *ratios* are physical, so the constraint is a
    gauge choice, not a physical input — without it `beta = 0` is the trivial minimiser.
    A comparison of the returned tensions against nominal ones is therefore only
    meaningful after gauge matching (see `foambryo_benchmarks.metrics.effective_tension_ratios`).

    A single anchor is correct here precisely because there is no line block. With two
    tension blocks present, anchoring only one sends the other towards zero under noise:
    to first order the error lies along the softest non-gauge singular direction, which
    mixes the two blocks at a fixed ratio, and the anchor pushes all of it into the
    un-anchored block. That is what `block_anchors` exists to avoid; `nj > 0` is accepted here only so the
    residual is reported consistently, and the caller is expected to pass `nj = 0`.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        ni (int): Number of interfaces (surface-tension columns).
        nj (int): Number of junctions (line-tension columns).
        mean_tension (float, optional): The gauge: the mean of the returned surface
            tensions. Defaults to 1.0, matching `foambryo.infer_tensions`' own convention
            and the `benchmarking-dataset` ground truth.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], float]: `(gamma_eff, p_eff,
            residual_fraction)`, where `residual_fraction` is
            `||M beta_eff|| / ||M_gamma gamma_eff||` — the force left unbalanced once the
            tensions themselves are free, i.e. the floor that `residual_at`'s `rho` at any
            *given* tensions cannot go below. `rho` far above this floor means the nominal
            tensions are wrong; both being large means the mesh is not an equilibrium of
            any tensions at all.
    """
    anchor = np.zeros(m.shape[1])
    anchor[:ni] = 1.0 / ni
    beta = solve_constrained(m, anchor, np.array([mean_tension]))
    gamma_eff = beta[:ni]
    p_eff = beta[ni + nj :]
    m_gamma = m[:, :ni]
    residual = float(np.linalg.norm(m @ beta) / np.linalg.norm(m_gamma @ gamma_eff))
    return gamma_eff, p_eff, residual


def singular_values(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sorted singular values of the column-normalised operator.

    The smallest is the overall-scale gauge; the next controls the noise amplification
    `1/sigma`.

    Args:
        m (NDArray[np.float64]): The force-balance operator.

    Returns:
        NDArray[np.float64]: Ascending singular values of `M / ||column||`.
    """
    scale = np.linalg.norm(m, axis=0)
    scale[scale == 0] = 1
    return np.sort(svd(m / scale, compute_uv=False))


def soft_mode(m: NDArray[np.float64], ni: int, nj: int) -> tuple[float, NDArray[np.float64]]:
    """Return `(sigma_soft, block_fractions)` for the worst-determined non-gauge direction.

    `sigma_soft` is its singular value; `block_fractions` is how much of it sits in the
    (surface, line, pressure) blocks. A line-heavy soft mode means the line/surface scale
    is the fragile quantity.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        ni (int): Number of interfaces (surface-tension columns).
        nj (int): Number of junctions (line-tension columns).

    Returns:
        tuple[float, NDArray[np.float64]]: `(sigma_soft, [surface_fraction, line_fraction, pressure_fraction])`.
    """
    scale = np.linalg.norm(m, axis=0)
    scale[scale == 0] = 1
    _, s, wt = svd(m / scale, full_matrices=False)
    order = np.argsort(s)
    w = wt[order[1]]  # second-smallest = softest non-gauge
    frac = np.array(
        [
            (w[:ni] ** 2).sum(),
            (w[ni : ni + nj] ** 2).sum(),
            (w[ni + nj :] ** 2).sum(),
        ],
    )
    return float(s[order[1]]), frac


def solve_constrained(
    m: NDArray[np.float64],
    anchors: list[NDArray[np.float64]] | NDArray[np.float64],
    targets: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Minimise `|M @ beta|^2` subject to `anchors @ beta = targets`, via the KKT system.

    `anchors` is a list of constraint vectors (e.g. the mean of a block, see
    `block_anchors`). The system solved is `[[2 M^T M, A^T], [A, 0]] @ (beta, lambda) =
    (0, targets)`, with `A` the stacked anchors. Anchoring one block leaves the other free to
    collapse under noise; anchoring both blocks (mean surface and mean line tension) pins
    both the global scale and the soft line/surface direction.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        anchors (list[NDArray[np.float64]] | NDArray[np.float64]): Constraint row vectors.
        targets (NDArray[np.float64]): Right-hand-side values for each constraint.

    Returns:
        NDArray[np.float64]: `beta`, the constrained least-squares solution.
    """
    a = np.atleast_2d(anchors)
    n, k = m.shape[1], len(a)
    kkt = np.zeros((n + k, n + k))
    kkt[:n, :n] = 2 * m.T @ m
    kkt[:n, n:] = a.T
    kkt[n:, :n] = a
    rhs = np.zeros(n + k)
    rhs[n:] = targets
    return np.linalg.solve(kkt, rhs)[:n]


def constrained_covariance(
    m: NDArray[np.float64],
    anchors: list[NDArray[np.float64]] | NDArray[np.float64],
    sigma_f: float,
) -> NDArray[np.float64]:
    """Covariance of the `solve_constrained` estimator under i.i.d. force-balance noise.

    Model: the vertex force residual `M @ beta` is not exactly zero but `epsilon ~ N(0,
    sigma_f^2 I)`, so `beta_hat` is a constrained generalised-least-squares estimator. Its
    covariance is the usual GLS covariance projected onto the constraint's null space,

        Sigma = sigma_f^2 (H^-1 - H^-1 A^T (A H^-1 A^T)^-1 A H^-1),     H = M^T M,

    which is **singular along every constraint direction** — variance exactly zero in the
    gauge that was fixed by hand, which is the correct statement: the overall tension scale
    is a convention, not a measured quantity. This is the error bar that belongs with the
    KKT solve, and it needs no prior; contrast `map_estimate`, whose posterior covariance is
    non-singular only because the prior supplies the missing information.

    **Honest caveat**: geometric noise on a good mesh is
    *correlated* between neighbouring vertices, since they share triangles, so an i.i.d.
    `epsilon` **understates** the true uncertainty. Use these as relative, comparable error
    bars across interfaces, not as calibrated absolute ones.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        anchors (list[NDArray[np.float64]] | NDArray[np.float64]): The same constraint row
            vectors passed to `solve_constrained`.
        sigma_f (float): Standard deviation of the per-component vertex force residual.
            `residual_noise_scale` estimates it from the fit itself.

    Returns:
        NDArray[np.float64]: Covariance of `beta`, shape `(ncols, ncols)`.
    """
    a = np.atleast_2d(anchors)
    h_inv = np.linalg.pinv(m.T @ m)
    h_inv_at = h_inv @ a.T
    correction = h_inv_at @ np.linalg.solve(a @ h_inv_at, h_inv_at.T)
    return sigma_f**2 * (h_inv - correction)


def residual_noise_scale(
    m: NDArray[np.float64],
    beta: NDArray[np.float64],
    n_constraints: int = 1,
) -> float:
    """Estimate `sigma_f` from the fit's own residual: `||M beta||^2 / dof`.

    The number of degrees of freedom is `n_rows - (n_unknowns - n_constraints)`; a
    constraint gives one unknown back. This is the textbook residual-variance estimator and
    it removes the only free hyperparameter from the KKT error bars.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        beta (NDArray[np.float64]): The solution whose residual is used.
        n_constraints (int, optional): Number of hard constraints imposed. Defaults to 1
            (the mean-tension gauge).

    Returns:
        float: The estimated per-component force-residual standard deviation.
    """
    n_rows, n_unknowns = m.shape
    dof = max(n_rows - (n_unknowns - n_constraints), 1)
    return float(np.sqrt(float(np.sum((m @ beta) ** 2)) / dof))


def block_anchors(ni: int, nj: int, ncols: int) -> list[NDArray[np.float64]]:
    """Build the two constraint vectors anchoring the mean surface and line tension.

    Passing both to `solve_constrained` fixes the scale of each block.

    Args:
        ni (int): Number of interfaces (surface-tension columns).
        nj (int): Number of junctions (line-tension columns).
        ncols (int): Total number of columns of `M`.

    Returns:
        list[NDArray[np.float64]]: `[a_surface, a_line]`.
    """
    a_surface = np.zeros(ncols)
    a_surface[:ni] = 1.0 / ni
    a_line = np.zeros(ncols)
    a_line[ni : ni + nj] = 1.0 / nj if nj else 0.0
    return [a_surface, a_line]


def map_estimate(
    m: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_cov: NDArray[np.float64],
    sigma_f: float = 1.0,
    nonneg_upto: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gaussian MAP estimate of `beta`, with posterior covariance.

    Minimises `|M @ beta|^2 / (2 * sigma_f^2) + 1/2 (beta - prior_mean)^T prior_cov^-1
    (beta - prior_mean)`. A prior with nonzero mean on both tension blocks softly anchors
    both scales, and the posterior covariance gives error bars.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        prior_mean (NDArray[np.float64]): Prior mean of `beta`.
        prior_cov (NDArray[np.float64]): Prior covariance of `beta`.
        sigma_f (float, optional): Force-balance noise scale. Defaults to 1.0.
        nonneg_upto (int | None, optional): If given (e.g. `n_surface + n_line`), those
            first entries are kept non-negative by a few projected-gradient steps
            (pressures left free). Defaults to `None`.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: `(beta, posterior_cov)`.
    """
    p_inv = np.linalg.inv(prior_cov)
    a = m.T @ m / sigma_f**2 + p_inv
    rhs = p_inv @ prior_mean
    post_cov = np.linalg.inv(a)
    beta = post_cov @ rhs
    if nonneg_upto is not None:
        step = 1.0 / np.linalg.eigvalsh(a).max()
        for _ in range(500):
            beta = beta - step * (a @ beta - rhs)
            beta[:nonneg_upto] = np.maximum(beta[:nonneg_upto], 0.0)
    return beta, post_cov
