"""Calibrated uncertainty and identifiability diagnostics for tension inference.

Part of the uncertainty work. This module is **additive**: no
existing signature changes, no existing number moves. `infer_tensions` keeps returning a
bare tension dict; this is the companion that also returns an interval and the diagnostics
needed to decide whether to trust it.

**Read this before using an error bar from here.**

The goal here is not "add error bars" — it is *produce an interval whose `error/std` is
O(1), or state plainly that we cannot*. `error/std` (the ratio of the actual error against
ground truth to the reported standard deviation, gauge-matched) is **the** calibration
metric: 1 is calibrated, >>1 is optimistic (dangerous), <<1 is conservative (safe but
uninformative). Measured over the 42 equilibrium cases of `benchmarking-dataset`
(`foambryo_benchmarks/run_uncertainty_calibration.py`):

| model | estimator | median `error/std` | worst case | verdict |
|---|---|---|---|---|
| **`ProbeResidual`** | force balance | **0.233** | **0.93** | **O(1), and conservative on 42/42** |
| `CoherentResidual` | force balance | 1.33 | 29.9 | O(1) at the median, **not uniformly** |
| `EdgeBootstrap` | `YoungDupreLocal` | 0.27 | 1.57 | ~4x conservative (the per-edge Young-Dupre
work, extended to all 47) |
| `BayesianMap` | force balance | 0.21 | 0.61 | O(1) **by choice of prior**, not by measurement |
| `AnglePropagation` | `YoungDupre` | 0.015 | 0.078 | ~65x conservative |
| `ClusterRobust` | force balance | 204 | 1.5e4 | optimistic; **do not ship** |
| `ClusterBootstrap` | force balance | 216 | 3e4 | optimistic; **do not ship** |
| `IidGls` | force balance | 246 | 5.4e3 | **do not ship** (the force-balance work's naive posterior) |
| `LineBootstrap` | `YoungDupre` | 1.3e-05 | — | ~1e5 conservative, weight-confounded; diagnostic only |

**So that goal is met: `ProbeResidual` is an interval with `error/std` = O(1), and it
errs on the safe side on every one of the 42 cases.**

**Why the naive posterior fails by 250x, and why the fix is one factor.** The i.i.d. model
says the per-component vertex force residual is `N(0, sigma_f^2)` with `sigma_f` estimated
as `||M beta|| / sqrt(dof)`. On a 11 000-vertex mesh that is ~34 000 rows, so the estimator
averages the noise down by `sqrt(34 000) ~ 185` and claims to know each tension to ~1e-4.
It does not: the force-balance work established that the residual is **systematic** — the meshes are the
equilibrium of the generator's volume-*constraint* projection, not of `M beta = 0` — so it
does not average down at all. Measured, the implied `n_eff` (the number of independent
observations the rows are actually worth, `dof / (error/std)^2`) has median **0.57** on the
42 and is **below 1 on 29 of them**: the ~33 000 rows are worth *less than one* independent
observation.

`CoherentResidual` replaces the probabilistic assumption with a deterministic one, and it
is the same linear algebra with `sigma_f -> ||M beta||`: treat the residual as **one
coherent perturbation of measured norm and unknown direction**. The worst-case-direction
first-order bound on the estimator is then

    |d gamma_i| <= sigma * sqrt(Sigma_unit_ii),

with `Sigma_unit` the constrained-GLS shape matrix of
`foambryo.validation.force_balance.constrained_covariance` at `sigma_f = 1`. That is the
`n_eff = 1` extreme, and it moves the median calibration from 246 to **1.33** — the whole
250x gap is that one `sqrt(dof)` factor, nothing else. But it is **not uniform**: 10 of the
42 cases still come out above 10, worst 29.9.

**`ProbeResidual` closes that, and the mechanism is measured rather than guessed.** The
remaining failure is *self-fitting*: `||M beta_hat||` is the residual at the estimate, and
`beta_hat` was chosen to minimise it. The residual at the truth is a median **5.4x** and up
to **87x** larger, and that ratio is what predicts the failures (Spearman 0.80). Worse, the
bound fails **worst on the best meshes** — a mesh close to `M`'s equilibrium lets the fit
drive `||M beta_hat||` to almost nothing while the true error stays put (Spearman against
`rho_free`: **-0.81**). `ProbeResidual` therefore takes `sigma` from the residual at an
**independent** tension estimate (`PROBE_METHOD`, `YoungDupre`, which never touches `M`).
Measured: the probe residual reproduces the residual at the ground truth to **0.1 % on every
case**, so this is the oracle bound without an oracle — and probe error can only *inflate*
`sigma`, so it fails safe. See `_probe_residual_norm`.

**Why resampling does not rescue the i.i.d. model.** `ClusterRobust` (a cluster-robust
sandwich) and `ClusterBootstrap` (resampling whole cells) both model *correlation*; neither
can model a *bias*. Measured, they move 246 to 204 and 216 — a factor 1.2, not 250. That is
the honest reason to prefer the deterministic bound: the dominant term is not variance, so
there is nothing for a resampling scheme to find.

**Why `ForceBalanceMAP`'s apparently-good 0.21 is not an answer.** Its interval's width is
set by `MAP_PRIOR_TENSION_STD`, a hyperparameter documented as "deliberately broad" and
never derived. Sweeping it over 0.03-1.0 moves `error/std` over 2.93-0.061, a log-log slope
of **-1.03**: the interval *is* the prior. Any calibration it shows was chosen. Both
`CoherentResidual` and `ProbeResidual` have no such knob — their scale is a measured residual
norm.

**The negative control behaves differently, and that is a check on the mechanism.** On the
five non-equilibrium cases excluded from every accuracy claim, the residual is genuinely large and spread over
interface interiors, so the implied `n_eff` is **370** rather than 0.57, `IidGls` improves to
9.0, and `ProbeResidual`/`CoherentResidual` become ~20x conservative. The models fail in the
*safe* direction exactly where the mesh is bad — which is the behaviour one wants and not
something that was fitted.

**Why the angle family is conservative rather than optimistic.** `YoungDupre` reads
triple-line angles and never touches `M`. Propagating the *measured* along-line angle
spread through the operator over-predicts its error by ~2 orders, because (as the force-balance
work established) the dataset's
angle deviations from Neumann lie nearly orthogonal to the Young-Dupre operator's column
space. So its interval is safe and nearly useless, and its point estimate (2.3e-05) is far
better than any interval we can defend around it.

**One caveat carried from the per-edge Young-Dupre work, and it is load-bearing.** `YoungDupre`'s accuracy is
**weighting-critical**: a factor-~4 change in relative inter-line weights costs two orders
of magnitude. `LineBootstrap` resamples whole triple lines, which *is* such a reweighting
(`n` copies of a line of weight `w` act as one row of weight `w sqrt(n)`), so its width is
dominated by weight perturbation rather than by angle noise. It is reported for that reason,
not despite it. **Nothing here changes the junction-length weighting of any shipped
method** — every point estimate returned by this module is bit-identical to the
corresponding `infer_tensions` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from foambryo.force_balance_inference import build_force_balance_operator
from foambryo.tension_inference import (
    TensionComputationMethod,
    _build_matrix_tension_symmetrical_yd,
)
from foambryo.validation import force_balance as fb

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


class UncertaintyModel(Enum):
    """How the interval around an inferred tension is produced.

    See the module docstring for the measured `error/std` of each. `ProbeResidual` is the one
    recommended for reporting — O(1) *and* conservative on every one of the 42 equilibrium
    cases. The others are kept because a calibration claim that cannot be reproduced against
    its alternatives is not a calibration claim, and because two of them (`IidGls`,
    `ClusterRobust`) are the models a reader would otherwise reach for.

    Attributes:
        ProbeResidual: `CoherentResidual`, but scaled by the residual at an **independent**
            tension estimate instead of at the fitted one, which removes the self-fitting.
            Force-balance / `Variational` family. **Default, and the only model measured to
            be conservative on every one of the 42 cases.**
        CoherentResidual: Deterministic worst-case-direction bound, `sigma_f = ||M beta||`
            (`n_eff = 1`). Force-balance / `Variational` family. O(1) at the median but
            **not uniformly** — see the module docstring.
        IidGls: the force-balance work's naive constrained-GLS posterior, `sigma_f = ||M beta|| / sqrt(dof)`.
            Retained for comparison and **not for reporting** — ~250x optimistic.
        ClusterRobust: Cluster-robust (sandwich) covariance, clusters = cells.
        ClusterBootstrap: Nonparametric bootstrap resampling whole cells.
        AnglePropagation: First-order propagation of the *measured* along-line angle spread
            through the Young-Dupre operator. Angle family (`YoungDupre`).
        LineBootstrap: Nonparametric bootstrap resampling whole triple lines. Angle family;
            weight-confounded, see the module docstring.
    """

    CoherentResidual = 0
    IidGls = 1
    ClusterRobust = 2
    ClusterBootstrap = 3
    AnglePropagation = 4
    LineBootstrap = 5
    ProbeResidual = 6


#: Models that apply to the force-balance / `Variational` estimator (they need `M`).
FORCE_BALANCE_MODELS = frozenset(
    {
        UncertaintyModel.ProbeResidual,
        UncertaintyModel.CoherentResidual,
        UncertaintyModel.IidGls,
        UncertaintyModel.ClusterRobust,
        UncertaintyModel.ClusterBootstrap,
    },
)

#: Estimator used as the independent probe by `UncertaintyModel.ProbeResidual`. `YoungDupre`
#: is the right choice on the evidence: it reads triple-line angles and **never touches `M`**,
#: so its error is not bounded by `rho_free` (per the force-balance work), and it is this dataset's most accurate
#: method by three orders of magnitude (2.3e-05). Its accuracy is not *required* for the bound
#: to be safe, only for it to be tight — see the module docstring.
PROBE_METHOD = TensionComputationMethod.YoungDupre

#: Models that apply to the angle-based `YoungDupre` estimator.
ANGLE_MODELS = frozenset({UncertaintyModel.AnglePropagation, UncertaintyModel.LineBootstrap})

#: Default number of bootstrap resamples. 200 is what the per-edge Young-Dupre work used; the reported spread is
#: stable to ~2 % against 400 (measured, `foambryo_benchmarks/run_uncertainty_calibration.py`).
DEFAULT_N_RESAMPLES = 200


@dataclass(frozen=True)
class InferenceDiagnostics:
    """Per-mesh diagnostics saying whether *any* tension inference on it can be trusted.

    These are properties of the mesh and the operator, not of a method, which is what makes
    them usable on `dw3d` output where no ground truth exists. `rho_free` is the one to read
    first: the force-balance work established that the force-balance / `Variational` family's error is
    limited by it (median `rho_free` 0.012 against a median error 0.021 on the 42), while
    `YoungDupre`'s is not.

    Attributes:
        n_interfaces (int): Number of interfaces (tension unknowns).
        n_cells (int): Number of cells (pressure unknowns; the exterior is the gauge).
        n_rows (int): Rows of `M` = `3 * n_vertices`.
        rho (float | None): `||P_perp M_gamma gamma|| / ||M_gamma gamma||` at the tensions
            passed in, pressures marginalised — the fraction of the surface-tension force
            this mesh leaves unbalanced *at those tensions*. `None` if no tensions were
            given. Gauge-invariant.
        rho_free (float): The same quantity with the tensions themselves free — the
            **distance of this mesh from any equilibrium of `M`**, and the floor `rho`
            cannot go below. Gauge-invariant. ~1e-10 for a mesh that minimises `M`;
            0.0007-0.036 on the 42 simulation meshes; 0.26-0.28 on the five non-equilibrium
            cases excluded from every accuracy claim. **On `dw3d`-reconstructed meshes of the
            same cases it is about 50x the reference value: median 0.578 (range 0.48-0.72) on the 40-case equilibrium
            cohort, every case above the 0.2 alarm below**, so the force-balance family's
            numbers on reconstructions stand on a floor no gauge choice can lower.
        sigma_gauge (float): Smallest singular value of the column-normalised `M` — the
            overall-scale gauge direction. Should be near zero; if it is not, the operator
            has no scale gauge and something is wrong.
        sigma_soft (float): Second-smallest singular value: the worst-determined *physical*
            direction. `1 / sigma_soft` is the noise amplification (1.2-5.5 on the 42).
        soft_mode_fractions (tuple[float, float, float]): How much of that soft direction
            sits in the (surface tension, line tension, pressure) blocks. The line entry is
            always 0 here — the harness operator carries no line-tension block, by the package
            author's design choice.
        condition_number (float): `cond(M)`, the operator actually inverted.
    """

    n_interfaces: int
    n_cells: int
    n_rows: int
    rho: float | None
    rho_free: float
    sigma_gauge: float
    sigma_soft: float
    soft_mode_fractions: tuple[float, float, float]
    condition_number: float

    def identifiability_report(self) -> str:
        """Human-readable identifiability and equilibrium summary.

        Returns:
            str: A short multi-line report, safe to print next to an inferred tension table.
        """
        lines = [
            f"interfaces {self.n_interfaces}, cells {self.n_cells}, operator rows {self.n_rows}",
            f"rho_free   = {self.rho_free:.4g}   (distance of this mesh from any equilibrium of M)",
        ]
        if self.rho is not None:
            lines.append(f"rho        = {self.rho:.4g}   (at the tensions supplied)")
        surface, line, pressure = self.soft_mode_fractions
        lines += [
            f"sigma_gauge= {self.sigma_gauge:.4g}   (the overall-scale gauge; expected ~0)",
            f"sigma_soft = {self.sigma_soft:.4g}   -> noise amplification 1/sigma_soft = {1 / self.sigma_soft:.3g}",
            f"soft mode  = {surface:.2f} surface / {line:.2f} line / {pressure:.2f} pressure",
            f"cond(M)    = {self.condition_number:.4g}",
        ]
        if self.rho_free > 0.2:
            lines.append(
                "WARNING: rho_free > 0.2 — no surface tensions balance this mesh at all. "
                "The force-balance / Variational family cannot be trusted on it; the "
                "angle-based family (YoungDupre) does not invert M and is not bounded this way.",
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class TensionUncertaintyResult:
    """An inferred tension field with an interval whose calibration is stated, not assumed.

    Attributes:
        interfaces (list[tuple[int, int]]): Interface keys, in the order of `tensions`.
        tensions (NDArray[np.float64]): Inferred tensions — **bit-identical** to the
            corresponding `infer_tensions` call. This module never changes a point estimate.
        tension_std (NDArray[np.float64]): Per-interface interval half-width. For
            `CoherentResidual` this is a worst-case-direction *bound*, not a standard
            deviation; read `calibration_note`.
        method (TensionComputationMethod): Which estimator produced `tensions`.
        model (UncertaintyModel): Which model produced `tension_std`.
        sigma_scale (float): The residual scale the interval was built on.
        residual_norm (float): `||M beta||` (force-balance family) or `||M gamma - b||`
            (angle family) at the returned estimate.
        n_rows (int): Rows of the operator inverted.
        diagnostics (InferenceDiagnostics): Mesh-level identifiability and equilibrium.
        calibration_note (str): The measured `error/std` of this model on the 42 equilibrium
            cases, and what it means. Always non-empty: an uncalibrated interval must not
            leave this module without saying so.
    """

    interfaces: list[tuple[int, int]]
    tensions: NDArray[np.float64]
    tension_std: NDArray[np.float64]
    method: TensionComputationMethod
    model: UncertaintyModel
    sigma_scale: float
    residual_norm: float
    n_rows: int
    diagnostics: InferenceDiagnostics
    calibration_note: str = field(default="")

    @property
    def dict_tensions(self) -> dict[tuple[int, int], float]:
        """Map interface key -> inferred tension."""
        return dict(zip(self.interfaces, self.tensions.tolist(), strict=True))

    @property
    def dict_tension_std(self) -> dict[tuple[int, int], float]:
        """Map interface key -> interval half-width on that tension."""
        return dict(zip(self.interfaces, self.tension_std.tolist(), strict=True))


#: Measured median `error/std` over the 42 equilibrium cases, keyed by model. Filled in from
#: `foambryo_benchmarks/baseline/uncertainty_calibration.json`; pinned by
#: `tests/test_uncertainty.py::test_calibration_notes_match_the_committed_measurement` so a
#: re-measurement cannot silently disagree with what this module tells users.
CALIBRATION_ERROR_OVER_STD: dict[UncertaintyModel, float] = {
    UncertaintyModel.ProbeResidual: 0.233,
    UncertaintyModel.CoherentResidual: 1.33,
    UncertaintyModel.IidGls: 246.0,
    UncertaintyModel.ClusterRobust: 204.0,
    UncertaintyModel.ClusterBootstrap: 216.0,
    UncertaintyModel.AnglePropagation: 0.0154,
    UncertaintyModel.LineBootstrap: 1.3e-05,
}


def _calibration_note(model: UncertaintyModel) -> str:
    """Build the mandatory calibration note for a model.

    Args:
        model (UncertaintyModel): The model.

    Returns:
        str: The note stored on the result.
    """
    ratio = CALIBRATION_ERROR_OVER_STD[model]
    measured = (
        f"Measured median error/std = {ratio:g} over the 42 equilibrium cases of "
        "benchmarking-dataset (foambryo_benchmarks/run_uncertainty_calibration.py). "
        "A calibrated interval gives 1."
    )
    if model is UncertaintyModel.ProbeResidual:
        return (
            f"{measured} This is a worst-case-direction BOUND, not a standard deviation, and "
            "it is the one interval this package recommends reporting: O(1) and conservative by ~4.3x on "
            "the median, with error/std <= 1 on 42 of the 42 equilibrium cases (worst 0.93, range 0.081-0.93). "
            "Its scale is the residual at an INDEPENDENT tension estimate (YoungDupre), which "
            "reproduces the residual at the ground truth to 0.1 %; a worse probe can only widen "
            "the interval, so it fails safe."
        )
    if model is UncertaintyModel.CoherentResidual:
        return (
            f"{measured} This is a worst-case-direction BOUND at n_eff = 1, not a standard "
            "deviation. O(1) at the median, but NOT UNIFORM: 10 of the 42 cases exceed 10 "
            "(worst 29.9), because it scales on the residual at its own fitted estimate, which "
            "is a median 5.4x smaller than the residual at the truth — and worst on the "
            "best-relaxed meshes (Spearman -0.81 against rho_free). Prefer ProbeResidual, which "
            "fixes exactly this."
        )
    if model is UncertaintyModel.IidGls:
        return (
            f"{measured} OPTIMISTIC BY ~250x, in the dangerous direction. DO NOT REPORT THIS "
            "AS AN ERROR BAR. Retained only as the comparison baseline for the force-balance residual: "
            "the i.i.d. force-noise assumption is false because the residual is systematic, "
            "so it does not average down over the ~34 000 operator rows."
        )
    if model in (UncertaintyModel.ClusterRobust, UncertaintyModel.ClusterBootstrap):
        return (
            f"{measured} Still two orders OPTIMISTIC. Modelling the residual's correlation "
            "recovers only a factor ~1.2 of the missing 250x (246 -> 204/216), because the "
            "dominant term is a bias and no resampling scheme can produce a bias. Not for "
            "reporting."
        )
    if model is UncertaintyModel.AnglePropagation:
        return (
            f"{measured} CONSERVATIVE by ~65x: the dataset's angle deviations from "
            "Neumann lie nearly orthogonal to the Young-Dupre operator's column space "
            "(as measured on the force-balance operator), so propagating their magnitude as i.i.d. noise vastly "
            "over-predicts "
            "the error. Safe, but nearly uninformative."
        )
    return (
        f"{measured} CONSERVATIVE by ~2 orders, and WEIGHT-CONFOUNDED: resampling whole "
        "triple lines is itself an inter-line reweighting, which the per-edge Young-Dupre work measured to be worth "
        "two orders of magnitude on this operator. Its width therefore reflects "
        "weight sensitivity more than angle noise. Diagnostic only."
    )


def compute_inference_diagnostics(
    mesh: DcelData,
    dict_tensions: dict[tuple[int, int], float] | None = None,
) -> InferenceDiagnostics:
    """Equilibrium and identifiability diagnostics for a mesh — the uncertainty work's user-facing report.

    This is the `rho` / `sigma_soft` block that the dataset-audit work computed inside the benchmark
    harness, promoted to a public function so it can be run on a mesh with no ground truth
    (i.e. on `dw3d` output). Everything here is a property of the mesh; nothing is fitted to
    a ground-truth tension.

    Args:
        mesh (DcelData): Mesh to analyse.
        dict_tensions (dict[tuple[int, int], float] | None, optional): Tensions at which to
            evaluate `rho`. Defaults to `None`, in which case `rho` is `None` and only
            `rho_free` (which needs no tensions) is reported.

    Returns:
        InferenceDiagnostics: See that class; call `identifiability_report()` for text.
    """
    m, interfaces, cells = build_force_balance_operator(mesh)
    ni = len(interfaces)

    rho = None
    if dict_tensions is not None:
        gamma = np.array([dict_tensions[key] for key in interfaces])
        rho = fb.residual_at(m, ni, 0, gamma, np.zeros(0))
    _gamma_eff, _p_eff, rho_free = fb.effective_surface_tensions(m, ni, 0, mean_tension=1.0)
    spectrum = fb.singular_values(m)
    sigma_soft, fractions = fb.soft_mode(m, ni, 0)

    return InferenceDiagnostics(
        n_interfaces=ni,
        n_cells=len(cells),
        n_rows=m.shape[0],
        rho=rho,
        rho_free=rho_free,
        sigma_gauge=float(spectrum[0]),
        sigma_soft=sigma_soft,
        soft_mode_fractions=(float(fractions[0]), float(fractions[1]), float(fractions[2])),
        condition_number=float(np.linalg.cond(m)),
    )


def _constrained_shape(m: NDArray[np.float64], anchor: NDArray[np.float64]) -> NDArray[np.float64]:
    """The constrained-GLS covariance at `sigma_f = 1` — the interval's shape, without its scale.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        anchor (NDArray[np.float64]): The gauge constraint row.

    Returns:
        NDArray[np.float64]: `Sigma_unit`, shape `(ncols, ncols)`. Singular along `anchor`,
            which is correct: the overall scale is a convention, not a measurement.
    """
    return fb.constrained_covariance(m, anchor, 1.0)


def _vertex_cell_clusters(mesh: DcelData) -> NDArray[np.int64]:
    """Partition vertices into non-overlapping clusters, one per cell.

    A vertex is assigned to the **lowest** region label incident on any of its triangles.
    Overlapping clusters would invalidate both the sandwich estimator and the block
    bootstrap, so an arbitrary-but-deterministic tie-break is required; the lowest label is
    that tie-break, and it makes the partition a function of the mesh alone.

    Args:
        mesh (DcelData): Mesh to analyse.

    Returns:
        NDArray[np.int64]: Cluster index per vertex, shape `(n_vertices,)`, values dense
            in `[0, n_clusters)`.
    """
    triangles, regions = mesh.f[:, :3], mesh.f[:, 3:5]
    label = np.full(len(mesh.v), np.iinfo(np.int64).max, dtype=np.int64)
    lowest = regions.min(axis=1)
    for corner in range(3):
        np.minimum.at(label, triangles[:, corner], lowest)
    _uniq, dense = np.unique(label, return_inverse=True)
    return dense.ravel().astype(np.int64)


def _cluster_robust_covariance(
    m: NDArray[np.float64],
    residual: NDArray[np.float64],
    shape: NDArray[np.float64],
    clusters: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Cluster-robust (sandwich) covariance `Sigma P Meat P Sigma` with cell clusters.

    `Meat = sum_g (X_g^T e_g)(X_g^T e_g)^T` over clusters, the standard Liang-Zeger form. It
    is consistent as the number of *clusters* grows; with 6-20 cells per mesh it is very
    noisy, and that noisiness is itself the finding — the effective sample size for a
    correlated residual is the cluster count, not the row count.

    Args:
        m (NDArray[np.float64]): The force-balance operator, shape `(3nv, ncols)`.
        residual (NDArray[np.float64]): `M beta`, shape `(3nv,)`.
        shape (NDArray[np.float64]): `Sigma_unit` from `_constrained_shape`.
        clusters (NDArray[np.int64]): Cluster index per vertex, shape `(nv,)`.

    Returns:
        NDArray[np.float64]: Covariance of `beta`, shape `(ncols, ncols)`.
    """
    meat = np.zeros((m.shape[1], m.shape[1]))
    for group in np.unique(clusters):
        rows = np.repeat(clusters == group, 3)
        score = m[rows].T @ residual[rows]
        meat += np.outer(score, score)
    return shape @ meat @ shape


def _cluster_bootstrap_std(
    m: NDArray[np.float64],
    anchor: NDArray[np.float64],
    clusters: NDArray[np.int64],
    ni: int,
    mean_tension: float,
    n_resamples: int,
    seed: int,
) -> NDArray[np.float64]:
    """Bootstrap standard deviation of the tensions, resampling whole cells with replacement.

    Each resample re-weights the rows by how many times their cluster was drawn and re-solves
    the same constrained problem, so the estimator is unchanged and only the data are
    perturbed. The gauge is re-imposed on every draw, so the spread is a spread of tension
    *ratios*, not of the scale convention.

    Args:
        m (NDArray[np.float64]): The force-balance operator.
        anchor (NDArray[np.float64]): The gauge constraint row.
        clusters (NDArray[np.int64]): Cluster index per vertex.
        ni (int): Number of interfaces.
        mean_tension (float): The gauge target.
        n_resamples (int): Number of bootstrap draws.
        seed (int): RNG seed, recorded so the number is reproducible.

    Returns:
        NDArray[np.float64]: Per-interface bootstrap standard deviation, shape `(ni,)`.
    """
    rng = np.random.default_rng(seed)
    n_vertices = len(clusters)
    n_groups = int(clusters.max()) + 1
    rows_of_group = [np.flatnonzero(clusters == g) for g in range(n_groups)]
    n_cols = m.shape[1]
    draws = []
    for _ in range(n_resamples):
        pick = rng.integers(0, n_groups, n_groups)
        weight = np.bincount(
            np.concatenate([rows_of_group[p] for p in pick]),
            minlength=n_vertices,
        ).astype(float)
        gram = (m * np.repeat(weight, 3)[:, None]).T @ m
        kkt = np.zeros((n_cols + 1, n_cols + 1))
        kkt[:n_cols, :n_cols] = 2 * gram
        kkt[:n_cols, n_cols] = anchor
        kkt[n_cols, :n_cols] = anchor
        rhs = np.zeros(n_cols + 1)
        rhs[-1] = mean_tension
        try:
            beta = np.linalg.solve(kkt, rhs)[:ni]
        except np.linalg.LinAlgError:  # pragma: no cover - a degenerate resample
            continue
        draws.append(beta)
    if not draws:  # pragma: no cover - would need every resample to be singular
        return np.full(ni, np.nan)
    return np.std(np.asarray(draws), axis=0)


def _probe_residual_norm(
    mesh: DcelData,
    m: NDArray[np.float64],
    ni: int,
    interfaces: list[tuple[int, int]],
) -> float:
    """`||P_perp M_gamma gamma_probe||` — the perturbation size, measured off an independent estimate.

    `CoherentResidual` scales its bound by `||M beta_hat||`, the residual at the *fitted*
    estimate. That is systematically too small, because `beta_hat` is chosen to minimise it:
    measured over the 42 cases, the residual at the truth is a **median 5.4x and up to 87x**
    larger, and that ratio is exactly what predicts where the bound fails (Spearman 0.80).
    It is also the reason the bound fails *worst on the best meshes* — a mesh close to `M`'s
    equilibrium lets the fit drive `||M beta_hat||` to almost nothing while the true error
    stays put (Spearman with `rho_free`: -0.81).

    The fix needs an estimate of `gamma` that was **not** obtained by minimising this residual.
    `PROBE_METHOD` (`YoungDupre`) is exactly that: it reads triple-line angles and never
    touches `M`. Measured, the probe residual reproduces the residual at the ground truth to
    **0.1 % on every case** — so the resulting bound is the oracle bound, without an oracle.

    **Why probe error fails safe rather than dangerous**, which is what makes this defensible
    beyond this dataset: by the triangle inequality,
    `||P_perp M_gamma gamma_probe|| <= ||P_perp M_gamma gamma*|| + ||P_perp M_gamma (gamma_probe - gamma*)||`,
    so a *worse* probe can only inflate the scale, widening the interval. The bound needs the
    probe to be good to be **tight**, not to be **valid**.

    Args:
        mesh (DcelData): Mesh to analyse.
        m (NDArray[np.float64]): The force-balance operator.
        ni (int): Number of interfaces.
        interfaces (list[tuple[int, int]]): Interface keys in the operator's column order.

    Returns:
        float: The probe residual norm, with pressures marginalised in closed form.
    """
    from foambryo.tension_inference import infer_tensions  # local import: avoids an import cycle

    probe = infer_tensions(mesh, mean_tension=1, mode=PROBE_METHOD)
    gamma = np.array([probe[key] for key in interfaces])
    gamma = gamma / gamma.mean()  # the probe's own gauge may be soft; put it in ours
    m_gamma, _m_t, m_p = fb.split(m, ni, 0)
    load = m_gamma @ gamma
    pressures, *_ = np.linalg.lstsq(m_p, -load, rcond=None)
    return float(np.linalg.norm(load + m_p @ pressures))


def _force_balance_uncertainty(
    mesh: DcelData,
    mean_tension: float,
    model: UncertaintyModel,
    n_resamples: int,
    seed: int,
) -> tuple[list[tuple[int, int]], NDArray[np.float64], NDArray[np.float64], float, float, int]:
    """Tensions and interval for the force-balance / `Variational` estimator.

    Args:
        mesh (DcelData): Mesh to analyse.
        mean_tension (float): The gauge.
        model (UncertaintyModel): One of `FORCE_BALANCE_MODELS`.
        n_resamples (int): Bootstrap draws, used by `ClusterBootstrap` only.
        seed (int): Bootstrap seed.

    Returns:
        tuple: `(interfaces, tensions, tension_std, sigma_scale, residual_norm, n_rows)`.
    """
    m, interfaces, _cells = build_force_balance_operator(mesh)
    ni = len(interfaces)
    n_rows, n_cols = m.shape
    anchor = np.zeros(n_cols)
    anchor[:ni] = 1.0 / ni
    beta = fb.solve_constrained(m, anchor, np.array([mean_tension]))
    residual = m @ beta
    residual_norm = float(np.linalg.norm(residual))
    shape = _constrained_shape(m, anchor)

    if model is UncertaintyModel.ProbeResidual:
        # The perturbation that displaced the estimate is the residual at the TRUTH, not at
        # the estimate; the latter is smaller by construction because beta_hat minimises it.
        # An independent estimator's tensions stand in for the truth as a probe.
        sigma = _probe_residual_norm(mesh, m, ni, interfaces)
        std = sigma * np.sqrt(np.clip(np.diag(shape)[:ni], 0.0, None))
    elif model is UncertaintyModel.CoherentResidual:
        # n_eff = 1: one coherent perturbation of measured norm, worst-case direction.
        sigma = residual_norm
        std = sigma * np.sqrt(np.clip(np.diag(shape)[:ni], 0.0, None))
    elif model is UncertaintyModel.IidGls:
        sigma = fb.residual_noise_scale(m, beta, n_constraints=1)
        std = sigma * np.sqrt(np.clip(np.diag(shape)[:ni], 0.0, None))
    elif model is UncertaintyModel.ClusterRobust:
        clusters = _vertex_cell_clusters(mesh)
        cov = _cluster_robust_covariance(m, residual, shape, clusters)
        sigma = residual_norm
        std = np.sqrt(np.clip(np.diag(cov)[:ni], 0.0, None))
    else:  # ClusterBootstrap
        clusters = _vertex_cell_clusters(mesh)
        sigma = residual_norm
        std = _cluster_bootstrap_std(m, anchor, clusters, ni, mean_tension, n_resamples, seed)

    return interfaces, beta[:ni], std, sigma, residual_norm, n_rows


def _young_dupre_operator(
    mesh: DcelData,
    mean_tension: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list[tuple[int, int]],
    dict[tuple[int, int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int, int], float],
]:
    """Assemble the shipped `YoungDupre` system, unchanged, plus the pieces the interval needs.

    Calls `_build_matrix_tension_symmetrical_yd` with exactly the arguments
    `_infer_tension_symmetrical_yd` passes, including `unique=False` — so the returned
    operator, its junction-length weights and hence its solution are bit-identical to
    `infer_tensions(mesh, mode=YoungDupre)`. **The weighting is load-bearing (the per-edge Young-Dupre work) and
    is not touched here.**

    Args:
        mesh (DcelData): Mesh to analyse.
        mean_tension (float): The gauge.

    Returns:
        tuple: `(matrix_m, vector_b, interfaces, dict_rad, dict_areas, dict_length)`.
    """
    dict_rad, _dict_deg, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_symmetrical_yd(
        dict_rad,
        dict_areas,
        dict_length,
        mean_tension,
    )
    interfaces = [tuple(int(label) for label in key) for key in sorted(dict_areas)]
    return matrix_m, vector_b, interfaces, dict_rad, dict_areas, dict_length


def _angle_propagation_std(
    mesh: DcelData,
    matrix_m: NDArray[np.float64],
    tensions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Propagate the *measured* along-line angle spread through the Young-Dupre operator.

    The system is `min ||M(alpha) gamma - b||`, and only `M` depends on the angles, so to
    first order `d gamma = -M^+ (dM) gamma`. Every angle enters exactly one place, as
    `cos(alpha) * factor`, so `d(dM)/d alpha = -sin(alpha) * factor` at those entries and the
    Jacobian columns can be assembled analytically. With the angle errors taken independent
    across (line, wedge),

        Cov(gamma) = J diag(sigma_alpha^2) J^T,    J[:, k] = -M^+ (dM/d alpha_k) gamma.

    `sigma_alpha` for a line is the **standard error of that line's mean wedge angle**,
    `s / sqrt(n_edges)`, measured from the per-edge angles of the per-edge Young-Dupre work's
    `compute_trijunction_edges` — the same along-line variation that work proved is error rather
    than signal. A line resolved by a single edge is given the mesh-wide median `s` instead
    of zero, so it is not silently treated as exact.

    Args:
        mesh (DcelData): Mesh to analyse.
        matrix_m (NDArray[np.float64]): The assembled Young-Dupre operator.
        tensions (NDArray[np.float64]): The solved tensions.

    Returns:
        tuple[NDArray[np.float64], float]: `(std, median_angle_standard_error_rad)`.
    """
    from foambryo.geometry import compute_trijunction_edges  # local import: avoids an import cycle

    dict_rad, _deg, dict_length = mesh.compute_angles_tri(unique=False)
    per_key: dict[tuple[int, int, int], list[float]] = {}
    for edge in compute_trijunction_edges(mesh):
        for key, angle in zip(edge.keys, edge.angles, strict=True):
            per_key.setdefault(key, []).append(float(angle))

    spreads = [float(np.std(v, ddof=1)) for v in per_key.values() if len(v) > 1]
    fallback = float(np.median(spreads)) if spreads else 0.0
    sigma_of_key: dict[tuple[int, int, int], float] = {}
    for key, angles in per_key.items():
        spread = float(np.std(angles, ddof=1)) if len(angles) > 1 else fallback
        sigma_of_key[key] = spread / np.sqrt(len(angles))

    # Rebuild the operator's row layout, exactly as the builder does, so each angle can be
    # located. The builder's own key bookkeeping is reproduced rather than guessed.
    n_interfaces = matrix_m.shape[1]
    n_junction_rows = matrix_m.shape[0] - 1
    n_junctions = n_junction_rows // 3

    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear = np.unique(linear)
    keys_t = np.array(sorted(mesh.compute_areas_interfaces().keys()))
    nb_zones = np.amax(keys_t) + 1
    reverse_map = {
        keys_t[i, 0] * nb_zones + keys_t[i, 1]: i for i in range(len(keys_t))
    }
    total_length = sum(dict_length.values()) / 3

    pinv = np.linalg.pinv(matrix_m)
    jacobian_columns = []
    variances = []
    for i, key in enumerate(linear[:n_junctions]):
        c = key // ka**2
        b = (key - (c * ka**2)) // ka
        a = key % ka
        factor = dict_length[(a, b, c)] / total_length
        idx_ab = reverse_map[min(a, b) * nb_zones + max(a, b)]
        idx_bc = reverse_map[min(b, c) * nb_zones + max(b, c)]
        idx_ac = reverse_map[min(a, c) * nb_zones + max(a, c)]
        # (wedge key, the (row, column) pairs its cosine occupies)
        layout = (
            ((a, b, c), ((3 * i, idx_bc), (3 * i + 1, idx_ab))),
            ((b, a, c), ((3 * i, idx_ac), (3 * i + 2, idx_ab))),
            ((a, c, b), ((3 * i + 1, idx_ac), (3 * i + 2, idx_bc))),
        )
        for wedge_key, cells in layout:
            angle = dict_rad[wedge_key]
            d_matrix = np.zeros_like(matrix_m)
            for row, column in cells:
                d_matrix[row, column] = -np.sin(angle) * factor
            jacobian_columns.append(-pinv @ (d_matrix @ tensions))
            variances.append(sigma_of_key.get(wedge_key, fallback) ** 2)

    jacobian = np.array(jacobian_columns).T if jacobian_columns else np.zeros((n_interfaces, 0))
    cov = jacobian @ np.diag(np.array(variances)) @ jacobian.T if jacobian.size else np.zeros(
        (n_interfaces, n_interfaces),
    )
    median_se = float(np.median(list(sigma_of_key.values()))) if sigma_of_key else 0.0
    return np.sqrt(np.clip(np.diag(cov), 0.0, None)), median_se


def _line_bootstrap_std(
    matrix_m: NDArray[np.float64],
    gauge_rhs: float,
    n_resamples: int,
    seed: int,
) -> NDArray[np.float64]:
    """Bootstrap the `YoungDupre` tensions by resampling whole triple lines with replacement.

    The gauge row (the last row) is kept in every resample; only the `n_junctions` 3-row
    physics blocks are drawn. Each block carries the junction-length weight the shipped
    builder gave it — **the weighting is not modified** — but drawing a line `n` times is
    equivalent to one row of weight `w sqrt(n)`, so this bootstrap perturbs the inter-line
    weights as well as the angle sample. The per-edge Young-Dupre work measured that reweighting to be worth two
    orders of magnitude here, so the resulting width is dominated by it. Reported for that
    reason; see `_calibration_note`.

    Args:
        matrix_m (NDArray[np.float64]): The assembled Young-Dupre operator.
        gauge_rhs (float): The gauge row's right-hand side, `n_interfaces * mean_tension`.
        n_resamples (int): Number of bootstrap draws.
        seed (int): RNG seed.

    Returns:
        NDArray[np.float64]: Per-interface bootstrap standard deviation.
    """
    n_junctions = (matrix_m.shape[0] - 1) // 3
    blocks = matrix_m[:-1].reshape(n_junctions, 3, matrix_m.shape[1])
    gauge_row = matrix_m[-1:]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        pick = rng.integers(0, n_junctions, n_junctions)
        augmented = np.vstack([blocks[pick].reshape(-1, matrix_m.shape[1]), gauge_row])
        rhs = np.zeros(augmented.shape[0])
        rhs[-1] = gauge_rhs
        solution, *_ = linalg.lstsq(augmented, rhs)
        draws.append(solution)
    return np.std(np.asarray(draws), axis=0)


def infer_tensions_with_uncertainty(
    mesh: DcelData,
    mean_tension: float = 1,
    method: TensionComputationMethod = TensionComputationMethod.ForceBalance,
    model: UncertaintyModel | None = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
) -> TensionUncertaintyResult:
    """Infer tensions **and** an interval whose calibration has been measured and is reported.

    Additive companion to `foambryo.infer_tensions`: the point estimate is bit-identical to
    the corresponding `infer_tensions` call (pinned by `tests/test_uncertainty.py`), and the
    returned `calibration_note` states the measured `error/std` of the interval rather than
    leaving the reader to assume it is 1.

    Only the estimators whose intervals the uncertainty work actually calibrated are accepted.
    That is deliberate: an interval for a method nobody measured would be exactly the failure
    mode this module exists to close.

    Args:
        mesh (DcelData): Mesh to analyse.
        mean_tension (float, optional): The gauge — the mean of the returned tensions.
            Defaults to 1.
        method (TensionComputationMethod, optional): The estimator. One of `ForceBalance`,
            `Variational` (both the force-balance family, same estimator up to the solve
            route) or `YoungDupre` (the angle family). Defaults to `ForceBalance`.
        model (UncertaintyModel | None, optional): How to build the interval. Defaults to
            `None`, which selects `CoherentResidual` for the force-balance family and
            `AnglePropagation` for the angle family — in both cases the best-calibrated
            model measured for that estimator.
        n_resamples (int, optional): Bootstrap draws, for the bootstrap models. Defaults to
            `DEFAULT_N_RESAMPLES`.
        seed (int, optional): Bootstrap RNG seed, recorded so any number is reproducible.
            Defaults to 0.

    Returns:
        TensionUncertaintyResult: Tensions, interval, diagnostics and calibration note.

    Raises:
        NotImplementedError: If `method` has no calibrated interval in the uncertainty work.
        ValueError: If `model` does not apply to `method`'s estimator family.
    """
    angle_family = {TensionComputationMethod.YoungDupre}
    balance_family = {
        TensionComputationMethod.ForceBalance,
        TensionComputationMethod.Variational,
    }
    if method in balance_family:
        allowed, default = FORCE_BALANCE_MODELS, UncertaintyModel.ProbeResidual
    elif method in angle_family:
        allowed, default = ANGLE_MODELS, UncertaintyModel.AnglePropagation
    else:
        message = (
            f"no calibrated interval exists for {method.name}. Error/std was measured "
            "only for the force-balance family (ForceBalance, Variational) and YoungDupre; "
            "shipping an interval for an estimator whose calibration was never measured "
            "would be a fabricated number, not a measured one. "
            "For YoungDupreLocal use foambryo.infer_tensions_young_dupre_local and its "
            "per-edge residual field."
        )
        raise NotImplementedError(message)

    chosen = default if model is None else model
    if chosen not in allowed:
        message = (
            f"{chosen.name} does not apply to {method.name}: it belongs to the other "
            f"estimator family. Allowed here: {sorted(m.name for m in allowed)}."
        )
        raise ValueError(message)

    if method in balance_family:
        interfaces, tensions, std, sigma, residual_norm, n_rows = _force_balance_uncertainty(
            mesh,
            mean_tension,
            chosen,
            n_resamples,
            seed,
        )
    else:
        matrix_m, vector_b, interfaces, _rad, _areas, _length = _young_dupre_operator(
            mesh,
            mean_tension,
        )
        tensions, _res, _rank, _sv = linalg.lstsq(matrix_m, vector_b)
        residual_norm = float(np.linalg.norm(matrix_m @ tensions - vector_b))
        n_rows = matrix_m.shape[0]
        if chosen is UncertaintyModel.AnglePropagation:
            std, sigma = _angle_propagation_std(mesh, matrix_m, tensions)
        else:
            std = _line_bootstrap_std(matrix_m, float(vector_b[-1]), n_resamples, seed)
            sigma = residual_norm

    return TensionUncertaintyResult(
        interfaces=interfaces,
        tensions=np.asarray(tensions, dtype=float),
        tension_std=np.asarray(std, dtype=float),
        method=method,
        model=chosen,
        sigma_scale=float(sigma),
        residual_norm=residual_norm,
        n_rows=int(n_rows),
        diagnostics=compute_inference_diagnostics(mesh),
        calibration_note=_calibration_note(chosen),
    )


def error_over_std(
    result: TensionUncertaintyResult,
    dict_truth_tensions: dict[tuple[int, int], float],
) -> dict[str, float]:
    """**The** calibration metric of the uncertainty work: `|error| / std`, gauge-matched, per interface.

    Gauge matching is mandatory (the rule every accuracy comparison in this package follows): only
    tension *ratios* are physical,
    so the inferred tensions are rescaled by the single factor making their mean match the
    truth's, and **the same factor is applied to the interval** — an interval is in the same
    units as the quantity it bounds, so rescaling one without the other would be a units
    error dressed up as a calibration result.

    Args:
        result (TensionUncertaintyResult): The result to score.
        dict_truth_tensions (dict[tuple[int, int], float]): Ground-truth tension per interface.

    Returns:
        dict[str, float]: `{"median", "p95", "max", "n", "gauge_matching_scale",
            "median_std", "median_abs_error"}`. Values are `nan` where the interval is zero.
    """
    truth = np.array([dict_truth_tensions[key] for key in result.interfaces])
    scale = float(np.mean(truth) / np.mean(result.tensions))
    error = np.abs(scale * result.tensions - truth)
    std = scale * result.tension_std
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(std > 0, error / std, np.nan)
    finite = ratio[np.isfinite(ratio)]
    return {
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "n": int(finite.size),
        "gauge_matching_scale": scale,
        "median_std": float(np.median(std)),
        "median_abs_error": float(np.median(error)),
    }
