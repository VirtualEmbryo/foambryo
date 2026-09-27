"""Tension inference from the unified discrete force-balance operator.

Part of the force-balance work. Where the eight classical methods
in `foambryo.tension_inference` each impose a *local* balance written in measured angles
(Young-Dupre, Lami, cotangent) or in per-interface area gradients (`Variational`), this
module inverts the **global** discrete equilibrium condition directly:

    E(v) = sum_interfaces gamma_m A_m(v) - sum_cells p_c V_c(v),
    equilibrium  <=>  grad_v E = 0  <=>  M beta = 0,     beta = (gamma, p),

with `M = [M_gamma | M_p]` assembled by
`foambryo.validation.force_balance.force_balance_operator` (ported, algorithm unchanged, from the
reference implementation that accompanies the technical note on tension inference (H. Turlier,
in preparation; not distributed)).
By the package author's design choice the line-tension block `M_t` is **not** used: `beta = (gamma, p)`.

**Sign convention.** The operator's pressure columns are `+dV_c/dv`, so the energy above
puts pressure in with a minus and the solved pressure-block entries are `-p`. This module
negates them, so `ForceBalanceResult.pressures` are physical pressures on foambryo's usual
convention (exterior medium at 0, higher pressure inside a convex cell). The exterior
label is not a column at all — `mesh_io.cell_labels` drops it — which is what fixes the
additive pressure gauge; without that the constant-pressure shift would be an exact null
direction of `M_p` and every solve below would be singular.

**The gauge, and the three variants.** `M beta = 0` is homogeneous: `beta = 0` minimises it
and only tension *ratios* are physical, so one scale convention has to be added. That is
the open question this module exists to answer, and the three answers are
`ForceBalanceNormalisation`:

- `Kkt` — `mean(gamma) = mean_tension` as a **hard** constraint: minimise `||M beta||^2`
  subject to `a^T beta = mean_tension` (`a` = `1/ni` on the surface-tension entries, 0 on the
  pressures) by solving the KKT system `[[2 M^T M, a], [a^T, 0]] @ (beta, lambda) =
  (0, mean_tension)`. Exact by construction, and the constraint cannot compete with the
  physics rows.
- `SoftRow` — the convention every existing foambryo method uses: one extra least-squares
  row `sum(gamma) = ni * mean_tension`. The normalisation then *competes* with the physics
  rows, so the returned mean is only approximately `mean_tension`.
- `BayesianMap` — Gaussian prior on `beta`, MAP estimate and posterior covariance
  (`map_estimate`). The prior anchors the scale *softly* and shrinks the tensions toward
  the prior mean, so this variant's gauge is deliberately not exact; report it
  gauge-matched (the rule every accuracy comparison in this package follows).

All three produce a per-interface standard deviation. For `Kkt` it is the constrained-GLS
covariance with `sigma_f` estimated from the fit's own residual, so it carries no free
hyperparameter and is exactly zero along the fixed gauge direction; see
`foambryo.validation.force_balance.constrained_covariance` for the model and for the
caveat that i.i.d. force noise *understates* the true (spatially correlated) uncertainty.

**Reliability on reconstructed meshes -- read every number from this module with this.** The
operator is inverted on the mesh it is given, and a `dw3d`-reconstructed mesh is not a force
balance for *any* tension assignment: its `rho_free` (`foambryo.uncertainty`) is about fifty
times the reference meshes' own value, on every case and independent of which sampler
produced it. Measured on the 40 equilibrium cases of the benchmark cohort against their own
reference meshes (shipped default reconstruction: dithered point placement, seed 42, `min_distance =
3`): `rho_free` median 0.578, range 0.479-0.718, against the references' 0.0118, range 0.00066-0.036
-- a ratio of medians of 49, a median per-case ratio of 50, per-case ratios from 17 to 850 -- and
every one of the 40 above the 0.2 alarm threshold; the deterministic variant reads 0.623. The five
non-equilibrium cases' references already sit at 0.26 and their reconstructions at 0.59.
`rho` at the `Kkt` solution equals `rho_free` by construction, so the solve cannot lower it.
Every `ForceBalance*` tension and pressure inferred on a reconstruction, and every comparison
of one reconstruction against another through this family, stands on that floor until the
reconstruction itself moves it; the angle-reading family (`YoungDupre` and relatives) never
inverts `M` and is not bounded this way. Whether the family is repairable on reconstructed
meshes is an open scientific question, not addressed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from foambryo.validation import force_balance as fb
from foambryo.validation import mesh_io

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


class ForceBalanceNormalisation(Enum):
    """How the overall tension scale (the gauge) is fixed when inverting `M beta = 0`.

    See the module docstring. `Kkt` is the default and the one
    `TensionComputationMethod.ForceBalance` uses.
    """

    Kkt = 0
    SoftRow = 1
    BayesianMap = 2


#: Prior standard deviation on each tension for `ForceBalanceNormalisation.BayesianMap`,
#: in units of `mean_tension`. 0.3 is deliberately broad — the tensions of a foam are
#: positive and O(1) in this gauge, and `benchmarking-dataset`'s span 0.075 to ~1.5 — but
#: it is a *choice*, and it is the reason the MAP variant's mean tension is not 1: any
#: proper prior shrinks the estimate toward its mean. Recorded as a module constant rather
#: than buried in a default argument so that the sensitivity can be swept.
MAP_PRIOR_TENSION_STD = 0.3

#: Prior standard deviation on each pressure, in units of `mean_tension` per unit length.
#: Large on purpose: pressures are a nuisance block here and should be left essentially
#: unconstrained, so the prior acts on the tension gauge alone.
MAP_PRIOR_PRESSURE_STD = 1.0e3


@dataclass(frozen=True)
class ForceBalanceResult:
    """Everything one force-balance solve produces, including its own diagnostics.

    Attributes:
        interfaces (list[tuple[int, int]]): Interface keys, in the order of `tensions`.
        cells (list[int]): Cell labels, in the order of `pressures`; the exterior medium is
            absent (it is the pressure gauge, fixed at 0).
        tensions (NDArray[np.float64]): Inferred surface tensions, shape `(ni,)`.
        pressures (NDArray[np.float64]): Inferred pressures, shape `(nc,)`, physical sign,
            exterior at 0. A byproduct of the same solve, not a separate inference.
        tension_std (NDArray[np.float64]): Per-interface standard deviation, shape `(ni,)`.
        normalisation (ForceBalanceNormalisation): Which gauge treatment produced this.
        mean_tension_deviation (float): `|mean(tensions) / mean_tension - 1|`. Exactly the
            quantity the three variants are compared on; ~1e-16 for `Kkt` by construction.
        residual_norm (float): `||M beta||`.
        rho (float): `||M beta|| / ||M_gamma gamma||` at the *inferred* tensions — the
            fraction of the surface-tension force this mesh leaves unbalanced under the
            best possible tensions. It is the floor that no choice of gauge can go below,
            and it is `rho_free` of the dataset-audit work.
        condition_number (float): Condition number of the linear system actually solved
            (the KKT matrix, the augmented operator, or the posterior precision).
        sigma_f (float): Force-residual noise scale used for the error bars.
    """

    interfaces: list[tuple[int, int]]
    cells: list[int]
    tensions: NDArray[np.float64]
    pressures: NDArray[np.float64]
    tension_std: NDArray[np.float64]
    normalisation: ForceBalanceNormalisation
    mean_tension_deviation: float
    residual_norm: float
    rho: float
    condition_number: float
    sigma_f: float

    @property
    def dict_tensions(self) -> dict[tuple[int, int], float]:
        """Map interface key -> inferred tension."""
        return dict(zip(self.interfaces, self.tensions.tolist(), strict=True))

    @property
    def dict_pressures(self) -> dict[int, float]:
        """Map cell label -> inferred pressure, with the exterior medium at 0."""
        pressures = {0: 0.0}
        pressures.update(dict(zip(self.cells, self.pressures.tolist(), strict=True)))
        return pressures

    @property
    def dict_tension_std(self) -> dict[tuple[int, int], float]:
        """Map interface key -> standard deviation of the inferred tension."""
        return dict(zip(self.interfaces, self.tension_std.tolist(), strict=True))


def build_force_balance_operator(
    mesh: DcelData,
) -> tuple[NDArray[np.float64], list[tuple[int, int]], list[int]]:
    """Assemble `M = [M_gamma | M_p]` for a `DcelData` mesh.

    `DcelData` normalises every triangle so that its stored winding puts the right-hand-rule
    normal outward from the *first* (lower) label (`dcel._normalize_label_order`), which is
    exactly the convention `force_balance_operator` assumes for the pressure signs — so the
    operator can be assembled from `mesh.f` directly, and gives the same matrix as
    assembling it from the raw `.rec` arrays (pinned by
    `test_operator_agrees_between_dcel_and_raw_rec`).

    Args:
        mesh (DcelData): Mesh to analyse.

    Returns:
        tuple[NDArray[np.float64], list[tuple[int, int]], list[int]]: `(M, interfaces,
            cells)`, with `M` of shape `(3 nv, ni + nc)` and both key lists sorted.
    """
    regions = mesh.f[:, 3:5]
    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    m = fb.force_balance_operator(mesh.v, mesh.f[:, :3], regions, interfaces, [], {}, cells)
    return m, interfaces, cells


def _solve_kkt(
    m: NDArray[np.float64],
    ni: int,
    mean_tension: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Hard mean-tension constraint through the KKT system. Returns `(beta, cov, cond, sigma_f)`."""
    anchor = np.zeros(m.shape[1])
    anchor[:ni] = 1.0 / ni
    beta = fb.solve_constrained(m, anchor, np.array([mean_tension]))
    sigma_f = fb.residual_noise_scale(m, beta, n_constraints=1)
    cov = fb.constrained_covariance(m, anchor, sigma_f)
    n = m.shape[1]
    kkt = np.zeros((n + 1, n + 1))
    kkt[:n, :n] = 2 * m.T @ m
    kkt[:n, n:] = anchor[:, None]
    kkt[n:, :n] = anchor[None, :]
    return beta, cov, float(np.linalg.cond(kkt)), sigma_f


def _solve_soft_row(
    m: NDArray[np.float64],
    ni: int,
    mean_tension: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Today's convention: the normalisation as one more least-squares row.

    The row and its right-hand side reproduce every existing builder in
    `foambryo.tension_inference` exactly — `matrix_m[-1, :ni] = 1`,
    `vector_b[-1] = ni * mean_tension` — so the comparison against `Kkt` isolates the gauge
    treatment and nothing else. Note what this row's *weight* is: 1 per tension against
    physics rows carrying discrete area gradients, i.e. the relative influence of the
    normalisation depends on the mesh's absolute scale. That arbitrariness is the structural
    objection to the soft row, independent of how large the resulting gauge error turns out
    to be.
    """
    row = np.zeros((1, m.shape[1]))
    row[0, :ni] = 1.0
    augmented = np.vstack([m, row])
    rhs = np.zeros(augmented.shape[0])
    rhs[-1] = ni * mean_tension
    beta, _residuals, _rank, _sv = linalg.lstsq(augmented, rhs)
    sigma_f = fb.residual_noise_scale(m, beta, n_constraints=1)
    cov = sigma_f**2 * np.linalg.pinv(augmented.T @ augmented)
    return beta, cov, float(np.linalg.cond(augmented)), sigma_f


def _solve_bayesian_map(
    m: NDArray[np.float64],
    ni: int,
    mean_tension: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Gaussian MAP with posterior covariance (`map_estimate`).

    `sigma_f` is set to the operator's own row scale, `||M||_2 / sqrt(n_rows)`, so that the
    data/prior balance is invariant under a change of the mesh's units and does not depend
    on the number of vertices through the row count alone. This is a documented choice, not
    a derivation: the MAP mean's distance from the prior is set by it, which is why this
    variant's `mean_tension_deviation` is O(10%) rather than machine precision.
    """
    n = m.shape[1]
    prior_mean = np.zeros(n)
    prior_mean[:ni] = mean_tension
    prior_var = np.concatenate(
        [
            np.full(ni, (MAP_PRIOR_TENSION_STD * mean_tension) ** 2),
            np.full(n - ni, (MAP_PRIOR_PRESSURE_STD * mean_tension) ** 2),
        ],
    )
    sigma_f = float(np.linalg.norm(m, ord=2) / np.sqrt(m.shape[0]))
    beta, cov = fb.map_estimate(m, prior_mean, np.diag(prior_var), sigma_f=sigma_f)
    precision = m.T @ m / sigma_f**2 + np.diag(1.0 / prior_var)
    return beta, cov, float(np.linalg.cond(precision)), sigma_f


_SOLVERS = {
    ForceBalanceNormalisation.Kkt: _solve_kkt,
    ForceBalanceNormalisation.SoftRow: _solve_soft_row,
    ForceBalanceNormalisation.BayesianMap: _solve_bayesian_map,
}


def infer_force_balance(
    mesh: DcelData,
    mean_tension: float = 1,
    normalisation: ForceBalanceNormalisation = ForceBalanceNormalisation.Kkt,
) -> ForceBalanceResult:
    """Infer tensions *and* pressures by inverting the discrete force-balance operator.

    One solve produces both blocks of `beta = (gamma, p)`, so unlike the classical path
    (tensions from angles, then pressures from a second variational solve) the two are
    consistent by construction.

    **Warning -- reconstructed meshes.** On `dw3d` output the returned `rho` is about fifty
    times what the same operator reads on the simulator's reference meshes (median 0.578
    against 0.0118 on the 40-case equilibrium cohort, all 40 above the 0.2 alarm threshold
    of `InferenceDiagnostics.identifiability_report`); no tension assignment balances such a
    mesh, so the tensions and pressures returned are the least-bad fit to a geometry that is
    not an equilibrium. Read `rho` before reading the tensions. See the module docstring.

    Args:
        mesh (DcelData): Mesh to analyse.
        mean_tension (float, optional): The gauge — the mean of the returned tensions.
            Defaults to 1, matching `foambryo.infer_tensions`.
        normalisation (ForceBalanceNormalisation, optional): How the gauge is imposed.
            Defaults to `Kkt` (exact by construction).

    Returns:
        ForceBalanceResult: Tensions, pressures, per-interface error bars and diagnostics.
    """
    m, interfaces, cells = build_force_balance_operator(mesh)
    ni = len(interfaces)
    beta, cov, condition_number, sigma_f = _SOLVERS[normalisation](m, ni, mean_tension)

    tensions = beta[:ni]
    # The operator's pressure columns are +dV/dv while the energy carries -p V, so the
    # solved entries are -p. Pinned by test_inferred_pressures_match_the_ground_truth_sign.
    pressures = -beta[ni:]
    tension_std = np.sqrt(np.clip(np.diag(cov)[:ni], 0.0, None))
    load = m[:, :ni] @ tensions
    residual_norm = float(np.linalg.norm(m @ beta))

    return ForceBalanceResult(
        interfaces=interfaces,
        cells=cells,
        tensions=tensions,
        pressures=pressures,
        tension_std=tension_std,
        normalisation=normalisation,
        mean_tension_deviation=abs(float(np.mean(tensions)) / mean_tension - 1.0),
        residual_norm=residual_norm,
        rho=residual_norm / float(np.linalg.norm(load)),
        condition_number=condition_number,
        sigma_f=sigma_f,
    )


def infer_tensions_force_balance(
    mesh: DcelData,
    mean_tension: float = 1,
    normalisation: ForceBalanceNormalisation = ForceBalanceNormalisation.Kkt,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Adapter putting `infer_force_balance` on the `infer_tensions_and_residuals` signature.

    Args:
        mesh (DcelData): Mesh to analyse.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        normalisation (ForceBalanceNormalisation, optional): How the gauge is imposed.
            Defaults to `Kkt`.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions;
            - map interface id (label 1, label 2) -> tension on this interface;
            - the sum of squared residuals `||M beta||^2`, as a length-1 array, which is
              what the other methods' `lstsq` returns for an over-determined system.
    """
    result = infer_force_balance(mesh, mean_tension, normalisation)
    return (
        result.tensions,
        result.dict_tensions,
        np.array([result.residual_norm**2]),
    )
