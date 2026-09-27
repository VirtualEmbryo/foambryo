"""The attribution benchmark's mesh-population split — one place, so no aggregate can mix them.

The attribution benchmark compares meshes from three
different origins that are **not interchangeable**:

- **P1 — simulator ground truth** (`benchmarking-dataset/*_mesh.rec`). Minimises the same
  operator `M` the `ForceBalance` methods invert, but by gradient descent and under a volume
  *constraint* rather than a volume *energy term* (the generator projects the force onto the
  volume-constraint manifold, with pressures as the Lagrange multipliers, so `M beta = 0` is
  never the condition it drives to zero) — so it is **under-converged**:
  `rho_free` 0.0007-0.036 on the 42 equilibrium cases (the dataset-audit work).
- **P2 — `dw3d` reconstruction** of the segmentation mask derived from the same shape.
  Nothing in `dw3d` targets `M`; its point set and connectivity are voxel-quantised and
  have **no vertex correspondence with P1**.
- **P3 — analytic phantoms** (the doublet, the tetrahedral vertex; `foambryo_benchmarks/phantoms.py`).
  Exact continuum equilibrium by construction, the only population with an uncontaminated
  ground truth.

The generative chain is `P1 -> artificial image -> segmentation -> mask -> dw3d -> P2`. P1
and P2 are two discretisations of the *same* underlying shape, not two versions of the same
mesh, which is why every attribution metric already goes through label-tuple keys
(`foambryo.infer_tensions` returns `dict[(label_a, label_b), tension]`, never a per-vertex
array) rather than any vertex correspondence.

**Why this module exists rather than a comment in each driver.** The failure mode is the
same one `foambryo_benchmarks/case_groups.py` guards against for the equilibrium split: an aggregate
across P1 and P2 records reads as one number and is in fact two different experiments (an
inference-only measurement and an inference+meshing measurement) silently averaged
together. `require_single_population` makes that structural.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Simulator ground-truth mesh (`benchmarking-dataset/*_mesh.rec`). Minimises `M`, under-converged.
P1_SIMULATOR_GROUND_TRUTH = "P1_simulator_ground_truth"

#: `dw3d` reconstruction of the segmentation mask. Nothing in `dw3d` targets `M`.
P2_DW3D_RECONSTRUCTION = "P2_dw3d_reconstruction"

#: Analytic phantom (doublet, tetrahedral vertex). Exact continuum equilibrium.
P3_ANALYTIC_PHANTOM = "P3_analytic_phantom"

#: Population order used in every report.
POPULATION_ORDER: tuple[str, str, str] = (
    P1_SIMULATOR_GROUND_TRUTH,
    P2_DW3D_RECONSTRUCTION,
    P3_ANALYTIC_PHANTOM,
)

_KNOWN_POPULATIONS = frozenset(POPULATION_ORDER)


def require_single_population(records: Sequence[dict], population_key: str = "population") -> str:
    """Assert that `records` are all from one mesh population, and return its name.

    Call this at the top of any function that reduces several records to one number
    (a mean, a median, a correlation). It is what stops an aggregate from silently mixing
    P1 (contaminated by the simulator's under-convergence) with P2 (contaminated by
    meshing error) and reporting the mixture as either population's accuracy.

    Args:
        records (Sequence[dict]): Per-record dicts each carrying a population tag under
            `population_key`.
        population_key (str, optional): Key holding the population tag. Defaults to
            `"population"`.

    Returns:
        str: The single population all records belong to (one of `POPULATION_ORDER`).

    Raises:
        ValueError: If `records` is empty, spans more than one population, or contains an
            unrecognised population tag (a typo here would otherwise silently create a
            fourth, unaudited "population").
    """
    if not records:
        message = "cannot aggregate an empty record list: the population is undefined"
        raise ValueError(message)

    present = {record[population_key] for record in records}
    unknown = present - _KNOWN_POPULATIONS
    if unknown:
        message = (
            f"unrecognised population tag(s) {sorted(unknown)} - expected one of "
            f"{sorted(_KNOWN_POPULATIONS)}. A typo here would silently create an "
            "unaudited fourth population."
        )
        raise ValueError(message)
    if len(present) > 1:
        message = (
            "refusing to aggregate across mesh populations: these records come from "
            f"different, non-interchangeable origins -> {sorted(present)}. P1 and P2 have "
            "no vertex correspondence and different ground-truth contamination; use "
            "per-population lists (see this module's docstring)."
        )
        raise ValueError(message)
    return present.pop()


def tag(record: dict, population: str, population_key: str = "population") -> dict:
    """Return a copy of `record` with its population tag set, validating the tag itself.

    Args:
        record (dict): Record to tag.
        population (str): One of `POPULATION_ORDER`.
        population_key (str, optional): Key to store the tag under. Defaults to
            `"population"`.

    Returns:
        dict: `record` with `population_key` set to `population`.

    Raises:
        ValueError: If `population` is not one of `POPULATION_ORDER`.
    """
    if population not in _KNOWN_POPULATIONS:
        message = f"unrecognised population {population!r} - expected one of {sorted(_KNOWN_POPULATIONS)}"
        raise ValueError(message)
    return {**record, population_key: population}
