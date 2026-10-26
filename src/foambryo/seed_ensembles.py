"""Tensions from several reconstructions of one mask: seed ensembles.

delaunay-watershed-3d places its tessellation points with a small random dither, fixed by a seed (42 by
default). Two seeds give two slightly different meshes of the same mask, and the tensions inferred from them
differ. How much depends on cell size: measured with the default reconstruction (`min_distance` 3, junction
relocation on) and `YoungDupre` with the chord reading, a tension varies from seed to seed by

* about 5 % (median coefficient of variation) on the 42 equilibrium cases of the benchmarking dataset, whose
  cells are about ten times larger in volume than real ones;
* about 13-14 % on synthetic tissues voxelised at a real cell size (median cell 69 370 voxels, radius ~25)
  and on embryo frames;
* about 21 % on a 110-cell and on a 1 866-cell tissue meshed on their native anisotropic grid (on meshes from
  dw3d's 0.6.0 code, not on 0.7.1's, which differ slightly; not measured again: one reconstruction of the larger
  tissue needs about 31 GiB).

Averaging the tensions of several seeds removes part of that variation. Against ground truth, an
ensemble of 8 seeds lowers the median relative tension error (gauge-matched, median over interfaces, then
over tissues) from 0.152 to 0.123 on 12 synthetic tissues at real cell size (-23 % per tissue, lower on
all 12), and from 0.077 to 0.064 on the benchmark (-9 %, lower on 35 of 42 cases), where most of the error
is bias rather than seed noise (both measured on meshes from dw3d's 0.6.0 code, never published, part of 0.7.0;
not measured again on 0.7.1's). The contact-angle error falls by about 2 degrees at real cell size.

**What this module offers.** :func:`infer_tensions_over_seeds` reconstructs a mask at a fixed list of seeds
(0 to 7 by default), infers the tensions of each mesh, and combines them:

* `combine="tensions"` (the default): the mean, per interface, of the seeds' tensions. Each seed's solve meets
  the normalisation (a mean tension of 1) exactly, so the means share one scale.
* `combine="angles"` (**experimental**, `YoungDupre` only): the mean, per junction line, of the seeds'
  line-averaged contact angles and line weights, then one `YoungDupre` solve. With `angle_reading="quadric"` it
  was the most accurate ensemble measured on synthetic tissues with known tensions, voxelised at the cell size of
  real embryos (median tension error 45 % below the default ensemble's on 46 tissues), but its margin over
  averaging quadric-read tensions is small and inconsistent, and with the chord reading it is inconsistent on the
  larger tissues. On synthetic masks degraded by a smooth random warp of about one voxel, `combine="angles"` with
  the quadric reading lost its advantage over the default on larger tissues of 90 and 165 cells while keeping it on
  small aggregates. It is not the default; it warns with :class:`SeedEnsembleExperimentalWarning`. Use it as a
  comparison, not as a replacement.

**The seed list is fixed by the caller and never chosen on a result.** Choosing the seed that gives the
"best-looking" tensions would tune on the quantity being measured. The default list is 0 to 7. Seed 42, the
default of every single-seed function, is not in it, so an ensemble never silently includes the default
mesh. `seeds=[42]` reproduces :func:`foambryo.infer_tensions` on
:func:`foambryo.dcel_mesh_from_segmentation_mask`'s mesh, bit for bit.

**The seed spread is a lower bound on the error, not an error bar.** `tension_seed_sd` is the standard
deviation over the seeds. It sees the part of the error that changes with the seed, never the bias every
seed shares. Against ground truth at real cell size, a single seed's ±1 SD interval contained the true
tension for about half of the interfaces (nominal 68 %), and on the benchmark for about a third.

**Interfaces found by only some seeds.** A seed can create an interface that another seed does not (two
cells that barely touch, or do not touch at all). At real cell size about a fifth of the interfaces are
found by only some of 16 seeds. On synthetic tissues with known geometry, most of those found by fewer than
half of the seeds do not exist in the true tissue. They are kept in the result with the flag
`found_by_fewer_than_half_the_seeds`; read their tensions with that in mind.

**Cost.** One reconstruction per seed, so N times the single-seed cost. Measured per seed:

| tissue | time | peak memory |
|---|---|---|
| benchmark case | ~3 s | ~1.5 GiB |
| 165-cell tissue at real cell size | ~40-65 s | ~8 GiB (up to 11) |
| 1 866-cell tissue | ~11 min | ~30-35 GiB |

`max_workers` runs seeds in parallel processes. Each holds one reconstruction in memory, so choose it from
the memory, not only from the number of cores.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from dw3d import get_dithered_mesh_reconstruction_algorithm

from foambryo import linear_solves
from foambryo.crop_faces import remove_crop_interfaces
from foambryo.io import _dcel_mesh_from_mask_with
from foambryo.junction_graph import (
    JunctionGraphPartition,
    solve_gauged_least_squares,
    warn_on_junction_graph_fragmentation,
)
from foambryo.second_order_angle_reading import CHORD_READING, validate_reading
from foambryo.tension_inference import (
    TensionComputationMethod,
    _as_legacy_tuple,
    _build_matrix_tension_symmetrical_yd,
    infer_tensions,
)
from foambryo.wedge_refusal import line_averaged_angles

Interface = tuple[int, int]
Wedge = tuple[int, int, int]

#: The default seed list: fixed, and never chosen on a result.
DEFAULT_ENSEMBLE_SEEDS: tuple[int, ...] = tuple(range(8))
COMBINE_TENSIONS = "tensions"
COMBINE_ANGLES = "angles"
COMBINE_OPTIONS = (COMBINE_TENSIONS, COMBINE_ANGLES)


class SeedEnsembleExperimentalWarning(UserWarning):
    """Emitted by `infer_tensions_over_seeds(..., combine="angles")`, which is experimental.

    Its advantage over the default `combine="tensions"` was small and inconsistent on synthetic
    tissues with known tensions, and was lost on larger tissues degraded by a smooth random warp;
    see `foambryo.seed_ensembles`.
    """


@dataclass(frozen=True)
class SeedReconstruction:
    """What one seed's reconstruction produced.

    Attributes:
        seed: the dither seed.
        n_points: mesh vertices.
        n_triangles: mesh triangles.
        n_crop_faces: triangles on a face of the image box that the tissue reaches (see `foambryo.crop_faces`).
        seconds: wall time of the reconstruction and inference of this seed.
        warnings: the warnings this seed raised, as `"Category: message"`, in order. All but deprecation notices
            are also re-emitted, prefixed with the seed, in seed order, whatever `max_workers` is.
    """

    seed: int
    n_points: int
    n_triangles: int
    n_crop_faces: int
    seconds: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SeedEnsembleResult:
    """Tensions combined over several reconstructions of one mask.

    Every map is keyed by interface `(label 1, label 2)`, `label 1 < label 2`, over the union of the interfaces
    the seeds found, in sorted order.

    Attributes:
        seeds: the seeds, in the order given.
        mode: the tension method.
        angle_reading: the angle reading the angle-reading methods used.
        combine: `"tensions"` or `"angles"`.
        tensions: the ensemble tension, relative: each seed's tensions are normalised to a mean of 1 (in each
            connected piece of its junction graph), and so is the single `"angles"` solve. With `"tensions"`, the
            mean over the seeds that found the interface with a finite value; with `"angles"`, the single solve on
            the averaged angles. `nan` when no seed determined it.
        tension_seed_sd: standard deviation (`ddof=1`) of the seeds' own tensions, over the seeds that found the
            interface; `nan` when fewer than two did. A lower bound on the error, not an error bar (module
            docstring).
        relative_tension_seed_sd: `tension_seed_sd / |tensions|`, the seed spread as a fraction of the ensemble
            tension (0.1 = 10 %); `nan` where either is `nan` or the ensemble tension is 0. A lower bound on the
            relative error, like `tension_seed_sd`.
        n_seeds_found: how many seeds found the interface with a finite tension.
        found_by_fewer_than_half_the_seeds: `n_seeds_found < len(seeds) / 2`. Most such interfaces do not exist
            in the true tissue (module docstring); they are kept, flagged.
        per_seed_tensions: each seed's own tension map (`foambryo.infer_tensions` on that seed's mesh).
        reconstructions: one :class:`SeedReconstruction` per seed, in seed order.
    """

    seeds: tuple[int, ...]
    mode: TensionComputationMethod
    angle_reading: str
    combine: str
    tensions: dict[Interface, float]
    tension_seed_sd: dict[Interface, float]
    relative_tension_seed_sd: dict[Interface, float]
    n_seeds_found: dict[Interface, int]
    found_by_fewer_than_half_the_seeds: dict[Interface, bool]
    per_seed_tensions: dict[int, dict[Interface, float]]
    reconstructions: tuple[SeedReconstruction, ...]

    @property
    def interfaces(self) -> tuple[Interface, ...]:
        """The interfaces of the result, sorted."""
        return tuple(self.tensions)


@dataclass(frozen=True)
class _SeedOutput:
    """One seed's contribution, returned by a worker (picklable)."""

    tensions: dict[Interface, float]
    rad: list[tuple[Wedge, float]]
    length: list[tuple[Wedge, np.float64]]
    interfaces: tuple[Interface, ...]
    components: tuple[tuple[Interface, ...], ...]
    crop_interfaces: tuple[Interface, ...]
    reconstruction: SeedReconstruction
    caught: tuple[tuple[type[Warning], str], ...]


def _key(pair: Iterable[Any]) -> Interface:
    a, b = (int(x) for x in pair)
    return (a, b) if a < b else (b, a)


def _one_seed(
    mask: np.ndarray,
    seed: int,
    min_distance: int,
    mode: TensionComputationMethod,
    angle_reading: str,
    combine: str,
    reconstruction_options: dict[str, Any],
    spacing: tuple[float, float, float] | None,
) -> _SeedOutput:
    """Reconstruct `mask` at `seed`, infer its tensions and, for `"angles"`, keep the line-averaged maps."""
    t0 = time.perf_counter()
    options = dict(reconstruction_options)
    print_info = bool(options.pop("print_info", False))
    physical_image_faces = options.pop("physical_image_faces", ())
    if options:
        message = f"unknown reconstruction option(s): {sorted(options)}; accepted: print_info, physical_image_faces"
        raise TypeError(message)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algorithm = get_dithered_mesh_reconstruction_algorithm(
            min_distance=min_distance, print_info=print_info, seed=seed,
        )
        mesh = _dcel_mesh_from_mask_with(algorithm, mask, physical_image_faces, spacing=spacing)
        tensions = {_key(k): float(v) for k, v in infer_tensions(mesh, 1, mode, angle_reading=angle_reading).items()}
        rad: list[tuple[Wedge, float]] = []
        length: list[tuple[Wedge, np.float64]] = []
        interfaces: tuple[Interface, ...] = ()
        components: tuple[tuple[Interface, ...], ...] = ()
        if combine == COMBINE_ANGLES:
            angles = line_averaged_angles(mesh, reading=angle_reading)
            rad = [(tuple(int(x) for x in k), float(v)) for k, v in angles.dict_rad.items()]
            # In the map's own order and as numpy floats: the builder sums the line weights with the built-in `sum`,
            # which compensates Python floats (since 3.12) but not numpy floats, so both are part of the result.
            length = [(tuple(int(x) for x in k), v) for k, v in angles.dict_length.items()]
            interfaces = tuple(_key(k) for k in angles.partition.interfaces)
            components = tuple(tuple(_key(k) for k in component) for component in angles.partition.components)
    crop = tuple(sorted(_key(k) for k in mesh.crop_interfaces)) if mesh.has_crop_faces else ()
    record = SeedReconstruction(
        seed=int(seed),
        n_points=len(mesh.v),
        n_triangles=len(mesh.f),
        n_crop_faces=int(np.count_nonzero(mesh.crop_faces)) if mesh.has_crop_faces else 0,
        seconds=time.perf_counter() - t0,
        warnings=tuple(f"{w.category.__name__}: {w.message}" for w in caught),
    )
    return _SeedOutput(tensions, rad, length, interfaces, components, crop, record,
                       tuple((w.category, str(w.message)) for w in caught))


def _relative(spread: float, tension: float) -> float:
    """`spread / |tension|`, or `nan` when either is not finite or the tension is 0."""
    if not (np.isfinite(spread) and np.isfinite(tension)) or tension == 0:
        return float("nan")
    return float(spread / abs(tension))


def _union_partition(outputs: list[_SeedOutput]) -> JunctionGraphPartition:
    """The junction-graph partition of the union of the seeds' lines.

    Two interfaces are connected in the union when some seed connects them, so the components are the
    connected components of the union of the seeds' component relations. Same conventions as
    `foambryo.junction_graph.partition_by_junction_graph`: sorted members, largest component first, ties broken by
    the smallest key.
    """
    interfaces = sorted({k for out in outputs for k in out.interfaces})
    parent = {k: k for k in interfaces}

    def find(x: Interface) -> Interface:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    touched: set[Interface] = set()
    for out in outputs:
        for component in out.components:
            touched.update(component)
            for other in component[1:]:
                ra, rb = find(component[0]), find(other)
                if ra != rb:
                    parent[ra] = rb
    groups: dict[Interface, list[Interface]] = {}
    for key in touched:
        groups.setdefault(find(key), []).append(key)
    components = tuple(tuple(sorted(g)) for g in sorted(groups.values(), key=lambda g: (-len(g), min(g))))
    return JunctionGraphPartition(
        interfaces=tuple(interfaces),
        components=components,
        component_of={k: i for i, c in enumerate(components) for k in c},
        isolated=tuple(sorted(set(interfaces) - touched)),
    )


def _solve_on_averaged_angles(outputs: list[_SeedOutput]) -> dict[Interface, float]:
    """`YoungDupre` on the per-wedge mean of the seeds' line-averaged angles and line weights: one solve.

    The maps keep the first seed's key order (then the next seeds' new keys), so that with one seed the system,
    including the sum of the line weights, is the production one bit for bit.
    """
    rad: dict[Wedge, list[float]] = {}
    length: dict[Wedge, list[np.float64]] = {}
    for out in outputs:
        for k, v in out.rad:
            rad.setdefault(k, []).append(v)
        for k, v in out.length:
            length.setdefault(k, []).append(v)
    partition = _union_partition(outputs)
    keys_interface = list(partition.interfaces)
    if not rad:
        return dict.fromkeys(keys_interface, float("nan"))
    dict_rad = {k: float(np.mean(v)) for k, v in rad.items()}
    dict_length = {k: np.mean(v) for k, v in length.items()}  # numpy floats, as the single-mesh builder gets them
    as_sparse = linear_solves.use_sparse_solver(len(keys_interface))
    if not as_sparse:
        n_lines = len({tuple(sorted(k)) for k in dict_rad})
        linear_solves.warn_if_forced_dense_exceeds_memory(
            3 * n_lines + 1, len(keys_interface), "YoungDupre tension system", stacklevel=4,
        )
    matrix_m, vector_b = _build_matrix_tension_symmetrical_yd(
        dict_rad, dict.fromkeys(keys_interface, 1.0), dict_length, 1, as_sparse=as_sparse,
    )
    solution = solve_gauged_least_squares(
        matrix_m, vector_b, keys_interface, 1, partition, system="YoungDupre tension system",
    )
    crop = tuple(sorted({k for out in outputs for k in out.crop_interfaces}))
    solution = remove_crop_interfaces(solution, crop)
    warn_on_junction_graph_fragmentation(
        solution.fragmented, solution.n_components, solution.undetermined, stacklevel=4,
    )
    return {_key(k): float(v) for k, v in _as_legacy_tuple(solution)[1].items()}


def _checked_arguments(
    seeds: Iterable[int],
    mode: TensionComputationMethod,
    angle_reading: str,
    combine: str,
    max_workers: int,
) -> tuple[int, ...]:
    """Check `infer_tensions_over_seeds`' arguments, warn for the experimental option, and return the seeds."""
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        message = "seeds is empty"
        raise ValueError(message)
    if len(set(seeds)) != len(seeds):
        message = f"seeds must be distinct, got {list(seeds)}: a repeated seed would count one mesh twice"
        raise ValueError(message)
    if combine not in COMBINE_OPTIONS:
        message = f"combine must be one of {COMBINE_OPTIONS}, got {combine!r}"
        raise ValueError(message)
    if combine == COMBINE_ANGLES and mode != TensionComputationMethod.YoungDupre:
        message = f"combine='angles' averages the angles YoungDupre reads; it is not defined for {mode.name}"
        raise ValueError(message)
    validate_reading(angle_reading)
    if max_workers < 1:
        message = f"max_workers must be at least 1, got {max_workers}"
        raise ValueError(message)
    if combine == COMBINE_ANGLES:
        warnings.warn(
            "combine='angles' is experimental: with angle_reading='quadric' it was the most accurate ensemble on "
            "synthetic tissues with known tensions, but not by a margin consistent enough to be the default; use it "
            "as a comparison, not as a replacement (see foambryo.seed_ensembles)",
            SeedEnsembleExperimentalWarning,
            stacklevel=3,
        )
    return seeds


def _report_progress(progress: Callable[[int, int, int], None] | None, outputs: list[_SeedOutput], total: int) -> None:
    """Call `progress(seeds done, seeds in all, the seed just done)`, if a callback was given."""
    if progress is not None:
        progress(len(outputs), total, outputs[-1].reconstruction.seed)


def infer_tensions_over_seeds(
    mask: np.ndarray,
    seeds: Iterable[int] = DEFAULT_ENSEMBLE_SEEDS,
    *,
    min_distance: int = 3,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    angle_reading: str = CHORD_READING,
    combine: str = COMBINE_TENSIONS,
    max_workers: int = 1,
    spacing: tuple[float, float, float] | None = None,
    progress: Callable[[int, int, int], None] | None = None,
    **reconstruction_options: object,
) -> SeedEnsembleResult:
    """Reconstruct a mask at several dither seeds and combine the tensions inferred from each mesh.

    See `foambryo.seed_ensembles` for what the seeds change, the measured gain, and why the seed spread is a
    lower bound on the error rather than an error bar: `tension_seed_sd` sees only the part of the error that
    changes with the seed, never the bias every seed shares.

    Each seed's mesh is reconstructed as `dcel_mesh_from_segmentation_mask` reconstructs it, at that dither
    seed (crop faces included), and its tensions are `infer_tensions(mesh, 1, mode, angle_reading=angle_reading)`:
    normalised to a mean of 1, chord-read unless `angle_reading` says otherwise, with the default solver options.

    Args:
        mask (np.ndarray): instance segmentation image, as for `foambryo.dcel_mesh_from_segmentation_mask`:
            3-D integer labels, axes (z, y, x), label 0 the medium.
        seeds (Iterable[int], optional): the dither seeds, distinct; 0 to 7 by default. Fix the list in advance:
            never choose seeds by their result.
        min_distance (int, optional): minimum distance between tessellation points, in voxels. Defaults to 3,
            as `dcel_mesh_from_segmentation_mask`. Keyword-only.
        mode (TensionComputationMethod, optional): the tension method, `YoungDupre` by default. Any method works
            with `combine="tensions"`; `combine="angles"` needs `YoungDupre`. Keyword-only.
        angle_reading (str, optional): the angle reading of the angle-reading methods, `"chord"` by default.
            Keyword-only.
        combine (str, optional): `"tensions"` (the default): average the seeds' tensions. `"angles"`
            (experimental, `YoungDupre` only, emits `SeedEnsembleExperimentalWarning`): average the contact angles
            over the seeds, then infer once. With `angle_reading="quadric"` it was the most accurate option on
            synthetic tissues (median error -45 % against the default on 46 tissues), but its advantage over
            averaging quadric-read tensions is small and inconsistent, so it is not the default. On synthetic
            masks degraded by a smooth random warp of about one voxel, `combine="angles"` with the quadric reading
            lost its advantage over the default on larger tissues of 90 and 165 cells while keeping it on small
            aggregates. Keyword-only.
        max_workers (int, optional): seeds reconstructed at once, in separate processes; 1 (the default) runs
            them one after another in this process. The result does not depend on it. Each worker holds one
            reconstruction: about 8 GiB for a 165-cell tissue at real cell size, about 30-35 GiB for a 1 866-cell
            tissue. Keyword-only.
        spacing (tuple[float, float, float] | None, optional): the voxel size along the mask's axes 0, 1, 2, as for
            `dcel_mesh_from_segmentation_mask`: given, every seed's reconstruction resamples a mask with one
            longer axis to cubic voxels of the shortest spacing and returns the mesh in the spacing's unit
            (`min_distance` then counts cubic voxels). Defaults to `None`, the voxel grid. Keyword-only.
        progress (Callable[[int, int, int], None] | None, optional): called once per seed, as its tensions come
            back and in seed order, with `(seeds done, seeds in all, that seed)`, for a front end's progress
            display; it changes no number. Defaults to `None`. Keyword-only.
        **reconstruction_options: `print_info` and `physical_image_faces`, passed to the reconstruction as
            `dcel_mesh_from_segmentation_mask` passes them.

    Returns:
        SeedEnsembleResult: the ensemble tensions, the seed spread (absolute and relative to the tension; a lower
            bound on the error, not an error bar), how many seeds found each interface, the flag for those found
            by fewer than half, and each seed's own tensions and reconstruction record. Every map is keyed by
            interface `(label_1, label_2)`, `label_1 < label_2`, over the union of the seeds' interfaces.

    Raises:
        ValueError: on an empty or repeated seed list, an unknown `combine`, `combine="angles"` with a method
            other than `YoungDupre`, an unknown `angle_reading`, or `max_workers < 1`; and, from a seed's
            reconstruction or inference, as `dcel_mesh_from_segmentation_mask` and `infer_tensions` (for
            example a non-chord `angle_reading` with a method that reads no angle).
        TypeError: on an unknown reconstruction option.

    Warns:
        SeedEnsembleExperimentalWarning: with `combine="angles"`.
        UserWarning: every warning a seed's reconstruction or inference emits (for example
            `JunctionGraphGaugeWarning`), re-emitted in seed order with its own category and the prefix
            `"seed N: "`, whatever `max_workers` is; deprecation notices stay in the seed's record only. With
            `combine="angles"`, `JunctionGraphGaugeWarning` for the single solve too.
    """
    seeds = _checked_arguments(seeds, mode, angle_reading, combine, max_workers)

    spacing = None if spacing is None else tuple(float(v) for v in spacing)
    arguments = [(mask, s, min_distance, mode, angle_reading, combine, reconstruction_options, spacing) for s in seeds]
    outputs: list[_SeedOutput] = []
    if max_workers == 1:
        for a in arguments:
            outputs.append(_one_seed(*a))
            _report_progress(progress, outputs, len(arguments))
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(seeds))) as pool:
            for out in pool.map(_one_seed, *zip(*arguments, strict=True)):
                outputs.append(out)
                _report_progress(progress, outputs, len(arguments))
    for out in outputs:
        for category, text in out.caught:
            # Deprecation notices concern the libraries' own internals, which a single-seed call would not show the
            # caller either; they stay in the seed's record.
            if not issubclass(category, (DeprecationWarning, PendingDeprecationWarning)):
                warnings.warn(f"seed {out.reconstruction.seed}: {text}", category, stacklevel=2)

    per_seed = {out.reconstruction.seed: out.tensions for out in outputs}
    interfaces = sorted({k for out in outputs for k in out.tensions})
    found = {k: [t[k] for t in per_seed.values() if k in t and np.isfinite(t[k])] for k in interfaces}
    n_found = {k: len(v) for k, v in found.items()}
    sd = {k: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan") for k, v in found.items()}
    if combine == COMBINE_TENSIONS:
        tensions = {k: float(np.mean(v)) if v else float("nan") for k, v in found.items()}
    else:
        solved = _solve_on_averaged_angles(outputs)
        tensions = {k: solved.get(k, float("nan")) for k in interfaces}
    return SeedEnsembleResult(
        seeds=seeds,
        mode=mode,
        angle_reading=angle_reading,
        combine=combine,
        tensions=tensions,
        tension_seed_sd=sd,
        relative_tension_seed_sd={k: _relative(sd[k], tensions[k]) for k in interfaces},
        n_seeds_found=n_found,
        found_by_fewer_than_half_the_seeds={k: n_found[k] < len(seeds) / 2 for k in interfaces},
        per_seed_tensions=per_seed,
        reconstructions=tuple(out.reconstruction for out in outputs),
    )
