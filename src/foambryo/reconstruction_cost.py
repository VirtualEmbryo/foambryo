"""The estimated wall time and peak memory of a reconstruction, before it runs.

A front end that reconstructs a mask (the napari plugin, `foambryo view MASK`) tells its user, before the run, how
long it should take and how much memory it should need; on a large tissue it can be minutes and tens of GiB. This
module is the one place the estimate is made, so every front end gives the same one.

The estimate is a power law of **the voxels handed to delaunay-watershed-3d**: the mask's voxels times the
anisotropy, because an anisotropic mask is resampled to cubic voxels of its finest spacing before meshing
(`voxels_handed_to_dw3d`). Its constants (`RECONSTRUCTION_COST_FIT`) were fitted by least squares in log-log to
the median wall time and peak memory of `delaunay-watershed-3d`'s default reconstruction (`min_distance` 3,
junction relocation on) at seven anchors (`RECONSTRUCTION_COST_ANCHORS`): two medians over the benchmark's
simulated tissues of 2 to 10 cells, and five reconstructions of single tissues of 90 to 1 866 cells, on one machine
(Apple M3 Max, 16 cores, 128 GiB; timings mostly on a shared, loaded machine, so provisional; peak memory
reliable).

Each estimate is a central value and a band: the band factor is the largest in-sample ratio between an anchor and
the fit, **not** a validated prediction interval. Time and memory also depend on the number of cells and on the
tissue's shape, which the estimate does not see.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "RECONSTRUCTION_COST_ANCHORS",
    "RECONSTRUCTION_COST_FIT",
    "ReconstructionCostEstimate",
    "estimate_reconstruction_cost",
    "voxels_handed_to_dw3d",
]

#: The measured reconstructions the fit was made on: `(what, voxels handed to dw3d, median seconds, peak GiB)`.
RECONSTRUCTION_COST_ANCHORS: tuple[tuple[str, float, float, float], ...] = (
    ("simulated tissues at real cell size, median", 1.5e6, 0.8, 0.6),
    ("simulated tissues at the benchmark's default size, median", 7.72e6, 2.3, 1.2),
    ("a 90-cell simulated tissue", 4.2e7, 21.0, 4.7),
    ("a 110-cell tissue, native grid", 7.99e7, 35.0, 5.1),
    ("a 165-cell simulated tissue", 1.17e8, 76.0, 11.6),
    ("a 1 866-cell tissue, native grid", 4.98e8, 610.0, 29.1),
    ("a 1 866-cell tissue, cubic voxels", 8.3e8, 807.0, 45.6),
)

#: The fit, per quantity: `central = prefactor * voxels ** exponent`, band `[central / band_factor,
#: central * band_factor]`.
RECONSTRUCTION_COST_FIT: dict[str, dict[str, float]] = {
    "seconds": {"prefactor": 3.6358636769611024e-08, "exponent": 1.1567209231530782,
                "band_factor": 1.5793137370008161},
    "peak_gib": {"prefactor": 2.1422186523855385e-05, "exponent": 0.7021459857892676,
                 "band_factor": 1.4860558062942626},
}


def voxels_handed_to_dw3d(shape: tuple[int, ...], spacing: tuple[float, float, float]) -> float:
    """The voxels a reconstruction meets: the mask's (its last three axes), times the anisotropy.

    An anisotropic mask is resampled to cubic voxels of its finest spacing before meshing, so its voxel count grows
    by `max(spacing) / min(spacing)`.

    Args:
        shape: the mask's shape `(nz, ny, nx)`; only the last three axes are counted.
        spacing: the voxel size `(dz, dy, dx)`, positive, in any one unit (it is not checked).

    Returns:
        float: the number of voxels handed to delaunay-watershed-3d.
    """
    z, y, x = (float(n) for n in shape[-3:])
    return z * y * x * max(spacing) / min(spacing)


@dataclass(frozen=True)
class ReconstructionCostEstimate:
    """A reconstruction's estimated wall time and peak memory, each `(low, central, high)`.

    Attributes:
        voxels: the voxels handed to delaunay-watershed-3d.
        seconds: wall time, in seconds.
        peak_gib: peak memory, in GiB (2**30 bytes).
    """

    voxels: float
    seconds: tuple[float, float, float]
    peak_gib: tuple[float, float, float]

    def text(self, dash: str = "-") -> str:
        """In words, for example `about 13 min (8 min-20 min) and 39 GiB (26 GiB-58 GiB)`.

        A quantity whose band reads the same at both ends and in the middle is said once, without a band: a tiny
        tissue reads `under 1 s and under 1 MiB`, not `about under 1 s (under 1 s-under 1 s)`. A central value under
        1 s whose high end is not reads `up to 2 s`.

        Args:
            dash: the range's dash: a hyphen for an ASCII-only font such as the Polyscope panel's, an en dash
                (U+2013) elsewhere. Defaults to a hyphen.

        Returns:
            str: the central time and memory, each with its band; a time in s, min or h, a memory in MiB or GiB.
        """
        return f"{_band(self.seconds, _duration, dash, 'about ')} and {_band(self.peak_gib, _memory, dash, '')}"


def estimate_reconstruction_cost(voxels: float) -> ReconstructionCostEstimate:
    """The estimated wall time and peak memory of reconstructing this many voxels (`voxels_handed_to_dw3d`).

    Args:
        voxels: the voxels handed to delaunay-watershed-3d (after resampling to cubic voxels).

    Returns:
        ReconstructionCostEstimate: `seconds` and `peak_gib`, each `(central / band_factor, central,
        central * band_factor)` with `central = prefactor * voxels ** exponent` (`RECONSTRUCTION_COST_FIT`).
    """
    voxels = float(voxels)
    out = {}
    for key in ("seconds", "peak_gib"):
        fit = RECONSTRUCTION_COST_FIT[key]
        central = fit["prefactor"] * voxels ** fit["exponent"]
        out[key] = (central / fit["band_factor"], central, central * fit["band_factor"])
    return ReconstructionCostEstimate(voxels, out["seconds"], out["peak_gib"])


def _duration(seconds: float) -> str:
    if seconds < 1:
        return "under 1 s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _memory(gib: float) -> str:
    if gib * 1024 < 1:
        return "under 1 MiB"
    return f"{gib * 1024:.0f} MiB" if gib < 1 else f"{gib:.0f} GiB"


def _band(values: tuple[float, float, float], word: Callable[[float], str], dash: str, about: str) -> str:
    """`(low, central, high)` in words: `{about}C (L-H)`, or one phrase when the band adds nothing.

    `under 1 s` is a bound, not a value, so it is never preceded by `about`: when the central value reads `under ...`
    and the high end does not, the phrase is the high end as a bound (`up to 2 s`); when all three read the same, the
    one phrase.
    """
    low, central, high = (word(v) for v in values)
    if low == central == high:
        return central
    if central.startswith("under"):
        return f"up to {high}"
    return f"{about}{central} ({low}{dash}{high})"


def _about_duration(seconds: float) -> str:
    """A central time alone in words: `about 3 min`, or `under 1 s` (never `about under 1 s`)."""
    text = _duration(seconds)
    return text if text.startswith("under") else f"about {text}"
