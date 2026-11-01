r"""The trijunction angle readings: the chord, the second-order (quadric) reading, and their oriented forms.

The `angle_reading` (or `reading`) selector of every angle-reading function takes one of
`ANGLE_READINGS`: `"chord"` (the default), `"quadric"`, `"oriented_chord"` and `"oriented_quadric"`.
Angles are in radians in every returned map.

Why this module exists
----------------------
`foambryo.geometry._wedges_at_half_edge` reads a trijunction wedge angle as
`arccos(n_i . n_j)` between two incident triangles' **face normals**. Both normals are
perpendicular to the shared junction edge, so both lie in the plane normal to it, and the
angle between them equals the angle between the two triangles' in-plane **traces** — the
straight lines from the junction out towards each incident triangle's apex. Production
therefore reads a **chord** of each interface where the geometry has a **tangent**.

A flat P1 facet has no curvature to offer, so it cannot report a tangent; a circular arc's
chord is rotated from its tangent by exactly `kappa * L / 2`, with `L` the in-plane distance
from the junction-edge midpoint to the incident apex. This chord-for-tangent substitution was
measured to account for **68.4 %** of the length-weighted flattening slope of the reconstructed
meshes.

This module supplies the second reading: fit a **generic quadric height field** to the
interface's own mesh vertices in a 3-D window around each junction-edge midpoint, read that
surface's normal-plane section curvature `kappa` analytically, and rotate the face normal about
the junction edge by `-kappa * L / 2`. It is a local patch fit, keyed to `foambryo`'s own trijunction
identification.

The oriented wedge reading
--------------------------
`arccos` returns `[0, pi]`, so a wedge whose interior angle exceeds 180 degrees — a **reflex**
wedge — is reported as `360 - theta`. The three wedges of one junction edge then sum to
`360 - 2 (theta - 180)` instead of 360, and every average along the line inherits the deficit.
This is not rare on a coarse reconstruction: measured on the benchmark meshes, 2 % to 25 % of
trijunctional edges carry one, at a rate that differs by a factor of six between reconstruction
strategies, so a comparison of two meshes through the unoriented reading compares two different
amounts of clipping as if they were geometry.

`ORIENTED_CHORD_READING` and `ORIENTED_QUADRIC_READING` return the **oriented** wedge angle in
`[0, 360)`: the arc from film `e|f` to film `f|g` measured *through* region `f`. The magnitude
is still `arccos` of the same dot product, evaluated by the same expression; the oriented sine
`dot(t, cross(n_first, n_second))` is used only to choose between `theta` and `360 - theta`. So
the oriented readings are **bitwise identical** to their unoriented counterparts on every wedge
that is not reflex, and the three wedges of an edge sum to 360 degrees.

The axis `t` runs from the lower to the higher vertex key, as everywhere else in this module.
Its sign is then resolved from the geometry, not asserted: three positive angles summing to 360
contain at most one greater than 180, so under the reversed axis the three oriented wedges sum
to 720 rather than 360, and the reader takes whichever assignment sums to 360. A wedge whose
oriented sine is exactly zero — an interior angle of exactly 0 or exactly 180 degrees — reads as
non-reflex under either axis, which reproduces the unoriented reader there.

Which inference methods this changes, and one that it breaks
------------------------------------------------------------
`MEASURED` on the benchmark meshes. A method's exposure follows from the trigonometric function
its rows use, because the reflection `theta -> 360 - theta` leaves `cos` unchanged and flips the
sign of `sin`:

* `YoungDupreLocal` builds every row from `cos` of a *per-edge* wedge, so it is **exactly**
  invariant -- its gauge-matched relative error moved by at most `1.6e-14` over 1035
  reconstructed benchmark meshes. It could never see the defect and cannot benefit from it
  either; a reflex wedge and its reflection are the same row to it.
* `YoungDupre` averages the wedge *along the line* and then takes `cos` of the mean, so the
  reflection does move it: up to `0.24` in absolute relative error, worse on 35 of the 46
  reconstruction-variant cohorts measured and better on the rest.
* `ProjectionYoungDupre`, `Lami` and `InvLami` use `sin`, which changes sign. Their error rises
  by 5-42 %. That is not a defect of the reading: three tensions whose wedge is reflex admit no
  planar equilibrium at all, and the negative sine is that fact surfacing where the unoriented
  reading manufactured a plausible positive number instead.
* **`LogLami` fails outright** on a mesh with a reflex wedge: its rows take `log(sin(theta))`,
  which is `nan` for a negative sine, and the solve then raises. Measured on 115 of 360
  benchmark meshes. Pair `LogLami` with an unoriented reading, or repair the reflex edges
  upstream.

The quadric window needs no reflex guard. It is defined per incident *face*, transverse to the
junction along that face's own chord, and never refers to a wedge angle at all; a reflex wedge
is a relation between two faces, so it cannot enter one face's window. What the quadric reading
does do is rotate normals about the junction edge, which moves some wedges *across* 180 degrees
in either direction, so its reflex count differs from the chord reading's on the same mesh. That
is a property of the corrected geometry, not a defect of the window.

The two junction-edge endpoints
------------------------------
The fit keeps a sample when it is inside the window and on the interface's own side of the
junction, `eta > 0`. Two of the candidate samples are the junction edge's **own two endpoints**,
and in exact arithmetic their `eta` is exactly zero — they sit on the `eta = 0` axis of the frame
by construction, being the two vertices whose midpoint is the frame's origin. A strict `eta > 0.0`
therefore resolves them on the sign of a rounding error, and **the sign of a rounding error depends
on the units the mesh is expressed in**.

`MEASURED`, on the benchmark's reference meshes: rescaling one mesh by `117x` — the ratio between
the normalised frame those meshes are stored in and the voxel frame their reconstructions live in —
changed the kept-sample count on `652` of `1 065` incident faces, changed the admission outcome on
`19`, and moved line-mean wedge angles by up to `0.14` degrees. The chord reading moves by
`2.5e-13` degrees under the same rescaling. That is about eleven orders of magnitude of difference
between two readings of one mesh, and it is why the same reference meshes have been published at
`2.032` and at `2.046` degrees of contact-angle error under one name: the frame a mesh is in was
part of that reading's construction and was never stated with it.

An angle between two surfaces cannot depend on the units the surfaces are measured in, so this is a
correctness defect independently of any angle metric. `JUNCTION_ENDPOINTS` decides those two
samples **by vertex identity** instead. They are **included**: they are exact, noise-free samples of
the interface at the very point the frame is anchored, they constrain the along-edge coefficients
that nothing else near `eta = 0` constrains, and including them is a statement rather than an
accident. `"exclude"` and the historical `"rounding"` remain reachable through
`quadric_angle_rotations` so that the repair can be *measured*; neither is reachable through any
`reading` selector or any inference entry point.

`MEASURED` after the repair, on the same reference mesh: rescaling by `10`, `117` and `1000` moves
the line-mean wedge angles by at most `2.3e-13` degrees, against the chord reading's own `2.5e-13`.
The two readings are now equally invariant, to rounding. On a smooth surface whose contact angles
are a closed form the repair moves the reading itself by at most `0.016` degrees and by about
`0.001` degrees at typical spacings — so it is a repair whose value is the invariance, not an
accuracy gain, and saying so is part of the repair.

`MEASURED` on reconstructed meshes, the repair does change readings, because the two endpoint samples
bring more faces over the fit's minimum sample count: on 40 benchmark reconstructions (dw3d 0.4.2,
dithered, seed 42, `min_distance` 3, without junction relocation), the median over cases of the
line-mean angle error under the oriented second-order reading moved from `5.270` to `5.403` degrees,
and single cases by up to `0.917` degrees. The `0.14` degrees above is the unit dependence removed,
not the change of numbers published before the repair.

The constant term, and the false reason it used to be omitted
-------------------------------------------------------------
The height field is fitted **without** a constant term. The reason recorded here used to be that
"the junction-edge midpoint is the midpoint of two mesh vertices lying on this interface, so the
interface passes through the origin by construction". **That reason is false whenever the junction
line is curved**, which on any real mesh it is: the midpoint of two vertices lies on the junction
edge's *chord*, and the line leaves that chord by about `kappa_line * L^2 / 8`, as does the
interface bounded by it. The frame's origin is a point the surface does not contain.

The real reason, `MEASURED` rather than argued, on a smooth surface whose contact angles are a
closed form and whose triangulation was varied over fourteen edge lengths spanning a factor of
seventeen:

* fitting the constant term makes the reading **better at every fine triangulation** — up to `43 %`
  lower line-mean error at the finest, and better at ten of the fourteen edge lengths tried;
* and **worse at the four coarsest**, by up to `7.4` degrees, because at those spacings the extra
  free parameter is estimated from a window holding barely twice as many samples as the fit has
  parameters, and the variance it adds exceeds the bias it removes;
* the coarse spacings the software is actually run at may or may not be on the worsening side: that
  depends on how curved real interfaces are relative to their triangulation, which is not a number
  this project has measured.

So the term stays off — not because the surface passes through the origin, which it does not, but
because switching it on is an improvement at fine triangulations bought with a degradation at coarse
ones, and the operating point is not known well enough to take that trade. `fit_constant_term=True`
turns it on for measurement. Turning it on also raises the minimum admissible sample count from `10`
to `12`, which lowers coverage on meshes whose windows are thinly populated. It does not change
`_section_curvature`, because translating a plane curve does not change its curvature.

**What would settle it**: the same fourteen-rung measurement with the analytic films' curvature
matched to real interfaces', or the equivalent measurement on reconstructions with a ground-truth
angle. Either would say which side of the crossover the shipped spacings are on.

Units and conventions
---------------------
* Coordinates are in the mesh's own units: **voxel indices** on `dw3d` output reconstructed without a
  `spacing`, the spacing's unit with one. `kappa` is per mesh length unit and `kappa * L / 2` is
  dimensionless (an angle in radians); the reading does not depend on the unit (see "The two
  junction-edge endpoints").
* `kappa` is the signed in-plane curvature of the interface's normal-plane section at the
  junction, positive when the section turns **counter-clockwise about the junction-edge
  direction** (lower vertex key -> higher vertex key) as it leaves the junction.
* The junction-edge axis runs from the **lower** to the **higher** vertex key. Using a
  half-edge's own direction instead would flip the sign of the correction on half the
  half-edges.

Which correction form is applied, and which is not
--------------------------------------------------
Only the closed form `-kappa * L / 2` is applied. It was measured that the alternative —
rotating the mesh's chord onto the *fitted patch's own tangent* — injects a zero-mean scatter of
about 3.3 degrees at the median and 14.4 degrees at p90 on this patch fit,
because a patch fit's absolute orientation carries the discrepancy between the fitted surface
and the mesh's local orientation, which is not a chord-versus-tangent error and must not be
rotated away. That form is not implemented here. The cost of the form that *is* implemented is
that the circular-arc model becomes load-bearing: `-kappa * L / 2` is exact for an arc and
first-order in `kappa * L` otherwise.

What enters the fit, exhaustively
---------------------------------
The fit is **generic**: a quadric height field is the second-order Taylor statement about
whatever surface is there, and injects only smoothness. **No constant-mean-curvature
constraint, no Laplace or Neumann relaxation, no tension, no pressure, no area, no volume, no
ground-truth angle, no reference mesh, no `.rec` file, no segmentation mask.** The complete
list of inputs is:

1. `mesh.v`, the mesh's vertex coordinates;
2. `mesh.f`, its triangles and their two material labels;

and nothing else. `quadric_angle_rotations` takes a `DcelData` and, besides the window's
measurement parameters, no other argument, so "no ground truth in the estimator" is structural
here: there is no parameter through which a reference mesh, a ground-truth angle, a tension or a
pressure could reach it.

Identity at zero curvature
--------------------------
A fitted `kappa` of exactly `0.0` produces a rotation of exactly `-0.0`, and
`geometry._wedges_at_half_edge` skips the Rodrigues call on a falsy angle, so the chord angle is
returned **bitwise**. Pinned by `tests/test_second_order_angle_reading.py`, which also checks the
stronger statement that an explicitly all-zero rotation map, and an empty one, are the identity to
the bit.

The honest qualification, MEASURED rather than asserted: a *geometrically* flat interface does
not fit to an exactly-zero `c_ee`, because the least-squares design matrix is not exactly
rank-consistent in binary floating point. On a planar three-film phantom the residual rotation
peaks at `4.1e-17` rad (`2.3e-15` degrees) and moves a reported wedge angle by at most `2.8e-14`
degrees — sixteen orders of magnitude below the ~5 degree per-line errors this reading exists to
reduce, and below `float64`'s own resolution at 120 degrees. So the bitwise statement is about
`kappa == 0`, and the practical statement about a flat surface is "identity to `1e-14` degrees".

Hyperparameters are fixed in production, and measurable outside it
------------------------------------------------------------------
`QUADRIC_WINDOW_FACTOR = 4.0` and `QUADRIC_AXIAL_FACTOR = 1.0` are fixed a priori (the axial
one is the isotropic choice, which assumes least). Re-selecting them by held-out fit residual,
with at least 85 % of incident faces covered, returns exactly these values, so they carry no
selection freedom with respect to the metrics they are judged on.

They remain module constants, and **no production path can change them**: no `reading` selector,
no inference entry point, no public convenience function and not `rotations_for_reading` takes a
window. `quadric_angle_rotations` alone accepts them, keyword-only, defaulting to these
constants, so that the cost of the window can be *measured* without a second implementation of
the fit living outside this module — two implementations of one fit diverge, and this project has
been bitten by that. Called at the defaults it is bitwise what it was before the parameters
existed, which `tests/test_second_order_angle_reading.py` executes rather than asserts.

The prohibition that motivated keeping them constants is unchanged and is about *selection*, not
about reachability: a window must never be chosen on the contact-angle flattening slope, the tension contrast or the
per-line angle error of a mesh whose accuracy is being judged. A window may be chosen only on
meshes that are not on trial — which is what makes measuring it admissible at all.

Coverage, and why the fallback is reported rather than hidden
--------------------------------------------------------------
The fit is not admissible everywhere: an incident face whose window holds fewer than two mesh
vertices per fit parameter, or whose design matrix is rank-deficient, gets **no** rotation and
falls back to the chord reading. Measured on dw3d's benchmark (0.7.1's default meshes, the 42
equilibrium cases) at `min_distance = 3`, coverage is `0.932` of incident faces; at
`min_distance = 4` it is `0.913`. A result that mixes
corrected and uncorrected angles without saying so would be unreadable as science, so
`quadric_angle_rotations` returns an `AngleReadingReport` naming the coverage and every
trijunction line that is not fully corrected, and the report-less convenience paths in
`foambryo.geometry` and `foambryo.tension_inference` emit an `AngleReadingCoverageWarning`
instead of staying silent.

Measured behaviour
------------------
The measured behaviour of the reading is in `BENCHMARKS.md` (foambryo repository).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from foambryo.dcel import DcelData

#: The chord reading: `foambryo`'s historical and default trijunction angle convention.
CHORD_READING = "chord"
#: The second-order reading: the quadric patch fit described in this module's docstring.
QUADRIC_READING = "quadric"
#: The chord reading with the **oriented** wedge angle, in `[0, 360)` degrees rather than
#: `[0, 180]`. See "The oriented wedge reading" in this module's docstring.
ORIENTED_CHORD_READING = "oriented_chord"
#: The second-order reading with the oriented wedge angle.
ORIENTED_QUADRIC_READING = "oriented_quadric"
#: Every accepted value of the `reading` / `angle_reading` selector.
ANGLE_READINGS = (CHORD_READING, QUADRIC_READING, ORIENTED_CHORD_READING, ORIENTED_QUADRIC_READING)
#: `reading -> (does it fit quadric tangents?, does it return the oriented wedge angle?)`. An
#: explicit registry rather than string surgery, so a newly-added selector cannot silently
#: acquire a default answer to either question.
_READING_TRAITS: dict[str, tuple[bool, bool]] = {
    CHORD_READING: (False, False),
    QUADRIC_READING: (True, False),
    ORIENTED_CHORD_READING: (False, True),
    ORIENTED_QUADRIC_READING: (True, True),
}
#: A wedge counts as reflex when it exceeds 180 degrees by more than this. Diagnostic only: the
#: reader itself branches on the strict sign of the oriented sine and applies no tolerance, so
#: this constant changes no angle. Chosen to match the tolerance the measurement that motivated
#: the oriented reading used, so the counts are comparable with it.
REFLEX_TOLERANCE_DEG = 1e-6

#: Transverse window, as a multiple of the incident face's own in-plane lever. Fixed a priori;
#: re-selecting it by held-out fit residual returns the same value.
#: **Fixed, not a tuning knob** — see this module's docstring.
QUADRIC_WINDOW_FACTOR = 4.0
#: Extent along the junction edge, as a multiple of the transverse window. `1.0` makes the
#: patch isotropic in the two surface directions, which is the choice that assumes least.
#: **Fixed, not a tuning knob.**
QUADRIC_AXIAL_FACTOR = 1.0
#: Minimum samples per fitted parameter. An a-priori rule, not a tuned value: two samples per
#: parameter is the smallest ratio at which a least-squares fit is a fit rather than an
#: interpolation.
_SAMPLES_PER_PARAMETER = 2
#: The quadric height field `zeta = c_xi*xi + c_eta*eta + c_xx*xi^2 + c_xe*xi*eta + c_ee*eta^2`
#: has five parameters, plus a constant term when `FIT_CONSTANT_TERM` is set. See
#: "The constant term" in this module's docstring for why the ground on which it used to be
#: omitted -- that the interface passes through the origin by construction -- is false on a
#: curved junction line.
_N_QUADRIC_PARAMETERS = 5

#: Whether the quadric height field fits a constant term. **Off**, and not for the reason this
#: module used to give: the junction-edge midpoint lies on the *chord* of the junction line, so the
#: interface does not pass through the frame's origin at all. It is off because fitting it was
#: measured to improve the reading at fine triangulations and degrade it at coarse ones, and which
#: side the shipped spacings fall on is not known. See "The constant term, and the false reason it
#: used to be omitted" in this module's docstring.
FIT_CONSTANT_TERM = False

#: `"include"`: the frame's own two junction-edge endpoints are kept as fit samples, decided by
#: vertex identity. `"exclude"`: they are dropped, also by identity. `"rounding"`: the historical
#: behaviour, in which their membership is decided by the strict test `eta > 0.0` on a quantity
#: that is exactly zero in exact arithmetic -- so by the sign of a rounding error, which changes
#: with the units the mesh is expressed in. See "The two junction-edge endpoints" in this module's
#: docstring.
JUNCTION_ENDPOINTS = "include"
#: Every accepted value of the `junction_endpoints` selector.
JUNCTION_ENDPOINT_POLICIES = ("include", "exclude", "rounding")
#: A trijunctional edge has exactly three incident faces; anything else is a quadrijunction
#: (or higher) and production skips it.
_FACES_PER_TRIJUNCTION_EDGE = 3
#: How many uncorrected trijunction lines the coverage warning names before it elides.
_MAX_LINES_IN_WARNING = 8
#: Slack added to the k-d tree query radius so the ball certainly contains the anisotropic
#: keep-region below, in mesh units.
_QUERY_RADIUS_SLACK = 2.0


class AngleReadingCoverageWarning(UserWarning):
    """Emitted when a quadric reading corrects only part of a mesh's trijunction faces.

    Applies to `"quadric"` and `"oriented_quadric"`; the chord readings always cover every face.

    The uncorrected faces silently fall back to the chord reading, which mixes two angle
    conventions in one result. That is admissible — it is strictly better than dropping the
    lines — but it must never be invisible. Use the `*_with_report` entry points in
    `foambryo.geometry` and `foambryo.tension_inference` to receive the numbers instead of
    the warning.
    """


@dataclass(frozen=True)
class AngleReadingReport:
    """What a trijunction angle reading covered, and what it fell back on.

    Attributes:
        reading: one of `ANGLE_READINGS`. Either chord reading always reports full coverage: it
            has no fit to fail, and it does not walk the junctions a second time, so its
            per-face counts are `0` and carry no information — only `coverage_fraction`,
            `uncorrected_lines` and `is_complete` are meaningful on a chord report.
        n_incident_faces: `(trijunctional geometric edge, incident face)` pairs visited. Three
            per trijunctional edge; quadrijunctional edges are skipped, exactly as
            `foambryo.geometry.compute_angles_tri` skips them.
        n_corrected: pairs that received a second-order correction.
        n_no_samples: pairs whose window held too few interface vertices to fit.
        n_no_fit: pairs whose design matrix was rank-deficient or whose curvature was not
            finite.
        coverage_fraction: `n_corrected / n_incident_faces`, or `None` on a mesh with no
            trijunctional edge at all.
        uncorrected_lines: every trijunction line — identified by its sorted material triple,
            the same three materials whose wedge keys `compute_angles_tri` returns — carrying
            at least one uncorrected face. Sorted.
        line_coverage: `material triple -> (corrected faces, total faces)` for every
            trijunction line in the mesh.
        window_factor: the transverse window multiplier actually used.
        axial_factor: the axial window multiplier actually used.
        junction_endpoints: the policy actually used for the frame's own two junction-edge
            endpoints, one of `JUNCTION_ENDPOINT_POLICIES`.
        fit_constant_term: whether the quadric height field actually fitted a constant term.

    A trijunction line with partial coverage (some but not all of its faces corrected) is
    not, by itself, a known source of extra angle error: measured against the chord
    reading's own error on the same lines, the per-line displacement this correction makes
    does not trend with how much of the line was covered (a near-zero, statistically
    insignificant slope, most of what little apparent trend there is already present in the
    chord reading, which has no coverage concept at all). A low `line_coverage_fraction`
    is a coverage fact about the mesh, not a warning sign about the correction's accuracy.
    """

    reading: str
    n_incident_faces: int
    n_corrected: int
    n_no_samples: int
    n_no_fit: int
    coverage_fraction: float | None
    uncorrected_lines: tuple[tuple[int, ...], ...]
    line_coverage: dict[tuple[int, ...], tuple[int, int]] = field(default_factory=dict)
    window_factor: float = QUADRIC_WINDOW_FACTOR
    axial_factor: float = QUADRIC_AXIAL_FACTOR
    junction_endpoints: str = JUNCTION_ENDPOINTS
    fit_constant_term: bool = FIT_CONSTANT_TERM

    @property
    def is_complete(self) -> bool:
        """True when every trijunctional incident face was read with the selected reading."""
        return not self.uncorrected_lines

    @property
    def line_coverage_fraction(self) -> float | None:
        """Fraction of trijunction *lines* every one of whose faces was corrected.

        The line analogue of `coverage_fraction`, which is per-*face*: a line with 2 of its
        3 faces corrected counts as 0 here (it is in `uncorrected_lines`) but contributes
        2/3 to `coverage_fraction`. `None` on a mesh with no trijunctional edge at all,
        mirroring `coverage_fraction`'s own convention.
        """
        if not self.line_coverage:
            return None
        n_fully_covered = sum(1 for corrected, total in self.line_coverage.values() if corrected == total)
        return n_fully_covered / len(self.line_coverage)

    def describe(self) -> str:
        """A one-line human-readable summary, used as the coverage warning's message.

        Leads with the two numbers a reader needs to judge the result: the measured face
        coverage fraction and how many trijunction lines were not fully corrected (out of how
        many), then lists those lines (at most `_MAX_LINES_IN_WARNING` of them).
        """
        if not fits_quadric_tangents(self.reading):
            return (
                f"angle reading {self.reading!r}: every trijunction face read the same way, no fallback"
            )
        coverage = "n/a" if self.coverage_fraction is None else f"{self.coverage_fraction:.4f}"
        line_coverage = "n/a" if self.line_coverage_fraction is None else f"{self.line_coverage_fraction:.4f}"
        lines = ", ".join(str(line) for line in self.uncorrected_lines[:_MAX_LINES_IN_WARNING])
        if len(self.uncorrected_lines) > _MAX_LINES_IN_WARNING:
            lines += f", ... ({len(self.uncorrected_lines)} lines in total)"
        n_lines = len(self.line_coverage)
        n_not_fully_corrected = len(self.uncorrected_lines)
        lines_total = "?" if not n_lines else str(n_lines)
        return (
            f"angle reading '{self.reading}': measured coverage {coverage} "
            f"({self.n_corrected}/{self.n_incident_faces} incident trijunction faces corrected); "
            f"{n_not_fully_corrected} of {lines_total} trijunction lines not fully corrected "
            f"(line coverage {line_coverage}). "
            f"{self.n_no_samples} faces had too few samples and {self.n_no_fit} had no admissible fit. "
            f"The uncovered faces fall back to the chord reading, so this result mixes the two "
            f"conventions on these trijunction lines: {lines}"
        )


#: The report for the chord reading. Full coverage by construction: the chord reading has no
#: fit that can fail, so there is nothing to fall back on and nothing to count. It is a module
#: constant rather than a computed value precisely so that selecting the default reading costs
#: nothing — the default path must not pay for a junction walk it does not need.
CHORD_READING_REPORT = AngleReadingReport(
    reading=CHORD_READING,
    n_incident_faces=0,
    n_corrected=0,
    n_no_samples=0,
    n_no_fit=0,
    coverage_fraction=1.0,
    uncorrected_lines=(),
    line_coverage={},
)


#: The report for the oriented chord reading. Full coverage for the same reason
#: `CHORD_READING_REPORT` is: reorienting a wedge is a sign choice, not a fit, so there is
#: nothing that can fail and nothing to fall back on.
ORIENTED_CHORD_READING_REPORT = AngleReadingReport(
    reading=ORIENTED_CHORD_READING,
    n_incident_faces=0,
    n_corrected=0,
    n_no_samples=0,
    n_no_fit=0,
    coverage_fraction=1.0,
    uncorrected_lines=(),
    line_coverage={},
)


def fits_quadric_tangents(reading: str) -> bool:
    """Whether `reading` replaces each triangle's chord by a fitted quadric tangent.

    Args:
        reading (str): one of `ANGLE_READINGS`.

    Returns:
        bool: True for `"quadric"` and `"oriented_quadric"`, False for the two chord readings.

    Raises:
        ValueError: if `reading` is not one of `ANGLE_READINGS`.
    """
    validate_reading(reading)
    return _READING_TRAITS[reading][0]


def reads_oriented_wedges(reading: str) -> bool:
    """Whether `reading` returns the oriented wedge angle in `[0, 360)` rather than `[0, 180]`.

    See "The oriented wedge reading" in this module's docstring for what the two ranges mean and
    why they differ only on reflex wedges.

    Args:
        reading (str): one of `ANGLE_READINGS`.

    Returns:
        bool: True for `"oriented_chord"` and `"oriented_quadric"`.

    Raises:
        ValueError: if `reading` is not one of `ANGLE_READINGS`.
    """
    validate_reading(reading)
    return _READING_TRAITS[reading][1]


def warn_on_partial_coverage(report: AngleReadingReport, stacklevel: int = 3) -> None:
    """Emit an `AngleReadingCoverageWarning` when a reading fell back on part of the mesh.

    Called only from the entry points that drop the report on the floor — the report-returning
    ones hand the caller the numbers the warning would have described, and stay silent.
    Complete coverage is silent either way.

    Args:
        report (AngleReadingReport): the reading's coverage report.
        stacklevel (int, optional): passed through to `warnings.warn`, so the warning points at
            the caller's own line rather than at this module. Defaults to 3.
    """
    if not report.is_complete:
        warnings.warn(report.describe(), AngleReadingCoverageWarning, stacklevel=stacklevel)


def validate_reading(reading: str) -> str:
    """Check an angle-reading selector and return it.

    Args:
        reading: the selector to check.

    Returns:
        str: `reading`, unchanged.

    Raises:
        ValueError: if `reading` is not one of `ANGLE_READINGS`.
    """
    if reading not in ANGLE_READINGS:
        message = f"unknown angle reading {reading!r}; expected one of {ANGLE_READINGS}"
        raise ValueError(message)
    return reading


def orthonormal_basis(normal: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A right-handed in-plane basis `(e1, e2)` with `cross(e1, e2) = normal`.

    `e1` is built from the world axis least aligned with `normal`, which is deterministic and
    never degenerate. Handedness matters: it fixes the sign of every curvature this module
    returns, and it is the reason `rodrigues_about` and the in-plane angles agree
    (`cross(axis, e1) = e2` for this basis).

    Args:
        normal (NDArray[np.float64]): the plane's normal, any non-zero length.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: the unit vectors `(e1, e2)`.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(normal)))] = 1.0
    e1 = np.cross(axis, normal)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return e1, e2


def rodrigues_about(
    axis: NDArray[np.float64],
    vector: NDArray[np.float64],
    angle: float,
) -> NDArray[np.float64]:
    """Rotate `vector` about the unit `axis` by `angle`, counter-clockwise seen from `+axis`.

    For a vector already perpendicular to the axis — which every face normal at a junction edge
    is — this reduces to `v cos a + (axis x v) sin a`, and it agrees with a counter-clockwise
    rotation in the `(e1, e2)` basis `orthonormal_basis` returns, because `cross(axis, e1) = e2`
    for that right-handed basis.

    The rotation is linear, so it acts identically on `n` and on `-n`: applying it before
    `_wedges_at_half_edge`'s relabelling flip and applying it after are the same operation, and
    the sign the mesh happens to carry on a face normal does not enter.

    Args:
        axis (NDArray[np.float64]): unit rotation axis.
        vector (NDArray[np.float64]): the vector to rotate.
        angle (float): rotation angle in radians.

    Returns:
        NDArray[np.float64]: the rotated vector.
    """
    cosine, sine = np.cos(angle), np.sin(angle)
    return vector * cosine + np.cross(axis, vector) * sine + axis * float(axis @ vector) * (1.0 - cosine)


def _fit_quadric_height_field(
    xi: NDArray[np.float64],
    eta: NDArray[np.float64],
    zeta: NDArray[np.float64],
    *,
    constant_term: bool = FIT_CONSTANT_TERM,
) -> dict[str, float] | None:
    r"""Least squares on `zeta = c_0 + c_xi*xi + c_eta*eta + c_xx*xi^2 + c_xe*xi*eta + c_ee*eta^2`.

    A **generic** second-order surface — the Taylor statement and nothing more.

    The constant term `c_0` used to be omitted, on the ground that the junction-edge midpoint is
    the midpoint of two mesh vertices lying on this interface and therefore that the interface
    passes through the origin by construction. **That ground is false on a curved junction line**:
    the midpoint of two vertices lies on the junction edge's *chord*, and the line leaves that
    chord by about `kappa_line * L^2 / 8`, so the surface misses the frame's origin by that much.
    Forcing the fit through a point the surface does not contain bends it, and the bend enters
    `c_ee`, which is the one coefficient the reading consumes. Fitting `c_0` costs one parameter
    and two samples of the minimum, and `_section_curvature` is unchanged by it: the section's
    curvature at `eta = 0` is `2 c_ee / (1 + c_eta^2)^{3/2}` whatever the section's height there is.

    `constant_term=False` restores the historical five-parameter fit exactly, for measurement.

    Args:
        xi (NDArray[np.float64]): sample coordinate along the junction edge.
        eta (NDArray[np.float64]): sample coordinate along the chord, towards the apex.
        zeta (NDArray[np.float64]): sample coordinate along `axis x chord`, the height.
        constant_term (bool, optional): fit `c_0`. Keyword-only. Defaults to `FIT_CONSTANT_TERM`.

    Returns:
        dict[str, float] | None: the coefficients (including `c_0`, which is `0.0` exactly when it
            was not fitted) plus `residual_rms`, `n_samples` and `n_parameters`, or `None` when
            there are fewer than `_SAMPLES_PER_PARAMETER` samples per parameter or the design
            matrix is rank-deficient. `None` means "no admissible fit": the caller falls back to
            the chord reading for that face.
    """
    columns = [xi, eta, xi * xi, xi * eta, eta * eta]
    names = ["c_xi", "c_eta", "c_xx", "c_xe", "c_ee"]
    if constant_term:
        columns.insert(0, np.ones_like(xi))
        names.insert(0, "c_0")
    design = np.column_stack(columns)
    if len(zeta) < _SAMPLES_PER_PARAMETER * design.shape[1]:
        return None
    solution, _residuals, rank, _singular = np.linalg.lstsq(design, zeta, rcond=None)
    if rank < design.shape[1]:
        return None
    residual = zeta - design @ solution
    coefficients = dict(zip(names, (float(v) for v in solution), strict=True))
    coefficients.setdefault("c_0", 0.0)
    return {
        **coefficients,
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "n_samples": len(zeta),
        "n_parameters": int(design.shape[1]),
    }


def _section_curvature(fit: dict[str, float]) -> float:
    r"""Signed curvature of the `xi = 0` section of a fitted height field, at `eta = 0`.

    The section is `zeta(eta) = c_0 + c_eta*eta + c_ee*eta^2`, a plane curve, whose signed
    curvature at `eta = 0` is the standard `2*c_ee / (1 + c_eta^2)^{3/2}`. The constant term does
    not appear: translating a curve does not change its curvature, so fitting `c_0` changes this
    value only through the least-squares solution for `c_eta` and `c_ee`, which is the point.

    Args:
        fit (dict[str, float]): a `_fit_quadric_height_field` result.

    Returns:
        float: the signed in-plane curvature `kappa`, in inverse mesh units.
    """
    c_eta, c_ee = fit["c_eta"], fit["c_ee"]
    return 2.0 * c_ee / (1.0 + c_eta**2) ** 1.5


def _interface_vertices(
    mesh: DcelData,
) -> dict[tuple[int, int], tuple[NDArray[np.int64], NDArray[np.float64], cKDTree]]:
    """Every interface's own mesh **vertices**, with their keys and a k-d tree, by material pair.

    Vertices, not triangle centroids: a centroid of a flat P1 triangle lies on the chord plane
    of the true surface and is therefore systematically displaced towards the chord, which would
    bias the fitted curvature towards zero — the very error being corrected. The vertices are
    the mesh's actual samples of the interface.

    The **vertex keys** are returned beside the coordinates because the fit's sample set has to be
    able to name two particular vertices — the frame's own junction-edge endpoints — rather than
    recognise them by a coordinate that is exactly zero in exact arithmetic. See "The two
    junction-edge endpoints" in this module's docstring.

    Args:
        mesh (DcelData): the mesh to index.

    Returns:
        dict: `(min, max)` material pair to `(vertex keys, vertex coordinates, k-d tree over the
            coordinates)`. The keys and the coordinates are in the same order, which is what makes
            a k-d tree result usable as an index into both.
    """
    triangles_and_labels = mesh.f
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(triangles_and_labels):
        a, b = int(row[3]), int(row[4])
        grouped[(min(a, b), max(a, b))].append(index)

    indexed: dict[tuple[int, int], tuple[NDArray[np.int64], NDArray[np.float64], cKDTree]] = {}
    for pair, indices in grouped.items():
        vertices = np.unique(triangles_and_labels[np.asarray(indices, dtype=np.int64)][:, :3].reshape(-1))
        coordinates = mesh.v[vertices]
        indexed[pair] = (vertices, coordinates, cKDTree(coordinates))
    return indexed


def junction_frames(mesh: DcelData) -> list[dict]:
    r"""One local frame per `(trijunctional geometric edge, incident face)`, from the DCEL.

    **This walk is production's own trijunction identification, not a second one.** A geometric
    edge is visited when at least one of its half-edges has more than one twin (the test
    `compute_angles_tri` and `compute_trijunction_edges` use), it is de-duplicated by its
    `(min, max)` vertex-key pair keeping the lowest incident half-edge, and its three incident
    faces are collected in exactly the order `geometry._wedges_at_half_edge` collects them —
    the half-edge's own face first, then its twins'. An edge with any number of incident faces
    other than three is a quadrijunction (or higher) and is skipped, exactly as production skips
    it. Keying the frames this way is what makes every corrected angle attributable to the same
    `(trijunction, material)` key the chord reading produces.

    The frame at one edge:

    * `axis` runs from the **lower** to the **higher** vertex key, normalised;
    * `midpoint` is the edge midpoint, which is where production's chord is anchored;
    * `(e1, e2)` is `orthonormal_basis(axis)`, whose handedness fixes every curvature sign;
    * `chord` is the in-plane unit vector from `midpoint` towards the incident face's apex —
      production's own reading direction — and `lever` is the in-plane distance to that apex.

    Args:
        mesh (DcelData): the mesh to walk.

    Returns:
        list[dict]: one entry per `(edge, face)` with keys `edge`, `face`, `axis`, `midpoint`,
            `e1`, `e2`, `lever`, `chord`, `edge_length`, `pair` (the face's sorted material
            pair) and `materials` (the edge's sorted material triple). Faces whose apex lies on
            the junction axis (`lever == 0`) are dropped: they have no chord to correct.
    """
    incident_half_edges: dict[tuple[int, int], list[int]] = {}
    for edge in mesh.half_edges:
        if len(edge.twin) > 1:
            geometric_edge = (min(edge.origin.key, edge.destination.key), max(edge.origin.key, edge.destination.key))
            incident_half_edges.setdefault(geometric_edge, []).append(edge.key)

    points = mesh.v
    triangles_and_labels = mesh.f
    frames: list[dict] = []
    for geometric_edge, half_edge_keys in incident_half_edges.items():
        edge = mesh.half_edges[min(half_edge_keys)]
        faces = [edge.incident_face, *(mesh.half_edges[twin].incident_face for twin in edge.twin)]
        if len(faces) != _FACES_PER_TRIJUNCTION_EDGE:
            continue

        low, high = geometric_edge
        axis = points[high] - points[low]
        norm = float(np.linalg.norm(axis))
        if norm <= 0:
            continue
        axis = axis / norm
        midpoint = 0.5 * (points[low] + points[high])
        e1, e2 = orthonormal_basis(axis)
        materials = tuple(sorted({int(m) for face in faces for m in (face.material_1, face.material_2)}))

        for face in faces:
            key = int(face.key)
            apex = next(int(v) for v in triangles_and_labels[key, :3] if int(v) not in geometric_edge)
            offset = points[apex] - midpoint
            in_plane = np.array([float(offset @ e1), float(offset @ e2)])
            lever = float(np.linalg.norm(in_plane))
            if lever <= 0:
                continue
            frames.append({
                "edge": geometric_edge,
                "face": key,
                "axis": axis,
                "midpoint": midpoint,
                "e1": e1,
                "e2": e2,
                "lever": lever,
                "chord": in_plane / lever,
                "edge_length": norm,
                "pair": (
                    min(int(face.material_1), int(face.material_2)),
                    max(int(face.material_1), int(face.material_2)),
                ),
                "materials": materials,
            })
    return frames


def _local_coordinates(
    samples: NDArray[np.float64],
    frame: dict,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """`(xi, eta, zeta)` of 3-D samples in the frame's chord-aligned basis.

    `xi` runs along the junction edge, `eta` along the chord towards the incident apex, and
    `zeta` along `cross(axis, chord)`, the in-plane direction a quarter turn counter-clockwise
    from the chord. The chord-aligned basis is `(e1, e2)` rotated by the chord's own polar
    angle — a proper rotation with `det = +1` — so the signed curvature is the same number in
    both frames and no sign convention is re-derived.

    Args:
        samples (NDArray[np.float64]): `(n, 3)` sample coordinates in mesh units.
        frame (dict): one `junction_frames` entry.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: `(xi, eta, zeta)`.
    """
    e1, e2, axis, chord = frame["e1"], frame["e2"], frame["axis"], frame["chord"]
    offset = samples - frame["midpoint"]
    xi = offset @ axis
    x = offset @ e1
    y = offset @ e2
    eta = x * chord[0] + y * chord[1]
    zeta = -x * chord[1] + y * chord[0]
    return xi, eta, zeta


def quadric_angle_rotations(
    mesh: DcelData,
    *,
    window_factor: float = QUADRIC_WINDOW_FACTOR,
    axial_factor: float = QUADRIC_AXIAL_FACTOR,
    junction_endpoints: str = JUNCTION_ENDPOINTS,
    fit_constant_term: bool = FIT_CONSTANT_TERM,
) -> tuple[dict[tuple[tuple[int, int], int], float], AngleReadingReport]:
    r"""Per-`(edge, face)` normal rotations that turn the chord reading into the quadric one.

    For each `(trijunctional geometric edge, incident face)` this fits a generic quadric height
    field to the incident interface's own mesh vertices inside an anisotropic window around the
    junction-edge midpoint, reads the normal-plane section's signed curvature `kappa` off the
    fitted surface, and returns the rotation `-kappa * lever / 2` to apply to that face's normal
    about the junction-edge axis. That is the exact chord-to-tangent rotation for a circular arc
    of curvature `kappa` whose chord has in-plane length `lever`.

    The window is `window_factor * lever` transversely and `axial_factor` times that along the
    junction edge. A sample is kept when `|xi| <= axial window` and `hypot(eta, zeta) <= window`,
    and then when `eta > 0` — which keeps the fit on the side of the junction where the interface
    actually lies, since an interface is bounded by its junction line — **except** for the frame's
    own two junction-edge endpoints, whose membership `junction_endpoints` decides explicitly. See
    "The two junction-edge endpoints" in this module's docstring for why that exception is a
    correctness fix and not a tuning knob.

    Failure modes, all of which fall back to the chord reading for that face and are counted in
    the returned report: fewer than `_SAMPLES_PER_PARAMETER * 5` interface vertices in the
    window, a rank-deficient design matrix, or a non-finite curvature.

    Determinism: the walk, the k-d trees and the least-squares solve are all deterministic
    functions of `mesh.v`, `mesh.f` and the four keyword arguments; there is no random seed
    anywhere in this module.

    **The two window factors are measurement affordances, not tuning knobs, and no production path
    can reach them.** They default to the module constants, `rotations_for_reading` does not
    forward them, and no `reading` selector, inference entry point or public convenience function
    exposes them; a caller who wants a different window has to name this function and say so. They
    exist because measuring what the window costs otherwise requires re-implementing the fit
    outside this module, and two implementations of one fit diverge. Called at the defaults the
    function is **bitwise** what it was before the parameters existed — the same float in every
    entry of the same map — and `tests/test_second_order_angle_reading.py` executes that identity
    rather than asserting it. Choosing a window on any quantity an arm is judged by would be the
    tuning this module's own docstring prohibits; a window may be chosen only on meshes that are
    not being judged.

    Args:
        mesh (DcelData): the mesh to read, in any length unit. **The only mesh-derived
            argument**: see "What enters the fit" in the module docstring. No reference mesh,
            target angle, tension or pressure can reach this function.
        window_factor (float, optional): transverse window, as a multiple of the incident face's
            own in-plane lever. Keyword-only. Defaults to `QUADRIC_WINDOW_FACTOR`, and production
            never passes anything else.
        axial_factor (float, optional): extent along the junction edge, as a multiple of the
            transverse window. Keyword-only. Defaults to `QUADRIC_AXIAL_FACTOR`.
        junction_endpoints (str, optional): one of `JUNCTION_ENDPOINT_POLICIES`. Keyword-only.
            Defaults to `JUNCTION_ENDPOINTS`, which is `"include"`. `"rounding"` reproduces the
            historical behaviour, in which these two samples' membership was decided by the sign of
            a rounding error; it exists so that the repair can be measured, not so that it can be
            chosen.
        fit_constant_term (bool, optional): fit the quadric's constant term. Keyword-only. Defaults
            to `FIT_CONSTANT_TERM`. `False` reproduces the historical five-parameter fit, including
            its lower minimum sample count, exactly.

    Raises:
        ValueError: if `junction_endpoints` is not one of `JUNCTION_ENDPOINT_POLICIES`.

    Returns:
        tuple[dict[tuple[tuple[int, int], int], float], AngleReadingReport]: the rotation angles
            in radians keyed by `((low vertex key, high vertex key), face key)`, and the
            coverage report. A key absent from the map means "no admissible fit here, read the
            chord".
    """
    if junction_endpoints not in JUNCTION_ENDPOINT_POLICIES:
        message = f"unknown junction_endpoints policy {junction_endpoints!r}; expected {JUNCTION_ENDPOINT_POLICIES}"
        raise ValueError(message)

    frames = junction_frames(mesh)
    indexed = _interface_vertices(mesh)

    rotations: dict[tuple[tuple[int, int], int], float] = {}
    per_line: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0, 0])
    n_no_samples = 0
    n_no_fit = 0
    minimum = _SAMPLES_PER_PARAMETER * (_N_QUADRIC_PARAMETERS + (1 if fit_constant_term else 0))

    for frame in frames:
        line = frame["materials"]
        per_line[line][1] += 1
        window = window_factor * frame["lever"]
        radius = window * np.hypot(1.0, axial_factor) + _QUERY_RADIUS_SLACK
        keys, coordinates, tree = indexed[frame["pair"]]
        found = tree.query_ball_point(frame["midpoint"], radius)
        found = np.asarray(found, dtype=np.int64) if found else np.zeros(0, dtype=np.int64)
        samples = coordinates[found] if len(found) else np.zeros((0, 3))
        if len(samples) < minimum:
            n_no_samples += 1
            continue

        xi, eta, zeta = _local_coordinates(samples, frame)
        in_window = (np.abs(xi) <= axial_factor * window) & (np.hypot(eta, zeta) <= window)
        on_the_interface_side = eta > 0.0
        if junction_endpoints == "rounding":
            keep = in_window & on_the_interface_side
        else:
            # Decide these two samples by **identity**, never by the sign of a computed zero: in
            # exact arithmetic their `eta` is exactly `0`, so `eta > 0.0` resolves them on a
            # rounding error whose sign changes with the units the mesh is expressed in.
            is_junction_endpoint = np.isin(keys[found], np.asarray(frame["edge"], dtype=keys.dtype))
            keep = in_window & (
                (on_the_interface_side | is_junction_endpoint)
                if junction_endpoints == "include"
                else (on_the_interface_side & ~is_junction_endpoint)
            )
        if int(keep.sum()) < minimum:
            n_no_samples += 1
            continue

        fit = _fit_quadric_height_field(xi[keep], eta[keep], zeta[keep], constant_term=fit_constant_term)
        if fit is None:
            n_no_fit += 1
            continue
        kappa = _section_curvature(fit)
        if not np.isfinite(kappa):
            n_no_fit += 1
            continue

        rotations[(frame["edge"], frame["face"])] = -0.5 * kappa * frame["lever"]
        per_line[line][0] += 1

    n_incident_faces = len(frames)
    line_coverage = {line: (counts[0], counts[1]) for line, counts in sorted(per_line.items())}
    report = AngleReadingReport(
        reading=QUADRIC_READING,
        n_incident_faces=n_incident_faces,
        n_corrected=len(rotations),
        n_no_samples=n_no_samples,
        n_no_fit=n_no_fit,
        coverage_fraction=(len(rotations) / n_incident_faces) if n_incident_faces else None,
        uncorrected_lines=tuple(line for line, (done, total) in line_coverage.items() if done < total),
        line_coverage=line_coverage,
        window_factor=window_factor,
        axial_factor=axial_factor,
        junction_endpoints=junction_endpoints,
        fit_constant_term=fit_constant_term,
    )
    return rotations, report


def rotations_for_reading(
    mesh: DcelData,
    reading: str,
) -> tuple[dict[tuple[tuple[int, int], int], float] | None, AngleReadingReport]:
    """Dispatch on the reading selector, with a per-mesh cache for the quadric fit.

    The quadric fit costs one k-d tree query and one 5-parameter least-squares solve per
    incident trijunction face — measured at roughly 0.5 s on a 47-case benchmark mesh, which is
    the same order as one inference. Every angle-reading tension method calls
    `compute_angles_tri` once, so a benchmark that runs eight methods over one mesh would pay
    for the fit eight times. The result is cached on the mesh instance under
    `_angle_reading_rotations`.

    **Cache validity.** `DcelData` never mutates `v` or `f` after `__init__`, so the cache
    cannot go stale through any public method of this package. If you mutate those arrays
    yourself, `del mesh._angle_reading_rotations` first.

    The cache is shared by `"quadric"` and `"oriented_quadric"`: the rotations depend only on
    the mesh, and whether a wedge is then reported oriented is decided downstream, in
    `foambryo.geometry`. Only the returned report's `reading` label differs.

    **This function does not forward a window.** It calls `quadric_angle_rotations` with no window
    arguments, so every reading reachable through a `reading` selector uses
    `QUADRIC_WINDOW_FACTOR` and `QUADRIC_AXIAL_FACTOR`. A measurement that wants another window
    calls `quadric_angle_rotations` itself; it may then seed this function's cache by assigning
    the pair to `mesh._angle_reading_rotations`, and the report it stores names the window it
    used, so a result read that way says so rather than looking like production's.

    Args:
        mesh (DcelData): the mesh to read.
        reading (str): one of `ANGLE_READINGS`.

    Returns:
        tuple[dict | None, AngleReadingReport]: the rotation map (`None` for either chord
            reading, which applies no rotation at all) and the coverage report.

    Raises:
        ValueError: if `reading` is not one of `ANGLE_READINGS`.
    """
    validate_reading(reading)
    if not fits_quadric_tangents(reading):
        return None, (ORIENTED_CHORD_READING_REPORT if reading == ORIENTED_CHORD_READING else CHORD_READING_REPORT)

    cached = getattr(mesh, "_angle_reading_rotations", None)
    if cached is None:
        cached = quadric_angle_rotations(mesh)
        mesh._angle_reading_rotations = cached
    rotations, report = cached
    if reading != report.reading:
        report = replace(report, reading=reading)
    return rotations, report
