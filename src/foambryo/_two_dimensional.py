r"""Relative tensions and pressures of a 2D tissue from its interfaces: the implementation behind foambryo's
package-level functions for 2-D meshes (private module; provisional).

**Use the package-level names**: `foambryo.infer_tensions`, `foambryo.infer_tensions_by_junction_component`,
`foambryo.infer_pressures` and `foambryo.infer_forces` dispatch a 2-D mesh (`foambryo.PlanarMesh`, dw2d's `DCEL_Data`)
here, and the 2-D-only names (`interface_arcs`, `junction_table`, `infer_forces_by_equilibrium_tiling`, the reports
and warnings) are exported by `foambryo` itself. **Provisional:** the 2-D names and signatures may change in any
release until they are declared stable.

The method is foambryo's, in two dimensions: the force balance at every junction gives the ratios of the
interface tensions (Young-Dupré), and Laplace's law on every interface gives the pressure differences. Please cite,
if you use the 2D inference: Yamamoto K., Ichbiah S., Perez M., Borrego-Pinto J., Delbary F., Goehring N. W.,
Northrop P., Turlier H., Charras G., *Spatiotemporal mapping of the contractile and adhesive forces sculpting early
C. elegans embryos*, Developmental Cell **61**(6), 1273-1288.e4 (2026), doi:10.1016/j.devcel.2026.04.013. The 3D
method is Ichbiah S. et al., *Embryo mechanics cartography: inference of 3D force atlases from fluorescence
microscopy*, Nature Methods **20**(12), 1989-1999 (2023), doi:10.1038/s41592-023-02084-7. The
2D inference of foambryo-2d (per-junction Young-Dupré with an exact gauge), written by Sacha Ichbiah and the
Turlier lab, runs here on foambryo's own row writer, junction graph, exact gauge and least-squares solve,
so that 2D and 3D share one implementation of each. Reading each interface as a circular arc and taking
its tangent at the point of the circle closest to the junction is the rule of CellFIT (Brodland G. W. et
al., PLoS ONE 9(6), e99116, 2014, doi:10.1371/journal.pone.0099116).

**Input.** A 2D multimaterial polyline mesh, as delaunay-watershed-2d (`dw2d`) builds it from a label
image: `v`, the `(V, 2)` vertex positions in pixels, in the image's (row, column) frame, and `e`, the
`(E, 4)` edges `[v1, v2, m1, m2]`, material 0 being the medium. Any object with these two attributes
(dw2d's `DCEL_Data`) or a `(vertices, edges)` pair is accepted; foambryo reads only the two arrays and
does not import dw2d. The pressures, and nothing else, rely on the **orientation rule** of foambryo's meshes in
both dimensions: on every edge `m1 < m2`, and the edge is the positively oriented boundary of `m1` in the index
frame, i.e. `m1` lies on the left of `v1 -> v2` in `(row, column)` (dw2d 0.8.0 states and tests it on every edge,
the image border included). A mesh in the former orientation (dw2d before 0.8.0: `m1` on the right in
`(row, column)`) is measured as such from its geometry and reversed on input, so it gives the same results; a mesh
whose cells disagree is refused (`foambryo.planar_mesh`'s docstring).

**Interfaces.** The edges of one material pair are joined into interfaces: maximal chains of edges between
junctions or excluded vertices (`interface_arcs`, which returns them as read-only `InterfaceArc` records with
the circle read on each). A chain with an edge whose two ends are both excluded (an image-border edge, by
default) is not an interface and is dropped. A closed chain (a cell enclosed by one other material) is an
interface without a junction.

**Angle readings** (`angle_reading`, the angle inside each material at a junction):

* `"arc"` (the default): each interface is read as a circular arc, as it is at mechanical equilibrium
  (uniform tension and pressure difference give a constant curvature): the geometric least-squares circle
  through all its vertices (an algebraic fit refined by Levenberg-Marquardt on the distances; a straight
  line when the curvature vanishes), and its tangent at the circle's point closest to the junction,
  oriented toward the interface's middle vertex. An interface with fewer than 4 vertices, or whose fit
  fails, is read by its whole-interface chord (junction to its other end). A junction where one of its
  three material pairs has no single interface ending there has no wedges (`nan`): it is counted as not
  finite and gives no row. Measured on synthetic equilibrium foams with known tensions and pressures
  (53 held-out tissues, 661 images: exact masks, crops and smooth boundary noise of 0.5 to 2 px, cells
  of 10 to 40 px radius), with dw2d 0.8.0's meshes, against the chord: the relative tension error is lower
  by 0.060 (95 % interval 0.054 to 0.067) and the pressure error by 0.085 of the RMS pressure (0.081 to
  0.091), relocation off; 0.068 and 0.091 with dw2d's junction relocation (its default since 0.8.0); in
  no tier is it significantly worse, including 0.75 px of noise, the level measured on real 2D masks.
* `"chord"`: the angle at the junction vertex between the two edges of the material that leave it, as
  `arccos(clip(u1 . u2, -1, 1))` with `u1`, `u2` unit vectors to the far ends of the two edges (the
  clip keeps two collinear edges at exactly 180 degrees instead of NaN). foambryo-2d's reading.
* `"fitted_tangent"`: a parabola through the junction and the next four vertices of each incident
  interface: more accurate than the chord on exactly rasterised masks, worse from 1 px of noise.

**Rows.** Each junction (a vertex where exactly three materials meet; vertices with four or more are
counted, never used) contributes its own three Young-Dupré rows
(`foambryo.tension_inference.young_dupre_entries`) with unit weight: a 2D junction is a point, where a 3D
trijunction line is weighted by its length. Junctions on excluded vertices give no row.

**Lenses: interfaces the angles cannot separate.** When two interfaces take their Young-Dupré rows from the
same junctions and from no other, their two tensions are determined only as far as the block of their two
columns in those rows is well conditioned. The usual case is a small lens (a cell, or a pocket of medium, with
only two sides): read by their chords (under the arc reading, sides of fewer than 4 vertices), its two sides
leave each junction along the same direction, the wedge inside the lens reads 0, the two columns are equal and
the system is singular. The solve would return two opposite tensions of order 1e11 to 1e12, and the rows of those
junctions would also pull the third interface's tension towards zero. This happens with one junction as well,
when the lens's other end is not a junction (for instance a vertex where four materials meet on the image
border). Where the condition number of the block exceeds `MAX_LENS_CONDITION` (20: an angle error of 1/20 rad,
about 3 degrees, could then change the two tensions by their own size), those junctions give no row, the
interfaces' tensions are `nan`, the report lists the group (`TwoDimensionalReport.refused_lenses`) and an
`IllConditionedLensWarning` names it. The search is repeated on the junctions left until none is refused.

**Gauge and solve.** Only ratios of tensions are determined, and only within one connected component of
the junction graph (interfaces linked through shared used junctions). Each component is gauged on its
own, `mean(tension) = mean_tension` over its interfaces, met exactly, as foambryo does in 3D
(`foambryo.junction_graph.solve_gauged_least_squares`, dense or sparse as `foambryo.linear_solves`'s
options say). Interfaces that no used junction touches are `nan`, never a number. `infer_tensions`
returns one flat map and warns (`JunctionGraphGaugeWarning`) when it holds several components or
undetermined interfaces; `infer_tensions_by_junction_component` returns each interface's component.

**Pressures** (`infer_pressures`): Laplace's law on every interface read by a circle,
`gamma * kappa = p_left - p_right` (`kappa` the signed curvature, positive when the circle's centre is on
the interface's left), in weighted least squares with the medium's pressure fixed. The weight of an
interface's row is the ratio of the typical junction-row noise to its curvature noise for independent
vertex noise (`interface_arcs`' noise factors), so long, well-sampled interfaces count more. The pressures
are on the scale of the tensions given: given `infer_tensions_by_junction_component`'s solution, each
junction-graph component is solved on its own scale.

**Image border.** Where the image border cuts a cell-cell interface, the medium edge along the border
makes a straight 180-degree wedge that is not a contact angle: by default (`exclude_image_border=True`)
the vertices on the image border (`image_border_vertices`) are excluded. On a dw2d 0.8.0 `DCEL_Data`
(`return_dcel()`) they are dw2d's mark, taken from its triangulation before any vertex moves and never
moved by its junction relocation; on arrays, the vertices on the image rectangle.

**Opt-in: a global fit** (`infer_forces_by_equilibrium_tiling`). Instead of reading each junction and each interface alone, it fits the
label image's crack points by one equilibrium tiling of circular arcs (Noll N. et al., Phys. Rev. X 10, 011072, 2020,
doi:10.1103/PhysRevX.10.011072), started from the arc reading of the mesh; it is more accurate on synthetic
equilibrium foams but still refuses some crops whole, so it is offered beside the arc reading (its docstring gives the
measurements).

**Reports.** With `diagnostics`, the entry points also return a frozen report, `TwoDimensionalReport` for
the tensions and `TwoDimensionalPressureReport` for the pressures, each holding the
`foambryo.linear_solves.SolveDiagnostics` of its solves.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from foambryo import linear_solves
from foambryo.junction_graph import (
    ComponentTensionSolution,
    JunctionGraphGaugeWarning,
    JunctionGraphPartition,
    _partition_over,
    _solve_least_squares,
    solve_gauged_least_squares,
)
from foambryo.linear_solves import SolveDiagnostics
from foambryo.pressure_inference import UndeterminedInferenceError
from foambryo.tension_inference import TensionComputationMethod, young_dupre_entries

ANGLE_READINGS = ("arc", "chord", "fitted_tangent")
DEFAULT_ANGLE_READING = "arc"
MEDIUM = 0
#: An interface with fewer vertices is read by its whole-interface chord and gives no curvature.
MIN_ARC_VERTICES = 4
Interface = tuple[int, int]
#: Interfaces whose rows come from the same junctions only (a lens) are refused (those junctions give no row, the
#: tensions are `nan`) when the block of two of their columns in those rows has a condition number above this: an
#: angle error of 1/20 rad (about 3 degrees) could then change the two tensions by their own size.
MAX_LENS_CONDITION = 20.0

__all__ = [
    "ANGLE_READINGS",
    "EQUILIBRIUM_TILING_MAX_EVALUATIONS",
    "DEFAULT_ANGLE_READING",
    "MAX_LENS_CONDITION",
    "EquilibriumTilingReport",
    "IllConditionedLensWarning",
    "InterfaceArc",
    "NoJunctionError",
    "TwoDimensionalPressureReport",
    "TwoDimensionalReport",
    "image_border_vertices",
    "infer_forces_by_equilibrium_tiling",
    "infer_pressures",
    "infer_tensions",
    "infer_tensions_by_junction_component",
    "interface_arcs",
    "junction_table",
]


class NoJunctionError(UndeterminedInferenceError):
    """The 2D mesh has no junction, so no tension can be inferred: its cells never meet.

    A tension is read from the angles at a junction, a vertex where three or more regions meet. On a
    label mask whose cells are separated from one another by background pixels (a common output of
    segmentation tools), every interface is a cell-medium loop, no junction exists, and every tension
    would be undetermined. Such a result carries no value, so the inference refuses it. The remedy is
    to grow the labels until neighbouring cells meet, with dw2d's `expansion_labels` argument,
    `geometry_reconstruction_2d(labels, expansion_labels=...)` (a distance in pixels).

    Fires only when the mesh has no vertex where three or more regions meet; a mesh whose interfaces
    are only partly undetermined is not refused (those interfaces are `nan`).

    Attributes:
        interfaces: the interfaces of the mesh (sorted material pairs), every one undetermined.
    """

    def __init__(self, interfaces: object) -> None:
        """Build the error and its message from the interfaces of the mesh."""
        self.interfaces = tuple(sorted(tuple(sorted(map(int, pair))) for pair in interfaces))
        shown = ", ".join(str(key) for key in self.interfaces[:8])
        if len(self.interfaces) > 8:
            shown += f", ... ({len(self.interfaces)} in total)"
        message = (
            f"the mesh has no junction (no vertex where three regions meet), so none of its "
            f"{len(self.interfaces)} interface tension(s) can be inferred: {shown}. The cells of the "
            "mask never meet: background pixels separate them. Grow the labels until neighbouring "
            "cells touch with dw2d's expansion_labels argument, e.g. "
            "geometry_reconstruction_2d(labels, expansion_labels=6) (a distance in pixels)."
        )
        super().__init__(message)


class IllConditionedLensWarning(UserWarning):
    """Interfaces whose rows come from the same junctions only were refused: the angles cannot separate their tensions.

    Their tensions are `nan` and those junctions give no row (module docstring, "Lenses"). The usual cause is a small
    lens (a cell or a pocket of medium with only two sides) whose two sides are read by their chords, which
    coincide. The report's `refused_lenses` lists each group, its junction vertices and its condition number.
    """


@dataclass(frozen=True)
class TwoDimensionalReport:
    """What a 2D tension inference read and solved: the second value returned with `diagnostics`.

    Frozen: its fields cannot be reassigned (the dicts it holds are plain dicts, as in 3D's
    `ComponentTensionSolution`, so that it can be pickled; they are not meant to be modified).

    Attributes:
        angle_reading: the reading of the junction angles (`"arc"`, `"chord"` or `"fitted_tangent"`).
        image_border_from: how the image border was found: `"dw2d mark"` (the mesh's
            `image_border_vertices`), `"vertex positions"` (`image_border_vertices`), or `None` when
            `exclude_image_border` was off.
        n_junctions_used: junctions that gave their three Young-Dupre rows.
        n_junctions_excluded: junctions on an excluded vertex (the image border or `excluded_vertices`).
        n_junctions_nonfinite: junctions not excluded whose wedges could not all be read.
        n_higher_order_vertices: vertices where four or more materials meet (counted, never used).
        n_rows: Young-Dupre rows (three per used junction).
        n_components: connected components of the junction graph, each gauged alone.
        n_interfaces_undetermined: interfaces touched by no used junction (`nan`).
        component_of: interface -> component index (`None` where undetermined).
        components: one dict per component, largest first (ties by the smallest interface): `index`,
            `n_interfaces`, `n_junctions`, `n_rows`, `gauge`, `mean_tension`, `rank_deficiency` (the dimension
            the exact gauge leaves undetermined, 0 when the ratios are unique; `None` on the sparse path, which
            does not compute it), `solver`.
        n_interfaces_by_fit: for the arc reading, how many interfaces were read by a `"circle"`, a `"line"` or
            their `"chord"`; `None` for the other readings.
        solves: the `foambryo.linear_solves.SolveDiagnostics` of the solves this call ran, as in 3D.
        n_junctions_refused: junctions that gave no row because they are the junctions of a refused lens.
        refused_lenses: one dict per refused group of interfaces whose rows came from the same junctions only
            (module docstring, "Lenses"): `interfaces` (their material pairs, sorted), `junction_vertices` (the
            vertex indices of those junctions, one or more) and `condition_number` (the largest over the pairs of
            interfaces in the group, `inf` when two columns are equal to rounding).
    """

    angle_reading: str
    image_border_from: str | None
    n_junctions_used: int
    n_junctions_excluded: int
    n_junctions_nonfinite: int
    n_higher_order_vertices: int
    n_rows: int
    n_components: int
    n_interfaces_undetermined: int
    component_of: dict[Interface, int | None]
    components: tuple[dict[str, object], ...]
    n_interfaces_by_fit: dict[str, int] | None
    solves: SolveDiagnostics
    n_junctions_refused: int = 0
    refused_lenses: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class TwoDimensionalPressureReport:
    """What a 2D pressure inference read and solved: the second value returned with `diagnostics`.

    Frozen, as `TwoDimensionalReport`.

    Attributes:
        image_border_from: as in `TwoDimensionalReport`.
        per_component: whether the tensions were a `ComponentTensionSolution`, each component's Laplace rows
            solved alone (on that component's tension scale).
        n_interfaces: interfaces read (`interface_arcs`).
        n_rows: Laplace rows solved.
        n_rows_without_tension: interfaces read by a circle or a line whose tension is not finite (no row).
        n_interfaces_without_curvature: interfaces read by their chord (no curvature, no row).
        n_cells_determined: cells given a pressure.
        n_cells_undetermined: cells left `nan`, including those of `n_cells_in_several_components`.
        n_cells_in_several_components: cells that the rows of two or more components determine (per component
            only); their pressures are in `pressures_by_component`, and `nan` in the returned map.
        n_edges_against_the_majority: edges that put the other material on the left of their interface (0 on a
            mesh that follows the orientation rule, module docstring).
        junction_noise_reference: the median junction-row noise that sets the Laplace weights (rad per pixel of
            vertex noise).
        component_of_cell: cell -> the component that determined it (per component only; `None` when
            undetermined or determined by several); empty when `per_component` is false.
        pressures_by_component: component -> {cell: pressure} (per component only); empty otherwise.
        solver: the record of the last least-squares solve (as `foambryo.junction_graph`'s solver notes).
        solves: the `foambryo.linear_solves.SolveDiagnostics` of the solves this call ran.
    """

    image_border_from: str | None
    per_component: bool
    n_interfaces: int
    n_rows: int
    n_rows_without_tension: int
    n_interfaces_without_curvature: int
    n_cells_determined: int
    n_cells_undetermined: int
    n_cells_in_several_components: int
    n_edges_against_the_majority: int
    junction_noise_reference: float
    solver: dict[str, object]
    solves: SolveDiagnostics
    component_of_cell: dict[int, int | None] = field(default_factory=dict)
    pressures_by_component: dict[int, dict[int, float]] = field(default_factory=dict)


# ==================================================================================================== mesh input
def _mesh_arrays(mesh: object) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """`(vertices (V, 2) float, edges (E, 4) int)` from a `foambryo.PlanarMesh`, a dw2d `DCEL_Data`, a
    `(points, segments, labels)` triple or a `(vertices, edges)` pair, the edges in the one orientation rule
    (reversed when the mesh is measured to be in the former one; refused when its cells disagree)."""
    from foambryo.planar_mesh import PlanarMesh, _segments_in_the_one_orientation

    if isinstance(mesh, PlanarMesh):
        return mesh.points, mesh.e
    if hasattr(mesh, "v") and hasattr(mesh, "e"):
        vertices, edges = mesh.v, mesh.e
    elif isinstance(mesh, (tuple, list)) and len(mesh) == 3:
        vertices, edges = mesh[0], np.hstack((np.asarray(mesh[1]), np.asarray(mesh[2])))
    else:
        vertices, edges = mesh
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or edges.ndim != 2 or edges.shape[1] != 4:
        message = f"expected (V, 2) vertices and (E, 4) edges, got {vertices.shape} and {edges.shape}"
        raise ValueError(message)
    segments, reversed_on_input = _segments_in_the_one_orientation(
        vertices, edges[:, :2], edges[:, 2:], getattr(mesh, "image_border_vertices", None),
    )
    if reversed_on_input:
        edges = np.hstack((segments, edges[:, 2:]))
    return vertices, edges


def _walked(edges: NDArray[np.int64]) -> NDArray[np.int64]:
    """The edges as the interface tracers walk them: from each edge's second vertex to its first.

    Under the orientation rule `m1` lies on the left of `v1 -> v2` in `(row, column)`, so on the left of `v2 -> v1` in
    the frame `(x, y) = (column, row)` the arcs are read in. Walking from the second vertex also traces every interface
    from the same end and in the same vertex order as foambryo did on dw2d's meshes before the orientation rule was
    unified (their edges were stored the other way round), so the circle fits, hence the tensions, are bit for bit
    those of that walk.
    """
    return np.ascontiguousarray(edges[:, [1, 0, 2, 3]])


def _check_angle_reading(angle_reading: str) -> None:
    if angle_reading not in ANGLE_READINGS:
        message = f"angle_reading must be one of {ANGLE_READINGS}, got {angle_reading!r}"
        raise ValueError(message)


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    return vector / np.linalg.norm(vector)


def _interfaces(edges: NDArray[np.int64]) -> list[Interface]:
    return sorted({(int(min(a, b)), int(max(a, b))) for a, b in edges[:, 2:4].tolist()})


def image_border_vertices(mesh: object) -> set[int]:
    """Vertices of a 2D mesh that lie on the image border: dw2d's mark, or else the vertex positions.

    Where a junction lies on the image border, the border cuts a cell-cell interface and dw2d's medium
    edge along the border makes a straight 180-degree wedge that is not a contact angle.

    * **dw2d's mark** (a mesh with an `image_border_vertices` attribute that is not None: dw2d 0.8.0's
      `DCEL_Data`, from `geometry_reconstruction_2d(...).return_dcel()`): those vertices, which dw2d
      takes from its triangulation's structure (the endpoints of its hull edges) before any vertex moves,
      and which its junction relocation never moves.
    * **Otherwise, the positions** (a `(vertices, edges)` pair, or an object without the mark): dw2d
      adds the four image corners to its Delaunay points and keeps them as mesh vertices, so the image
      rectangle is the bounding box of the vertices and the border vertices are those on its sides.
      Exact on a mesh whose border vertices did not move. If the four corners of the bounding box are
      not vertices (a mesh that did not come from dw2d), the border cannot be located: a `UserWarning`
      is raised and no vertex is returned. To exclude the border of a relocated mesh given as arrays,
      pass dw2d's mark as `excluded_vertices` (with `exclude_image_border=False`).
    """
    mark = getattr(mesh, "image_border_vertices", None)
    if mark is not None:
        return {int(v) for v in np.asarray(mark).ravel().tolist()}
    vertices, _ = _mesh_arrays(mesh)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    corners = {(lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])}
    if not corners <= {tuple(p) for p in vertices.tolist()}:
        warnings.warn(
            "the image corners are not vertices of this mesh, so the image border cannot be located; "
            "no vertex is excluded as image border (pass excluded_vertices explicitly)",
            UserWarning,
            stacklevel=2,
        )
        return set()
    on_x = (vertices[:, 0] == lo[0]) | (vertices[:, 0] == hi[0])
    on_border = on_x | (vertices[:, 1] == lo[1]) | (vertices[:, 1] == hi[1])
    return set(np.flatnonzero(on_border).tolist())


def _excluded(mesh: object, exclude_image_border: bool, excluded_vertices: Collection[int]) -> set[int]:
    """The union of the image border (when asked) and the vertices the caller names."""
    excluded = {int(v) for v in excluded_vertices}
    if exclude_image_border:
        excluded |= image_border_vertices(mesh)
    return excluded


def _image_border_from(mesh: object, exclude_image_border: bool) -> str | None:
    if not exclude_image_border:
        return None
    return "dw2d mark" if getattr(mesh, "image_border_vertices", None) is not None else "vertex positions"


# ==================================================================================================== circle fits
@dataclass(frozen=True)
class _Circle:
    """A circle (`centre`, `radius`) or its zero-curvature limit (`centre` None: `point`, unit `normal`), in (x, y)."""

    centre: NDArray[np.float64] | None
    radius: float
    point: NDArray[np.float64] | None = None
    normal: NDArray[np.float64] | None = None

    def closest_point_and_tangent(self, q: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The point of the circle nearest `q` and the circle's unit tangent there (unoriented)."""
        if self.centre is None:
            n, on_line = self.normal, self.point
            if n is None or on_line is None:
                message = "a line needs a point and a normal"
                raise ValueError(message)
            point = q - float(np.dot(q - on_line, n)) * n
            return point, np.array([-n[1], n[0]])
        d = q - self.centre
        norm = float(np.hypot(*d))
        if norm == 0.0:
            message = "the point is the circle's centre"
            raise ValueError(message)
        radial = d / norm
        return self.centre + self.radius * radial, np.array([-radial[1], radial[0]])


#: The geometric circle fit's fixed stopping rule (`_refine_circle`), in two phases from Pratt's fit.
#: (1) Levenberg-Marquardt, at most `_REFINEMENT_MAX_ITERATIONS` accepted steps: damping `lambda` starts at
#: `_REFINEMENT_INITIAL_DAMPING`, is divided by 10 after a step that lowers the sum of squared distances (accepted)
#: and multiplied by 10 after one that does not (rejected); it stops after an accepted step of at most
#: `_REFINEMENT_STEP_TOLERANCE * (1 + max |p|)` in every parameter, or when the damping exceeds
#: `_REFINEMENT_MAX_DAMPING`. Comparing sums of squares stops it where they agree to rounding, which fixes the
#: parameters only to about the square root of the machine epsilon on flat minima (few points, shallow arcs).
#: (2) Gauss-Newton polish, at most `_POLISH_MAX_STEPS` undamped steps, each taken while its largest component is
#: strictly smaller than the previous one's: it converges on the gradient `J^T d = 0`, which rounding resolves
#: to about the machine epsilon times the problem's condition number.
_REFINEMENT_MAX_ITERATIONS = 100
_REFINEMENT_INITIAL_DAMPING = 1e-3
_REFINEMENT_MAX_DAMPING = 1e16
_REFINEMENT_STEP_TOLERANCE = 4.0 * float(np.finfo(np.float64).eps)
_POLISH_MAX_STEPS = 20
#: Newton's iteration on Pratt's characteristic quartic: at most this many steps (it converges in a few).
_PRATT_MAX_NEWTON_STEPS = 100


def _algebraic_circle(points: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Pratt's algebraic fit on centred, scaled points: `(A, B, C, D)` with `A |x|^2 + B x + C y + D = 0`.

    Normalised by `B^2 + C^2 - 4 A D = 1` (a line has `A = 0`). The fit is the generalised eigenvector of
    `(M, N)` (`M` the moment matrix of `(|x|^2, x, y, 1)`, `N` the constraint's matrix) with the smallest
    non-negative eigenvalue `eta`. For points centred on their centroid, `det(M - eta N) = -Q(eta)` with the
    quartic `Q(eta) = 4 eta^4 + A2 eta^2 + A1 eta + A0`, whose smallest non-negative root Newton's iteration
    from `eta = 0` reaches monotonically (Chernov's form of Pratt's fit); the eigenvector is then the largest
    column of the adjugate of `M - eta N`, by cofactors. Only elementwise arithmetic and `np.sum`: no BLAS or
    LAPACK call, so the result does not depend on the linear-algebra library's version or build.
    """
    x, y = points[:, 0], points[:, 1]
    z = x * x + y * y
    n = len(points)
    mxx, myy, mxy = float(np.sum(x * x)) / n, float(np.sum(y * y)) / n, float(np.sum(x * y)) / n
    mxz, myz, mzz = float(np.sum(x * z)) / n, float(np.sum(y * z)) / n, float(np.sum(z * z)) / n
    mz = mxx + myy
    cov_xy = mxx * myy - mxy * mxy
    var_z = mzz - mz * mz
    a2 = 4.0 * cov_xy - 3.0 * mz * mz - mzz
    a1 = var_z * mz + 4.0 * cov_xy * mz - mxz * mxz - myz * myz
    a0 = mxz * (mxz * myy - myz * mxy) + myz * (myz * mxx - mxz * mxy) - var_z * cov_xy
    eta, q = 0.0, a0
    for _ in range(_PRATT_MAX_NEWTON_STEPS):
        slope = a1 + eta * (2.0 * a2 + 16.0 * eta * eta)
        if slope == 0.0:
            break
        trial = eta - q / slope
        q_trial = a0 + trial * (a1 + trial * (a2 + 4.0 * trial * trial))
        if not np.isfinite(trial) or trial == eta or abs(q_trial) >= abs(q):
            break
        eta, q = trial, q_trial
    if not (np.isfinite(eta) and eta >= -1e-10):  # rounding puts the root of exact points just below 0
        message = "no admissible algebraic circle"
        raise ValueError(message)
    # The null vector of K = M - eta N (rank 3): the largest column of its adjugate, by cofactors.
    k = [
        [mzz, mxz, myz, mz + 2.0 * eta],
        [mxz, mxx - eta, mxy, 0.0],
        [myz, mxy, myy - eta, 0.0],
        [mz + 2.0 * eta, 0.0, 0.0, 1.0],
    ]
    best, best_size = (0.0, 0.0, 0.0, 0.0), 0.0
    for j in range(4):
        column = tuple(_cofactor(k, j, i) for i in range(4))
        size = sum(v * v for v in column)
        if size > best_size:
            best, best_size = column, size
    a, b, c, d = best
    norm = b * b + c * c - 4.0 * a * d
    if not (np.isfinite(norm) and norm > 0.0):
        message = "no admissible algebraic circle"
        raise ValueError(message)
    root = math.sqrt(norm)
    return a / root, b / root, c / root, d / root


def _cofactor(matrix: list[list[float]], row: int, column: int) -> float:
    """`(-1)^(row + column)` times the determinant of `matrix` (4 x 4) without that row and column."""
    m = [[matrix[i][j] for j in range(4) if j != column] for i in range(4) if i != row]
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    return -det if (row + column) % 2 else det


def _geometric_residuals_and_jacobian(
    params: tuple[float, float, float],
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Signed distances of the points to the circle `(A, delta, theta)`, and their Jacobian (n x 3).

    `B = u cos(theta)`, `C = u sin(theta)`, `u = 1 + 2 A delta`, `D = delta (1 + A delta)`, so that
    `B^2 + C^2 - 4 A D = 1` identically and `delta` is the signed distance from the origin to the circle.
    With `P = A |x|^2 + B x + C y + D` and `S = sqrt(1 + 4 A P)`, the distance is `d = 2 P / (1 + S)`;
    `dd/dP = 1 / S` and `dd/dA` at fixed `P` is `-4 P^2 / (S (1 + S)^2)`.
    """
    a, delta, theta = params
    ct, st = math.cos(theta), math.sin(theta)
    u = 1.0 + 2.0 * a * delta
    x, y = points[:, 0], points[:, 1]
    z = x * x + y * y
    w = x * ct + y * st
    p = a * z + u * w + delta * (1.0 + a * delta)
    s = np.sqrt(np.maximum(1.0 + 4.0 * a * p, 1e-300))
    d = 2.0 * p / (1.0 + s)
    dp_da = z + 2.0 * delta * w + delta * delta
    dp_ddelta = 2.0 * a * w + u
    dp_dtheta = u * (-x * st + y * ct)
    jacobian = np.column_stack([dp_da / s - 4.0 * p * p / (s * (1.0 + s) ** 2), dp_ddelta / s, dp_dtheta / s])
    return d, jacobian


def _solve_damped_normal_equations(
    jac: NDArray[np.float64],
    residuals: NDArray[np.float64],
    damping: float,
) -> tuple[float, float, float] | None:
    """The step `h` of `(J^T J + damping diag(J^T J)) h = -J^T d`, by a written-out 3 x 3 Cholesky; None if singular.

    The products are elementwise sums (`np.sum`), not BLAS calls. A zero diagonal entry of `J^T J` is damped with
    `eps` times the largest one.
    """
    columns = (jac[:, 0], jac[:, 1], jac[:, 2])
    h = [[float(np.sum(columns[i] * columns[j])) for j in range(3)] for i in range(3)]
    g = [-float(np.sum(columns[i] * residuals)) for i in range(3)]
    floor = float(np.finfo(np.float64).eps) * max(h[0][0], h[1][1], h[2][2])
    for i in range(3):
        h[i][i] += damping * max(h[i][i], floor)
    l00 = h[0][0]
    if not l00 > 0.0:
        return None
    l00 = math.sqrt(l00)
    l10, l20 = h[1][0] / l00, h[2][0] / l00
    l11 = h[1][1] - l10 * l10
    if not l11 > 0.0:
        return None
    l11 = math.sqrt(l11)
    l21 = (h[2][1] - l20 * l10) / l11
    l22 = h[2][2] - l20 * l20 - l21 * l21
    if not l22 > 0.0:
        return None
    l22 = math.sqrt(l22)
    y0 = g[0] / l00
    y1 = (g[1] - l10 * y0) / l11
    y2 = (g[2] - l20 * y0 - l21 * y1) / l22
    x2 = y2 / l22
    x1 = (y1 - l21 * x2) / l11
    x0 = (y0 - l10 * x1 - l20 * x2) / l00
    return x0, x1, x2


def _refine_circle(start: tuple[float, float, float], points: NDArray[np.float64]) -> tuple[float, float, float]:
    """The geometric least-squares circle `(A, delta, theta)` from `start`, with the fixed stopping rule above.

    Levenberg-Marquardt (`_levenberg_marquardt`), then a Gauss-Newton polish (`_polish`). Deterministic: the same
    points and start give the same circle bit for bit whatever SciPy's (or the BLAS's) version, since only
    elementwise NumPy arithmetic, `np.sum` and `math` are used.
    """
    return _polish(_levenberg_marquardt(start, points), points)


def _polish(start: tuple[float, float, float], points: NDArray[np.float64]) -> tuple[float, float, float]:
    """Undamped Gauss-Newton steps from `start`, each taken while its largest component strictly decreases."""
    params, previous = start, math.inf
    for _ in range(_POLISH_MAX_STEPS):
        residuals, jac = _geometric_residuals_and_jacobian(params, points)
        step = _solve_damped_normal_equations(jac, residuals, 0.0)
        if step is None:
            break
        size = max(abs(step[0]), abs(step[1]), abs(step[2]))
        if not size < previous:  # also stops on nan
            break
        params, previous = (params[0] + step[0], params[1] + step[1], params[2] + step[2]), size
    return params


def _levenberg_marquardt(start: tuple[float, float, float], points: NDArray[np.float64]) -> tuple[float, float, float]:
    """Levenberg-Marquardt on the geometric distances from `start`, phase (1) of the stopping rule above."""
    params = start
    residuals, jac = _geometric_residuals_and_jacobian(params, points)
    cost = float(np.sum(residuals * residuals))
    damping = _REFINEMENT_INITIAL_DAMPING
    for _ in range(_REFINEMENT_MAX_ITERATIONS):
        while True:
            step = _solve_damped_normal_equations(jac, residuals, damping)
            if step is not None:
                trial = (params[0] + step[0], params[1] + step[1], params[2] + step[2])
                trial_residuals, trial_jac = _geometric_residuals_and_jacobian(trial, points)
                trial_cost = float(np.sum(trial_residuals * trial_residuals))
                if trial_cost < cost:  # False for nan
                    break
            damping *= 10.0
            if damping > _REFINEMENT_MAX_DAMPING:
                return params
        params, residuals, jac, cost = trial, trial_residuals, trial_jac, trial_cost
        damping /= 10.0
        size = 1.0 + max(abs(params[0]), abs(params[1]), abs(params[2]))
        if max(abs(step[0]), abs(step[1]), abs(step[2])) <= _REFINEMENT_STEP_TOLERANCE * size:
            return params
    return params


def _fit_circle(points: NDArray[np.float64]) -> _Circle | None:
    """Geometric least-squares circle (or line) through `points` (n x 2, x y), or None.

    None with fewer than 3 distinct points or a fit that does not converge to a finite circle or line.
    The points are centred on their centroid and scaled by their RMS distance to it; Pratt's algebraic fit
    (`_algebraic_circle`) starts a Levenberg-Marquardt refinement of the distances and a Gauss-Newton polish
    (`_refine_circle`), written in NumPy with a fixed stopping rule, so that the circle does not depend on SciPy's
    version.
    """
    unique = np.unique(np.asarray(points, float), axis=0)
    if len(unique) < 3:
        return None
    n = len(unique)
    mean = np.array([float(np.sum(unique[:, 0])) / n, float(np.sum(unique[:, 1])) / n])
    centred = unique - mean
    scale = math.sqrt(float(np.sum(centred * centred)) / n)
    if not scale > 0.0:
        return None
    scaled = centred / scale
    try:
        a, b, c, d = _algebraic_circle(scaled)
    except ValueError:
        return None
    s0 = math.sqrt(max(1.0 + 4.0 * a * d, 0.0))
    delta0 = 2.0 * d / (1.0 + s0)
    u0 = 1.0 + 2.0 * a * delta0
    theta0 = math.atan2(c, b) if u0 >= 0 else math.atan2(-c, -b)
    a, delta, theta = _refine_circle((a, delta0, theta0), scaled)
    if not all(math.isfinite(v) for v in (a, delta, theta)):
        return None
    u = 1.0 + 2.0 * a * delta
    b, c = u * math.cos(theta), u * math.sin(theta)
    if abs(a) < 1e-14:  # the zero-curvature limit, in scaled units
        normal = np.array([b, c]) / math.hypot(b, c)
        return _Circle(centre=None, radius=np.inf, point=mean - scale * delta * normal, normal=normal)
    return _Circle(centre=mean + scale * (-np.array([b, c]) / (2.0 * a)), radius=scale / (2.0 * abs(a)))


# ============================================================================================ interfaces as arcs
@dataclass(frozen=True)
class InterfaceArc:
    """One interface of a 2D mesh (a chain of edges of one material pair) and the circular arc read on it.

    Read-only data (a frozen dataclass of numbers and tuples), returned by `interface_arcs`.

    Vectors and positions are in the mesh's own (row, column) frame, in pixels; "left" and "right" are
    those of the traversal `vertices[0] -> vertices[-1]` in the frame `(x, y) = (column, row)` (where `m1` lies on the
    right of each edge `v1 -> v2`, being on its left in `(row, column)`).

    Attributes:
        materials: the sorted material pair.
        left: the material on the left of the traversal (the majority over its edges; on a dw2d mesh every
            edge agrees, see `n_edges_against_the_majority`); `right` the other one.
        right: the material on the right of the traversal.
        vertices: the vertex indices along the interface (`vertices[0] == vertices[-1]` when `closed`).
        closed: whether the interface is a closed loop without junction (a cell enclosed by one material).
        fit: `"circle"`, `"line"` (zero curvature) or `"chord"` (fewer than `MIN_ARC_VERTICES` vertices, or
            no circle could be fitted: tangents along the whole-interface chord, no curvature).
        curvature: the signed curvature (1/px): `1/R`, positive when the circle's centre lies on the left of
            the traversal (the interface bulges into its right material); 0 for a line; `nan` for `"chord"`.
            Laplace's law reads `tension * curvature = p_left - p_right`.
        centre: the circle's centre, `(row, column)`, or None for a line or a chord.
        radius: the circle's radius (px); `inf` for a line, `nan` for a chord.
        tangent_start: unit tangent at `vertices[0]`, pointing into the interface.
        tangent_end: unit tangent at `vertices[-1]`, pointing into the interface.
        length: the length of the polyline (px).
        n_points: the number of distinct vertices fitted (a closed loop counts its first vertex once).
        tangent_noise: the standard deviation of an end tangent's angle (rad) per pixel of independent
            vertex noise, `sqrt(192 / n) / length` (a shallow arc of evenly spaced vertices); for a chord
            `sqrt(2) / chord length`.
        curvature_noise: the standard deviation of the curvature (1/px) per pixel of independent vertex
            noise, `sqrt(720 / n) / length^2`; `nan` for a chord.
        n_edges_against_the_majority: how many of its edges put the other material on the left.
    """

    materials: Interface
    left: int
    right: int
    vertices: tuple[int, ...]
    closed: bool
    fit: str
    curvature: float
    centre: tuple[float, float] | None
    radius: float
    tangent_start: tuple[float, float]
    tangent_end: tuple[float, float]
    length: float
    n_points: int
    tangent_noise: float
    curvature_noise: float
    n_edges_against_the_majority: int


def _trace_pieces(  # noqa: C901 -- one tracing loop for open chains, one for closed loops, sharing `used`
    edges: NDArray[np.int64],
    pair_edges: list[int],
    stop: set[int],
) -> list[tuple[list[int], list[int], bool]]:
    """The chains of one material pair's edges: open ones from every stop or branch vertex, then closed loops."""
    adjacent: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for k in pair_edges:
        v1, v2 = int(edges[k, 0]), int(edges[k, 1])
        adjacent[v1].append((v2, k))
        adjacent[v2].append((v1, k))
    used: set[int] = set()

    def follow(path: list[int], current: int, kk: int, until: Callable[[int], bool]) -> list[int]:
        path_edges = []
        while True:
            used.add(kk)
            path.append(current)
            path_edges.append(kk)
            if until(current):
                return path_edges
            candidates = [(w, j) for w, j in adjacent[current] if j not in used]
            if not candidates:
                return path_edges
            current, kk = candidates[0]

    pieces = []
    for start in [v for v in adjacent if v in stop or len(adjacent[v]) != 2]:
        for following, k in adjacent[start]:
            if k not in used:
                path = [start]
                path_edges = follow(path, following, k, lambda v: v in stop or len(adjacent[v]) != 2)
                pieces.append((path, path_edges, False))
    for k in pair_edges:  # closed loops without any stop vertex
        if k not in used:
            first = int(edges[k, 0])
            path = [first]
            path_edges = follow(path, int(edges[k, 1]), k, lambda v, first=first: v == first)
            pieces.append((path, path_edges, path[0] == path[-1]))
    return pieces


def _left_material(edges: NDArray[np.int64], path: list[int], path_edges: list[int]) -> tuple[int, int]:
    """The material on the left of the traversal in `(x, y)` by majority over its edges, and the edges against it.

    `edges` are walked edges (`_walked`): `m1` on the left of `v1 -> v2` in `(x, y) = (column, row)`.
    """
    votes: dict[int, int] = defaultdict(int)
    for here, k in zip(path[:-1], path_edges, strict=True):
        v1, _, m1, m2 = (int(x) for x in edges[k])
        votes[m1 if here == v1 else m2] += 1
    left = max(votes, key=lambda m: (votes[m], -m))
    return int(left), int(sum(n for m, n in votes.items() if m != left))


def _chains(
    edges: NDArray[np.int64],
    stops_besides_junctions: set[int],
) -> list[tuple[Interface, list[int], int, bool, int]]:
    """Every interface: `(pair, vertex path, left material, closed, edges against the majority)`.

    A chain stops at junctions (three or more materials) and at `stops_besides_junctions`; a chain with an
    edge whose two ends are both in `stops_besides_junctions` is dropped (an image-border edge). The edges are walked
    from their second vertex (`_walked`).
    """
    edges = _walked(edges)
    materials_at: dict[int, set[int]] = defaultdict(set)
    for v1, v2, a, b in edges.tolist():
        materials_at[v1] |= {a, b}
        materials_at[v2] |= {a, b}
    stop = {v for v, s in materials_at.items() if len(s) >= 3} | stops_besides_junctions
    by_pair: dict[Interface, list[int]] = defaultdict(list)
    for k, (_, _, a, b) in enumerate(edges.tolist()):
        by_pair[(min(a, b), max(a, b))].append(k)
    out = []
    for pair, pair_edges in by_pair.items():
        for path, path_edges, closed in _trace_pieces(edges, pair_edges, stop):
            if any(
                int(edges[k, 0]) in stops_besides_junctions and int(edges[k, 1]) in stops_besides_junctions
                for k in path_edges
            ):
                continue
            left, against = _left_material(edges, path, path_edges)
            out.append((pair, [int(v) for v in path], left, bool(closed), against))
    return out


def _polyline_length(points: NDArray[np.float64]) -> float:
    return float(np.sum(np.hypot(*np.diff(points, axis=0).T)))


def _arc_of(
    pair: Interface,
    path: list[int],
    left: int,
    closed: bool,
    against: int,
    xy: NDArray[np.float64],
) -> InterfaceArc:
    """The free circle of one interface (in (x, y) internally), or its whole-interface chord."""
    points = xy[path]
    right = pair[0] if left == pair[1] else pair[1]
    length = _polyline_length(points)
    chord = float(np.hypot(*(points[-1] - points[0])))

    def rowcol(vector: NDArray[np.float64]) -> tuple[float, float]:
        return float(vector[1]), float(vector[0])

    def chord_reading() -> InterfaceArc:
        e = points[-1] - points[0] if chord > 0 else points[1] - points[0]
        e = e / float(np.hypot(*e))
        return InterfaceArc(
            pair,
            left,
            right,
            tuple(path),
            closed,
            "chord",
            float("nan"),
            None,
            float("nan"),
            rowcol(e),
            rowcol(-e),
            length,
            len(points),
            float(np.sqrt(2.0) / max(chord, 1e-12)),
            float("nan"),
            against,
        )

    if len(points) < MIN_ARC_VERTICES:
        return chord_reading()
    fitted = points[:-1] if closed else points
    circle = _fit_circle(fitted)
    if circle is None:
        return chord_reading()
    middle = points[len(points) // 2]
    tangents = []
    for end in (points[0], points[-1]):
        try:
            closest, tangent = circle.closest_point_and_tangent(end)
        except ValueError:
            return chord_reading()
        if float(np.dot(tangent, middle - closest)) < 0:
            tangent = -tangent
        tangents.append(tangent)
    if circle.centre is None:
        curvature, centre, fit = 0.0, None, "line"
    else:
        m = len(points) // 2
        local = points[min(m + 1, len(points) - 1)] - points[max(m - 1, 0)]
        local = local / np.hypot(*local)
        towards_centre = circle.centre - points[m]
        curvature = float(np.sign(local[0] * towards_centre[1] - local[1] * towards_centre[0]) / circle.radius)
        centre, fit = rowcol(circle.centre), "circle"
    n = len(fitted)
    return InterfaceArc(
        pair,
        left,
        right,
        tuple(path),
        closed,
        fit,
        curvature,
        centre,
        float(circle.radius),
        rowcol(tangents[0]),
        rowcol(tangents[1]),
        length,
        n,
        float(np.sqrt(192.0 / n) / length),
        float(np.sqrt(720.0 / n) / length**2),
        against,
    )


def _arcs(vertices: NDArray[np.float64], edges: NDArray[np.int64], excluded: set[int]) -> list[InterfaceArc]:
    xy = vertices[:, ::-1]
    return [
        _arc_of(pair, path, left, closed, against, xy) for pair, path, left, closed, against in _chains(edges, excluded)
    ]


def interface_arcs(
    mesh: object,
    *,
    exclude_image_border: bool = True,
    excluded_vertices: Collection[int] = (),
) -> list[InterfaceArc]:
    """Every interface of a 2D mesh with the circular arc `angle_reading="arc"` reads on it.

    Args:
        mesh: a dw2d `DCEL_Data`, an object with `v` and `e`, or a `(vertices, edges)` pair.
        exclude_image_border: also stop interfaces at the vertices on the image border and drop the
            interfaces with an edge along it (`image_border_vertices`). Keyword-only.
        excluded_vertices: further vertices where interfaces stop (and whose junctions give no row).
            Keyword-only.

    Returns:
        list[InterfaceArc]: one per interface (module docstring, "Interfaces").
    """
    vertices, edges = _mesh_arrays(mesh)
    return _arcs(vertices, edges, _excluded(mesh, exclude_image_border, excluded_vertices))


# ==================================================================================================== junctions
def _fitted_tangent(points: NDArray[np.float64], k: int = 4) -> NDArray[np.float64]:
    """Unit tangent at `points[0]` of a parabola through `points[0]` and the next `k` points."""
    p = np.asarray(points[: k + 1], float) - points[0]
    if len(p) < 3:
        return _unit(p[1])
    ex = _unit(p[-1])
    ey = np.array([-ex[1], ex[0]])
    x, y = p[1:] @ ex, p[1:] @ ey
    (a, _b), *_ = np.linalg.lstsq(np.column_stack([x, x * x]), y, rcond=None)
    return _unit(ex + a * ey)


def _interface_chains(edges: NDArray[np.int64]) -> dict[Interface, list[list[int]]]:
    """Each interface's edges joined into vertex chains (foambryo-2d's construction, for `"fitted_tangent"`).

    The edges are walked from their second vertex (`_walked`).
    """
    by_pair: dict[Interface, list[tuple[int, int]]] = defaultdict(list)
    for v1, v2, m1, m2 in _walked(edges).tolist():
        by_pair[(min(m1, m2), max(m1, m2))].append((v1, v2))
    chains = {}
    for pair, pair_edges in by_pair.items():
        adjacent: dict[int, list[int]] = defaultdict(list)
        for a, b in pair_edges:
            adjacent[a].append(b)
            adjacent[b].append(a)
        ends = [v for v, neighbours in adjacent.items() if len(neighbours) != 2]
        seen: set[int] = set()
        out = []
        for start in ends + list(adjacent):
            if start in seen:
                continue
            chain, previous, current = [start], None, start
            seen.add(start)
            while True:
                following = [w for w in adjacent[current] if w != previous and w not in seen]
                if not following:
                    break
                previous, current = current, following[0]
                chain.append(current)
                seen.add(current)
            out.append(chain)
        chains[pair] = out
    return chains


def _wedges_from_tangents(materials: tuple[int, int, int], tangents: dict[Interface, NDArray[np.float64]]) -> dict:
    a, b, c = materials

    def wedge(p: Interface, q: Interface) -> float:
        return float(np.arccos(np.clip(float(np.dot(tangents[p], tangents[q])), -1.0, 1.0)))

    return {a: wedge((a, b), (a, c)), b: wedge((a, b), (b, c)), c: wedge((a, c), (b, c))}


def _read_fitted_tangents(vertices: NDArray[np.float64], edges: NDArray[np.int64], junctions: list[dict]) -> None:
    """Replace each junction's chord wedges by fitted-tangent ones where its three interfaces can be followed."""
    chains = _interface_chains(edges)
    for junction in junctions:
        a, b, c = junction["materials"]
        tangents = {}
        for pair in ((a, b), (a, c), (b, c)):
            for chain in chains.get(pair, []):
                if chain[0] == junction["vertex"]:
                    tangents[pair] = _fitted_tangent(vertices[chain])
                elif chain[-1] == junction["vertex"]:
                    tangents[pair] = _fitted_tangent(vertices[chain[::-1]])
        if len(tangents) == 3:  # else keep the chord reading where an incident chain cannot be followed
            junction["wedges_rad"] = _wedges_from_tangents((a, b, c), tangents)


def _read_arcs(arcs: list[InterfaceArc], junctions: list[dict]) -> None:
    """Replace each junction's wedges by those between its three interfaces' arc tangents (nan if not three)."""
    ends: dict[int, dict[Interface, list[NDArray[np.float64]]]] = defaultdict(lambda: defaultdict(list))
    for arc in arcs:
        if arc.closed:
            continue
        # tangents back in (x, y), where the wedges were measured when the reading was designed
        ends[arc.vertices[0]][arc.materials].append(np.array(arc.tangent_start[::-1]))
        ends[arc.vertices[-1]][arc.materials].append(np.array(arc.tangent_end[::-1]))
    for junction in junctions:
        a, b, c = junction["materials"]
        at = ends.get(junction["vertex"], {})
        pairs = ((a, b), (a, c), (b, c))
        if any(len(at.get(pair, [])) != 1 for pair in pairs):
            junction["wedges_rad"] = dict.fromkeys((a, b, c), float("nan"))
            continue
        junction["wedges_rad"] = _wedges_from_tangents((a, b, c), {pair: at[pair][0] for pair in pairs})


def _junction_table(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int64],
    angle_reading: str,
    excluded: set[int],
) -> tuple[list[dict], list[dict], list[InterfaceArc] | None]:
    far_ends: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for v1, v2, m1, m2 in edges.tolist():
        for here, there in ((v1, v2), (v2, v1)):
            far_ends[here][m1].append(there)
            far_ends[here][m2].append(there)
    junctions, higher_order = [], []
    for vertex in sorted(far_ends):
        by_material = far_ends[vertex]
        if len(by_material) < 3:
            continue
        materials = tuple(sorted(int(m) for m in by_material))
        if len(materials) > 3:
            higher_order.append({"vertex": int(vertex), "materials": materials})
            continue
        here = vertices[vertex]
        wedges = {}
        for material in materials:
            ends = by_material[material]
            if len(ends) != 2:
                wedges[material] = float("nan")
                continue
            u1, u2 = _unit(vertices[ends[0]] - here), _unit(vertices[ends[1]] - here)
            wedges[material] = float(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))
        junctions.append({"vertex": int(vertex), "materials": materials, "wedges_rad": wedges})
    arcs = None
    if angle_reading == "fitted_tangent":
        _read_fitted_tangents(vertices, edges, junctions)
    elif angle_reading == "arc":
        arcs = _arcs(vertices, edges, excluded)
        _read_arcs(arcs, junctions)
    return junctions, higher_order, arcs


def junction_table(
    mesh: object,
    angle_reading: str = DEFAULT_ANGLE_READING,
    *,
    exclude_image_border: bool = True,
    excluded_vertices: Collection[int] = (),
) -> tuple[list[dict], list[dict]]:
    """The junctions of a 2D mesh with their three wedge angles, and the vertices where four or more regions meet.

    Args:
        mesh: a dw2d `DCEL_Data`, or any object with `v` (`(V, 2)`) and `e` (`(E, 4)`), or a
            `(vertices, edges)` pair (module docstring).
        angle_reading: `"arc"` (default), `"chord"` or `"fitted_tangent"` (module docstring).
        exclude_image_border: for `"arc"`, interfaces stop at the image border and those along it are
            dropped, as in `interface_arcs`. The junctions on excluded vertices are listed all the same
            (`infer_tensions` does not use them). Keyword-only.
        excluded_vertices: for `"arc"`, further vertices where interfaces stop. Keyword-only.

    Returns:
        `(junctions, higher_order)`, sorted by vertex index. Each junction is a dict with `vertex`
        (index into the vertices), `materials` (the sorted triple) and `wedges_rad` (material ->
        angle inside it, radians; the three sum to 2 pi on a junction whose interfaces do not cross).
        Each higher-order vertex is a dict with `vertex` and `materials`. A wedge is `nan` where it
        cannot be read: a material whose boundary passes through the junction more than once (a pinch),
        or, for `"arc"`, a material pair without exactly one interface ending at the junction.
    """
    _check_angle_reading(angle_reading)
    vertices, edges = _mesh_arrays(mesh)
    excluded = _excluded(mesh, exclude_image_border, excluded_vertices) if angle_reading == "arc" else set()
    junctions, higher_order, _ = _junction_table(vertices, edges, angle_reading, excluded)
    return junctions, higher_order


# ==================================================================================================== tensions
def _rank_deficiency_under_the_gauge(physics: NDArray[np.float64]) -> int:
    """`n - 1 - rank(physics @ B)`, B an orthonormal basis of the directions that keep the mean.

    The dimension of the set of solutions the exact gauge leaves undetermined (0 when the ratios
    are unique), computed as foambryo-2d computes it: the gauge removed by one Householder
    reflector, the rank from `scipy.linalg.lstsq`'s default cut-off.
    """
    n = physics.shape[1]
    anchor = np.full(n, 1.0 / n)
    reflector = anchor / np.linalg.norm(anchor) - np.eye(n)[0]
    norm = np.linalg.norm(reflector)
    basis = np.eye(n)[:, 1:] if norm < 1e-14 else (np.eye(n) - 2 * np.outer(reflector / norm, reflector / norm))[:, 1:]
    reduced = physics @ basis
    if reduced.shape[1] == 0:
        return 0
    _, _, rank, _ = linalg.lstsq(reduced, np.zeros(reduced.shape[0]))
    return int(reduced.shape[1] - rank)


def _young_dupre_columns(junction: dict) -> dict[Interface, tuple[float, float, float]]:
    """The values each interface of the junction takes in the junction's three rows (`young_dupre_entries`)."""
    a, b, c = junction["materials"]
    wedge = junction["wedges_rad"]
    cos_a, cos_b, cos_c = (float(np.cos(wedge[m])) for m in (a, b, c))
    return {(a, b): (1.0, cos_b, cos_a), (b, c): (cos_b, 1.0, cos_c), (a, c): (cos_a, cos_c, 1.0)}


def _condition_of_two_columns(first: list[float], second: list[float]) -> float:
    """`sigma_max / sigma_min` of the two columns, from the 2 x 2 matrix of their dot products in closed form.

    `inf` when its determinant is not positive in floating point (the columns are parallel to rounding).
    """
    m11 = math.fsum(x * x for x in first)
    m22 = math.fsum(y * y for y in second)
    m12 = math.fsum(x * y for x, y in zip(first, second, strict=True))
    determinant = m11 * m22 - m12 * m12
    if not determinant > 0.0:
        return math.inf
    trace = m11 + m22
    largest = 0.5 * (trace + math.sqrt(max(trace * trace - 4.0 * determinant, 0.0)))
    return math.sqrt(largest / (determinant / largest))


def _refuse_ill_conditioned_lenses(used: list[dict]) -> tuple[list[dict], list[dict[str, object]]]:
    """The used junctions without those of the ill-conditioned lenses, and a record of each refused lens.

    A lens: two or more interfaces whose rows come from the same used junctions and no other. Refused when the
    block of two of their columns in those rows has a condition number above `MAX_LENS_CONDITION`; its junctions
    are removed, and the search is repeated on the remaining junctions. The order of the junctions kept is
    unchanged, so nothing changes where nothing is refused.
    """
    kept = list(used)
    refused: list[dict[str, object]] = []
    while True:
        columns = [_young_dupre_columns(junction) for junction in kept]
        support: dict[Interface, list[int]] = defaultdict(list)
        for index, values in enumerate(columns):
            for pair in values:
                support[pair].append(index)
        by_junctions: dict[tuple[int, ...], list[Interface]] = defaultdict(list)
        for pair, indices in support.items():
            by_junctions[tuple(indices)].append(pair)
        drop: set[int] = set()
        for indices, pairs in sorted(by_junctions.items()):
            if len(pairs) < 2:
                continue
            pairs = sorted(pairs)
            blocks = {pair: [value for index in indices for value in columns[index][pair]] for pair in pairs}
            condition = max(
                _condition_of_two_columns(blocks[p], blocks[q]) for i, p in enumerate(pairs) for q in pairs[i + 1 :]
            )
            if condition > MAX_LENS_CONDITION:
                refused.append(
                    {
                        "interfaces": tuple(pairs),
                        "junction_vertices": tuple(int(kept[index]["vertex"]) for index in indices),
                        "condition_number": float(condition),
                    },
                )
                drop |= set(indices)
        if not drop:
            return kept, refused
        kept = [junction for index, junction in enumerate(kept) if index not in drop]


def _warn_of_refused_lenses(refused: list[dict[str, object]]) -> None:
    if not refused:
        return
    shown = "; ".join(
        f"{' and '.join(map(str, record['interfaces']))} at the junction(s) of vertex "
        f"{', '.join(map(str, record['junction_vertices']))} (condition number {record['condition_number']:.3g})"
        for record in refused
    )
    message = (
        f"{len(refused)} lens(es) refused: interfaces whose rows come from the same junctions only, and whose angles "
        f"there cannot separate their tensions: {shown}. Their tensions are nan and those junctions give no row. The "
        "usual cause is a small lens (a cell with only two sides) whose two sides are read by their chords, which "
        f"coincide; refused when the condition number of their rows exceeds {MAX_LENS_CONDITION:g}."
    )
    warnings.warn(message, IllConditionedLensWarning, stacklevel=4)


def _component_records(
    matrix: sparse.csr_matrix,
    column: dict[Interface, int],
    partition: JunctionGraphPartition,
    used: list[dict],
    solution: ComponentTensionSolution,
    mean_tension: float,
) -> list[dict[str, object]]:
    """One record per junction-graph component, largest first: its size, its gauge and its solve."""
    junction_component = [partition.component_of[(j["materials"][0], j["materials"][1])] for j in used]
    records: list[dict[str, object]] = []
    for index, members in enumerate(partition.components):
        cols = [column[pair] for pair in members]
        rows = [3 * i + k for i, c in enumerate(junction_component) if c == index for k in range(3)]
        rank_deficiency = None
        if not linear_solves.use_sparse_solver(len(cols)):
            rank_deficiency = _rank_deficiency_under_the_gauge(matrix[rows][:, cols].toarray())
        records.append(
            {
                "index": index,
                "n_interfaces": len(members),
                "n_junctions": len(rows) // 3,
                "n_rows": len(rows),
                "gauge": solution.gauge_notes.get(index, ""),
                "mean_tension": float(mean_tension),
                "rank_deficiency": rank_deficiency,
                "solver": solution.solver_notes.get(index, {}),
            },
        )
    return records


def _tension_solution(
    mesh: object,
    mean_tension: float,
    mode: TensionComputationMethod,
    angle_reading: str,
    exclude_image_border: bool,
    excluded_vertices: Collection[int],
    collected: SolveDiagnostics | None,
) -> tuple[ComponentTensionSolution, TwoDimensionalReport | None, JunctionGraphPartition]:
    """The gauged per-component solve and its report, `None` without diagnostics (shared by both entry points).

    `collected` is the `SolveDiagnostics` that `linear_solves.solve_options` fills during the solves.
    """
    if mode is not TensionComputationMethod.YoungDupre:
        message = (
            f"mode {mode!r} is not offered in 2D: foambryo offers in 2D only the methods whose rows "
            "foambryo's shared builders write for both dimensions, i.e. TensionComputationMethod.YoungDupre"
        )
        raise ValueError(message)
    _check_angle_reading(angle_reading)
    vertices, edges = _mesh_arrays(mesh)
    interfaces = _interfaces(edges)
    materials_at: dict[int, set[int]] = defaultdict(set)
    for v1, v2, m1, m2 in edges.tolist():
        materials_at[v1] |= {m1, m2}
        materials_at[v2] |= {m1, m2}
    if all(len(materials) < 3 for materials in materials_at.values()):
        raise NoJunctionError(interfaces)
    excluded = _excluded(mesh, exclude_image_border, excluded_vertices)
    junctions, higher_order, arcs = _junction_table(vertices, edges, angle_reading, excluded)
    used, n_excluded, n_nonfinite = [], 0, 0
    for junction in junctions:
        if junction["vertex"] in excluded:
            n_excluded += 1
        elif not all(np.isfinite(junction["wedges_rad"][m]) for m in junction["materials"]):
            n_nonfinite += 1
        else:
            used.append(junction)
    n_considered = len(used)
    used, refused = _refuse_ill_conditioned_lenses(used)
    _warn_of_refused_lenses(refused)
    partition = _partition_over(interfaces, [j["materials"] for j in used], {})
    n_rows = 3 * len(used)
    components: list[dict[str, object]] = []
    if partition.components:
        column = {pair: i for i, pair in enumerate(interfaces)}
        entries = []
        for junction in used:
            a, b, c = junction["materials"]
            wedge = junction["wedges_rad"]
            entries.append((column[(a, b)], column[(b, c)], column[(a, c)], wedge[a], wedge[b], wedge[c], 1.0))
        rows, cols, values = young_dupre_entries(entries)
        n = len(interfaces)
        matrix = sparse.coo_matrix(
            (
                np.concatenate([values, np.ones(n)]),
                (np.concatenate([rows, np.full(n, n_rows)]), np.concatenate([cols, np.arange(n)])),
            ),
            shape=(n_rows + 1, n),
        ).tocsr()
        vector = np.zeros(n_rows + 1)
        vector[-1] = n * mean_tension
        solution = solve_gauged_least_squares(
            matrix,
            vector,
            interfaces,
            mean_tension,
            partition,
            system="2D Young-Dupre tension system",
        )
        components = _component_records(matrix, column, partition, used, solution, mean_tension)
    else:
        solution = ComponentTensionSolution(
            tensions=np.full(len(interfaces), np.nan),
            dict_tensions=dict.fromkeys(interfaces, float("nan")),
            component_of=dict.fromkeys(interfaces),
            gauge_notes={},
            undetermined=tuple(interfaces),
            lstsq_residuals=np.zeros(0),
            fragmented=bool(interfaces),
            n_components=0,
        )
    if collected is None:
        return solution, None, partition
    report = TwoDimensionalReport(
        angle_reading=angle_reading,
        image_border_from=_image_border_from(mesh, exclude_image_border),
        n_junctions_used=len(used),
        n_junctions_excluded=n_excluded,
        n_junctions_nonfinite=n_nonfinite,
        n_higher_order_vertices=len(higher_order),
        n_rows=n_rows,
        n_components=partition.n_components,
        n_interfaces_undetermined=int(sum(not np.isfinite(t) for t in solution.dict_tensions.values())),
        component_of=dict(solution.component_of),
        components=tuple(components),
        n_interfaces_by_fit=None
        if arcs is None
        else {f: sum(arc.fit == f for arc in arcs) for f in ("circle", "line", "chord")},
        solves=collected,
        n_junctions_refused=n_considered - len(used),
        refused_lenses=tuple(refused),
    )
    return solution, report, partition


def _warn_if_fragmented(partition: JunctionGraphPartition, n_undetermined: int) -> None:
    if partition.n_components <= 1 and n_undetermined == 0:
        return
    message = (
        f"this 2D mesh's junction graph has {partition.n_components} component(s) and {n_undetermined} "
        "undetermined interface(s) (touched by no used junction: a closed loop, an interface cut by the image "
        "border, one whose junctions were excluded, or one of a refused lens); they "
        "are nan. Tensions of different components are on "
        "independent scales and must not be compared or averaged; use infer_tensions_by_junction_component to "
        "see which interface belongs to which component."
    )
    warnings.warn(message, JunctionGraphGaugeWarning, stacklevel=3)


def infer_tensions(
    mesh: object,
    mean_tension: float = 1.0,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    angle_reading: str = DEFAULT_ANGLE_READING,
    exclude_image_border: bool = True,
    excluded_vertices: Collection[int] = (),
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> dict[Interface, float] | tuple[dict[Interface, float], TwoDimensionalReport]:
    """Relative tensions of a 2D mesh from per-junction Young-Dupré rows, each junction-graph component gauged alone.

    Args:
        mesh: a dw2d `DCEL_Data`, an object with `v` and `e`, or a `(vertices, edges)` pair.
        mean_tension: the gauge of each component: the mean of its tensions (only ratios within a
            component are physical).
        mode: the method. Only `TensionComputationMethod.YoungDupre` is offered: it is the one method
            whose rows foambryo's shared builder (`foambryo.tension_inference.young_dupre_entries`)
            writes for both dimensions; any other raises `ValueError`.
        angle_reading: `"arc"` (default), `"chord"` or `"fitted_tangent"` (module docstring). Keyword-only.
        exclude_image_border: the junctions on the image border give no row (`image_border_vertices`), and
            for `"arc"` interfaces stop there. Keyword-only.
        excluded_vertices: further vertex indices whose junctions give no row (the union with the image
            border is excluded). Keyword-only.
        solver: `"auto"` (the default), `"dense"` or `"sparse"` (`foambryo.linear_solves`). Keyword-only.
        sparse_threshold: unknowns at which `"auto"` goes sparse; `None` is
            `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS` (2 000). Keyword-only.
        diagnostics: `False` (the default), `True` or `"spectrum"`: also return a `TwoDimensionalReport`, which
            holds the solves' `foambryo.linear_solves.SolveDiagnostics` as `solves`. Keyword-only.

    Returns:
        `dict_tensions`: every interface (sorted material pair) of the mesh mapped to its tension, or to
        `nan` where no used junction touches it. Tensions of different components are on different,
        independent scales: compare them only within a component (`infer_tensions_by_junction_component`).
        With `diagnostics`, `(dict_tensions, report)`, `report` a `TwoDimensionalReport`.

    Raises:
        NoJunctionError: if the mesh has no junction at all (the cells of the mask never meet).
        ValueError: for a `mode` other than Young-Dupré, an unknown `angle_reading`, malformed arrays or an
            invalid solver option.

    Warns:
        JunctionGraphGaugeWarning: when the junction graph has several components or undetermined interfaces.
        IllConditionedLensWarning: when a lens is refused (module docstring, "Lenses").
    """
    with linear_solves.solve_options(solver, sparse_threshold, diagnostics) as collected:
        solution, report, partition = _tension_solution(
            mesh,
            mean_tension,
            mode,
            angle_reading,
            exclude_image_border,
            excluded_vertices,
            collected,
        )
    tensions = {pair: float(value) for pair, value in solution.dict_tensions.items()}
    _warn_if_fragmented(partition, sum(not np.isfinite(t) for t in tensions.values()))
    if report is None:
        return tensions
    return tensions, report


def infer_tensions_by_junction_component(
    mesh: object,
    mean_tension: float = 1.0,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    angle_reading: str = DEFAULT_ANGLE_READING,
    exclude_image_border: bool = True,
    excluded_vertices: Collection[int] = (),
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> ComponentTensionSolution | tuple[ComponentTensionSolution, TwoDimensionalReport]:
    """`infer_tensions`, with each connected piece of the junction graph made explicit, as in 3D.

    Same arguments and report as `infer_tensions`; emits no `JunctionGraphGaugeWarning`, since the result
    carries what the warning would say.

    Returns:
        foambryo.junction_graph.ComponentTensionSolution: `dict_tensions` (interface -> tension, `nan` if
        undetermined), `component_of` (interface -> component index, `None` if undetermined), `gauge_notes`
        (the gauge applied to each component), `undetermined`, `n_components`, `solver_notes`. With
        `diagnostics`, `(solution, report)`, `report` a `TwoDimensionalReport`. Pass the solution to
        `infer_pressures` to solve the pressures of each component on its own scale.
    """
    with linear_solves.solve_options(solver, sparse_threshold, diagnostics) as collected:
        solution, report, _ = _tension_solution(
            mesh,
            mean_tension,
            mode,
            angle_reading,
            exclude_image_border,
            excluded_vertices,
            collected,
        )
    if report is None:
        return solution
    return solution, report


# ==================================================================================================== pressures
def _laplace_weights(arcs: list[InterfaceArc], junction_arcs: list[tuple[int, int, int]]) -> tuple[NDArray, float]:
    """Each interface's Laplace-row weight `median_j(sigma_J) / sigma_kappa` (0 without a curvature).

    A junction row's residual is `sum_k gamma_k delta_theta_k` over its three interfaces: per projected row its
    variance is `sigma_J^2 = (1/2) sum_k sigma_theta_k^2` (tensions near the gauge, 1); a Laplace row's residual
    is `gamma delta_kappa`, of standard deviation `sigma_kappa`. Weighting each Laplace row by
    `median_j(sigma_J) / sigma_kappa` gives it the noise of a typical junction row. The vertex noise cancels.
    """
    sigma_j = [np.sqrt(0.5 * sum(arcs[i].tangent_noise ** 2 for i in three)) for three in junction_arcs]
    reference = float(np.median(sigma_j)) if sigma_j else 1.0
    weights = np.array(
        [
            reference / arc.curvature_noise if np.isfinite(arc.curvature_noise) and arc.curvature_noise > 0 else 0.0
            for arc in arcs
        ],
    )
    return weights, reference


def _linked_to_the_medium(pairs: list[Interface], cells: list[int]) -> set[int]:
    parent = {m: m for m in [*cells, MEDIUM]}

    def find(m: int) -> int:
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m

    for a, b in pairs:
        parent[find(a)] = find(b)
    return {m for m in cells if find(m) == find(MEDIUM)}


def infer_pressures(  # noqa: C901 -- one entry point for a plain map and for a per-component solution
    mesh: object,
    dict_tensions: dict[Interface, float] | ComponentTensionSolution,
    base_pressure: float = 0.0,
    *,
    exclude_image_border: bool = True,
    excluded_vertices: Collection[int] = (),
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> dict[int, float] | tuple[dict[int, float], TwoDimensionalPressureReport]:
    """Pressures of the cells of a 2D mesh from Laplace's law on the circles read on its interfaces.

    Every interface read by a circle or a line (`interface_arcs`) whose tension is finite gives one row,
    `w (p_left - p_right) = w gamma kappa`, weighted as `_laplace_weights` says; the medium (material 0) is
    held at `base_pressure` and the other pressures solve the rows in least squares (`foambryo.linear_solves`'
    shared solve, its columns scaled on the sparse path). Relies on the orientation rule (module
    docstring) to know which material is on which side; an interface whose edges disagree is read by the
    majority of its edges and counted in the report.

    The pressures are in the unit of the tensions divided by the pixel, relative to the medium, and on the
    scale of the tensions given. Tensions of different junction-graph components are on independent scales:

    * Given a `ComponentTensionSolution` (`infer_tensions_by_junction_component`), the rows of each component
      are solved alone, the medium held at `base_pressure` in each, so every pressure is on the scale of one
      component. A cell that the rows of two or more components determine has one pressure per component
      (in the report's `pressures_by_component`) and none that is right on a common scale: it is `nan` in the
      returned map, with a `JunctionGraphGaugeWarning`.
    * Given a plain map (`infer_tensions`), all rows are solved together, and a `JunctionGraphGaugeWarning`
      says so when the interfaces giving rows belong to several components of the junction graph (rebuilt from
      the junctions that are not excluded and whose three interfaces have finite tensions).

    Args:
        mesh: a dw2d `DCEL_Data`, an object with `v` and `e`, or a `(vertices, edges)` pair.
        dict_tensions: interface -> tension, e.g. from `infer_tensions`, or the `ComponentTensionSolution` of
            `infer_tensions_by_junction_component`. An interface with no finite tension gives no row.
        base_pressure: the medium's pressure.
        exclude_image_border: as in `interface_arcs`: the interfaces along the image border are dropped and
            interfaces stop on it; the junctions there do not set the weights. Keyword-only.
        excluded_vertices: further vertices where interfaces stop. Keyword-only.
        solver: `"auto"` (the default), `"dense"` or `"sparse"`. Keyword-only.
        sparse_threshold: unknowns (cells) at which `"auto"` goes sparse; `None` is
            `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS`. Keyword-only.
        diagnostics: also return a `TwoDimensionalPressureReport`. Keyword-only.

    Returns:
        `dict_pressures`: every material of the mesh mapped to its pressure: the medium to `base_pressure`, a
        cell linked to the medium by rows to its pressure, any other cell (no row, rows only among cells not
        linked to the medium, or rows of several components) to `nan`. With `diagnostics`,
        `(dict_pressures, report)`.

    Warns:
        UserWarning: when an interface's edges disagree on which material is on their left (a mesh that does
            not follow the orientation rule edge by edge, though its cells' boundaries passed the measure on input).
        JunctionGraphGaugeWarning: as said above, when the tensions come from several components.
    """
    vertices, edges = _mesh_arrays(mesh)
    excluded = _excluded(mesh, exclude_image_border, excluded_vertices)
    arcs = _arcs(vertices, edges, excluded)
    against = sum(arc.n_edges_against_the_majority for arc in arcs)
    if against:
        warnings.warn(
            f"{against} edge(s) put the other material on the left of their interface: the mesh does not follow "
            "the orientation rule on every edge (m1 on the left of v1 -> v2 in (row, column)); each interface "
            "is read by the majority of its edges",
            UserWarning,
            stacklevel=2,
        )
    per_component = isinstance(dict_tensions, ComponentTensionSolution)
    given = dict_tensions.dict_tensions if isinstance(dict_tensions, ComponentTensionSolution) else dict_tensions
    tensions = {(min(int(a), int(b)), max(int(a), int(b))): float(v) for (a, b), v in given.items()}
    triples = _junction_arc_triples(edges, arcs, excluded)
    weights, reference = _laplace_weights(arcs, triples)
    rows = [(arc, float(w)) for arc, w in zip(arcs, weights, strict=True) if w > 0 and np.isfinite(arc.curvature)]
    without_tension = sum(not np.isfinite(tensions.get(arc.materials, np.nan)) for arc, _ in rows)
    rows = [(arc, w) for arc, w in rows if np.isfinite(tensions.get(arc.materials, np.nan))]
    materials = sorted({int(m) for m in np.unique(edges[:, 2:4])})
    pressures = {m: float("nan") for m in materials}
    if MEDIUM in pressures:
        pressures[MEDIUM] = float(base_pressure)
    component_of_cell: dict[int, int | None] = {}
    by_component: dict[int, dict[int, float]] = {}
    n_rows, record, n_several = 0, {}, 0
    with linear_solves.solve_options(solver, sparse_threshold, diagnostics) as collected:
        if isinstance(dict_tensions, ComponentTensionSolution):
            groups: dict[int, list[tuple[InterfaceArc, float]]] = defaultdict(list)
            for arc, w in rows:
                component = dict_tensions.component_of.get(arc.materials)
                if component is not None:
                    groups[int(component)].append((arc, w))
            for component in sorted(groups):
                solved, n, record = _solve_laplace_rows(groups[component], tensions)
                n_rows += n
                by_component[component] = {m: float(base_pressure + p) for m, p in solved.items()}
            owners: dict[int, list[int]] = defaultdict(list)
            for component, solved_cells in by_component.items():
                for m in solved_cells:
                    owners[m].append(component)
            for m in materials:
                if m == MEDIUM:
                    continue
                if len(owners.get(m, [])) == 1:
                    component_of_cell[m] = owners[m][0]
                    pressures[m] = by_component[owners[m][0]][m]
                else:
                    component_of_cell[m] = None
            n_several = sum(len(c) > 1 for c in owners.values())
        else:
            solved, n_rows, record = _solve_laplace_rows(rows, tensions)
            pressures.update({m: float(base_pressure + p) for m, p in solved.items()})
    if n_several:
        warnings.warn(
            f"{n_several} cell(s) are determined by the Laplace rows of several junction-graph components, whose "
            "tensions are on independent scales: their pressures are nan (each component's are in the report's "
            "pressures_by_component, with diagnostics=True)",
            JunctionGraphGaugeWarning,
            stacklevel=2,
        )
    if not per_component and _components_of_the_rows(edges, excluded, tensions, [arc for arc, _ in rows]) > 1:
        warnings.warn(
            "the interfaces that give Laplace rows belong to several components of the junction graph, whose tensions "
            "are on independent scales; their rows were solved together. Pass the ComponentTensionSolution of "
            "infer_tensions_by_junction_component to solve each component on its own scale.",
            JunctionGraphGaugeWarning,
            stacklevel=2,
        )
    if collected is None:
        return pressures
    n_determined = sum(1 for m in materials if m != MEDIUM and np.isfinite(pressures[m]))
    report = TwoDimensionalPressureReport(
        image_border_from=_image_border_from(mesh, exclude_image_border),
        per_component=per_component,
        n_interfaces=len(arcs),
        n_rows=n_rows,
        n_rows_without_tension=int(without_tension),
        n_interfaces_without_curvature=int(sum(not np.isfinite(arc.curvature) for arc in arcs)),
        n_cells_determined=n_determined,
        n_cells_undetermined=int(sum(1 for m in materials if m != MEDIUM) - n_determined),
        n_cells_in_several_components=int(n_several),
        n_edges_against_the_majority=int(against),
        junction_noise_reference=reference,
        solver=record,
        solves=collected,
        component_of_cell=component_of_cell,
        pressures_by_component=by_component,
    )
    return pressures, report


def _components_of_the_rows(
    edges: NDArray[np.int64],
    excluded: set[int],
    tensions: dict[Interface, float],
    row_arcs: list[InterfaceArc],
) -> int:
    """How many junction-graph components the row interfaces span, the graph rebuilt from the mesh.

    Interfaces are linked through the junctions (vertices of exactly three materials) that are not excluded and
    whose three interfaces have finite tensions: the junctions a tension solve could have used.
    """
    materials_at: dict[int, set[int]] = defaultdict(set)
    for v1, v2, a, b in edges.tolist():
        materials_at[v1] |= {a, b}
        materials_at[v2] |= {a, b}
    junctions = []
    for vertex, at in materials_at.items():
        if len(at) != 3 or vertex in excluded:
            continue
        a, b, c = sorted(at)
        if all(np.isfinite(tensions.get(pair, np.nan)) for pair in ((a, b), (a, c), (b, c))):
            junctions.append((a, b, c))
    row_interfaces = sorted({arc.materials for arc in row_arcs})
    if not row_interfaces:
        return 0
    partition = _partition_over(sorted(set(row_interfaces) | set(_interfaces(edges))), junctions, {})
    return len({partition.component_of.get(pair, ("isolated", pair)) for pair in row_interfaces})


def _junction_arc_triples(
    edges: NDArray[np.int64],
    arcs: list[InterfaceArc],
    excluded: set[int],
) -> list[tuple[int, int, int]]:
    """The junctions the arc reading uses (three materials, not excluded, one interface end per pair): arc indices."""
    ends: dict[int, dict[Interface, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, arc in enumerate(arcs):
        if not arc.closed:
            ends[arc.vertices[0]][arc.materials].append(i)
            ends[arc.vertices[-1]][arc.materials].append(i)
    materials_at: dict[int, set[int]] = defaultdict(set)
    for v1, v2, a, b in edges.tolist():
        materials_at[v1] |= {a, b}
        materials_at[v2] |= {a, b}
    triples = []
    for vertex in sorted(materials_at):
        if len(materials_at[vertex]) != 3 or vertex in excluded:
            continue
        a, b, c = sorted(materials_at[vertex])
        at = ends.get(vertex, {})
        pairs = ((a, b), (a, c), (b, c))
        if all(len(at.get(pair, [])) == 1 for pair in pairs):
            triples.append((at[pairs[0]][0], at[pairs[1]][0], at[pairs[2]][0]))
    return triples


def _solve_laplace_rows(
    rows: list[tuple[InterfaceArc, float]],
    tensions: dict[Interface, float],
) -> tuple[dict[int, float], int, dict[str, object]]:
    """Pressures relative to the medium of the cells linked to it by `rows`, the rows used, and the solve record."""
    cells = sorted({m for arc, _ in rows for m in arc.materials if m != MEDIUM})
    column = {m: i for i, m in enumerate(sorted(_linked_to_the_medium([arc.materials for arc, _ in rows], cells)))}
    system_rows = [(arc, w) for arc, w in rows if all(m == MEDIUM or m in column for m in arc.materials)]
    if not column:
        return {}, 0, {}
    r, c, v = [], [], []
    rhs = np.zeros(len(system_rows))
    for k, (arc, w) in enumerate(system_rows):
        for material, sign in ((arc.left, 1.0), (arc.right, -1.0)):
            if material != MEDIUM:
                r.append(k)
                c.append(column[material])
                v.append(sign * w)
        rhs[k] = w * tensions[arc.materials] * arc.curvature
    matrix = sparse.csr_matrix((v, (r, c)), shape=(len(system_rows), len(column)))
    x, _, record = _solve_least_squares(matrix, rhs, system="2D Laplace pressure system", scale_columns=True)
    return {m: float(x[i]) for m, i in column.items()}, len(system_rows), record


# ==================================================================================================== arc tiling
#: Largest number of residual evaluations of `infer_forces_by_equilibrium_tiling`'s fit (the setting chosen on development data).
EQUILIBRIUM_TILING_MAX_EVALUATIONS = 200


@dataclass(frozen=True)
class EquilibriumTilingReport:
    """What `infer_forces_by_equilibrium_tiling` fitted, checked and refused.

    Attributes:
        refused: interface -> the reason it has no tension (`nan`).
        pressure_refused: cell -> the reason it has no pressure (`nan`).
        converged: whether every block's fit met its stopping rule within `max_evaluations`.
        valid: whether the fitted tiling reproduces the data's junctions and gives every cell one bounded, connected
            region (the check of `_global_fit_2d.fit_tiling`); refusals are local, so a tiling can be invalid and
            still give tensions where it agrees with the data.
        validity: the check's details (missing and extra junctions, empty, disconnected and unbounded cells).
        blocks: per block (`component_of`'s index): its interfaces, its reference material, its fit's status.
        n_points: the crack points fitted.
        seconds: wall time of each stage.
    """

    refused: dict[Interface, str]
    pressure_refused: dict[int, str]
    converged: bool
    valid: bool
    validity: dict
    blocks: list[dict]
    n_points: int
    seconds: dict[str, float]


def _arc_tiling_readings(mesh: object) -> dict:
    """The 0.8.0 reading of every interface of `mesh` as the global fit's `LocalReading`s (circle or line, tension)."""
    from foambryo import _global_fit_2d as gf

    vertices, _ = _mesh_arrays(mesh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensions = infer_tensions_by_junction_component(mesh).dict_tensions
    out: dict = {}
    for arc in interface_arcs(mesh):
        t = np.asarray(arc.tangent_start, float)
        if arc.fit != "circle":
            t = vertices[arc.vertices[-1]] - vertices[arc.vertices[0]] if arc.fit == "chord" else t
            t = t / np.linalg.norm(t)
        out.setdefault(arc.materials, []).append(
            gf.LocalReading(
                left=int(arc.left),
                right=int(arc.right),
                curvature=float(arc.curvature) if arc.fit == "circle" else 0.0,
                centre=tuple(arc.centre) if arc.fit == "circle" else None,
                point=tuple(vertices[arc.vertices[0]]),
                normal=(-float(t[1]), float(t[0])),
                tension=float(tensions.get(arc.materials, math.nan)),
            ),
        )
    return out


def infer_forces_by_equilibrium_tiling(
    mesh: object,
    labels: NDArray[np.integer],
    mean_tension: float = 1.0,
    *,
    max_evaluations: int = EQUILIBRIUM_TILING_MAX_EVALUATIONS,
    diagnostics: bool = False,
) -> (
    tuple[ComponentTensionSolution, dict[int, float]]
    | tuple[ComponentTensionSolution, dict[int, float], EquilibriumTilingReport]
):
    r"""Tensions and pressures by a global fit of an equilibrium tiling of circular arcs (provisional, opt-in).

    The label image's crack points (the middle of every side shared by two 4-adjacent pixels of different labels) are
    fitted by a Möbius tiling (Noll N. et al., Phys. Rev. X 10, 011072, 2020, doi:10.1103/PhysRevX.10.011072): every
    material carries `f(x) = A |x|^2 + B . x + C`, every interface is the circle where two `f` are equal, Young's and
    Laplace's laws hold exactly, and the fit minimises the squared distance of the points to their circles
    (Levenberg-Marquardt, sparse analytic Jacobian). Start: Noll's algebraic problem with each interface's residual
    divided by the tension of foambryo's arc reading of `mesh` (`infer_tensions_by_junction_component` at its defaults)
    and one anchor per interface, so that the start is measured in pixels; this is what makes large tissues and crops
    converge to the true tiling. Tensions share a scale only within a block (biconnected component) of the
    graph of materials linked by interfaces: each block is fitted alone and gauged to `mean(tension) = mean_tension`.

    Named after what it returns, tensions and pressures in one call, like `foambryo.infer_forces`; the mesh comes first,
    as in every `infer_*` function (`infer_by_arc_tiling(labels, mesh)` before 0.8.0, never released).

    Args:
        mesh: the 2-D mesh of `labels` (a `foambryo.PlanarMesh`, dw2d's `DCEL_Data`, ...), for the start. A mesh in
            physical units (`PlanarMesh.spacing`) is read in pixels, and the pressures are returned per unit of length.
        labels: the label image (`(rows, columns)` integers, 0 the medium) that `mesh` was built from.
        mean_tension: each block's mean tension.
        max_evaluations: the fit's largest number of residual evaluations.
        diagnostics: also return an `EquilibriumTilingReport`.

    Returns:
        `(solution, pressures)`, or `(solution, pressures, report)` with `diagnostics`. `solution` is a
        `ComponentTensionSolution` (interface -> tension; `component_of` the block; `nan` where refused, with the
        reason in `report.refused`). `pressures`: cell -> pressure relative to the medium, on the scale of its block's
        tensions, per pixel; `nan` for a cell of a block without the medium or one the check refuses (reason in
        `report.pressure_refused`); the medium is 0.

    Raises:
        ValueError: when `labels` is not the image `mesh` was built from, as far as it can be checked: not 2-D, a shape
            other than the mesh's `image_shape` (when the mesh records it), or a cell of the mesh absent from `labels`.

    Refusals (local, never the whole image): an interface with fewer than 3 crack points; an interface that is a block
    by itself; an interface of a cell whose fitted pressure is not above the medium's (an unbounded region), or that is
    empty in the fitted tiling; an interface of the data that the fitted tiling does not have in the image.

    Measured on synthetic equilibrium foams with known tensions and pressures (307 held-out tissues of 4 to 64 cells,
    exact masks, crops, smooth boundary noise of 0.5 to 2 px, the 0.75 px measured on real 2D masks, Cellpose
    segmentations of rendered images, and images whose membranes are displaced 0.75 px off equilibrium; cells of 10 to
    40 px radius), against the arc reading of `infer_tensions_by_junction_component`: the median relative tension error
    is lower in every tier and scale, by 0.03 to 0.11 (95 % intervals below 0), except at 2 px of noise and 20 px radius
    (-0.005, interval -0.014 to +0.005); on a 4 096-cell tissue (10 px radius) 0.077 against 0.156, every interface
    kept, in 74 s and 0.6 GiB. It refuses whole images where the arc reading refuses none: 11 of 453 crops (the fit
    slides along the one direction of the cost that a cut tissue barely determines, the ratio of the cell-medium to
    the cell-cell tensions), so it is offered beside the arc reading, not as the default.
    """
    import time

    from foambryo import _global_fit_2d as gf

    labels = np.asarray(labels)
    _check_the_label_image_of(mesh, labels)
    spacing = getattr(mesh, "spacing", None)
    if spacing is not None:
        from foambryo.planar_mesh import PlanarMesh

        planar = PlanarMesh.from_any(mesh)
        mesh = PlanarMesh(
            planar.points / spacing, planar.segments, planar.labels, image_border_vertices=planar.image_border_vertices,
        )
    seconds = {}
    t0 = time.perf_counter()
    data = gf.data_from_label_image(labels)
    readings = _arc_tiling_readings(mesh)
    seconds["data and reading"] = time.perf_counter() - t0
    res = gf.fit_tiling(
        data,
        starts=("anchored",),
        max_evaluations=max_evaluations,
        readings=readings,
        validity="pixels",
    )
    seconds.update({f"fit: {k}": v for k, v in res.report["seconds"].items()})
    keys = sorted(res.tensions)
    dict_tensions = {p: float(res.tensions[p]) * mean_tension for p in keys}
    component_of = {p: (res.report["component_of"][p] if math.isfinite(dict_tensions[p]) else None) for p in keys}
    blocks = res.report["blocks"]
    n_blocks = res.report["n_blocks"]
    solution = ComponentTensionSolution(
        tensions=np.array([dict_tensions[p] for p in keys]),
        dict_tensions=dict_tensions,
        component_of=component_of,
        gauge_notes={b["block"]: f"mean(tension) = {mean_tension:g} over the block's finite tensions" for b in blocks},
        undetermined=tuple(p for p in keys if not math.isfinite(dict_tensions[p])),
        lstsq_residuals=np.array([b["final_cost"] for b in blocks]),
        fragmented=n_blocks > 1,
        n_components=n_blocks,
        solver_notes={
            b["block"]: dict(solver="Levenberg-Marquardt", **{k: v for k, v in b.items() if k != "block"})
            for b in blocks
        },
    )
    per_unit = 1.0 if spacing is None else 1.0 / spacing
    pressures = {
        int(m): (float(x) * mean_tension * per_unit if math.isfinite(x) else math.nan) for m, x in res.pressures.items()
    }
    if not diagnostics:
        return solution, pressures
    cells = res.report["validity"]["cells"]
    pressure_refused = {}
    for m, x in pressures.items():
        if m == MEDIUM or math.isfinite(x):
            continue
        if m in cells["unbounded"]:
            pressure_refused[m] = "unbounded in the fit (pressure not above the medium's)"
        elif m in cells["empty"]:
            pressure_refused[m] = "empty in the fitted tiling"
        else:
            pressure_refused[m] = "not in a block with the medium (no pressure relative to it)"
    report = EquilibriumTilingReport(
        refused=dict(res.refused),
        pressure_refused=pressure_refused,
        converged=bool(res.report["converged"]),
        valid=bool(res.report["validity"]["valid"]),
        validity=res.report["validity"],
        blocks=blocks,
        n_points=int(res.report["n_points"]),
        seconds={**seconds, "total": time.perf_counter() - t0},
    )
    return solution, pressures, report


def _check_the_label_image_of(mesh: object, labels: NDArray[np.integer]) -> None:
    """Refuse a label image that cannot be the one `mesh` was built from (see `infer_forces_by_equilibrium_tiling`)."""
    if labels.ndim != 2:
        message = f"labels must be the 2-D label image the mesh was built from, got an array of shape {labels.shape}"
        raise ValueError(message)
    shape = getattr(mesh, "image_shape", None)
    if shape is not None and tuple(labels.shape) != tuple(shape):
        message = f"labels has shape {labels.shape}, but the mesh was built from an image of shape {tuple(shape)}"
        raise ValueError(message)
    _, edges = _mesh_arrays(mesh)
    missing = sorted(set(np.unique(edges[:, 2:]).tolist()) - set(np.unique(labels).tolist()) - {MEDIUM})
    if missing:
        message = f"cells {missing[:8]} of the mesh are absent from labels: it is not the image the mesh was built from"
        raise ValueError(message)
