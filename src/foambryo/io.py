"""The meshes every inference reads: a `DcelData` (3D) or a `PlanarMesh` (2D) from a mask, mesh arrays or a file.

Inputs can be:
- a segmentation mask (3D or 2D integer array; its `ndim` decides) ;
- a multimaterial mesh: points, triangles (or segments), labels ;
- a file (.rec, or opened by meshio) containing a multimaterial mesh; a 2-D `.rec` file is recognised by the
  `dimension` its appendix records, else by its layout (`dw3d.io.rec_dimension`).

Output is a DcelData mesh in 3D, a `foambryo.PlanarMesh` in 2D (points in the mask's `(row, column)` order, in pixels
unless an isotropic `spacing` is given). Conventions: label 0 is the exterior medium; a mask's axes are
(z, y, x) and the mesh's coordinates follow them, column 0 along axis 0; the mesh is in voxel units
unless a `spacing` gives a physical one, and the inferred pressures are in the tension unit divided
by that length unit. Mesh files keep that column order, (z, y, x): a program that reads the columns as
(x, y, z), as ParaView and MeshLab do, shows the mirror image of the tissue. foambryo's Polyscope viewer
draws (x, y, z) = the columns (2, 1, 0), the tissue as it is (`foambryo.viewing`).

Matthieu Perez 2024.
"""

import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.io import load_meshio_mesh, load_rec, rec_dimension
from numpy.typing import NDArray

from foambryo.crop_faces import warn_if_crop_faces_unrecorded
from foambryo.dcel import DcelData
from foambryo.planar_mesh import PlanarMesh

# `load_rec_appendix` only exists on dw3d >= 0.4.0's `dw3d.io`. An unguarded
# `from dw3d.io import ... load_rec_appendix` here would make the whole `foambryo`
# package fail to import against an older dw3d, since `foambryo/__init__.py` imports from
# this module. Guard just this one name instead of reimplementing or dropping the feature:
# everything else in foambryo works normally against an older dw3d, and only a call to
# `load_rec_appendix` itself fails, with a message naming the required dw3d version
# instead of a bare ImportError at package import time.
try:
    from dw3d.io import load_rec_appendix
except ImportError:

    def load_rec_appendix(*_args: object, **_kwargs: object) -> None:
        """Raise because the installed dw3d does not provide the .rec metadata appendix reader.

        This stands in for `dw3d.io.load_rec_appendix`, which is only available on
        dw3d >= 0.4.0.

        Raises:
            ImportError: always; the installed dw3d lacks `load_rec_appendix`.
        """
        msg = (
            "load_rec_appendix() requires dw3d >= 0.4.0, which is not installed. "
            "Upgrade dw3d to use this function."
        )
        raise ImportError(msg)


# Re-exported, not reimplemented -- dw3d.io is the single implementation of the .rec
# metadata appendix format. dcel_mesh_from_file reads one thing from it besides the crop
# faces: the source voxel size of a mesh in physical units (DcelData.source_spacing), which
# no foambryo geometry or inference code uses.
__all__ = [
    "MaskReconstruction",
    "dcel_mesh_from_file",
    "dcel_mesh_from_multimaterial_mesh",
    "dcel_mesh_from_segmentation_mask",
    "default_min_distance",
    "load_rec_appendix",
    "reconstruct_mask",
]


def default_min_distance(dimension: int = 3) -> int:
    """The reconstruction's point spacing when none is given (`min_distance=None`): delaunay-watershed's own defaults.

    Args:
        dimension: 3 for a 3-D mask, 2 for a 2-D one.

    Returns:
        int: 3 voxels in 3D (dw3d's), 4 pixels in 2D (dw2d's).
    """
    if dimension == 2:
        from dw2d import DEFAULT_MIN_DISTANCE as two_dimensional_default

        return int(two_dimensional_default)
    from dw3d.reconstruction_algorithm_factory import DEFAULT_MIN_DISTANCE as three_dimensional_default

    return int(three_dimensional_default)


@dataclass(frozen=True)
class MaskReconstruction:
    """A mask's mesh as `foambryo mesh` reconstructs it, held as plain arrays and the bytes of its `.rec` file.

    Returned by `reconstruct_mask`. It holds no reconstruction algorithm, so it pickles to the size of the mesh
    (a front end can reconstruct in a child process and send it back), and it writes the `.rec` file `foambryo
    mesh` writes, byte for byte, because it holds the bytes delaunay-watershed-3d's own export wrote.

    Attributes:
        points: the mesh's points, in the unit of `spacing`, axis order (z, y, x).
        triangles: the triangles, as the reconstruction returned them; in 2D the segments `(m, 2)`.
        labels: the two materials of each triangle.
        crop_faces: `True` on the crop faces, aligned with `triangles`, or `None` when the reconstruction
            records none.
        spacing: the voxel size the mask was reconstructed with, `(dz, dy, dx)`; in 2D the pixel size `(dy, dx)`.
        spacing_unit: its unit.
        min_distance: the reconstruction's `min_distance`, in cubic voxels of the shortest spacing.
        rec_bytes: the `.rec` file (crop faces as its last triangle rows, metadata appendix included).
        n_cells: the labels that are cells (neither the exterior 0 nor an enclosed medium) that the mesh holds.
        n_cut_cells: the cells the image boundary cuts.
        media: the labels that are media: 0, then the enclosed media (lumens) the reconstruction made its own
            regions (delaunay-watershed-3d 0.7.1 and later; `(0,)` with an earlier one).
        unresolved: labels of the mask absent from the mesh, because no tetrahedron of the reconstruction lay in
            their voxels (delaunay-watershed-3d 0.7.1 and later), with their voxel counts (pixel counts in 2D).
        dimension: 3, or 2 for a 2-D mask (meshed by delaunay-watershed's dw2d).
        image_border_vertices: 2D only: the vertices on the image border (dw2d's mark).
        image_shape: 2D only: the shape of the label image.
    """

    points: NDArray[np.float64]
    triangles: NDArray[np.int64]
    labels: NDArray[np.int64]
    crop_faces: NDArray[np.bool_] | None
    spacing: tuple[float, float, float]
    spacing_unit: str
    min_distance: int
    rec_bytes: bytes = field(repr=False)
    n_cells: int
    n_cut_cells: int
    media: tuple[int, ...] = (0,)
    unresolved: dict[int, int] = field(default_factory=dict)
    dimension: int = 3
    image_border_vertices: NDArray[np.int64] | None = None
    image_shape: tuple[int, ...] | None = None

    def mesh(self) -> "DcelData | PlanarMesh":
        """The mesh for inference, as `dcel_mesh_from_segmentation_mask(mask, spacing=...)` builds it.

        Built from copies of the arrays, so the reconstruction is unchanged by any use of the mesh. In 2D
        (`dimension == 2`) a `PlanarMesh`, whose `triangles` are the segments.
        """
        if self.dimension == 2:
            return PlanarMesh(
                self.points.copy(), self.triangles.copy(), self.labels.copy(),
                image_border_vertices=self.image_border_vertices, image_shape=self.image_shape,
                spacing=self.spacing[0], unit=self.spacing_unit,
            )
        crop_faces = None if self.crop_faces is None else self.crop_faces.copy()
        return DcelData(self.points.copy(), self.triangles.copy(), self.labels.copy(), crop_faces=crop_faces,
                        source_spacing=self.spacing, media=self.media, unresolved=tuple(self.unresolved))

    def write_rec(self, filename: str | Path) -> None:
        """Write the `.rec` file, byte for byte what `foambryo mesh` writes for the same mask and options."""
        Path(filename).write_bytes(self.rec_bytes)

    def summary(self) -> str:
        """The counts `foambryo mesh` prints: points, triangles, cells, those cut by the image boundary, and media.

        The enclosed media (lumens) are named only when there is one, so a summary without is as before 0.7.1.
        """
        enclosed = [label for label in self.media if label != 0]
        lumens = ""
        if enclosed:
            plural = len(enclosed) != 1
            names = ", ".join(str(label) for label in enclosed)
            lumens = f"; {len(enclosed)} enclosed medi{'a' if plural else 'um'} (label{'s' if plural else ''} {names})"
        if self.unresolved:
            plural = len(self.unresolved) != 1
            names = ", ".join(f"{label}: {voxels} voxels" for label, voxels in sorted(self.unresolved.items()))
            lumens += (f"; {len(self.unresolved)} label{'s' if plural else ''} of the mask unresolved, absent from "
                       f"the mesh ({names})")
        if self.dimension == 2:
            return (f"{len(self.points)} points, {len(self.triangles)} segments, {self.n_cells} cells "
                    f"({self.n_cut_cells} cut by the image border){lumens}")
        return (f"{len(self.points)} points, {len(self.triangles)} triangles, {self.n_cells} cells "
                f"({self.n_cut_cells} cut by the image boundary){lumens}")


def reconstruct_mask(
    segmentation_mask: NDArray[np.uint],
    spacing: tuple[float, ...],
    *,
    spacing_unit: str = "um",
    min_distance: int | None = None,
    pad_thin_medium_margins: bool | None = None,
) -> MaskReconstruction:
    """Reconstruct a mask's mesh in physical units, exactly as `foambryo mesh` does.

    The call is delaunay-watershed-3d's entry point for masks with a voxel size,
    `get_default_mesh_reconstruction_algorithm(min_distance, False).construct_mesh_in_physical_units(mask, spacing,
    spacing_unit=spacing_unit, axis_order="zyx")`: a mask whose voxels are longer along one axis is resampled to
    cubic voxels of the shortest spacing, meshed, and the mesh is scaled to the unit of `spacing`; cubic voxels are
    meshed as they are. Every face of the image box is a crop. The mesh's numbers are those of
    `dcel_mesh_from_segmentation_mask(mask, min_distance, spacing=spacing)`; only the unit recorded in the `.rec`
    file depends on `spacing_unit`.

    Args:
        segmentation_mask: 3-D labels, axis 0 = z (see `foambryo.segmentation_masks.check_segmentation_mask`).
        spacing: `(dz, dy, dx)`, the voxel size along axes 0, 1, 2; one axis may be longer than the other two,
            which must be equal.
        spacing_unit: the unit of `spacing`, recorded in the `.rec` file. Keyword-only. Defaults to `"um"`, the
            command's.
        min_distance: minimum distance between the reconstruction's points, in cubic voxels of the shortest
            spacing. Keyword-only. Defaults to None: 3 voxels (delaunay-watershed-3d's), 4 pixels for a 2-D mask
            (dw2d's).
        pad_thin_medium_margins: whether delaunay-watershed pads with medium the faces of the image box (sides of
            the image in 2D) that no cell reaches when their margin of medium is thin, so that a cell a few voxels
            from such a face does not take the medium between them (`construct_mesh_in_physical_units`'
            keyword, `foambryo mesh --pad-thin-medium-margins` / `--no-pad-thin-medium-margins`). Keyword-only.
            Defaults to None, delaunay-watershed's own default: on for cubic voxels and square pixels, off for a
            mask it resamples (voxels longer along one axis), where the padding made its watershed lose cells more
            often.

    A 2-D mask (`segmentation_mask.ndim == 2`) is meshed by dw2d through the same call (delaunay-watershed-3d hands
    it over), with `spacing` the pixel size `(dy, dx)`, square pixels only, and `axis_order="yx"`; the result has
    `dimension == 2` and its `mesh()` is a `PlanarMesh`.

    Returns:
        MaskReconstruction: the mesh's arrays, its `.rec` bytes and its counts.

    Raises:
        ValueError: on a spacing the resampling cannot handle, or a mask the reconstruction refuses; or if the
            installed delaunay-watershed-3d has no `construct_mesh_in_physical_units`.
    """
    algorithm = get_default_mesh_reconstruction_algorithm(min_distance, False)
    if not hasattr(algorithm, "construct_mesh_in_physical_units"):
        message = ("reconstruct_mask needs delaunay-watershed 0.8.0 or later, as foambryo requires: the installed "
                   "delaunay-watershed-3d has no construct_mesh_in_physical_units")
        raise ValueError(message)
    # Passed only when given, so that the call is the library's own default call otherwise.
    padding = {} if pad_thin_medium_margins is None else {"pad_thin_medium_margins": bool(pad_thin_medium_margins)}
    if np.ndim(segmentation_mask) == 2:
        return _reconstruct_a_two_dimensional_mask(algorithm, segmentation_mask, spacing, spacing_unit, **padding)
    spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    points, triangles, labels = algorithm.construct_mesh_in_physical_units(
        segmentation_mask, spacing, spacing_unit=spacing_unit, axis_order="zyx", **padding,
    )
    with tempfile.TemporaryDirectory() as folder:
        rec_path = Path(folder) / "mesh.rec"
        algorithm.save_to_rec_mesh(rec_path)
        rec_bytes = rec_path.read_bytes()
    report = algorithm.crop_face_report
    crop_faces = getattr(algorithm, "crop_faces", None)
    media = tuple(int(label) for label in getattr(algorithm, "medium_labels", (0,)))
    report_of_media = getattr(algorithm, "enclosed_medium_report", None)
    unresolved = {int(u.label): int(u.voxels) for u in getattr(report_of_media, "unresolved", ())}
    present = np.unique(np.asarray(labels))
    return MaskReconstruction(
        points=points,
        triangles=cast("NDArray[np.int64]", triangles),  # dw3d's own arrays, passed on as the library passes them
        labels=cast("NDArray[np.int64]", labels),
        crop_faces=None if crop_faces is None else np.asarray(crop_faces, dtype=bool),
        spacing=spacing,
        spacing_unit=spacing_unit,
        min_distance=int(getattr(algorithm, "placement_min_distance", None) or default_min_distance(3)
                         if min_distance is None else min_distance),
        rec_bytes=rec_bytes,
        n_cells=int(np.count_nonzero(~np.isin(present, media))),
        n_cut_cells=len(report.cut_cells) if report is not None and report.meets_the_image_box else 0,
        media=tuple(label for label in media if label in present),
        unresolved=unresolved,
    )


def dcel_mesh_from_segmentation_mask(
    segmentation_mask: NDArray[np.uint],
    min_distance: int | None = None,
    print_info: bool = False,
    *,
    physical_image_faces: Iterable[str] = (),
    spacing: tuple[float, ...] | None = None,
) -> "DcelData | PlanarMesh":
    """Obtain a DcelData mesh from a segmentation mask using Delaunay-Watershed 3D.

    The default mesh reconstruction algorithm is used (its dither seed is 42; see
    `foambryo.infer_tensions_over_seeds` for other seeds), with its junction relocation (on by
    default since dw3d 0.5.0). The mesh carries the reconstruction's **crop faces**: the triangles a cell
    has on a face of the image box that the cell reaches in the mask, where the image cut the tissue
    (see `foambryo.crop_faces`).
    Every inference method keeps them out of its equations and reports the pressures of the cells
    they cut as `nan`.

    Args:
        segmentation_mask (NDArray[np.uint]): Instance segmentation image: 3-D integer labels,
            axes (z, y, x), label 0 the medium, one label per cell. The mesh's coordinates follow
            the mask's axes, column 0 along axis 0. **A 2-D mask** (axes (y, x)) is meshed by dw2d, through
            delaunay-watershed-3d's algorithm, and returned as a `foambryo.PlanarMesh` (with dw2d's image-border
            vertices, and its points in the unit of an isotropic `spacing` when given).
        min_distance (int | None, optional): minimum distance between points of the output mesh, in voxels.
            Defaults to None: 3 in 3D, dw3d's own default (it was 5 up to foambryo 0.4.2), and 4 pixels in 2D,
            dw2d's. Pass 5 to reproduce the 0.4 spacing; at 5 or more dw3d emits `CoarseSpacingRelocationWarning`
            once per reconstruction.
        print_info (bool, optional): print mesh construction details. Defaults to False.
        physical_image_faces (Iterable[str], optional): faces of the image box where the tissue
            really ends, for example on a coverslip: `"axis0_low"` (index 0 along the mask's first
            axis), `"axis0_high"`, ... `"axis2_high"`. Cell surfaces on them are contact with the
            medium. Keyword-only. Defaults to none: every image face is a crop.
        spacing (tuple[float, float, float] | None, optional): the voxel size along the mask's
            axes 0, 1, 2, in any one unit; one axis (a z step) may be longer than the other two,
            which must be equal. Keyword-only. Defaults to `None`: the mesh is reconstructed on
            the voxel grid and returned in voxel units.
            Given, the call is delaunay-watershed-3d's `construct_mesh_in_physical_units`: the mask
            is **resampled to cubic voxels** of the shortest spacing (shape-based interpolation),
            meshed with the same settings (`min_distance` then counts cubic voxels), and the mesh
            is returned in the unit of `spacing`, with `DcelData.source_spacing` set. Measured
            on 12 simulated tissues of 90 and 165 cells at a real cell size, down-sampled along z,
            against the same tissues meshed at full resolution: resampling first costs +0.83, +0.71
            and +1.17 degrees of contact-angle error at an anisotropy of 5/3, 2 and 3, against
            +1.38, +1.97 and +4.10 for meshing the native grid and scaling the mesh (dw3d 0.7.0's
            meshes, before 0.8.0 pads thin margins of medium on the native grid; not measured again). Above 3
            nothing was measured at that cell size, and delaunay-watershed-3d warns
            (`UnmeasuredAnisotropyWarning`); the mesh is built anyway. The cost is that of a mask
            acquired at the short spacing: the anisotropy times the voxels.

    Returns:
        DcelData: The mesh reconstructed from the segmentation and ready for force inference, in
            voxel units without `spacing` and in the spacing's unit with it.

    Raises:
        ValueError: if `physical_image_faces` names an unknown face, or is given to a dw3d that
            does not record crop faces; if `spacing` is not three positive finite numbers with
            at most one longer axis, or is given to a dw3d without
            `construct_mesh_in_physical_units`.

    Warns:
        UserWarning: the reconstruction's own warnings, from delaunay-watershed-3d (for example
            `CoarseSpacingRelocationWarning` at `min_distance` 5 or more, or
            `UnmeasuredAnisotropyWarning` above an anisotropy of 3).
    """
    # Get a mesh reconstruction algorithm
    mesh_reconstruction_algorithm = get_default_mesh_reconstruction_algorithm(min_distance, print_info)
    return _dcel_mesh_from_mask_with(
        mesh_reconstruction_algorithm, segmentation_mask, physical_image_faces, spacing=spacing,
    )


def _dcel_mesh_from_mask_with(
    mesh_reconstruction_algorithm: object,
    segmentation_mask: NDArray[np.uint],
    physical_image_faces: Iterable[str] = (),
    *,
    spacing: tuple[float, float, float] | None = None,
) -> DcelData:
    """Reconstruct a mask with a given dw3d algorithm, keeping its crop faces.

    The body of :func:`dcel_mesh_from_segmentation_mask`, shared with `foambryo.seed_ensembles`, which
    reconstructs the same mask with the default algorithm at several dither seeds.

    Args:
        mesh_reconstruction_algorithm (object): a dw3d `MeshReconstructionAlgorithm`.
        segmentation_mask (NDArray[np.uint]): Instance segmentation image.
        physical_image_faces (Iterable[str], optional): see :func:`dcel_mesh_from_segmentation_mask`.
        spacing (tuple[float, float, float] | None, optional): see :func:`dcel_mesh_from_segmentation_mask`.
            Keyword-only.

    Returns:
        DcelData: The mesh, with the reconstruction's crop faces (and `source_spacing` when `spacing` is given).

    Raises:
        ValueError: if `physical_image_faces` is given to a dw3d that does not record crop faces, or `spacing`
            to a dw3d without `construct_mesh_in_physical_units`.
    """
    physical_image_faces = tuple(physical_image_faces)
    if np.ndim(segmentation_mask) == 2 and physical_image_faces:
        message = "physical_image_faces applies to 3-D masks only (the mask is 2-D)"
        raise ValueError(message)
    records_crop = hasattr(mesh_reconstruction_algorithm, "physical_image_faces")
    if physical_image_faces and not records_crop:
        message = ("physical_image_faces needs delaunay-watershed 0.8.0 or later, as foambryo requires: the installed "
                   "delaunay-watershed-3d does not record crop faces")
        raise ValueError(message)
    if records_crop:
        mesh_reconstruction_algorithm.physical_image_faces = physical_image_faces

    # Reconstruct a multimaterial mesh from the mask using the mesh reconstruction algorithm
    if spacing is None:
        points, triangles, labels = mesh_reconstruction_algorithm.construct_mesh_from_segmentation_mask(
            segmentation_mask,
        )
    else:
        if not hasattr(mesh_reconstruction_algorithm, "construct_mesh_in_physical_units"):
            message = ("spacing needs delaunay-watershed 0.8.0 or later, as foambryo requires: the installed "
                       "delaunay-watershed-3d has no construct_mesh_in_physical_units")
            raise ValueError(message)
        points, triangles, labels = mesh_reconstruction_algorithm.construct_mesh_in_physical_units(
            segmentation_mask,
            spacing,
        )
    two_dimensional = getattr(mesh_reconstruction_algorithm, "two_dimensional_algorithm", None)
    if two_dimensional is not None:
        return PlanarMesh(
            points, triangles, labels, image_border_vertices=two_dimensional.image_border_vertices,
            image_shape=np.shape(segmentation_mask), spacing=None if spacing is None else float(spacing[0]),
        )
    crop_faces = getattr(mesh_reconstruction_algorithm, "crop_faces", None)
    media = [label for label in getattr(mesh_reconstruction_algorithm, "medium_labels", ()) if label != 0]
    present = set(np.unique(np.asarray(labels)).tolist())
    report_of_media = getattr(mesh_reconstruction_algorithm, "enclosed_medium_report", None)
    unresolved = [int(u.label) for u in getattr(report_of_media, "unresolved", ())]
    return DcelData(points, triangles, labels, crop_faces=crop_faces, source_spacing=spacing,
                    media=[label for label in media if label in present], unresolved=unresolved)


def _source_spacing_from_rec_appendix(filename: Path) -> tuple[float, float, float] | None:
    """The source voxel size a `.rec` file records for a mesh in physical units, or `None`.

    Only a file whose appendix says `coordinate_frame = physical` is read (delaunay-watershed-3d's
    `construct_mesh_in_physical_units` writes it, and so does `foambryo mesh`): a mesh in voxel-index
    coordinates with an anisotropic `spacing` beside it was meshed on the stretched grid, and nothing
    here should make it look like a mesh in physical units.
    """
    try:
        appendix = load_rec_appendix(filename)
    except ImportError:
        return None
    if appendix is None or appendix.get("coordinate_frame") != "physical" or "spacing" not in appendix:
        return None
    spacing = tuple(float(v) for v in appendix["spacing"])
    return spacing if len(spacing) == 3 else None


# For consistency
def dcel_mesh_from_multimaterial_mesh(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    *,
    crop_faces: NDArray[np.bool_] | NDArray[np.integer] | None = None,
    source_spacing: tuple[float, float, float] | None = None,
) -> "DcelData | PlanarMesh":
    """Obtain a DcelData mesh for force inference from raw multimaterial mesh data.

    `(n, 2)` points with `(m, 2)` segments in place of triangles give a 2-D mesh, `foambryo.PlanarMesh` (dw2d's
    layout; crop faces and a source spacing are 3-D only).

    Args:
        points (NDArray[np.float64]): Points of the multimaterial mesh, shape `(nv, 3)`, in any
            length unit (the pressures are then in the tension unit divided by it).
        triangles (NDArray[np.int64]): Triangles of the multimaterial mesh, as vertex indices,
            shape `(nf, 3)`.
        labels (NDArray[np.int64]): the two materials on the sides of each triangle, shape
            `(nf, 2)`, label 0 the exterior medium. A triangle's winding must put its
            right-hand-rule normal on the side of its second label (see `DcelData`); the label
            order is then normalised to `label_1 < label_2`.
        crop_faces (NDArray | None, optional): which triangles are crop faces, as a boolean mask
            or row indices (see `DcelData`). Keyword-only. Defaults to `None`, no crop face.
        source_spacing (tuple | None, optional): the voxel size of the source mask, recorded,
            not used (see `DcelData`). Keyword-only. Defaults to `None`, unknown.

    Returns:
        DcelData: The mesh ready for force inference.

    Raises:
        ValueError: as `DcelData`: a triangle with the same material on both sides, an invalid
            `crop_faces` or `source_spacing`.
    """
    if np.shape(triangles)[1:] == (2,):
        if np.shape(points)[1:] != (2,):
            message = f"(m, 2) segments need (n, 2) points, got points of shape {np.shape(points)}"
            raise ValueError(message)
        if crop_faces is not None or source_spacing is not None:
            message = "crop_faces and source_spacing apply to 3-D meshes only (the mesh is 2-D: segments)"
            raise ValueError(message)
        return PlanarMesh(points, triangles, labels)
    return DcelData(points, triangles, labels, crop_faces=crop_faces, source_spacing=source_spacing)


def dcel_mesh_from_file(filename: str | Path) -> "DcelData | PlanarMesh":
    """Obtain a DcelData mesh for force inference from a multimaterial mesh file.

    A 2-D `.rec` file (its appendix records `dimension = 2`, or its layout has two coordinates per point:
    `dw3d.io.rec_dimension`) is read by dw2d's reader into a `foambryo.PlanarMesh`, with the image-border vertices and,
    for a mesh in physical units, the pixel size and unit its appendix records.

    It can be either a `.rec` (or `.arec`) file created from Delaunay-Watershed 3D, or a mesh
    that can be opened by meshio and contains a "label1" and "label2" data.
    The .vtk meshes saved by Delaunay-Watershed 3D are an example of such meshes. The mesh is
    read in the length unit of the file.

    **Crop faces** are read from the file when it records them: in a `.rec` file, the metadata
    appendix's `crop_faces = <first row> <count>` (the last block of triangle rows); in a file
    meshio opens, a `crop_face` cell array. A file that records neither is read without crop
    faces; if its mesh has triangles that look like crop faces (between the medium and
    a cell, lying flat on a face of the mesh's bounding box over a large patch), an
    `UnrecordedCropFacesWarning` says so, because they are then read as contact with the medium.

    The **source voxel size** is read from a `.rec` file whose appendix says the mesh is in physical
    units (`coordinate_frame = physical`, `spacing = ...`), as `DcelData.source_spacing`; it changes
    no number. Any other file gives `None`.

    The **media** are read from a `.rec` file whose appendix records them (`media = 0 L ...`: the enclosed media,
    lumens, that delaunay-watershed-3d 0.7.1 made their own regions), as `DcelData.media`. Any other file has the
    exterior 0 alone; name further media with `DcelData(..., media=...)` or `foambryo infer --media`.

    Args:
        filename (str | Path): Path to the file on disk.

    Returns:
        DcelData: The mesh ready for force inference.

    Raises:
        ValueError: if the file's crop record is inconsistent with its triangles, or as
            `DcelData` (a triangle with the same material on both sides).

    Warns:
        UnrecordedCropFacesWarning: when the file records no crop faces but its mesh has
            triangles that look like them.
    """
    filename = Path(filename)
    source_spacing = None
    media = unresolved = None
    if filename.suffix in (".rec", ".arec") and rec_dimension(filename) == 2:
        return _planar_mesh_from_rec_file(filename)
    if filename.suffix in (".rec", ".arec"):
        points, triangles, labels = load_rec(filename)
        crop_faces = _crop_faces_from_rec_appendix(filename, labels)
        source_spacing = _source_spacing_from_rec_appendix(filename)
        media = _media_from_rec_appendix(filename)
        unresolved = _labels_from_rec_appendix(filename, "unresolved_regions")
    else:
        points, triangles, labels = load_meshio_mesh(filename)

        # ensure endian and type
        points = points.astype(np.float64)
        triangles = triangles.astype(np.int64)
        labels = labels.astype(np.int64)
        crop_faces = _crop_faces_from_meshio_file(filename)
    if crop_faces is None:
        warn_if_crop_faces_unrecorded(points, triangles, labels, f"{filename.name}", stacklevel=3)
    return DcelData(points, triangles, labels, crop_faces=crop_faces, source_spacing=source_spacing, media=media,
                    unresolved=unresolved)


def _media_from_rec_appendix(filename: Path) -> tuple[int, ...] | None:
    """The medium labels a `.rec` file's appendix records (`media = 0 L ...`), or `None` when it records none.

    delaunay-watershed-3d 0.7.1 writes the key only when the mesh has an enclosed medium (a lumen made its own
    region). Its own reader parses it to integers; an older one keeps it as a string, read here as well.
    """
    return _labels_from_rec_appendix(filename, "media")


def _labels_from_rec_appendix(filename: Path, key: str) -> tuple[int, ...] | None:
    """The integer labels a `.rec` file's appendix records under `key`, or `None` when it records none.

    Read from the typed tuple of delaunay-watershed-3d 0.7.1's reader, or from the string an older reader keeps.
    """
    try:
        appendix = load_rec_appendix(filename)
    except ImportError:
        return None
    if appendix is None or key not in appendix:
        return None
    value = appendix[key]
    labels = value.split() if isinstance(value, str) else value
    return tuple(int(label) for label in labels)


def _crop_faces_from_rec_appendix(filename: Path, labels: NDArray[np.int64]) -> NDArray[np.bool_] | None:
    """The crop-face mask a `.rec` file's appendix records, or `None` when it records none.

    Parsed here rather than by dw3d's own reader so that a dw3d that predates the key (and keeps
    it as a raw string) is read the same way.
    """
    try:
        appendix = load_rec_appendix(filename)
    except ImportError:
        return None
    if appendix is None or "crop_faces" not in appendix:
        return None
    raw = appendix["crop_faces"]
    values = tuple(int(v) for v in (raw.split() if isinstance(raw, str) else raw))
    n_triangles = len(labels)
    if len(values) != 2 or values[0] < 0 or values[1] < 0 or values[0] + values[1] != n_triangles:
        message = (
            f"{filename.name}: crop_faces = {raw} is not the last block of its {n_triangles} triangles; "
            "the file is inconsistent with its appendix"
        )
        raise ValueError(message)
    mask = np.zeros(n_triangles, dtype=bool)
    mask[values[0] :] = True
    return mask


def _crop_faces_from_meshio_file(filename: Path) -> NDArray[np.bool_] | None:
    """The crop-face mask a meshio-readable file's `crop_face` cell array records, or `None`.

    Read through dw3d's reader; a dw3d that predates crop faces cannot read the array, and the
    file is then read without them (and warned about if it looks cropped).
    """
    try:
        from dw3d.io import load_meshio_mesh_with_crop_faces
    except ImportError:
        return None
    return load_meshio_mesh_with_crop_faces(filename)[3]


def _reconstruct_a_two_dimensional_mask(
    algorithm: object,
    segmentation_mask: NDArray[np.integer],
    spacing: tuple[float, ...],
    spacing_unit: str,
    **padding: bool,
) -> MaskReconstruction:
    """`reconstruct_mask` for a 2-D mask: delaunay-watershed-3d's algorithm hands it to dw2d."""
    if len(spacing) != 2:
        message = f"a 2-D mask takes a pixel size of 2 values (dy, dx), got {len(spacing)}: {tuple(spacing)}"
        raise ValueError(message)
    spacing = (float(spacing[0]), float(spacing[1]))
    points, segments, labels = algorithm.construct_mesh_in_physical_units(
        segmentation_mask, spacing, spacing_unit=spacing_unit, axis_order="yx", **padding,
    )
    two_dimensional = algorithm.two_dimensional_algorithm
    with tempfile.TemporaryDirectory() as folder:
        rec_path = Path(folder) / "mesh.rec"
        algorithm.save_to_rec_mesh(rec_path)
        rec_bytes = rec_path.read_bytes()
    mask = np.asarray(segmentation_mask)
    unresolved = {int(label): int(np.count_nonzero(mask == label)) for label in two_dimensional.unresolved_cells}
    present = np.unique(np.asarray(labels))
    return MaskReconstruction(
        points=points,
        triangles=segments,
        labels=labels,
        crop_faces=None,
        spacing=spacing,
        spacing_unit=spacing_unit,
        min_distance=int(two_dimensional.min_distance),
        rec_bytes=rec_bytes,
        n_cells=int(np.count_nonzero(present != 0)),
        n_cut_cells=len(two_dimensional.cut_cells),
        media=(0,),
        unresolved=unresolved,
        dimension=2,
        image_border_vertices=two_dimensional.image_border_vertices,
        image_shape=tuple(mask.shape),
    )


def _planar_mesh_from_rec_file(filename: Path) -> PlanarMesh:
    """A 2-D `.rec` file as a `PlanarMesh`: dw2d's reader, and the appendix's border vertices and physical frame.

    The edges' orientation is measured from the geometry (`foambryo.planar_mesh`'s docstring): a file in the former
    orientation (dw2d before 0.8.0, which records none) is reversed on input; a file whose recorded `orientation`
    its geometry contradicts, or that records an unknown rule, is refused.
    """
    from dw2d.io import load_rec as load_two_dimensional_rec
    from dw2d.io import load_rec_appendix as load_two_dimensional_appendix

    from foambryo.planar_mesh import _segments_in_the_one_orientation

    points, segments, labels = load_two_dimensional_rec(filename)
    appendix = load_two_dimensional_appendix(filename) or {}
    spacing = None
    if appendix.get("coordinate_frame") == "physical" and "spacing" in appendix:
        spacing = float(appendix["spacing"][0])
    border = appendix.get("image_border_vertices")
    try:
        segments, _ = _segments_in_the_one_orientation(
            points, segments, labels, border, declared=appendix.get("orientation"),
        )
    except ValueError as error:
        message = f"{filename}: {error}"
        raise ValueError(message) from error
    shape = appendix.get("source_shape")
    return PlanarMesh(
        points, segments, labels, image_border_vertices=None if border is None else np.asarray(border, dtype=np.int64),
        image_shape=None if shape is None or len(shape) != 2 else tuple(shape), spacing=spacing,
        unit=appendix.get("spacing_unit"),
    )
