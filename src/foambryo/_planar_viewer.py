"""The Polyscope window of a 2-D tissue: interfaces as curves coloured by tension, cells as polygons by pressure.

`foambryo.viewing.plot_force_inference` (and its tension and pressure variants), `foambryo.viewing.view_mask` and
`foambryo view` open this window when the mesh, or the mask, is 2-D. It is the 3-D window (`viewing._ForceWindow`,
`viewing._MaskWindow`) with what differs in 2D replaced, and the same panel otherwise:

* **what is drawn** (`foambryo.plotting_utilities.planar_mesh_drawing`): every interface segment as a curve, coloured by
  its tension (`coolwarm`, centred on the mean tension, the library's colours: `foambryo.colour_scales`); an
  undetermined tension (`nan`) as a green curve; the segments that close a cut cell along the image border, which carry
  no value, in the glass's edge colour; each cell as a filled polygon coloured by its pressure (reversed `magma`,
  trimmed, from 0 to the `1 - alpha` quantile over the cells, drawn flat, as on the bars), shown when the panel colours
  by pressure; the junctions (three or more segments) as points;
* **the image under the mesh**, when one is given (`image=`; the mask in the window opened on a mask): a textured
  quad, one texel per pixel (nearest filtering), pixel `(row, column)` centred on the mesh point `(row, column)` times
  the pixel size, so the curves lie on the image's own pixels;
* **the frame**: Polyscope's `(x, y)` is `(column, -row)` (times the pixel size), a rotation of the mesh's
  `(row, column)` by a quarter turn, not a reflection: the window shows the image as napari and matplotlib's `imshow`
  do, row 0 at the top and column 0 at the left, and each polygon keeps its orientation. The mesh and its files are
  not changed (`foambryo._display_frame` maps what is handed to Polyscope; here the mesh's `(row, column)` enter it as
  the columns `(y, x)` of a 3-D point `(0, -row, column)`);
* **the camera**: planar navigation (pan and zoom, no rotation), looking straight down on the plane, framed on the
  tissue (Polyscope's own home view is tilted by a few degrees, which draws the image as a trapezoid);
* **the panel**: the 3-D panel without Cut; Explode moves each cell away from the tissue's centre by its area
  centroid; the methods are the 2-D ones (Young-Dupré tensions with the 2-D angle readings, Laplace pressures); the
  frame is `x, y` (column, row); the colour bars, Save figure, Save colour bars, Export and the equivalent command
  line are the 3-D window's.

* **the stresses** (the force viewer, with tensions and pressures, as in 3D): each cell's mean stress tensor
  (`foambryo._planar_curvature_and_stress`, 2 x 2 in `(row, column)`) drawn at its area centroid as its two principal
  axes, each scaled by its principal value and drawn both ways, compression in red and tension in teal, with 3D's sizes;
  a cell cut by the image border, or with an undetermined pressure or tension, has none. Like the pressures it is
  built on, it is a measurement only for cells deformed by nothing but their own tensions and pressures: not in an
  embryo confined by an eggshell, or cells on a substrate (`foambryo._planar_curvature_and_stress`). With
  `stress="deviatoric"`, each cell's deviatoric stress instead, from the tensions alone ("Deviatoric stress principal
  directions"; a cell with an undetermined pressure gets one): a measurement in a confined embryo too when the tensions
  are read near the junctions (`angle_reading="fitted_tangent"`);
* **the curvatures** (`scalar_quantities=True`, the force and tension viewers, as in 3D): two colour quantities of the
  interfaces, `jet` between the `alpha` and `1 - alpha` quantiles, `nan` green, off until chosen in Polyscope's
  structure panel: "Curvature (arc)", `|kappa|` of each interface's fitted circle, and "Curvature (local)", the vertex
  curvature `|K|` (`foambryo.curvature`). The sphere-fit residual and the area, volume and Gaussian-curvature
  quantities of 3D have no 2-D counterpart.

The graph view has no 2-D counterpart yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polyscope.imgui as psim

from foambryo._colour_scales import (
    CROP_FACE_EDGE_COLOR,
    UNDETERMINED_COLOR,
    colour_bar_ranges,
    pressure_colour_fractions,
    pressure_colours,
    quantity_colours,
    tension_colour_fractions,
    tension_colours,
)
from foambryo._polyscope_recording import polyscope_target as ps
from foambryo._viewer_figures import (
    COMPRESSIVE_AXIS_COLOR,
    TENSILE_AXIS_COLOR,
    format_length,
    project_to_pixels,
    round_scale_bar_length,
    scale_bar_for_points,
)
from foambryo.planar_mesh import PlanarMesh
from foambryo.plotting_utilities import PlanarMeshDrawing, planar_mesh_drawing
from foambryo.pressure_inference import PressureComputationMethod
from foambryo.tension_inference import TensionComputationMethod

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["PLANAR_STRUCTURES", "planar_scene_points", "prepare_image"]

#: The structures of the 2-D window, by name.
_INTERFACES = "Interfaces"
_BORDER = "Image border (no value)"
_CELLS = "Cells"
_JUNCTIONS = "Junctions"
_IMAGE = "Image"
PLANAR_STRUCTURES = (_INTERFACES, _BORDER, _CELLS, _JUNCTIONS, _IMAGE)
#: Curves 0.6 pixel in radius (1.2 px wide), junctions 1.4 px: legible at the tissue's scale, thin beside a cell.
_CURVE_RADIUS_PX = 0.6
_JUNCTION_RADIUS_PX = 1.4
#: The image lies this fraction of the tissue's extent below the plane of the mesh, so the curves are drawn over it.
_IMAGE_DEPTH = 1e-3
#: The interfaces' colour when no tension is drawn (a pressure viewer): dark grey, drawn in the background's ink.
_NO_TENSION_COLOR = (0.15, 0.15, 0.15)
#: The stress glyphs of the 3-D window (`viewing._plot_stress_tensor`): the point cloud's name, sizes and colours.
_STRESS = "Stress tensors principal directions"
#: The deviatoric stress glyphs (`plot_force_inference(..., stress="deviatoric")`), named as in 3D.
_DEVIATORIC_STRESS = "Deviatoric stress principal directions"
_STRESS_POINT_RADIUS = 0.004
_STRESS_AXIS_RADIUS = 0.005
_STRESS_AXIS_LENGTH = 0.1
_STRESS_COLOURS = (("compression", COMPRESSIVE_AXIS_COLOR), ("tension", TENSILE_AXIS_COLOR))
#: The curvature quantities of the interfaces (`scalar_quantities=True`).
_ARC_CURVATURE = "Curvature (arc)"
_LOCAL_CURVATURE = "Curvature (local)"
_FRAME_LABELS_2D = ("x", "y")


def planar_scene_points(points: NDArray[np.floating], depth: float = 0.0) -> NDArray[np.float64]:
    """The mesh's `(row, column)` points as the 3-D points handed to the drawn frame: `(depth, -row, column)`.

    `foambryo._display_frame` draws a point `(z, y, x)` at Polyscope's `(x, y, z)`: these land at `(column, -row,
    depth)`, a quarter-turn rotation of the image, rows downward.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.c_[np.full(len(p), float(depth)), -p[:, 0], p[:, 1]]


def prepare_image(image: NDArray | None) -> NDArray[np.float64] | None:
    """An image to draw under the mesh, as an `(rows, columns, 3)` RGB array in [0, 1], or `None`.

    A 2-D array is grey, scaled linearly between its 0.5 % and 99.5 % quantiles (a microscope image is rarely in
    [0, 1]); an `(rows, columns, 3)` or `(..., 4)` array is RGB(A), as floats in [0, 1] or integers in [0, 255].

    Raises:
        ValueError: if the array is neither.
    """
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 2:
        values = image.astype(np.float64)
        low, high = np.quantile(values, [0.005, 0.995]) if values.size else (0.0, 1.0)
        grey = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(values)
        return np.repeat(grey[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = image[..., :3]
        if np.issubdtype(rgb.dtype, np.integer):
            return rgb.astype(np.float64) / 255.0
        return np.clip(rgb.astype(np.float64), 0.0, 1.0)
    error = f"an image under a 2-D mesh is (rows, columns) or (rows, columns, 3 or 4), not {image.shape}"
    raise ValueError(error)


def check_image_on_mesh(image: NDArray[np.float64], mesh: PlanarMesh) -> None:
    """Refuse an image whose pixel grid cannot be the mesh's (another shape than the mesh's `image_shape`)."""
    if mesh.image_shape is not None and tuple(image.shape[:2]) != tuple(mesh.image_shape):
        error = (f"the image is {image.shape[0]} x {image.shape[1]} pixels, the mesh was built from a "
                 f"{mesh.image_shape[0]} x {mesh.image_shape[1]} image: it cannot be drawn under it")
        raise ValueError(error)


def _label_rgb(labels: NDArray[np.integer]) -> NDArray[np.float64]:
    from foambryo._mask_workflow import label_colours

    return label_colours(np.asarray(labels))


def planar_window_class() -> type:
    """`_PlanarForceWindow`, defined on first use (it subclasses `viewing._ForceWindow`, which needs Polyscope)."""
    return _planar_classes()[0]


def planar_mask_window_class() -> type:
    """`_PlanarMaskWindow`, defined on first use."""
    return _planar_classes()[1]


_CLASSES: list[type] = []


def _planar_classes() -> list[type]:  # noqa: C901 - the two classes, defined against viewing's
    if _CLASSES:
        return _CLASSES
    from foambryo import viewing

    class _PlanarForceWindow(viewing._ForceWindow):
        """One 2-D viewer: `viewing._ForceWindow` with a 2-D scene, frame, camera and panel (the module docstring)."""

        def __init__(self, mesh: object, kind: str, tensions: dict | None, pressures: dict | None,
                     options: viewing._InferenceOptions, values_from: str, *, image: NDArray | None = None,
                     length_unit: str | None = None, scale_bar: bool | float = True, **kwargs: Any) -> None:  # noqa: ANN401
            mesh = PlanarMesh.from_any(mesh)
            super().__init__(mesh, kind, tensions, pressures, options, values_from, length_unit=length_unit,
                             scale_bar=False, **kwargs)
            self.scene: PlanarMeshDrawing = planar_mesh_drawing(mesh)
            #: The mesh's unit per pixel: the pixel size of a mesh in physical units, else 1.
            self.pixel = float(mesh.spacing) if mesh.spacing else 1.0
            self.image = prepare_image(image)
            if self.image is not None:
                check_image_on_mesh(self.image, mesh)
            self.show_image = self.image is not None
            #: Flat colours by default: a plane facing the camera has no shape to shade, and shaded colours are
            #: lighter than the bars' (the panel's "flat colours" switches shading on).
            self.flat_colours = True
            self.show_junctions = True
            self.show_border = True
            used = np.unique(self.scene.segments) if len(self.scene.segments) else np.zeros(0, np.int64)
            self._used_points = self.scene.points[used] if len(used) else self.scene.points
            if length_unit is not None and scale_bar is not False:
                length = None if scale_bar is True else float(scale_bar)
                self.scale_bar = scale_bar_for_points(planar_scene_points(self._used_points), length_unit, length)
            self.show_scale_bar = self.scale_bar is not None
            self.tension_choice = 0
            readings = _readings()
            self.reading_choice = readings.index(options.angle_reading) if options.angle_reading in readings else 0

        # -- the scene ----------------------------------------------------------------------------------------
        def _draw_scene(self, scattered: bool, coeff: float) -> None:
            scene = self.scene
            moves = scene.displacements(coeff) if scattered else None
            if self.image is not None:
                self._draw_image(enabled=self.show_image and not scattered)
            self._draw_cells(moves)
            self._draw_interfaces(moves)
            if self.kind == "force" and self.tensions is not None and (
                    self.pressures is not None or self.stress == "deviatoric"):
                self._draw_stresses(moves)
            if not scattered and len(scene.junctions):
                cloud = ps.register_point_cloud(_JUNCTIONS, planar_scene_points(scene.points[scene.junctions]),
                                                color=viewing._on_background((0.0, 0.0, 0.0)),
                                                enabled=self.show_junctions)
                cloud.set_radius(_JUNCTION_RADIUS_PX * self.pixel, relative=False)
            self._draw_scale_bar()
            self.apply_ink()

        def _draw_image(self, *, enabled: bool) -> None:
            rows, columns = self.image.shape[:2]
            h = self.pixel
            corners = np.array([[-0.5, -0.5], [-0.5, columns - 0.5], [rows - 0.5, columns - 0.5],
                                [rows - 0.5, -0.5]]) * h
            uv = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
            # corners counter-clockwise on screen; handed reversed, as every triangle (`_display_frame`)
            quad = ps.register_surface_mesh(_IMAGE, planar_scene_points(corners, -_IMAGE_DEPTH * self.scene.extent),
                                            np.array([[0, 1, 2], [0, 2, 3]])[:, [0, 2, 1]], enabled=enabled)
            quad.add_parameterization_quantity("pixel coordinates", uv, defined_on="vertices")
            quad.add_color_quantity("pixels", self.image, defined_on="texture", param_name="pixel coordinates",
                                    image_origin="upper_left", enabled=True, filter_mode="nearest")
            quad.set_material("flat")
            quad.set_back_face_policy("identical")

        def _draw_cells(self, moves: dict | None) -> None:
            scene = self.scene
            if not len(scene.triangles):
                return
            n = len(scene.points)
            keys = scene.triangle_cell[:, None] * n + scene.triangles
            unique, inverse = np.unique(keys, return_inverse=True)
            vertex_cell = unique // n
            vertices = scene.points[unique % n].copy()
            if moves is not None:
                vertices += np.array([moves[int(c)] for c in vertex_cell])
            faces = inverse.reshape(-1, 3)[:, [0, 2, 1]]  # counter-clockwise on screen once drawn
            surface = ps.register_surface_mesh(_CELLS, planar_scene_points(vertices), faces, enabled=False)
            surface.set_back_face_policy("identical")
            surface.set_material("flat" if self.flat_colours else "clay")
            surface.add_color_quantity("label", _label_rgb(scene.triangle_cell), defined_on="faces", enabled=True)
            pressures = self._pressures_drawn
            if pressures is not None:
                values = np.array([pressures.get(int(c), np.nan) for c in scene.triangle_cell], dtype=np.float64)
                ranges = colour_bar_ranges(self.mesh, dict(self.tensions or {}), pressures, self.alpha)
                if ranges.pressure is None:
                    colours = np.tile(UNDETERMINED_COLOR, (len(values), 1))
                else:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        colours = pressure_colours(pressure_colour_fractions(values, ranges.pressure[1]))
                surface.add_color_quantity(viewing._PRESSURES, colours, defined_on="faces", enabled=True)

        def _draw_interfaces(self, moves: dict | None) -> None:
            scene = self.scene
            if moves is None:
                rows = np.arange(len(scene.segments))
                ends = scene.segments
                keys = ends
                positions = scene.points
            else:
                rows, side = scene.interface_sides()
                n = len(scene.points)
                keys = side[:, None] * n + scene.segments[rows]
                positions = None
            interior = ~scene.border[rows]
            for name, chosen in ((_INTERFACES, interior), (_BORDER, ~interior)):
                if not chosen.any():
                    continue
                unique, inverse = np.unique(keys[chosen], return_inverse=True)
                if moves is None:
                    nodes = positions[unique]
                else:
                    nodes = scene.points[unique % len(scene.points)] + np.array(
                        [moves[int(c)] for c in unique // len(scene.points)])
                network = ps.register_curve_network(name, planar_scene_points(nodes), inverse.reshape(-1, 2),
                                                    enabled=name == _INTERFACES or self.show_border)
                network.set_radius(_CURVE_RADIUS_PX * self.pixel, relative=False)
                network.set_material("flat")
                if name == _BORDER:
                    network.set_color(tuple(float(c) for c in CROP_FACE_EDGE_COLOR))
                    continue
                pairs = scene.segment_pair[rows[chosen]]
                if self.scalar_quantities and self.kind in ("force", "tension"):
                    vertices = unique if moves is None else unique % len(scene.points)
                    self._add_curvatures(network, pairs, vertices)
                if self.tensions is None:
                    network.set_color(viewing._on_background(_NO_TENSION_COLOR))
                    continue
                per_pair = np.array([_tension_of(self.tensions, scene.pairs[k]) for k in range(len(scene.pairs))],
                                    dtype=np.float64)
                values = per_pair[pairs]
                ranges = colour_bar_ranges(self.mesh, self.tensions, None, self.alpha)
                if ranges.tension is None:
                    colours = np.tile(UNDETERMINED_COLOR, (len(values), 1))
                else:
                    colours = tension_colours(tension_colour_fractions(values, ranges.tension))
                network.add_color_quantity(viewing._TENSIONS, colours, defined_on="edges", enabled=True)

        def _add_curvatures(self, network: Any, pairs: NDArray[np.int64], vertices: NDArray[np.int64]) -> None:  # noqa: ANN401
            """The interfaces' arc curvature (per segment) and vertex curvature (per node), `|kappa|`, in `jet`."""
            from foambryo.curvature import compute_curvature_interfaces, compute_curvature_vertices_cotan

            arc = compute_curvature_interfaces(self.mesh, reading="arc")
            per_pair = np.array([abs(arc.get(pair, np.nan)) for pair in self.scene.pairs], dtype=np.float64)
            local = compute_curvature_vertices_cotan(self.mesh)[0]
            for name, values, where in ((_ARC_CURVATURE, per_pair[pairs], "edges"),
                                        (_LOCAL_CURVATURE, local[vertices], "nodes")):
                network.add_color_quantity(name, quantity_colours(_quantile_fractions(values, self.alpha)),
                                           defined_on=where, enabled=False)

        def _draw_stresses(self, moves: dict | None) -> None:
            """Each cell's principal stresses at its centroid, as the 3-D window draws them (the module docstring)."""
            from foambryo._cell_stress import cell_mean_stress_tensors, split_principal_stresses
            from foambryo._planar_curvature_and_stress import cell_areas_and_centroids

            mesh, scene = self.mesh, self.scene
            areas, _ = cell_areas_and_centroids(mesh)
            cut = set(mesh.cut_cells)
            full = self.stress == "full"
            # The deviatoric stress needs no pressure: a cell with an undetermined pressure gets one.
            cells = [c for c in scene.cells if c not in cut and c in scene.centroids and areas.get(c, 0.0) > 0
                     and (not full or np.isfinite(self.pressures.get(c, np.nan)))]
            if full:
                tensors = cell_mean_stress_tensors(mesh, self.tensions, self.pressures, areas, cells,
                                                   lumen_materials=mesh.media)
            else:
                tensors = cell_mean_stress_tensors(mesh, self.tensions, None, areas, cells, lumen_materials=mesh.media,
                                                   part=self.stress)
            cells = [c for c in cells if np.all(np.isfinite(tensors[c]))]
            self._stress_principal_values = [float(v) for c in cells for v in np.linalg.eigvalsh(tensors[c])]
            if not cells:
                return
            positions = np.array([scene.centroids[c] + (moves[c] if moves is not None else 0.0) for c in cells])
            cloud = ps.register_point_cloud(_STRESS if full else _DEVIATORIC_STRESS, planar_scene_points(positions),
                                            color=(0, 0, 0), radius=_STRESS_POINT_RADIUS)
            axes = [split_principal_stresses(tensors[c]) for c in cells]
            for name, colour in _STRESS_COLOURS:
                for k in range(2):
                    axis = np.array([getattr(a, name)[:, k] for a in axes])
                    for suffix, sign in (("", 1), ("_down", -1)):
                        cloud.add_vector_quantity(f"{name}_axes_{k}{suffix}", planar_scene_points(sign * axis),
                                                  enabled=True, radius=_STRESS_AXIS_RADIUS,
                                                  length=_STRESS_AXIS_LENGTH, color=colour)

        def apply_ink(self) -> None:
            super().apply_ink()
            if ps.has_point_cloud(_JUNCTIONS):
                ps.get_point_cloud(_JUNCTIONS).set_color(viewing._on_background((0.0, 0.0, 0.0)))
            if self.tensions is None and ps.has_curve_network(_INTERFACES):
                ps.get_curve_network(_INTERFACES).set_color(viewing._on_background(_NO_TENSION_COLOR))

        def apply_showing(self) -> None:
            """Colour by tension: the curves alone (over the image); by pressure: the cells filled, under the curves."""
            if ps.has_surface_mesh(_CELLS):
                ps.get_surface_mesh(_CELLS).set_enabled(self.showing == "pressure" and self._pressures_drawn is not None)

        # -- frame, box, camera ----------------------------------------------------------------------------------
        def drawn_box(self) -> np.ndarray:
            """The `(2, 3)` box of the mesh's curves (unexploded), in the 3-D points handed to the drawn frame."""
            points = planar_scene_points(self._used_points)
            return np.stack([points.min(axis=0), points.max(axis=0)])

        def _frame_rc(self) -> tuple[np.ndarray, float] | None:
            low, high = self._used_points.min(axis=0), self._used_points.max(axis=0)
            extent = float(np.max(high - low))
            if not (np.isfinite(extent) and extent > 0):
                return None
            return low - 0.1 * extent, round_scale_bar_length(extent)

        def _draw_frame(self) -> None:
            """`x` along the columns (to the right) and `y` along the rows (downwards), from the top-left corner."""
            found = self._frame_rc()
            if found is None:
                self.frame = None
                return
            origin, length = found
            nodes = planar_scene_points(_frame_nodes(origin, length))
            network = ps.register_curve_network(viewing._FRAME, nodes, np.array([[0, 1], [0, 2]]),
                                                color=viewing._on_background(viewing._FRAME_COLOR))
            network.set_radius(_CURVE_RADIUS_PX * self.pixel, relative=False)
            network.set_enabled(self.show_frame)
            self._installed.add(("curve_network", viewing._FRAME))
            self.frame = (origin, length)

        def frame_labels(self, width: int, height: int) -> list[tuple[str, float, float]]:
            if self.frame is None or not self.show_frame:
                return []
            origin, length = self.frame
            points = planar_scene_points(_frame_nodes(origin, length))
            from foambryo._display_frame import drawn_points

            camera = ps.get_view_camera_parameters()
            pixels = project_to_pixels(drawn_points(points), camera.get_view_mat(), camera.get_fov_vertical_deg(),
                                       width, height)
            unit = format_length(length, self.length_unit) if self.length_unit else f"{length:g} (mesh units)"
            return [(text, float(x), float(y)) for text, (x, y) in zip((unit, *_FRAME_LABELS_2D), pixels, strict=True)
                    if np.isfinite(x) and np.isfinite(y)]

        def home_view(self) -> None:
            """Look straight down on the plane, framed on the tissue (and its image), rows downward."""
            box = [planar_scene_points(self._used_points)]
            if self.image is not None and self.show_image:
                rows, columns = self.image.shape[:2]
                box.append(planar_scene_points(np.array([[-0.5, -0.5], [rows - 0.5, columns - 0.5]]) * self.pixel))
            points = np.vstack(box)[:, [2, 1, 0]]  # Polyscope's x, y, z
            low, high = points.min(axis=0), points.max(axis=0)
            centre = (low + high) / 2
            width, height = ps.get_window_size()
            aspect = width / height if height else 1.0
            half = 0.5 * max(high[1] - low[1], (high[0] - low[0]) / aspect, 1e-9)
            distance = 1.15 * half / np.tan(np.radians(ps.get_vertical_fov_degrees()) / 2)
            ps.look_at((float(centre[0]), float(centre[1]), float(distance)),
                       (float(centre[0]), float(centre[1]), 0.0))

        def cut_pose(self) -> None:
            return None

        def apply_cut(self) -> None:
            return None

        def open(self, *, save_figure: Any, show: bool) -> None:  # noqa: ANN401
            previous = _enter_planar_mode()
            try:
                self.draw()
                self.home_view()
                if save_figure is not None:
                    self.save_figure(save_figure, render_first=True)
                if not show:
                    return
                ps.set_open_imgui_window_for_user_callback(False)
                ps.set_user_callback(self.panel)
                try:
                    ps.show()
                finally:
                    ps.clear_user_callback()
                    self.close()
            finally:
                _leave_planar_mode(previous)

        def save_figure(self, path: Any, *, render_first: bool = False) -> Any:  # noqa: ANN401
            if render_first:
                self.home_view()
                ps.show(forFrames=1)
            return super().save_figure(path, render_first=False)

        # -- the panel ---------------------------------------------------------------------------------------
        def panel_body(self, *, methods: bool = True) -> None:
            self.poll_redraw()
            psim.TextWrapped(f"Values: {self.values_from}")
            if self.preparing:
                psim.TextDisabled("drawing in the background ...")
            if self.scales is None:
                return
            psim.Separator()
            self._panel_showing()
            self._panel_bars()
            psim.Separator()
            self._panel_display()
            if psim.CollapsingHeader("Explode", psim.ImGuiTreeNodeFlags_DefaultOpen):
                self._panel_explode()
            if psim.CollapsingHeader("Save figure, colour bars", psim.ImGuiTreeNodeFlags_DefaultOpen):
                self._panel_saving()
            if psim.CollapsingHeader("Export (foambryo infer's files)", psim.ImGuiTreeNodeFlags_DefaultOpen):
                self._panel_export()
            if methods and psim.CollapsingHeader("Methods, solver, Recompute", psim.ImGuiTreeNodeFlags_DefaultOpen):
                self._panel_methods()
            if self.status:
                psim.TextWrapped(self.status)

        def _panel_showing(self) -> None:
            if self.tensions is None or self._pressures_drawn is None:
                return
            psim.TextUnformatted("colour by")
            for value, name in (("tension", "tensions (curves)"), ("pressure", "pressures (cells)")):
                psim.SameLine()
                if psim.RadioButton(name, self.showing == value) and self.showing != value:
                    self.showing = value
                    self.apply_showing()

        def _panel_bars(self) -> None:
            for scale in self.scales.scales:
                viewing._imgui_colour_bar(scale)
            if self.scales.undetermined:
                viewing._imgui_swatch(UNDETERMINED_COLOR, "undetermined (nan): a green curve (a cell: green)")
            if self.scales.crop:
                viewing._imgui_swatch(CROP_FACE_EDGE_COLOR, "image border (no value)")

        def _panel_explode(self) -> None:
            changed, scattered = psim.Checkbox("explode (each cell moved away from the centre)", self.scattered)
            changed_coeff, coeff = psim.SliderFloat("explode coefficient", self.scatter_coeff, 0.0, 1.0)
            if changed_coeff:
                self.scatter_coeff = coeff
            if changed or (changed_coeff and self.scattered):
                self.scattered = scattered
                self.request_redraw()

        def _panel_display(self) -> None:
            changed, self.flat_colours = psim.Checkbox("flat colours (as on the bars, no shading)", self.flat_colours)
            if changed and ps.has_surface_mesh(_CELLS):
                ps.get_surface_mesh(_CELLS).set_material("flat" if self.flat_colours else "clay")
            if self.scale_bar is not None:
                changed, self.show_scale_bar = psim.Checkbox(f"scale bar, {self.scale_bar.label}", self.show_scale_bar)
                if changed:
                    ps.get_curve_network(f"Scale bar ({self.scale_bar.label})").set_enabled(self.show_scale_bar)
            else:
                psim.TextDisabled("no scale bar: the mesh's length unit is not known")
            unit = f"{self.length_unit}" if self.length_unit else "mesh units"
            changed, self.show_frame = psim.Checkbox(f"x, y frame: column, row ({unit})", self.show_frame)
            if changed and ps.has_curve_network(viewing._FRAME):
                ps.get_curve_network(viewing._FRAME).set_enabled(self.show_frame)
            for attribute, name, label, kind in (
                    ("show_image", _IMAGE, "image under the mesh", "surface_mesh"),
                    ("show_junctions", _JUNCTIONS, "junctions (points)", "point_cloud"),
                    ("show_border", _BORDER, "image-border segments (no value)", "curve_network")):
                if getattr(ps, f"has_{kind}")(name):
                    changed, value = psim.Checkbox(label, getattr(self, attribute))
                    if changed:
                        setattr(self, attribute, value)
                        getattr(ps, f"get_{kind}")(name).set_enabled(value)
            viewing._panel_background(self)

        def _panel_methods(self) -> None:
            psim.TextUnformatted("tension method: young-dupre (the one offered in 2D)")
            readings = _readings()
            _, self.reading_choice = psim.Combo("angle reading", self.reading_choice, list(readings))
            if self.base_kind != "tension":
                _, self.pressures_choice = psim.Checkbox("pressures (off: the tensions only)", self.pressures_choice)
                if self.pressures_choice:
                    psim.TextDisabled("pressure method: laplace (the one offered in 2D)")
            _, self.solver_choice = psim.Combo("solver", self.solver_choice, list(viewing._SOLVERS))
            _, self.diagnostics_choice = psim.Combo("diagnostics (for Export)", self.diagnostics_choice,
                                                    list(viewing._DIAGNOSTICS))
            if psim.Button("Recompute"):
                self._try(self._recompute_and_say)
            command = self.command_line()
            if command:
                psim.TextDisabled("the same numbers as files:")
                psim.TextWrapped(command)

        def recompute(self) -> None:
            with_pressures = self.pressures_choice and self.base_kind != "tension"
            reading = _readings()[self.reading_choice]
            options = viewing._InferenceOptions(
                TensionComputationMethod.YoungDupre, PressureComputationMethod.Laplace, self.options.mean_tension,
                self.options.base_pressure, viewing._SOLVERS[self.solver_choice], pressures=with_pressures,
                diagnostics=viewing._DIAGNOSTICS[self.diagnostics_choice],
                angle_reading=None if reading == _default_reading() else reading, dimension=2)
            kind = ("force" if self.base_kind != "tension" else "tension") if with_pressures else "tension"
            tensions, pressures, values_from = options.infer(self.mesh, kind, None, None, None)
            self.options, self.tensions, self.values_from = options, tensions, values_from
            self.pressures = pressures if with_pressures else None
            self.kind = kind if self.base_kind != "pressure" or not with_pressures else "pressure"
            self.inference = options.inference
            if self.kind == "tension":
                self.showing = "tension"
            self.redraw()

    class _PlanarMaskWindow(viewing._MaskWindow):
        """The window of a 2-D mask: the mask as an image, its reconstruction as curves, then the values' window."""

        def __init__(self, workflow: Any, **kwargs: Any) -> None:  # noqa: ANN401
            self.workflow = workflow
            self.show_mask = True
            self.scattered = kwargs.get("scattered", False)
            self.scatter_coeff = kwargs.get("scatter_coeff", viewing.DEFAULT_SCATTER_COEFF)
            self.scale_bar, self.alpha = kwargs.get("scale_bar", True), kwargs.get("alpha", 0.05)
            self.stress = kwargs.get("stress", "full")
            self.window = None
            self.cut = None
            self.slice_plane = None
            self.image = mask_image(workflow.mask)
            self.pixel = float(workflow.spacing[0])

        def draw_mask(self) -> None:
            rows, columns = self.image.shape[:2]
            h = self.pixel
            corners = np.array([[-0.5, -0.5], [-0.5, columns - 0.5], [rows - 0.5, columns - 0.5],
                                [rows - 0.5, -0.5]]) * h
            extent = max(rows, columns) * h
            quad = ps.register_surface_mesh(viewing._MASK, planar_scene_points(corners, -_IMAGE_DEPTH * extent),
                                            np.array([[0, 1, 2], [0, 2, 3]])[:, [0, 2, 1]], enabled=self.show_mask)
            quad.add_parameterization_quantity("pixel coordinates", np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0],
                                                                              [0.0, 0.0]]), defined_on="vertices")
            quad.add_color_quantity("label", self.image, defined_on="texture", param_name="pixel coordinates",
                                    image_origin="upper_left", enabled=True, filter_mode="nearest")
            quad.set_material("flat")
            quad.set_back_face_policy("identical")

        def draw_mesh(self) -> None:
            mesh = PlanarMesh.from_any(self.workflow.mesh)
            used, inverse = np.unique(mesh.segments, return_inverse=True)
            network = ps.register_curve_network(viewing._RECONSTRUCTION, planar_scene_points(mesh.points[used]),
                                                inverse.reshape(-1, 2), color=(0.1, 0.1, 0.1))
            network.set_radius(_CURVE_RADIUS_PX * self.pixel, relative=False)

        def draw(self) -> None:
            ps.remove_all_structures()
            if self.window is not None:
                self.window.draw()
                return
            self.draw_mask()
            if self.workflow.mesh is not None:
                self.draw_mesh()

        def show_values(self) -> None:
            from foambryo._window_export import WindowInference
            from foambryo.results import PRESSURE_METHODS, TENSION_METHODS

            workflow, choices = self.workflow, self.workflow.inferred_with
            outcome = workflow.outcome
            if outcome is None or outcome.tensions is None:
                return
            reading = choices.angle_reading if choices.angle_reading != _default_reading() else None
            options = viewing._InferenceOptions(TENSION_METHODS[choices.tension_method],
                                                PRESSURE_METHODS[choices.pressure_method], choices.mean_tension,
                                                choices.base_pressure, choices.solver, pressures=choices.pressures,
                                                diagnostics=choices.diagnostics, angle_reading=reading, dimension=2)
            kind = "force" if outcome.pressures is not None else "tension"
            inference = WindowInference(outcome, choices.record_options(), workflow.inferred_date,
                                        workflow.inferred_seconds, workflow.warnings)
            window = _PlanarForceWindow(
                workflow.mesh, kind, outcome.tensions, outcome.pressures, options,
                f"inferred here: {outcome.library_call}", inference=inference,
                rec_bytes=workflow.reconstruction.rec_bytes,
                image=mask_image(workflow.mask, MASK_FADE_UNDER_VALUES), alpha=self.alpha,
                scattered=self.scattered, scatter_coeff=self.scatter_coeff, length_unit=workflow.unit,
                scale_bar=self.scale_bar, stress=self.stress)
            self.window = window
            window.request_redraw()

        def open(self) -> None:
            previous = _enter_planar_mode()
            try:
                ps.init()
                self.draw()
                self._home_view()
                ps.set_open_imgui_window_for_user_callback(False)
                ps.set_user_callback(self.panel)
                try:
                    ps.show()
                finally:
                    ps.clear_user_callback()
                    self.workflow.cancel()
                    if self.window is not None:
                        self.window.close()
            finally:
                _leave_planar_mode(previous)

        def _home_view(self) -> None:
            rows, columns = self.image.shape[:2]
            h = self.pixel
            centre = (0.5 * (columns - 1) * h, -0.5 * (rows - 1) * h)
            width, height = ps.get_window_size()
            aspect = width / height if height else 1.0
            half = 0.5 * max(rows * h, columns * h / aspect)
            distance = 1.15 * half / np.tan(np.radians(ps.get_vertical_fov_degrees()) / 2)
            ps.look_at((centre[0], centre[1], float(distance)), (centre[0], centre[1], 0.0))

        def panel(self) -> None:
            event = self.workflow.poll()
            if event == "reconstructed":
                self.window = None
                self.draw()
            elif event == "inferred":
                self.show_values()
            viewing._begin_panel()
            psim.TextWrapped(self.workflow.progress_text() or self.workflow.status)
            if self.window is not None:
                if psim.CollapsingHeader("Values", psim.ImGuiTreeNodeFlags_DefaultOpen):
                    self.window.panel_body(methods=False)
                self.scattered, self.scatter_coeff = self.window.scattered, self.window.scatter_coeff
            for title, section in (("Mask and reconstruction", self._panel_mask), ("Inference", self._panel_inference)):
                if psim.CollapsingHeader(title, psim.ImGuiTreeNodeFlags_DefaultOpen):
                    section()
            psim.End()
            if self.window is not None:
                self.window._label_overlays()

        def _panel_mask(self) -> None:
            workflow = self.workflow
            spacing = " x ".join(f"{v:g}" for v in workflow.spacing)
            psim.TextWrapped(f"Mask {workflow.mask_name}: {workflow.mask.shape[0]} x {workflow.mask.shape[1]} "
                             f"pixels of {spacing} {workflow.unit}")
            changed, self.show_mask = psim.Checkbox("show the mask (one colour per label)", self.show_mask)
            if changed and ps.has_surface_mesh(viewing._MASK):
                ps.get_surface_mesh(viewing._MASK).set_enabled(self.show_mask)
            busy = workflow.busy
            _, workflow.min_distance = psim.InputInt("min distance (square pixels)", workflow.min_distance)
            workflow.min_distance = max(1, workflow.min_distance)
            psim.TextDisabled("a 2-D mask is meshed by delaunay-watershed-2d in about a second; no estimate")
            if psim.Button("Reconstruct") and not busy:
                workflow.start_reconstruction()
            psim.SameLine()
            if psim.Button("Cancel") and busy:
                workflow.cancel()
            if self.window is None:
                viewing._panel_background(None)

        def _panel_inference(self) -> None:
            from dataclasses import replace

            workflow = self.workflow
            choices = workflow.choices
            psim.TextUnformatted("tension method: young-dupre (the one offered in 2D)")
            readings = list(_readings())
            index = readings.index(choices.angle_reading) if choices.angle_reading in readings else 0
            _, index = psim.Combo("angle reading", index, readings)
            _, with_pressures = psim.Checkbox("pressures (off: the tensions only)", choices.pressures)
            if with_pressures:
                psim.TextDisabled("pressure method: laplace (the one offered in 2D)")
            _, solver = psim.Combo("solver", viewing._SOLVERS.index(choices.solver), list(viewing._SOLVERS))
            _, diagnostics = psim.Combo("diagnostics (for Export)", viewing._DIAGNOSTICS.index(choices.diagnostics),
                                        list(viewing._DIAGNOSTICS))
            workflow.choices = replace(choices, angle_reading=readings[index], pressures=with_pressures,
                                       solver=viewing._SOLVERS[solver], diagnostics=viewing._DIAGNOSTICS[diagnostics])
            self._panel_infer_buttons()

    _CLASSES.extend([_PlanarForceWindow, _PlanarMaskWindow])
    return _CLASSES


def _quantile_fractions(values: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """`values` as fractions of the range between their `alpha` and `1 - alpha` quantiles (clipped); `nan` kept."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.full(len(values), np.nan)
    low, high = np.quantile(finite, [alpha, 1 - alpha])
    with np.errstate(invalid="ignore"):
        fractions = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else np.where(
            np.isfinite(values), 0.5, np.nan)
    return fractions


def _frame_nodes(origin: NDArray[np.float64], length: float) -> NDArray[np.float64]:
    """The frame's corner and the tips of `x` (along the columns) and `y` (along the rows), `(row, column)`."""
    return np.vstack([origin, origin + np.array([0.0, length]), origin + np.array([length, 0.0])])


def _readings() -> tuple[str, ...]:
    from foambryo._two_dimensional import ANGLE_READINGS

    return tuple(ANGLE_READINGS)


def _default_reading() -> str:
    from foambryo._two_dimensional import DEFAULT_ANGLE_READING

    return DEFAULT_ANGLE_READING


def _tension_of(tensions: dict, pair: tuple[int, int]) -> float:
    if pair in tensions:
        return float(tensions[pair])
    reverse = (pair[1], pair[0])
    return float(tensions[reverse]) if reverse in tensions else float("nan")


#: Under the values, the mask's colours are faded this far towards white, so that they do not compete with the
#: tension colours of the curves drawn over them.
MASK_FADE_UNDER_VALUES = 0.65


def mask_image(mask: NDArray[np.integer], fade: float = 0.0) -> NDArray[np.float64]:
    """A 2-D mask as an RGB image: each cell in its label colour (`label_colours`), the medium white.

    `fade` in [0, 1] mixes the cells' colours with white (0: unchanged).
    """
    mask = np.asarray(mask)
    image = np.ones((*mask.shape, 3), dtype=np.float64)
    cells = mask > 0
    if cells.any():
        image[cells] = (1.0 - fade) * _label_rgb(mask[cells]) + fade
    return image


def _enter_planar_mode() -> str | None:
    """Planar navigation and no ground plane, for a 2-D window; returns the navigation style to restore."""
    import polyscope

    polyscope.init()
    try:
        previous = polyscope.get_navigation_style()
    except Exception:  # an older Polyscope: restore the default
        previous = None
    polyscope.set_navigation_style("planar")
    polyscope.set_ground_plane_mode("none")
    return previous


def _leave_planar_mode(previous: str | None) -> None:
    """Restore the navigation style a later 3-D window of the same process expects."""
    import polyscope

    polyscope.set_navigation_style(previous if previous is not None else "turntable")
