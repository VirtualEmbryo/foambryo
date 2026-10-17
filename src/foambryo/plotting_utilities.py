"""Drawing helpers for the Polyscope viewer: values on a mesh's surface, crop faces, and triple lines.

Sacha Ichbiah 2021
Matthieu Perez 2024

`foambryo.viewing`'s viewers are built from these functions; they can also be called on their own. They need
Polyscope (`pip install "foambryo[viewing]"`), and `show=True` needs a display. Tensions are coloured with the
library's tension colours (`view_tensions_on_mesh`: `coolwarm`, centred on the mean tension); other per-interface
or per-vertex quantities with `jet` unless another map is given. An undetermined value (`nan`) is drawn in
`UNDETERMINED_COLOR` (green, #1b9e77); crop faces are drawn apart, as see-through glass (`view_crop_faces`).
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import cm
from numpy.typing import NDArray

# The colour of a face or a cell whose value is undetermined (`nan`): an interface no equation
# fixes, or a cell whose pressure is not reported (an undetermined or a cut cell), and the crop faces' glass.
# Defined with the colour scales, which the colour bars share without importing Polyscope.
from foambryo._colour_scales import (
    CROP_FACE_COLOR,
    CROP_FACE_EDGE_COLOR,
    CROP_FACE_OPACITY,
    UNDETERMINED_COLOR,
    clipped_colour_fractions,
    quantile_colour_range,
    tension_colour_fractions,
    tension_colour_range,
    tension_colours,
)

# `ps` is Polyscope, or, on a thread that prepares the viewer's scene, the recorder of its calls
# (`foambryo._polyscope_recording`).
from foambryo._polyscope_recording import polyscope_target as ps

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


##Support functions, simple mesh

#: The name of the per-face transparency (1 drawn, 0 hidden) that hides the crop faces of the coloured surface: they
#: are drawn as their own glass surface (`view_crop_faces`), never in a value's colour.
CROP_FACES_HIDDEN = "crop faces hidden (drawn as glass)"


def view_faces_values_on_embryo(
    mesh: "DcelData",
    face_values: NDArray[np.float64],
    name_values: str = "Values",
    ps_mesh: ps.SurfaceMesh | None = None,
    colormap: "cm.colors.ColorMap" = cm.jet,
    min_to_zero: bool = True,
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
    highlight_junctions: bool = False,
    adapt_values: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """Colour a surface mesh's triangles by one value each, as a Polyscope colour quantity.

    The values are mapped through `colormap` (expected in [0, 1]); a `nan` value is drawn `UNDETERMINED_COLOR`
    (green). The surface is registered as "Embryo" unless `ps_mesh` is given.

    Args:
        mesh (DcelData): the mesh.
        face_values (NDArray[np.float64]): one value per row of `mesh.f` (per triangle).
        name_values (str, optional): the colour quantity's name in the viewer. Defaults to "Values".
        ps_mesh (ps.SurfaceMesh | None, optional): a registered surface to add the quantity to; its triangles must
            be those drawn (`mesh.f`, or `mesh.f_scattered` with `scattered`). Defaults to None.
        colormap (matplotlib colormap, optional): the colour map. Defaults to `cm.jet`.
        min_to_zero (bool, optional): with `adapt_values`, subtract the minimum first. Defaults to True.
        clean_before (bool, optional): remove every Polyscope structure first. Defaults to True.
        clean_after (bool, optional): remove every Polyscope structure before returning (after the window, with
            `show`). Defaults to True.
        show (bool, optional): open Polyscope's window (needs a display); returns when it is closed. Defaults to
            True.
        highlight_junctions (bool, optional): also draw the triple lines, in white. Defaults to False.
        adapt_values (bool, optional): divide the values by their range (max - min), after subtracting the
            minimum with `min_to_zero`, so that they span [0, 1]. Defaults to True.
        scattered (bool, optional): draw the scattered (exploded) mesh; `mesh.compute_scattered_arrays` must have
            been called. Defaults to False.

    Returns:
        ps.SurfaceMesh: the surface the quantity was added to.
    """
    if scattered:
        v, f, idx = mesh.v_scattered, mesh.f_scattered, mesh.idx_scattered

    else:
        v, f, idx = mesh.v, mesh.f, np.arange(len(mesh.f))

    values = face_values.copy()
    if adapt_values:
        if min_to_zero:
            values -= np.amin(values)
        values /= np.amax(values) - np.amin(values)

    values = values[idx]
    colors_face = colormap(values)[:, :3]
    # A face whose value is undetermined (nan) is drawn `UNDETERMINED_COLOR` (green); with every value finite
    # this changes nothing.
    colors_face[np.isnan(values)] = UNDETERMINED_COLOR

    ps.init()

    if clean_before:
        ps.remove_all_structures()

    if ps_mesh is None:
        ps_mesh = ps.register_surface_mesh("Embryo", v, f[:, [0, 1, 2]])

    ps_mesh.set_color((0.3, 0.6, 0.8))  # rgb triple on [0,1]
    # ps_mesh.set_transparency(0.2)
    ps_mesh.add_color_quantity(name_values, colors_face, defined_on="faces", enabled=True)

    if highlight_junctions:
        plot_trijunctions(
            mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            color="uniform",
            value_color=np.ones(3),
        )

    ps.set_ground_plane_mode("none")

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()

    return ps_mesh


def view_vertex_values_on_embryo(  # noqa: C901
    mesh: "DcelData",
    value_on_vertices: NDArray[np.float64],
    remove_trijunctions: bool = True,
    ps_mesh: ps.SurfaceMesh | None = None,
    name_values: str = "Values",
    clean_after: bool = True,
    show: bool = True,
    highlight_junction: bool = False,
    scattered: bool = True,
) -> ps.SurfaceMesh:
    """Colour a surface mesh's triangles by values given on its vertices, with `jet`.

    Each triangle takes the mean of its vertices' values (without the vertices on a triple line when
    `remove_trijunctions`; a triangle with no vertex left takes the mean over the other triangles); the triangle
    values are then stretched to [0, 1] between their minimum and maximum.

    Args:
        mesh (DcelData): the mesh.
        value_on_vertices (NDArray[np.float64]): one value per vertex (per row of `mesh.v`).
        remove_trijunctions (bool, optional): leave out the vertices on a triple line, where these quantities are
            not defined. Defaults to True.
        ps_mesh (ps.SurfaceMesh | None, optional): a registered surface to add the quantity to; else "Embryo" is
            registered. Defaults to None.
        name_values (str, optional): the colour quantity's name in the viewer. Defaults to "Values".
        clean_after (bool, optional): remove every Polyscope structure before returning. Defaults to True.
        show (bool, optional): open Polyscope's window (needs a display); returns when it is closed. Defaults to
            True.
        highlight_junction (bool, optional): also draw the triple lines, in white. Defaults to False.
        scattered (bool, optional): draw the scattered (exploded) mesh; `mesh.compute_scattered_arrays` must have
            been called. Defaults to True.

    Returns:
        ps.SurfaceMesh: the surface the quantity was added to.
    """
    v, f = mesh.v, mesh.f

    values = np.zeros(len(mesh.f))

    mesh.mark_trijunctional_vertices()
    valid_values = []
    indices_nan = []
    for i, face in enumerate(f[:, [0, 1, 2]]):
        a, b, c = face
        liste = []
        for vert_idx in face:
            if remove_trijunctions:
                if not mesh.vertices[vert_idx].on_trijunction:
                    liste.append(value_on_vertices[vert_idx])
            else:
                liste.append(value_on_vertices[vert_idx])
        if len(liste) > 0:
            values[i] = np.mean(np.array(liste))
            valid_values.append(values[i])
        else:
            indices_nan.append(i)

    mean = np.mean(np.array(valid_values))
    for i in indices_nan:
        values[i] = mean

    ps.init()

    values -= np.amin(values)
    values /= np.amax(values)

    if scattered:
        v, f, idx = mesh.v_scattered, mesh.f_scattered, mesh.idx_scattered
        values = values[idx]

    colors_face = cm.jet(values)[:, :3]

    if ps_mesh is None:
        ps_mesh = ps.register_surface_mesh("Embryo", v, f[:, [0, 1, 2]])
    ps_mesh.add_color_quantity(name_values, colors_face, defined_on="faces", enabled=True)

    if highlight_junction:
        plot_trijunctions(
            mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            color="uniform",
            value_color=np.ones(3),
        )

    ps.set_ground_plane_mode("none")
    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()

    return ps_mesh


def view_dict_values_on_mesh(
    mesh: "DcelData",
    dict_values: dict[tuple[int, int], float],
    alpha: float = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
    name_values: str = "Values",
    alpha_values: bool = True,
    min_value: float | None = None,
    max_value: float | None = None,
    cmap: "cm.colors.ColorMap" = cm.jet,
) -> ps.SurfaceMesh:
    """Colour a surface mesh's triangles by a value per interface (a seed spread, a residual), with `jet`.

    Not for tensions: they have their own colours, `view_tensions_on_mesh`. The values are clipped to a range and
    stretched to [0, 1] (`clipped_colour_fractions`): by default between the `alpha` and `1 - alpha` quantiles of
    the finite values (`quantile_colour_range`, the range the viewer's colour bars draw). An interface whose value
    is `nan`, or that is absent from the map (a crop interface), is drawn `UNDETERMINED_COLOR` (green). The range
    is printed.

    Args:
        mesh (DcelData): the mesh.
        dict_values (dict[tuple[int, int], float]): one value per interface, keyed `(label_1, label_2)` with
            `label_1 < label_2`.
        alpha (float, optional): the quantile clipped at each end, in [0, 0.5). Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): a registered surface to add the quantity to; else "Embryo" is
            registered. Defaults to None.
        clean_before (bool, optional): remove every Polyscope structure first. Defaults to True.
        clean_after (bool, optional): remove every Polyscope structure before returning. Defaults to True.
        show (bool, optional): open Polyscope's window (needs a display); returns when it is closed. Defaults to
            True.
        scattered (bool, optional): draw the scattered (exploded) mesh; `mesh.compute_scattered_arrays` must have
            been called. Defaults to False.
        name_values (str, optional): the colour quantity's name in the viewer. Defaults to "Values".
        alpha_values (bool, optional): clip at the quantiles; `False` clips to `[min_value, max_value]` instead.
            Defaults to True.
        min_value (float | None, optional): the low end of the range when `alpha_values` is False. Defaults to
            None.
        max_value (float | None, optional): the high end of the range when `alpha_values` is False. Defaults to
            None.
        cmap (matplotlib colormap, optional): the colour map. Defaults to `cm.jet`.

    Returns:
        ps.SurfaceMesh: the surface the quantity was added to.

    Raises:
        IndexError: with `alpha_values` and no finite value (numpy's error on an empty quantile).
    """
    _, f = mesh.v, mesh.f

    def _find_values(triangles_and_labels: NDArray[np.int64]) -> float:
        # An interface absent from the map (a crop interface, which carries no tension) is drawn
        # like an undetermined one.
        return dict_values.get(tuple(triangles_and_labels[[3, 4]]), np.nan)

    values = np.array(list(map(_find_values, f)), dtype=np.float64)
    values_values = np.array(list(dict_values.values()), dtype=np.float64)
    # Undetermined values (nan) take no part in the scale; with every value finite this is the
    # same array.
    values_values = values_values[np.isfinite(values_values)]
    if alpha_values:
        # The same range `foambryo.viewing.save_colour_bars` draws.
        mint, maxt = quantile_colour_range(values_values, alpha)
    else:
        mint = min_value
        maxt = max_value
    print("Extremas of the " + name_values + " plotted : ", mint, maxt)
    values = clipped_colour_fractions(values, (mint, maxt))

    ps_mesh = view_faces_values_on_embryo(
        mesh,
        values,
        ps_mesh=ps_mesh,
        name_values=name_values,
        colormap=cmap,
        clean_before=clean_before,
        clean_after=clean_after,
        show=show,
        adapt_values=False,
        scattered=scattered,
    )

    return ps_mesh


def view_tensions_on_mesh(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    alpha: float = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    scattered: bool = False,
    name_values: str = "Surface Tensions",
) -> ps.SurfaceMesh:
    """Colour the surface's triangles by their interface's tension, with the library's tension colours.

    `coolwarm` over `tension_colour_range` (centred on the mean of the finite tensions, symmetric around it, its
    half-width the farther of the `alpha` and `1 - alpha` quantiles from the mean); an undetermined tension, or an
    interface absent from the map (a crop interface), is filled `UNDETERMINED_COLOR` (green) here: the viewer's
    window draws such an interface as its outline instead. With no finite tension, every triangle is green. The
    range is printed, as `view_dict_values_on_mesh` prints its own. The crop faces are hidden on this surface
    (`hide_crop_faces`). Does not open a window.

    Args:
        mesh (DcelData): the mesh.
        dict_tensions (dict[tuple[int, int], float]): tension per interface, keyed `(label_1, label_2)` with
            `label_1 < label_2`.
        alpha (float, optional): the quantile at each end, in [0, 0.5). Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): the surface to add the colours to, else "Embryo" is registered.
            Defaults to None.
        scattered (bool, optional): the scattered view's triangles; `mesh.compute_scattered_arrays` must have
            been called. Defaults to False.
        name_values (str, optional): the colour quantity's name. Defaults to "Surface Tensions".

    Returns:
        ps.SurfaceMesh: the surface.
    """
    values = np.array([dict_tensions.get(tuple(row), np.nan) for row in mesh.f[:, 3:5]], dtype=np.float64)
    tensions = np.array(list(dict_tensions.values()), dtype=np.float64)
    if np.isfinite(tensions).any():
        colour_range = tension_colour_range(tensions, alpha)
        print("Extremas of the " + name_values + " plotted : ", *colour_range)
        colours = tension_colours(tension_colour_fractions(values, colour_range))
    else:
        colours = np.tile(UNDETERMINED_COLOR, (len(values), 1))
    if scattered:
        v, f, idx = mesh.v_scattered, mesh.f_scattered, mesh.idx_scattered
    else:
        v, f, idx = mesh.v, mesh.f, np.arange(len(mesh.f))
    ps.init()
    if ps_mesh is None:
        ps_mesh = ps.register_surface_mesh("Embryo", v, f[:, [0, 1, 2]])
    ps_mesh.set_color((0.3, 0.6, 0.8))
    ps_mesh.add_color_quantity(name_values, colours[idx], defined_on="faces", enabled=True)
    hide_crop_faces(mesh, ps_mesh, scattered=scattered)
    return ps_mesh


def hide_crop_faces(mesh: "DcelData", ps_mesh: ps.SurfaceMesh, scattered: bool = False) -> None:
    """Hide the crop faces of a coloured surface (a per-face transparency, `CROP_FACES_HIDDEN`): they carry no value.

    They are drawn by `view_crop_faces` as glass instead. Does nothing on a mesh without crop faces.

    Args:
        mesh (DcelData): the mesh.
        ps_mesh (ps.SurfaceMesh): the coloured surface, with the triangles of `mesh` (or of its scattered view).
        scattered (bool, optional): `ps_mesh` has the scattered view's triangles. Defaults to False.
    """
    if not getattr(mesh, "has_crop_faces", False):
        return
    idx = mesh.idx_scattered if scattered else np.arange(len(mesh.f))
    crop = np.asarray(mesh.crop_faces, dtype=bool)[idx]
    ps_mesh.add_scalar_quantity(CROP_FACES_HIDDEN, np.where(crop, 0.0, 1.0), defined_on="faces")
    ps_mesh.set_transparency_quantity(CROP_FACES_HIDDEN)


def view_crop_faces(mesh: "DcelData", scattered: bool = False) -> ps.SurfaceMesh | None:
    """Draw the mesh's crop faces as their own surface, "Crop faces": see-through glass with its edges drawn.

    Crop faces lie on the boundary of the imaged volume, where the tissue was cut (see
    `foambryo.crop_faces`); they carry no tension, and the cells they touch no reported pressure.
    Drawing them apart, in `CROP_FACE_COLOR` at `CROP_FACE_OPACITY` with edges in `CROP_FACE_EDGE_COLOR`, keeps
    them from reading as cell surfaces with a value; uncheck the structure to hide them. Polyscope's transparency
    mode is set to "pretty" (depth peeling), the one that draws a see-through surface over others correctly. Does
    nothing on a mesh without crop faces.

    Args:
        mesh (DcelData): the mesh.
        scattered (bool, optional): draw the scattered mesh's crop faces. Defaults to False.

    Returns:
        ps.SurfaceMesh | None: the registered surface, or `None` without crop faces.
    """
    if not getattr(mesh, "has_crop_faces", False):
        return None
    if scattered:
        v, f, idx = mesh.v_scattered, mesh.f_scattered, mesh.idx_scattered
    else:
        v, f, idx = mesh.v, mesh.f, np.arange(len(mesh.f))
    crop = np.asarray(mesh.crop_faces)[idx]
    ps.init()
    ps_crop = ps.register_surface_mesh("Crop faces", v, f[crop][:, [0, 1, 2]])
    ps_crop.set_color(tuple(float(c) for c in CROP_FACE_COLOR))
    ps_crop.set_transparency(CROP_FACE_OPACITY)
    ps_crop.set_edge_width(1.0)
    ps_crop.set_edge_color(tuple(float(c) for c in CROP_FACE_EDGE_COLOR))
    ps_crop.set_back_face_policy("identical")
    ps.set_transparency_mode("pretty")
    return ps_crop


def plot_trijunctions(  # noqa: C901
    mesh: "DcelData",
    dict_trijunctional_values: dict[Iterable[int], float] | None = None,
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
    color: Literal["values"] | Literal["uniform"] = "values",
    value_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cmap: "cm.colors.ColorMap" = cm.jet,
) -> None:
    """Draw a mesh's triple lines as a Polyscope curve network, "trijunctions", in one colour or by value.

    A triple line is keyed by the sorted labels of the materials around it (three, or more at a quadruple edge).
    With `color="values"`, each line takes its value in `dict_trijunctional_values` (0 when absent), divided by the
    largest value and coloured with `cmap`.

    Args:
        mesh (DcelData): the mesh.
        dict_trijunctional_values (dict[Iterable[int], float] | None, optional): a value per triple line, keyed by
            its sorted labels `(a, b, c)`. Needed with `color="values"`. Defaults to None.
        clean_before (bool, optional): remove every Polyscope structure first. Defaults to True.
        clean_after (bool, optional): remove every Polyscope structure before returning. Defaults to True.
        show (bool, optional): open Polyscope's window (needs a display); returns when it is closed. Defaults to
            True.
        color (Literal["values"] | Literal["uniform"], optional): colour by value, or every line in `value_color`.
            Defaults to "values".
        value_color (tuple[float, float, float], optional): the RGB colour, in [0, 1], with `color="uniform"`.
            Defaults to (1.0, 1.0, 1.0).
        cmap (matplotlib colormap, optional): the colour map with `color="values"`. Defaults to `cm.jet`.
    """
    dict_trijunctions = {}

    for edge in mesh.half_edges:
        if len(edge.twin) > 1:
            list_materials = []
            for a in edge.twin:
                list_materials.append(mesh.half_edges[a].incident_face.material_1)
                list_materials.append(mesh.half_edges[a].incident_face.material_2)
            list_materials = np.unique(list_materials)
            key_junction = tuple(list_materials)
            dict_trijunctions[key_junction] = [
                *dict_trijunctions.get(key_junction, []),
                [edge.origin.key, edge.destination.key],
            ]

    ps.init()
    if clean_before:
        ps.remove_all_structures()

    if color == "uniform":
        plotted_edges = []
        edges = []
        verts = []
        i = 0
        for key in dict_trijunctions:
            if len(key) >= 3:
                for _ in range(len(np.array(dict_trijunctions[key]))):
                    plotted_edges.append([2 * i, 2 * i + 1])
                    i += 1
                edges.append(np.array(dict_trijunctions[key]))
                verts.append(mesh.v[edges[-1]].reshape(-1, 3))

        edges = np.vstack(edges)
        verts = np.vstack(verts)
        plotted_edges = np.array(plotted_edges)
        ps.register_curve_network("trijunctions", verts, plotted_edges, color=value_color)

    else:
        edges = []
        verts = []
        plotted_edges = []
        edges_values = []
        i = 0
        for key in dict_trijunctions:
            if len(key) >= 3:
                edges.append(np.array(dict_trijunctions[key]))
                verts.append(mesh.v[edges[-1]].reshape(-1, 3))
                for _ in range(len(edges[-1])):
                    plotted_edges.append([2 * i, 2 * i + 1])
                    i += 1
                    edges_values.append(dict_trijunctional_values.get(key, 0))

        edges = np.vstack(edges)
        verts = np.vstack(verts)
        plotted_edges = np.array(plotted_edges)
        curv_net = ps.register_curve_network(
            "trijunctions",
            verts,
            plotted_edges,
            color=np.random.default_rng().random(3),
        )

        edges_values = np.array(edges_values)
        edges_values /= np.amax(edges_values)
        color_values = cmap(edges_values)[:, :3]
        curv_net.add_color_quantity("line tensions", color_values, defined_on="edges", enabled=True)

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()


def plot_trijunctions_topo_change_viewer(  # noqa: C901
    mesh: "DcelData",
    dict_trijunctional_values: dict[Iterable[int], float] | None = None,
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
    color: Literal["values"] | Literal["uniform"] = "values",
    value_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Draw a mesh's triple lines in two groups: red where a line's value is 1, green elsewhere.

    The drawing behind `foambryo.viewing.plot_valid_junctions`: with `color="values"`, a line whose value in
    `dict_trijunctional_values` is 1 is drawn red ("bad trijunctions", opaque), every other line, an absent one
    included, green ("valid trijunctions", opacity 0.3). With `color="uniform"`, every line is drawn in
    `value_color` ("trijunctions"). A group with no line is not drawn (every line of a mesh can be bad).

    Args:
        mesh (DcelData): the mesh.
        dict_trijunctional_values (dict[Iterable[int], float] | None, optional): a value per triple line, keyed by
            its sorted labels `(a, b, c)`; 1 marks a line to draw red. Needed with `color="values"`. Defaults to
            None.
        clean_before (bool, optional): remove every Polyscope structure first. Defaults to True.
        clean_after (bool, optional): remove every Polyscope structure before returning. Defaults to True.
        show (bool, optional): open Polyscope's window (needs a display); returns when it is closed. Defaults to
            True.
        color (Literal["values"] | Literal["uniform"], optional): two groups by value, or every line in
            `value_color`. Defaults to "values".
        value_color (tuple[float, float, float], optional): the RGB colour, in [0, 1], with `color="uniform"`.
            Defaults to (1.0, 1.0, 1.0).
    """
    dict_trijunctions = {}

    for edge in mesh.half_edges:
        if len(edge.twin) > 1:
            list_materials = []
            for a in edge.twin:
                list_materials.append(mesh.half_edges[a].incident_face.material_1)
                list_materials.append(mesh.half_edges[a].incident_face.material_2)
            list_materials = np.unique(list_materials)
            key_junction = tuple(list_materials)
            dict_trijunctions[key_junction] = [
                *dict_trijunctions.get(key_junction, []),
                [edge.origin.key, edge.destination.key],
            ]

    ps.init()
    if clean_before:
        ps.remove_all_structures()

    if color == "uniform":
        plotted_edges = []
        edges = []
        verts = []
        i = 0
        for key in dict_trijunctions:
            if len(key) >= 3:
                for _ in range(len(np.array(dict_trijunctions[key]))):
                    plotted_edges.append([2 * i, 2 * i + 1])
                    i += 1
                edges.append(np.array(dict_trijunctions[key]))
                verts.append(mesh.v[edges[-1]].reshape(-1, 3))

        edges = np.vstack(edges)
        verts = np.vstack(verts)
        plotted_edges = np.array(plotted_edges)
        ps.register_curve_network("trijunctions", verts, plotted_edges, color=value_color)

    else:
        edges_r = []
        verts_r = []
        edges_g = []
        verts_g = []
        green_edges = []
        red_edges = []
        edges_values = []
        b = 0
        r = 0
        for key in dict_trijunctions:
            if len(key) >= 3:
                edges_values.append(np.clip(dict_trijunctional_values.get(key, 0), 0, None))
                if edges_values[-1] == 1:
                    for _ in range(len(dict_trijunctions[key])):
                        red_edges.append([2 * r, 2 * r + 1])
                        r += 1
                    edges_r.append(np.array(dict_trijunctions[key]))
                    verts_r.append(mesh.v[edges_r[-1]].reshape(-1, 3))
                else:
                    for _ in range(len(dict_trijunctions[key])):
                        green_edges.append([2 * b, 2 * b + 1])
                        b += 1

                    edges_g.append(np.array(dict_trijunctions[key]))
                    verts_g.append(mesh.v[edges_g[-1]].reshape(-1, 3))

        # A group with no line is not drawn: numpy cannot stack no line, and every line of a mesh can be bad.
        if verts_g:
            ps.register_curve_network(
                "valid trijunctions",
                np.vstack(verts_g),
                np.array(green_edges),
                color=(0, 1, 0),
                transparency=0.3,
            )
        if verts_r:
            ps.register_curve_network(
                "bad trijunctions",
                np.vstack(verts_r),
                np.array(red_edges),
                color=(1, 0, 0),
                transparency=1.0,
            )

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()
