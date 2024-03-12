"""Module defining several plotting scenarii for a polyscope viewer.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import polyscope as ps
from matplotlib import cm
from numpy.typing import NDArray

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


##Support functions, simple mesh


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
    """Visualize face_values data on a surface mesh.

    Args:
        mesh (DcelData): Mesh data to visualize.
        face_values (NDArray[np.float64]): Data array to visualize, per mesh triangle.
        name_values (str, optional): Name of the data in viewer. Defaults to "Values".
        ps_mesh (ps.SurfaceMesh|None, optional): If given, data is added to it and "mesh" is not used. Defaults to None.
        colormap ("cm.colors.ColorMap", optional): Matplotlib colormap for the data. Defaults to cm.jet.
        min_to_zero (bool, optional): Map the minimum value to 0, if adapt_values is True too. Defaults to True.
        clean_before (bool, optional): Clean polyscope viewer before adding the mesh. Defaults to True.
        clean_after (bool, optional): Clean polyscope viewer after adding the mesh. Defaults to True.
        show (bool, optional): Show polyscope viewer directly. Defaults to True.
        highlight_junctions (bool, optional): Show trijunctions on mesh. Defaults to False.
        adapt_values (bool, optional): Values are stretched so that min & max have a difference of 1. Defaults to True.
        scattered (bool, optional): Separate individual cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: _description_
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
    """Plot a Surface Mesh with colors based on data on the mesh's vertices.

    Args:
        mesh (DcelData): Mesh data to show.
        value_on_vertices (NDArray[np.float64]): Data to show.
        remove_trijunctions (bool, optional): Do not show data on trijunctions. Defaults to True.
        ps_mesh (ps.SurfaceMesh | None, optional): Polyscope mesh pre-existing replacing the mesh. Defaults to None.
        name_values (str, optional): Name of the data in the viewer. Defaults to "Values".
        clean_after (bool, optional): Clean polyscope viewer after this function. Defaults to True.
        show (bool, optional): Show polyscope viewer directly. Defaults to True.
        highlight_junction (bool, optional): Show junctions. Defaults to False.
        scattered (bool, optional): Use scattered version of the mesh, with cells separated. Defaults to True.

    Returns:
        ps.SurfaceMesh: Polyscope mesh with associated data visualized.
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
    """View the mesh with data defined per interface.

    Args:
        mesh (DcelData): Mesh to view.
        dict_values (dict[tuple[int, int], float]): Data on interfaces
        alpha (float, optional): Percentage to clip min & max value. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Polyscope mesh already instead of the mesh. Defaults to None.
        clean_before (bool, optional): Clear polyscope viewer before this function. Defaults to True.
        clean_after (bool, optional): Clear polyscope viewer after this function. Defaults to True.
        show (bool, optional): Show polyscope viewer directly. Defaults to True.
        scattered (bool, optional): Show scattered mesh. Defaults to False.
        name_values (str, optional): Name of the data in the viewer. Defaults to "Values".
        alpha_values (bool, optional): Use alpha parameter. Defaults to True.
        min_value (float | None, optional): If not alpha_values, then clip data with this value. Defaults to None.
        max_value (float | None, optional): If not alpha_values, then clip data with this value. Defaults to None.
        cmap ("cm.colors.ColorMap", optional): Colormap to color the data. Defaults to cm.jet.

    Returns:
        ps.SurfaceMesh: _description_
    """
    _, f = mesh.v, mesh.f

    def _find_values(triangles_and_labels: NDArray[np.int64]) -> float:
        return dict_values[tuple(triangles_and_labels[[3, 4]])]

    values = np.array(list(map(_find_values, f)))
    values_values = np.array(list(dict_values.values()))
    if alpha_values:
        mint = np.quantile(values_values, alpha)
        maxt = np.quantile(values_values, 1 - alpha)
    else:
        mint = min_value
        maxt = max_value
    values = values.clip(mint, maxt)
    print("Extremas of the " + name_values + " plotted : ", mint, maxt)
    values -= np.amin(mint)
    values /= np.amax(maxt - mint)

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
    """Plot trijunctions curve network of a mesh with optional data.

    Args:
        mesh (DcelData): Mesh to show.
        dict_trijunctional_values (dict[Iterable[int], float] | None, optional): Data to plot on junctions.
            Defaults to None.
        clean_before (bool, optional): Clean polyscope viewer before this function. Defaults to True.
        clean_after (bool, optional): Clean polyscope viewer after this function. Defaults to True.
        show (bool, optional): Sho directly polyscope viewer. Defaults to True.
        color (Literal["values"] | Literal["uniform"], optional): Show values from dict or one fixed color.
            Defaults to "values".
        value_color (tuple[float, float, float], optional): if color=="uniform", show junctions with this color.
            Defaults to (1.0, 1.0, 1.0).
        cmap ("cm.colors.ColorMap", optional): if color=="values" show junctions with this colormap. Defaults to cm.jet.
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
    """Plot trijunctions curve network of a mesh with optional data. The color are not from a colormap.

    TODO: find what this function does ;-) please document your own functions ;-)

    Args:
        mesh (DcelData): Mesh to show.
        dict_trijunctional_values (dict[Iterable[int], float] | None, optional): Data to plot on junctions.
            Defaults to None.
        clean_before (bool, optional): Clean polyscope viewer before this function. Defaults to True.
        clean_after (bool, optional): Clean polyscope viewer after this function. Defaults to True.
        show (bool, optional): Sho directly polyscope viewer. Defaults to True.
        color (Literal["values"] | Literal["uniform"], optional): Show values from dict or one fixed color.
            Defaults to "values".
        value_color (tuple[float, float, float], optional): if color=="uniform", show junctions with this color.
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

        verts_r = np.vstack(verts_r)
        verts_g = np.vstack(verts_g)
        red_edges = np.array(red_edges)
        green_edges = np.array(green_edges)
        ps.register_curve_network(
            "valid trijunctions",
            verts_g,
            green_edges,
            color=(0, 1, 0),
            transparency=0.3,
        )
        ps.register_curve_network("bad trijunctions", verts_r, red_edges, color=(1, 0, 0), transparency=1.0)

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()
