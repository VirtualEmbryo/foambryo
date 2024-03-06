"""Viewer part of foambryo.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import polyscope as ps
from matplotlib import cm
from numpy.linalg import eig
from numpy.typing import NDArray

from foambryo.curvature import (
    compute_curvature_vertices_cotan,
    compute_gaussian_curvature_vertices,
    compute_sphere_fit_residues_dict,
)
from foambryo.dcel import (
    compute_faces_areas,
    compute_normal_faces,
    update_graph_with_scattered_values,
)
from foambryo.plotting_utilities import (
    plot_trijunctions,
    plot_trijunctions_topo_change_viewer,
    view_dict_values_on_mesh,
    view_vertex_values_on_embryo,
)
from foambryo.tension_inference import (
    compute_residual_junctions_dict,
    infer_forces,
    infer_pressure,
    infer_tension,
)

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


def plot_force_inference(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float] | None = None,
    dict_pressure: dict[int, float] | None = None,
    alpha: float = 0.05,
    scalar_quantities: bool = False,
    scattered: bool = False,
    scatter_coeff: float = 0.2,
) -> None:
    """Polyscope plot of a mesh with tensions and pressures shown.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float] | None, optional):
            Tensions on the mesh interfaces (computed if None). Defaults to None.
        dict_pressure (dict[int, float] | None, optional):
            Pressures in the mesh cells (computed if None). Defaults to None.
        alpha (float, optional): Quantile to filter extreme values. Defaults to 0.05.
        scalar_quantities (bool, optional): Show scalar quantities such as curvatures. Defaults to False.
        scattered (bool, optional): Scatter cells. Defaults to False.
        scatter_coeff (float, optional): How much to scatter cells. Defaults to 0.2.
    """
    if dict_tensions is None:
        dict_tensions, dict_pressure = infer_forces(mesh)
    if dict_pressure is None:
        dict_pressure = infer_pressure(mesh, dict_tensions)

    nx_graph = mesh.compute_networkx_graph()
    nx.set_edge_attributes(nx_graph, dict_tensions, "tension")
    nx.set_node_attributes(nx_graph, dict_pressure, "pressure")

    if scattered:
        mesh.compute_scattered_arrays(coeff=scatter_coeff)
        nx_graph = update_graph_with_scattered_values(nx_graph, mesh)

    dict_sphere_fit_residual = compute_sphere_fit_residues_dict(nx_graph)

    ps_mesh = view_dict_values_on_mesh(
        mesh,
        dict_tensions,
        alpha=alpha,
        ps_mesh=None,
        scattered=scattered,
        clean_before=False,
        clean_after=False,
        show=False,
        name_values="Surface Tensions",
    )
    ps_mesh = view_pressures_on_mesh(
        mesh,
        dict_pressure,
        ps_mesh=ps_mesh,
        alpha=alpha,
        clean_before=False,
        clean_after=False,
        show=False,
        scattered=scattered,
    )
    plot_stress_tensor(
        mesh,
        nx_graph,
        dict_tensions,
        dict_pressure,
        clean_before=False,
        clean_after=False,
        show=False,
    )
    display_embryo_graph_forces(
        nx_graph,
        alpha=alpha,
        clean_before=False,
        clean_after=False,
        show=False,
        base_pressure=0,
        plot_pressures=True,
    )

    if scalar_quantities:
        ps_mesh = view_dict_values_on_mesh(
            mesh,
            dict_sphere_fit_residual,
            alpha=alpha,
            ps_mesh=ps_mesh,
            scattered=scattered,
            clean_before=False,
            clean_after=False,
            show=False,
            name_values="Sphere fit residual",
        )
        view_area_derivatives(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_volume_derivatives(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_mean_curvature_cotan(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        # view_mean_curvature_robust(
        #     mesh,
        #     alpha=alpha,
        #     ps_mesh=ps_mesh,
        #     clean_before=False,
        #     clean_after=False,
        #     show=False,
        #     scattered=scattered,
        # )
        view_gaussian_curvature(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_discrepancy_of_principal_curvatures(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )

    ps.show()


def plot_tension_inference(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float] | None = None,
    alpha: float = 0.05,
    scalar_quantities: bool = False,
    scattered: bool = False,
    scatter_coeff: float = 0.2,
) -> None:
    """Polyscope plot of a mesh with tensions and pressures shown.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float] | None, optional):
            Tensions on the mesh interfaces (computed if None). Defaults to None.
        alpha (float, optional): Quantile to filter extreme values. Defaults to 0.05.
        scalar_quantities (bool, optional): Show scalar quantities such as curvatures. Defaults to False.
        scattered (bool, optional): Scatter cells. Defaults to False.
        scatter_coeff (float, optional): How much to scatter cells. Defaults to 0.2.
    """
    ps.remove_all_structures()

    if dict_tensions is None:
        _, dict_tensions, _ = infer_tension(mesh, mean_tension=1)

    nx_graph = mesh.compute_networkx_graph()
    nx.set_edge_attributes(nx_graph, dict_tensions, "tension")

    if scattered:
        mesh.compute_scattered_arrays(coeff=scatter_coeff)
        nx_graph = update_graph_with_scattered_values(nx_graph, mesh)

    dict_sphere_fit_residual = compute_sphere_fit_residues_dict(nx_graph)

    ps_mesh = view_dict_values_on_mesh(
        mesh,
        dict_tensions,
        alpha=alpha,
        ps_mesh=None,
        scattered=scattered,
        clean_before=False,
        clean_after=False,
        show=False,
        name_values="Surface Tensions",
    )

    display_embryo_graph_forces(
        nx_graph,
        alpha=alpha,
        clean_before=False,
        clean_after=False,
        show=False,
        base_pressure=0,
        plot_pressures=False,
    )
    if scalar_quantities:
        ps_mesh = view_dict_values_on_mesh(
            mesh,
            dict_sphere_fit_residual,
            alpha=alpha,
            ps_mesh=ps_mesh,
            scattered=scattered,
            clean_before=False,
            clean_after=False,
            show=False,
            name_values="Sphere fit residual",
        )
        view_area_derivatives(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_volume_derivatives(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_mean_curvature_cotan(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        # view_mean_curvature_robust(
        #     Mesh,
        #     alpha=alpha,
        #     ps_mesh=ps_mesh,
        #     clean_before=False,
        #     clean_after=False,
        #     show=False,
        #     scattered=scattered,
        # )
        view_gaussian_curvature(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_discrepancy_of_principal_curvatures(
            mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )

    ps.show()


def view_pressures_on_mesh(
    mesh: "DcelData",
    dict_pressures: tuple[int, float],
    alpha: float = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """Get a polyscope SurfaceMesh with pressures data on cells.

    Args:
        mesh (DcelData): Mesh to show.
        dict_pressures (tuple[int, float]): Pressures already computed.
        alpha (float, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): SurfaceMesh to add the pressure data if it exists. Defaults to None.
        scattered (bool, optional): Scatters cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: A Polyscope SurfaceMesh with a pressure data on cells.
    """
    if scattered:
        v, f, cluster_idx = (
            mesh.v_scattered,
            mesh.f_scattered,
            mesh.cluster_idx_scattered,
        )  # , Mesh.idx_scattered

        def _find_pressure(idx: int) -> float:
            return dict_pressures[idx]

        pressures = np.array(list(map(_find_pressure, cluster_idx)))
    else:
        v, f = mesh.v, mesh.f

        def _find_pressure(face: NDArray[np.int64]) -> float:
            return dict_pressures[max(face[3], face[4])]

        pressures = np.array(list(map(_find_pressure, f)))

    maxp = np.quantile(pressures, 1 - alpha)

    print("Extremas of pressures plotted: ", 0, maxp)

    pressures = pressures.clip(0, maxp)
    pressures /= np.amax(pressures)
    pressures = 1 - pressures
    # ps_mesh = view_faces_values_on_embryo(Mesh,pressures,ps_mesh = ps_mesh,name_values = "Pressures Cells",colormap = cm.magma,clean_before = clean_before, clean_after=clean_after,show=show,adapt_values=False,scattered = False)  # noqa: E501

    if ps_mesh is None:
        ps_mesh = ps.register_surface_mesh("Embryo", v, f[:, [0, 1, 2]])
    ps_mesh.set_color((0.3, 0.6, 0.8))  # rgb triple on [0,1]
    colors_face = cm.magma(pressures)[:, :3]
    ps_mesh.add_color_quantity("Pressures Cells", colors_face, defined_on="faces", enabled=True)
    return ps_mesh


# def view_pressures_on_mesh(Mesh,dict_pressures,alpha = 0.05,ps_mesh = None, clean_before = True, clean_after=True,show=True, scattered = False):  # noqa: E501

#     v,f = Mesh.v,Mesh.f

#     def find_pressure(face):
#         return(dict_pressures[max(face[3],face[4])])

#     pressures = np.array(list(map(find_pressure,f)))
#     pressures_values = np.array(list(dict_pressures.values()))
#     maxp = np.quantile(pressures_values,1-alpha)

#     print("Extremas of pressures plotted: ",0,maxp)

#     pressures = pressures.clip(0,maxp)
#     pressures/=np.amax(pressures)
#     pressures = 1-pressures
#     ps_mesh = view_faces_values_on_embryo(Mesh,pressures,ps_mesh = ps_mesh,name_values = "Pressures Cells",colormap = cm.magma,clean_before = clean_before, clean_after=clean_after,show=show,adapt_values=False,scattered = scattered)  # noqa: E501

#     return(ps_mesh)


# def view_faces_values_on_embryo(Mesh,Vf,name_values = "Values",ps_mesh = None,colormap = cm.jet,min_to_zero=True,clean_before = True, clean_after=True,show=True,highlight_junctions=False,adapt_values = True, scattered = False):  # noqa: E501

#     if scattered :
#         v,f,idx = Mesh.v_scattered,Mesh.f_scattered, Mesh.idx_scattered

#     else :
#         v,f,idx = Mesh.v,Mesh.f, np.arange(len(Mesh.f))

#     Values = Vf.copy()
#     if adapt_values :
#         if min_to_zero :
#             Values-=np.amin(Values)
#         Values/=(np.amax(Values)-np.amin(Values))

#     Values = Values[idx]
#     colors_face = colormap(Values)[:,:3]

#     ps.init()

#     if clean_before :
#         ps.remove_all_structures()


#     if ps_mesh == None :
#         ps_mesh = ps.register_surface_mesh("Embryo", v,f[:,[0,1,2]])

#     ps_mesh.set_color((0.3, 0.6, 0.8)) # rgb triple on [0,1]
#     #ps_mesh.set_transparency(0.2)
#     ps_mesh.add_color_quantity(name_values, colors_face, defined_on='faces',enabled=True)


#     if highlight_junctions :
#         plot_trijunctions(Mesh,clean_before = False, clean_after = False, show=False, color = "uniform",value_color = np.ones(3))  # noqa: E501

#     ps.set_ground_plane_mode("none")

#     if show :
#         ps.show()

#     if clean_after :
#         ps.remove_all_structures()

#     return(ps_mesh)


###
# Scalar quantities
###


def view_area_derivatives(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add area derivatives data on a Polyscope mesh.

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the area derivatives added.
    """
    area_derivatives = mesh.compute_area_derivatives()
    derivatives_verts = np.zeros(mesh.v.shape)
    for key in area_derivatives:
        derivatives_verts += np.abs(area_derivatives[key])

    vmin, vmax = (
        np.quantile(derivatives_verts, alpha),
        np.quantile(derivatives_verts, 1 - alpha),
    )
    print("Extremas of area derivatives plotted: ", vmin, vmax)
    derivatives_verts = derivatives_verts.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        derivatives_verts,
        name_values="Area Derivatives",
        ps_mesh=ps_mesh,
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def view_volume_derivatives(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add volume derivatives data on a Polyscope mesh.

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the volume derivatives added.
    """
    volume_derivatives = mesh.compute_volume_derivatives()
    derivatives_verts = np.zeros(mesh.v.shape)
    for dv in volume_derivatives:
        derivatives_verts += np.abs(volume_derivatives[dv])

    vmin, vmax = (
        np.quantile(derivatives_verts, alpha),
        np.quantile(derivatives_verts, 1 - alpha),
    )
    print("Extremas of volume derivatives plotted: ", vmin, vmax)
    derivatives_verts = derivatives_verts.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        derivatives_verts,
        ps_mesh=ps_mesh,
        name_values="Volume Derivatives",
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def view_mean_curvature_cotan(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add mean curvature data on a Polyscope mesh.

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the mean curvature added.
    """
    mean_curvature, _, _ = compute_curvature_vertices_cotan(mesh)

    vmin, vmax = np.quantile(mean_curvature, alpha), np.quantile(mean_curvature, 1 - alpha)
    print("Extremas of mean curvature (cotan) plotted: ", vmin, vmax)
    mean_curvature = mean_curvature.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        mean_curvature,
        ps_mesh=ps_mesh,
        name_values="Mean Curvature Cotan",
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def view_gaussian_curvature(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add Gaussian curvature data on a Polyscope mesh.

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the gaussian curvature added.
    """
    gaussian_curvature = compute_gaussian_curvature_vertices(mesh)

    vmin, vmax = np.quantile(gaussian_curvature, alpha), np.quantile(gaussian_curvature, 1 - alpha)
    print("Extremas of gaussian curvature plotted: ", vmin, vmax)
    gaussian_curvature = gaussian_curvature.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        gaussian_curvature,
        ps_mesh=ps_mesh,
        name_values="Gaussian Curvature",
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def view_discrepancy_of_principal_curvatures(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add discrepancy of principal curvatures on a Polyscope mesh.

    If H is the mean curvature and G the gaussian curvature, it shows (H²-G)/H².

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the discrepancy of principal curvatures added.
    """
    mean_curvature, _, _ = compute_curvature_vertices_cotan(mesh)
    mean_curvature /= 9
    mean_curvature[mean_curvature == 0] = np.mean(mean_curvature)

    gaussian_curvature = compute_gaussian_curvature_vertices(mesh)
    kdiff = np.abs((mean_curvature**2 - gaussian_curvature) / (mean_curvature**2))

    vmin, vmax = np.quantile(kdiff, alpha), np.quantile(kdiff, 1 - alpha)
    print("Extremas of discrepancy of principal curvatures plotted: ", vmin, vmax)
    kdiff = kdiff.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        kdiff,
        ps_mesh=ps_mesh,
        name_values="Principal curvature discrepancy",
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


###
# Stress and graph
###


def plot_stress_tensor(  # noqa: C901
    mesh: "DcelData",
    nx_graph: nx.Graph,
    dict_tensions: dict[tuple[int, int], float],
    dict_pressure: dict[int, float],
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
    lumen_materials: int | list[int] = 0,
) -> None:
    """Show stress tensor on a polyscope viewer.

    Args:
        mesh (DcelData): Mesh to analyze.
        nx_graph (nx.Graph): Graph from mesh with data
        dict_tensions (dict[tuple[int, int], float]): Map interface to tension.
        dict_pressure (dict[int, float]): Map cell to volume.
        clean_before (bool, optional): Clean viewer before this function. Defaults to True.
        clean_after (bool, optional): Clean viewer after this function. Defaults to True.
        show (bool, optional): Show viewer. Defaults to True.
        lumen_materials (int | list[int], optional): Materials considered exterior. Defaults to 0.
    """
    # Formula from G. K. BATCHELOR, J Fluid Mech, 1970
    if isinstance(lumen_materials, int):
        lumen_materials = [lumen_materials]

    volumes = mesh.compute_volumes_cells()

    membrane_in_contact_with_cells = {key: [] for key in mesh.materials}

    for key in dict_tensions:
        a, b = key
        membrane_in_contact_with_cells[a].append(key)
        membrane_in_contact_with_cells[b].append(key)

    dict_faces_membrane = dict(zip(dict_tensions.keys(), [[] for i in range(len(dict_tensions.keys()))], strict=False))
    for i, face in enumerate(mesh.f):
        a, b = face[3:]
        dict_faces_membrane[(a, b)].append(i)

    areas_triangles = compute_faces_areas(mesh.v, mesh.f)
    normals_triangles = compute_normal_faces(mesh.v, mesh.f)
    delta = np.identity(3)

    def _tensor_normal_normal(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return delta - np.tensordot(x, x, axes=0)

    tensordot_normal_triangles = np.array(list(map(_tensor_normal_normal, normals_triangles)))

    for i in range(len(tensordot_normal_triangles)):
        tensordot_normal_triangles[i] *= areas_triangles[i]

    stress_vectors = np.zeros((mesh.n_materials - 1, 3, 3))
    compression_vectors = np.zeros((mesh.n_materials - 1, 3, 3))
    # compression_dots = []
    delta = np.identity(3)
    for i in range(1, mesh.n_materials):
        c = mesh.materials[i]
        if nx_graph.nodes.data("volume")[c] <= 0:
            continue
        mean_stress = np.zeros((3, 3))
        p = dict_pressure[c]
        v = volumes[c]
        mean_stress += -p * delta

        for m in membrane_in_contact_with_cells[c]:
            a, b = m

            # lumen_materials is by default 0
            # A membrane in contact with the exterior medium (i.e in lumen_materials) is counted once.
            # A membrane in contact with another cell is counted twice, thus we have to divide its surface tension by 2
            # See Supplementary Note for further explanations
            t = dict_tensions[m] if a in lumen_materials or b in lumen_materials else dict_tensions[m] / 2

            dotsum = 0
            for nt in dict_faces_membrane[m]:
                dotsum += tensordot_normal_triangles[nt]
            if a == c:  # we need to reverse the triangle !
                sign = -1
            elif b == c:
                sign = 1

            mean_stress += t / (v) * sign * dotsum

        vals, vects = eig(mean_stress)

        stress_vectors[i - 1] = vects.copy()
        compression_vectors[i - 1] = vects.copy()

        if vals[0] < 0:
            stress_vectors[i - 1, :, 0] *= vals[0]
            compression_vectors[i - 1, :, 0] *= 0
        else:
            stress_vectors[i - 1, :, 0] *= 0
            compression_vectors[i - 1, :, 0] *= vals[0]

        if vals[1] < 0:
            stress_vectors[i - 1, :, 1] *= vals[1]
            compression_vectors[i - 1, :, 1] *= 0
        else:
            stress_vectors[i - 1, :, 1] *= 0
            compression_vectors[i - 1, :, 2] *= vals[1]

        if vals[2] < 0:
            stress_vectors[i - 1, :, 2] *= vals[2]
            compression_vectors[i - 1, :, 2] *= 0
        else:
            stress_vectors[i - 1, :, 2] *= 0
            compression_vectors[i - 1, :, 2] *= vals[2]

        # if vals[0]>0 or vals[1]>0 or vals[2]>0 :
        #    compression_dots.append(G.nodes.data('centroid')[c])
    # compression_dots = np.array(compression_dots)

    ###
    # PLOTING
    ###

    ps.init()
    ps.set_ground_plane_mode("none")
    if clean_before:
        ps.remove_all_structures()

    # register a point cloud
    centroids = np.array([a[1] for a in nx_graph.nodes.data("centroid")])[
        np.array(nx_graph.nodes.data("volume"))[:, 1] > 0
    ]

    # ps_mesh = ps.register_surface_mesh("volume mesh", Mesh.v, Mesh.f[:,[0,1,2]])

    # ps_mesh.set_enabled() # default is true

    # ps_mesh.set_color((0.3, 0.6, 0.8)) # rgb triple on [0,1]
    # ps_mesh.set_transparency(0.2)

    ps_cloud = ps.register_point_cloud("Stress tensors principal directions", centroids, color=(0, 0, 0), radius=0.004)

    # For extensile stress:
    vecs_0 = stress_vectors[:, :, 0][np.array(nx_graph.nodes.data("volume"))[1:, 1] > 0]
    vecs_1 = stress_vectors[:, :, 1][np.array(nx_graph.nodes.data("volume"))[1:, 1] > 0]
    vecs_2 = stress_vectors[:, :, 2][np.array(nx_graph.nodes.data("volume"))[1:, 1] > 0]
    radius = 0.005
    length = 0.1
    color = (0.2, 0.5, 0.5)
    # basic visualization

    ps_cloud.add_vector_quantity(
        "principal_axes_0",
        vecs_0,
        enabled=True,
        radius=radius,
        length=length,
        color=color,
    )
    ps_cloud.add_vector_quantity(
        "principal_axes_0_down",
        -vecs_0,
        enabled=True,
        radius=radius,
        length=length,
        color=color,
    )
    ps_cloud.add_vector_quantity(
        "principal_axes_1",
        vecs_1,
        enabled=True,
        radius=radius,
        length=length,
        color=color,
    )
    ps_cloud.add_vector_quantity(
        "principal_axes_1_down",
        -vecs_1,
        enabled=True,
        radius=radius,
        length=length,
        color=color,
    )
    ps_cloud.add_vector_quantity(
        "principal_axes_2",
        vecs_2,
        enabled=True,
        radius=radius,
        length=length,
        color=color,
    )
    ps_cloud.add_vector_quantity(
        "principal_axes_2_down",
        -vecs_2,
        enabled=True,
        radius=radius,
        length=length,
        color=color,
    )

    # For compressive stress:
    vecs_comp_0 = compression_vectors[:, :, 0][np.array(nx_graph.nodes.data("volume"))[1:, 1] > 0]
    vecs_comp_1 = compression_vectors[:, :, 1][np.array(nx_graph.nodes.data("volume"))[1:, 1] > 0]
    vecs_comp_2 = compression_vectors[:, :, 2][np.array(nx_graph.nodes.data("volume"))[1:, 1] > 0]
    red = (0.7, 0.0, 0.0)

    ps_cloud.add_vector_quantity(
        "compressive_stress__axes_0",
        vecs_comp_0,
        enabled=True,
        radius=radius,
        length=length,
        color=red,
    )
    ps_cloud.add_vector_quantity(
        "compressive_stress__axes_0_down",
        -vecs_comp_0,
        enabled=True,
        radius=radius,
        length=length,
        color=red,
    )
    ps_cloud.add_vector_quantity(
        "compressive_stress__axes_1",
        vecs_comp_1,
        enabled=True,
        radius=radius,
        length=length,
        color=red,
    )
    ps_cloud.add_vector_quantity(
        "compressive_stress__axes_1_down",
        -vecs_comp_1,
        enabled=True,
        radius=radius,
        length=length,
        color=red,
    )
    ps_cloud.add_vector_quantity(
        "compressive_stress__axes_2",
        vecs_comp_2,
        enabled=True,
        radius=radius,
        length=length,
        color=red,
    )
    ps_cloud.add_vector_quantity(
        "compressive_stress__axes_2_down",
        -vecs_comp_2,
        enabled=True,
        radius=radius,
        length=length,
        color=red,
    )

    if show:
        ps.show()
    if clean_after:
        ps.remove_all_structures()


def display_embryo_graph(
    mesh: "DcelData",
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
) -> None:
    """Show the embryo as a graph.

    Args:
        mesh (DcelData): Mesh to analyze.
        clean_before (bool, optional): Clean polyscope viewer before this function. Defaults to True.
        clean_after (bool, optional): Clean polyscope viewer before this function. Defaults to True.
        show (bool, optional): Show polyscope viewer. Defaults to True.
    """
    ps.init()
    if clean_before:
        ps.remove_all_structures()

    nx_graph = mesh.compute_networkx_graph()
    edges_for_plotting = []
    for edge in list(nx_graph.edges):
        if edge[0] == 0 or edge[1] == 1:
            continue
        edges_for_plotting.append(np.array(edge) - 1)
    edges_for_plotting = np.array(edges_for_plotting)
    centroids_for_plotting = np.array([a[1] for a in nx_graph.nodes.data("centroid")])[1:]
    ps.register_point_cloud("Nodes", centroids_for_plotting, color=[0, 0, 0], radius=0.07)
    ps.register_curve_network(
        "Edges",
        centroids_for_plotting,
        edges_for_plotting,
        color=[0.5, 0.5, 0.5],
        radius=0.02,
    )

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()


def display_embryo_graph_forces(
    nx_graph: nx.Graph,
    alpha: float = 0.05,
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
    base_pressure: float = 0,
    plot_pressures: bool = True,
) -> None:
    """Show polyscope viewer with embryo as a graph colored wrt to tensions and pressures.

    Args:
        nx_graph (nx.Graph): Graph from a mesh.
        alpha (float, optional): Quantile to ignore extreme values. Defaults to 0.05.
        clean_before (bool, optional): Clean viewer before this function. Defaults to True.
        clean_after (bool, optional): Clean viewer after this function. Defaults to True.
        show (bool, optional): Show viewer. Defaults to True.
        base_pressure (float, optional): Base value for exterior pressure. Defaults to 0.
        plot_pressures (bool, optional): Show pressures on cell nodess. Defaults to True.
    """
    ps.init()
    ps.set_ground_plane_mode("none")

    if clean_before:
        ps.remove_all_structures()

    keys_edges = list(nx_graph.edges.keys())
    vertices = []
    edges = []
    values = []
    v_p0_list = []
    for i, key in enumerate(keys_edges):
        a, b = key

        v2 = nx_graph.nodes[b]["centroid"]
        if a == 0:
            v1 = np.mean(nx_graph.edges[key]["verts"], axis=0)  # + (np.mean(G.edges[key]['verts'],axis=0)-v2)*1.7
            v_p0_list.append(v1.copy())
        else:
            v1 = nx_graph.nodes[a]["centroid"]

        vertices.append(v1)
        vertices.append(v2)

        edges.append([2 * i, 2 * i + 1])
        values.append(nx_graph.edges[key]["tension"])

    vertices = np.array(vertices)
    edges = np.array(edges)
    values = np.array(values)
    v_p0_list = np.array(v_p0_list)

    values = values.clip(np.quantile(values, alpha), np.quantile(values, 1 - alpha))
    values -= np.amin(values)
    values /= np.amax(values)

    colors_tensions = cm.jet(values)[:, :3]

    ps_net = ps.register_curve_network(
        "Graph Edges",
        vertices,
        edges,
        radius=0.007,
        color=np.array([92, 85, 141]) / 255,
    )  # ,color = "000000")
    ps_net.add_color_quantity("Surface tensions", colors_tensions, defined_on="edges", enabled=True)

    centroids_for_plotting = np.array([a[1] for a in nx_graph.nodes.data("centroid")])[
        np.array(nx_graph.nodes.data("volume"))[:, 1] > 0
    ]  # [1:]
    ps_cloud = ps.register_point_cloud("Graph Nodes", centroids_for_plotting, radius=0.02, color=[0, 0, 0])  # )

    if plot_pressures:
        pressure_for_plotting = np.array([a[1] for a in nx_graph.nodes.data("pressure")])[
            np.array(nx_graph.nodes.data("volume"))[:, 1] > 0
        ]  # [1:]
        values_p = pressure_for_plotting.copy()
        values_p -= base_pressure

        values_p = values_p.clip(0, np.quantile(values_p, 1 - alpha))
        # values_p = values_p.clip(np.quantile(values_p,alpha),np.quantile(values_p,1-alpha))
        # print("Extremas of the pressures plotted : ",np.amin(values_p),np.amax(values_p))
        # values_p-=np.amin(values_p)
        values_p /= np.amax(values_p)
        values_p = np.array(1 - values_p, dtype=np.float64)

        colors_pressure = cm.magma(values_p)[:, :3]
        ps_cloud.add_color_quantity("Pressure", colors_pressure, enabled=True)

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()


####
# Residuals
####


def plot_valid_junctions(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float] | None = None,
) -> None:
    """Plot which trijunctions are considered valid or not.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float] | None, optional): Tensions map, computed if None. Defaults to None.
    """
    if dict_tensions is None:
        _, dict_tensions, _ = infer_tension(mesh, mean_tension=1)
    dict_length = mesh.compute_length_trijunctions()
    dict_validity = {}
    for key in dict_length:
        a, b, c = key
        ga, gb, gc = dict_tensions[(a, b)], dict_tensions[(a, c)], dict_tensions[(b, c)]
        if gc > ga + gb or ga > gc + gb or gb > gc + ga:
            dict_validity[key] = 1.0
        else:
            dict_validity[key] = 0.0

    ps.set_ground_plane_mode("none")
    ps.remove_all_structures()
    ps.register_surface_mesh("mesh", mesh.v, mesh.f[:, :3], color=(1, 1, 1), transparency=0.1)
    if np.amax(list(dict_validity.values())) == 0:
        plot_trijunctions_topo_change_viewer(
            mesh,
            dict_trijunctional_values=dict_validity,
            color="uniform",
            value_color=(0, 1, 0),
            clean_before=False,
        )
    else:
        plot_trijunctions_topo_change_viewer(
            mesh,
            dict_trijunctional_values=dict_validity,
            color="values",
            clean_before=False,
        )


def plot_residual_junctions(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float] | None = None,
    alpha: float = 0.05,
) -> None:
    """Plot residuals on juntions.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float] | None, optional): Tensions map, computed if None. Defaults to None.
        alpha (float, optional): Quantile to ignore extreme values. Defaults to 0.05.
    """
    if dict_tensions is None:
        _, dict_tensions, _ = infer_tension(mesh, mean_tension=1)
    dict_residuals = compute_residual_junctions_dict(mesh, dict_tensions, alpha=alpha)

    print(
        "Extremas of the residuals plotted : ",
        np.amin(list(dict_residuals.values())),
        np.amax(list(dict_residuals.values())),
    )

    ps.set_ground_plane_mode("none")
    ps.remove_all_structures()
    ps.register_surface_mesh("mesh", mesh.v, mesh.f[:, :3], color=(1, 1, 1), transparency=0.1)

    plot_trijunctions(mesh, dict_trijunctional_values=dict_residuals, clean_before=False, cmap=cm.jet)


# def plot_force_inference_with_lt(Mesh, force_inference_dicts=None, alpha=0.05, scalar_quantities=False):
#     if force_inference_dicts == None:
#         _, dict_tensions, dict_line_tensions, dict_pressure, _ = infer_forces_variational_lt(Mesh, mean_tension=1)
#     else:
#         dict_tensions, dict_line_tensions, dict_pressure = force_inference_dicts

#     G = Mesh.compute_networkx_graph()
#     nx.set_edge_attributes(G, dict_tensions, "tension")
#     nx.set_node_attributes(G, dict_pressure, "pressure")

#     ps_mesh = view_dict_values_on_mesh(
#         Mesh,
#         dict_tensions,
#         alpha=alpha,
#         ps_mesh=None,
#         clean_before=False,
#         clean_after=False,
#         show=False,
#         name_values="Surface Tensions",
#     )
#     ps_mesh = view_pressures_on_mesh(
#         Mesh,
#         dict_pressure,
#         ps_mesh=ps_mesh,
#         alpha=alpha,
#         clean_before=False,
#         clean_after=False,
#         show=False,
#     )
#     ps_mesh = plot_trijunctions(
#         Mesh,
#         Dict_trijunctional_values=dict_line_tensions,
#         clean_before=False,
#         clean_after=False,
#         show=False,
#         cmap=cm.jet,
#     )

#     plot_stress_tensor(Mesh, G, dict_tensions, dict_pressure, clean_before=False, clean_after=False, show=False)
#     display_embryo_graph_forces(
#         G,
#         alpha=alpha,
#         clean_before=False,
#         clean_after=False,
#         show=False,
#         P0=0,
#         Plot_pressures=True,
#     )
#     if scalar_quantities:
#         view_area_derivatives(Mesh, alpha=alpha, ps_mesh=ps_mesh, clean_before=False, clean_after=False, show=False)
#         view_volume_derivatives(Mesh, alpha=alpha, ps_mesh=ps_mesh, clean_before=False, clean_after=False, show=False)
#         view_mean_curvature_cotan(Mesh, alpha=alpha, ps_mesh=ps_mesh, clean_before=False, clean_after=False, show=False)  # noqa: E501
#         # view_mean_curvature_robust(
#         #     Mesh,
#         #     alpha=alpha,
#         #     ps_mesh=ps_mesh,
#         #     clean_before=False,
#         #     clean_after=False,
#         #     show=False,
#         # )
#         view_gaussian_curvature(Mesh, alpha=alpha, ps_mesh=ps_mesh, clean_before=False, clean_after=False, show=False)
#         view_discrepancy_of_principal_curvatures(
#             Mesh,
#             alpha=alpha,
#             ps_mesh=ps_mesh,
#             clean_before=False,
#             clean_after=False,
#             show=False,
#         )

#     ps.show()


# from foambryo.curvature import compute_curvature_vertices_robust_laplacian


# def view_mean_curvature_robust(
#     Mesh,
#     alpha=0.05,
#     ps_mesh=None,
#     remove_trijunctions=True,
#     clean_before=True,
#     clean_after=True,
#     show=True,
#     scattered=False,
# ):
#     H, _, _ = compute_curvature_vertices_robust_laplacian(Mesh)

#     vmin, vmax = np.quantile(H, alpha), np.quantile(H, 1 - alpha)
#     print("Extremas of mean curvature (robust) plotted: ", vmin, vmax)
#     H = H.clip(vmin, vmax)

#     ps_mesh = view_vertex_values_on_embryo(
#         Mesh,
#         H,
#         ps_mesh=ps_mesh,
#         name_values="Mean Curvature Robust",
#         remove_trijunctions=remove_trijunctions,
#         clean_before=clean_before,
#         clean_after=clean_after,
#         show=show,
#         scattered=scattered,
#     )

#     return ps_mesh
