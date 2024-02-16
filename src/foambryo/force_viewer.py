from numpy.linalg import eig
import networkx as nx
from .tension_inference import (
    infer_forces,
    infer_tension,
    infer_pressure,
    compute_residual_junctions_dict,
)

from dw3d.Dcel import (
    compute_faces_areas,
    compute_normal_Faces,
    update_graph_with_scattered_values,
)
from dw3d.Curvature import (
    compute_gaussian_curvature_vertices,
    compute_curvature_vertices_cotan,
    compute_sphere_fit_residues_dict,
)

from .plotting_utilities import *


def plot_force_inference(
    Mesh,
    dict_tensions=None,
    dict_pressure=None,
    alpha=0.05,
    scalar_quantities=False,
    scattered=False,
    scatter_coeff=0.2,
):
    if dict_tensions == None:
        dict_tensions, dict_pressure = infer_forces(Mesh)
    if dict_pressure == None:
        dict_pressure = infer_pressure(Mesh, dict_tensions)

    G = Mesh.compute_networkx_graph()
    nx.set_edge_attributes(G, dict_tensions, "tension")
    nx.set_node_attributes(G, dict_pressure, "pressure")

    if scattered:
        Mesh.compute_scattered_arrays(coeff=scatter_coeff)
        G = update_graph_with_scattered_values(G, Mesh)

    dict_sphere_fit_residual = compute_sphere_fit_residues_dict(G)

    ps_mesh = view_dict_values_on_mesh(
        Mesh,
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
        Mesh,
        dict_pressure,
        ps_mesh=ps_mesh,
        alpha=alpha,
        clean_before=False,
        clean_after=False,
        show=False,
        scattered=scattered,
    )
    plot_stress_tensor(
        Mesh,
        G,
        dict_tensions,
        dict_pressure,
        clean_before=False,
        clean_after=False,
        show=False,
    )
    display_embryo_graph_forces(
        G,
        alpha=alpha,
        clean_before=False,
        clean_after=False,
        show=False,
        P0=0,
        Plot_pressures=True,
    )

    if scalar_quantities:
        ps_mesh = view_dict_values_on_mesh(
            Mesh,
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
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_volume_derivatives(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_mean_curvature_cotan(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_mean_curvature_robust(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_gaussian_curvature(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_discrepancy_of_principal_curvatures(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )

    ps.show()


def plot_tension_inference(
    Mesh,
    dict_tensions=None,
    alpha=0.05,
    scalar_quantities=False,
    scattered=False,
    scatter_coeff=0.2,
):
    ps.remove_all_structures()

    if dict_tensions == None:
        _, dict_tensions, _ = infer_tension(Mesh, mean_tension=1)

    G = Mesh.compute_networkx_graph()
    nx.set_edge_attributes(G, dict_tensions, "tension")

    if scattered:
        Mesh.compute_scattered_arrays(coeff=scatter_coeff)
        G = update_graph_with_scattered_values(G, Mesh)

    dict_sphere_fit_residual = compute_sphere_fit_residues_dict(G)

    ps_mesh = view_dict_values_on_mesh(
        Mesh,
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
        G,
        alpha=alpha,
        clean_before=False,
        clean_after=False,
        show=False,
        P0=0,
        Plot_pressures=False,
    )
    if scalar_quantities:
        ps_mesh = view_dict_values_on_mesh(
            Mesh,
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
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_volume_derivatives(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_mean_curvature_cotan(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_mean_curvature_robust(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_gaussian_curvature(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )
        view_discrepancy_of_principal_curvatures(
            Mesh,
            alpha=alpha,
            ps_mesh=ps_mesh,
            clean_before=False,
            clean_after=False,
            show=False,
            scattered=scattered,
        )

    ps.show()


def view_pressures_on_mesh(
    Mesh,
    dict_pressures,
    alpha=0.05,
    ps_mesh=None,
    clean_before=True,
    clean_after=True,
    show=True,
    scattered=False,
):
    if scattered:
        v, f, cluster_idx = (
            Mesh.v_scattered,
            Mesh.f_scattered,
            Mesh.cluster_idx_scattered,
        )  # , Mesh.idx_scattered

        def find_pressure(idx):
            return dict_pressures[idx]

        pressures = np.array(list(map(find_pressure, cluster_idx)))
    else:
        v, f = Mesh.v, Mesh.f

        def find_pressure(face):
            return dict_pressures[max(face[3], face[4])]

        pressures = np.array(list(map(find_pressure, f)))

    maxp = np.quantile(pressures, 1 - alpha)

    print("Extremas of pressures plotted: ", 0, maxp)

    pressures = pressures.clip(0, maxp)
    pressures /= np.amax(pressures)
    pressures = 1 - pressures
    # ps_mesh = view_faces_values_on_embryo(Mesh,pressures,ps_mesh = ps_mesh,name_values = "Pressures Cells",colormap = cm.magma,clean_before = clean_before, clean_after=clean_after,show=show,adapt_values=False,scattered = False)

    if ps_mesh == None:
        ps_mesh = ps.register_surface_mesh("Embryo", v, f[:, [0, 1, 2]])
    ps_mesh.set_color((0.3, 0.6, 0.8))  # rgb triple on [0,1]
    colors_face = cm.magma(pressures)[:, :3]
    ps_mesh.add_color_quantity(
        "Pressures Cells", colors_face, defined_on="faces", enabled=True
    )
    return ps_mesh


"""
def view_pressures_on_mesh(Mesh,dict_pressures,alpha = 0.05,ps_mesh = None, clean_before = True, clean_after=True,show=True, scattered = False):

    v,f = Mesh.v,Mesh.f

    def find_pressure(face): 
        return(dict_pressures[max(face[3],face[4])])

    pressures = np.array(list(map(find_pressure,f)))
    pressures_values = np.array(list(dict_pressures.values()))
    maxp = np.quantile(pressures_values,1-alpha)

    print("Extremas of pressures plotted: ",0,maxp)

    pressures = pressures.clip(0,maxp)
    pressures/=np.amax(pressures)
    pressures = 1-pressures
    ps_mesh = view_faces_values_on_embryo(Mesh,pressures,ps_mesh = ps_mesh,name_values = "Pressures Cells",colormap = cm.magma,clean_before = clean_before, clean_after=clean_after,show=show,adapt_values=False,scattered = scattered)

    return(ps_mesh)



def view_faces_values_on_embryo(Mesh,Vf,name_values = "Values",ps_mesh = None,colormap = cm.jet,min_to_zero=True,clean_before = True, clean_after=True,show=True,highlight_junctions=False,adapt_values = True, scattered = False):
    
    if scattered : 
        v,f,idx = Mesh.v_scattered,Mesh.f_scattered, Mesh.idx_scattered
    
    else : 
        v,f,idx = Mesh.v,Mesh.f, np.arange(len(Mesh.f))
        
    Values = Vf.copy()
    if adapt_values : 
        if min_to_zero : 
            Values-=np.amin(Values)
        Values/=(np.amax(Values)-np.amin(Values))

    Values = Values[idx]
    colors_face = colormap(Values)[:,:3]

    ps.init()
    
    if clean_before : 
        ps.remove_all_structures()


    if ps_mesh == None : 
        ps_mesh = ps.register_surface_mesh("Embryo", v,f[:,[0,1,2]])

    ps_mesh.set_color((0.3, 0.6, 0.8)) # rgb triple on [0,1]
    #ps_mesh.set_transparency(0.2)
    ps_mesh.add_color_quantity(name_values, colors_face, defined_on='faces',enabled=True)
    
    
    if highlight_junctions : 
        plot_trijunctions(Mesh,clean_before = False, clean_after = False, show=False, color = "uniform",value_color = np.ones(3))
    
    ps.set_ground_plane_mode("none") 
    
    if show : 
        ps.show()
    
    if clean_after : 
        ps.remove_all_structures()
        
    return(ps_mesh)

"""
###
# Scalar quantities
###


def view_area_derivatives(
    Mesh,
    alpha=0.05,
    ps_mesh=None,
    remove_trijunctions=True,
    clean_before=True,
    clean_after=True,
    show=True,
    scattered=False,
):
    DA = Mesh.compute_area_derivatives()
    Derivatives_verts = np.zeros(Mesh.v.shape)
    for key in DA.keys():
        Derivatives_verts += np.abs(DA[key])

    vmin, vmax = (
        np.quantile(Derivatives_verts, alpha),
        np.quantile(Derivatives_verts, 1 - alpha),
    )
    print("Extremas of area derivatives plotted: ", vmin, vmax)
    Derivatives_verts = Derivatives_verts.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        Mesh,
        Derivatives_verts,
        name_values="Area Derivatives",
        ps_mesh=ps_mesh,
        remove_trijunctions=remove_trijunctions,
        clean_before=clean_before,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def view_volume_derivatives(
    Mesh,
    alpha=0.05,
    ps_mesh=None,
    remove_trijunctions=True,
    clean_before=True,
    clean_after=True,
    show=True,
    scattered=False,
):
    DV = Mesh.compute_volume_derivatives()
    Derivatives_verts = np.zeros(Mesh.v.shape)
    for dv in DV.keys():
        Derivatives_verts += np.abs(DV[dv])

    vmin, vmax = (
        np.quantile(Derivatives_verts, alpha),
        np.quantile(Derivatives_verts, 1 - alpha),
    )
    print("Extremas of volume derivatives plotted: ", vmin, vmax)
    Derivatives_verts = Derivatives_verts.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        Mesh,
        Derivatives_verts,
        ps_mesh=ps_mesh,
        name_values="Volume Derivatives",
        remove_trijunctions=remove_trijunctions,
        clean_before=clean_before,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def view_mean_curvature_cotan(
    Mesh,
    alpha=0.05,
    ps_mesh=None,
    remove_trijunctions=True,
    clean_before=True,
    clean_after=True,
    show=True,
    scattered=False,
):
    H, _, _ = compute_curvature_vertices_cotan(Mesh)

    vmin, vmax = np.quantile(H, alpha), np.quantile(H, 1 - alpha)
    print("Extremas of mean curvature (cotan) plotted: ", vmin, vmax)
    H = H.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        Mesh,
        H,
        ps_mesh=ps_mesh,
        name_values="Mean Curvature Cotan",
        remove_trijunctions=remove_trijunctions,
        clean_before=clean_before,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def view_gaussian_curvature(
    Mesh,
    alpha=0.05,
    ps_mesh=None,
    remove_trijunctions=True,
    clean_before=True,
    clean_after=True,
    show=True,
    scattered=False,
):
    H = compute_gaussian_curvature_vertices(Mesh)

    vmin, vmax = np.quantile(H, alpha), np.quantile(H, 1 - alpha)
    print("Extremas of gaussian curvature plotted: ", vmin, vmax)
    H = H.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        Mesh,
        H,
        ps_mesh=ps_mesh,
        name_values="Gaussian Curvature",
        remove_trijunctions=remove_trijunctions,
        clean_before=clean_before,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def view_discrepancy_of_principal_curvatures(
    Mesh,
    alpha=0.05,
    ps_mesh=None,
    remove_trijunctions=True,
    clean_before=True,
    clean_after=True,
    show=True,
    scattered=False,
):
    H, _, _ = compute_curvature_vertices_cotan(Mesh)
    H /= 9
    H[H == 0] = np.mean(H)

    G = compute_gaussian_curvature_vertices(Mesh)
    Kdiff = np.abs((H**2 - G) / (H**2))

    vmin, vmax = np.quantile(Kdiff, alpha), np.quantile(Kdiff, 1 - alpha)
    print("Extremas of discrepancy of principal curvatures plotted: ", vmin, vmax)
    Kdiff = Kdiff.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        Mesh,
        Kdiff,
        ps_mesh=ps_mesh,
        name_values="Principal curvature discrepancy",
        remove_trijunctions=remove_trijunctions,
        clean_before=clean_before,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


###
# Stress and graph
###


def plot_stress_tensor(
    Mesh,
    G,
    dict_tensions,
    dict_pressure,
    clean_before=True,
    clean_after=True,
    show=True,
    lumen_materials=[0],
):
    # Formula from G. K. BATCHELOR, J Fluid Mech, 1970

    Vols = Mesh.compute_volumes_cells()

    Membrane_in_contact_with_cells = {key: [] for key in Mesh.materials}

    for key in dict_tensions.keys():
        a, b = key
        Membrane_in_contact_with_cells[a].append(key)
        Membrane_in_contact_with_cells[b].append(key)

    dict_faces_membrane = dict(
        zip(dict_tensions.keys(), [[] for i in range(len(dict_tensions.keys()))])
    )
    for i, face in enumerate(Mesh.f):
        a, b = face[3:]
        dict_faces_membrane[(a, b)].append(i)

    Areas_triangles = compute_faces_areas(Mesh.v, Mesh.f)
    Normals_triangles = compute_normal_Faces(Mesh.v, Mesh.f)
    delta = np.identity(3)
    tensor_normal_normal = lambda x: delta - np.tensordot(x, x, axes=0)
    Tensordot_normal_triangles = np.array(
        list(map(tensor_normal_normal, Normals_triangles))
    )

    for i in range(len(Tensordot_normal_triangles)):
        Tensordot_normal_triangles[i] *= Areas_triangles[i]

    Stress_vectors = np.zeros((Mesh.n_materials - 1, 3, 3))
    Compression_vectors = np.zeros((Mesh.n_materials - 1, 3, 3))
    # compression_dots = []
    delta = np.identity(3)
    for i in range(1, Mesh.n_materials):
        c = Mesh.materials[i]
        if G.nodes.data("volume")[c] <= 0:
            continue
        mean_stress = np.zeros((3, 3))
        p = dict_pressure[c]
        v = Vols[c]
        mean_stress += -p * delta

        for m in Membrane_in_contact_with_cells[c]:
            a, b = m

            # lumen_materials is by default 0
            # A membrane in contact with the exterior medium (i.e in lumen_materials) is counted once.
            # A membrane in contact with another cell is counted twice, thus we have to divide its surface tension by 2
            # See Supplementary Note for further explanations
            if (a in lumen_materials) or (b in lumen_materials):
                t = dict_tensions[m]
            else:
                t = dict_tensions[m] / 2

            Sum = 0
            for nt in dict_faces_membrane[m]:
                Sum += Tensordot_normal_triangles[nt]
            if a == c:  # we need to reverse the triangle !
                sign = -1
            elif b == c:
                sign = 1

            mean_stress += t / (v) * sign * Sum

        vals, vects = eig(mean_stress)

        Stress_vectors[i - 1] = vects.copy()
        Compression_vectors[i - 1] = vects.copy()

        if vals[0] < 0:
            Stress_vectors[i - 1, :, 0] *= vals[0]
            Compression_vectors[i - 1, :, 0] *= 0
        else:
            Stress_vectors[i - 1, :, 0] *= 0
            Compression_vectors[i - 1, :, 0] *= vals[0]

        if vals[1] < 0:
            Stress_vectors[i - 1, :, 1] *= vals[1]
            Compression_vectors[i - 1, :, 1] *= 0
        else:
            Stress_vectors[i - 1, :, 1] *= 0
            Compression_vectors[i - 1, :, 2] *= vals[1]

        if vals[2] < 0:
            Stress_vectors[i - 1, :, 2] *= vals[2]
            Compression_vectors[i - 1, :, 2] *= 0
        else:
            Stress_vectors[i - 1, :, 2] *= 0
            Compression_vectors[i - 1, :, 2] *= vals[2]

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
    N = Mesh.n_materials - 1
    centroids = np.array([a[1] for a in G.nodes.data("centroid")])[
        np.array(G.nodes.data("volume"))[:, 1] > 0
    ]

    # ps_mesh = ps.register_surface_mesh("volume mesh", Mesh.v, Mesh.f[:,[0,1,2]])

    # ps_mesh.set_enabled() # default is true

    # ps_mesh.set_color((0.3, 0.6, 0.8)) # rgb triple on [0,1]
    # ps_mesh.set_transparency(0.2)

    ps_cloud = ps.register_point_cloud(
        "Stress tensors principal directions", centroids, color=(0, 0, 0), radius=0.004
    )

    # For extensile stress:
    vecs_0 = Stress_vectors[:, :, 0][np.array(G.nodes.data("volume"))[1:, 1] > 0]
    vecs_1 = Stress_vectors[:, :, 1][np.array(G.nodes.data("volume"))[1:, 1] > 0]
    vecs_2 = Stress_vectors[:, :, 2][np.array(G.nodes.data("volume"))[1:, 1] > 0]
    radius = 0.005
    length = 0.1
    color = (0.2, 0.5, 0.5)
    # basic visualization
    all_vecs = np.vstack((vecs_0, -vecs_0, vecs_1, -vecs_1, vecs_2, -vecs_2))
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
    vecs_comp_0 = Compression_vectors[:, :, 0][
        np.array(G.nodes.data("volume"))[1:, 1] > 0
    ]
    vecs_comp_1 = Compression_vectors[:, :, 1][
        np.array(G.nodes.data("volume"))[1:, 1] > 0
    ]
    vecs_comp_2 = Compression_vectors[:, :, 2][
        np.array(G.nodes.data("volume"))[1:, 1] > 0
    ]
    red = (0.7, 0.0, 0.0)
    all_vecs = np.vstack((vecs_0, -vecs_0, vecs_1, -vecs_1, vecs_2, -vecs_2))
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


def display_embryo_graph(Mesh, clean_before=True, clean_after=True, show=True):
    ps.init()
    if clean_before:
        ps.remove_all_structures()

    G = Mesh.compute_networkx_graph()
    Edges_for_plotting = []
    for edge in list(G.edges):
        if edge[0] == 0 or edge[1] == 1:
            continue
        Edges_for_plotting.append(np.array(edge) - 1)
    Edges_for_plotting = np.array(Edges_for_plotting)
    Centroids_for_plotting = np.array([a[1] for a in G.nodes.data("centroid")])[1:]
    ps.register_point_cloud(
        "Nodes", Centroids_for_plotting, color=[0, 0, 0], radius=0.07
    )
    ps.register_curve_network(
        "Edges",
        Centroids_for_plotting,
        Edges_for_plotting,
        color=[0.5, 0.5, 0.5],
        radius=0.02,
    )

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()


def display_embryo_graph_forces(
    G,
    alpha=0.05,
    clean_before=True,
    clean_after=True,
    show=True,
    P0=0,
    Plot_pressures=True,
):
    ps.init()
    ps.set_ground_plane_mode("none")

    if clean_before:
        ps.remove_all_structures()

    keys_edges = list(G.edges.keys())
    vertices = []
    edges = []
    values = []
    v_p0_list = []
    for i, key in enumerate(keys_edges):
        a, b = key

        v2 = G.nodes[b]["centroid"]
        if a == 0:
            v1 = np.mean(
                G.edges[key]["verts"], axis=0
            )  # + (np.mean(G.edges[key]['verts'],axis=0)-v2)*1.7
            v_p0_list.append(v1.copy())
        else:
            v1 = G.nodes[a]["centroid"]

        vertices.append(v1)
        vertices.append(v2)

        edges.append([2 * i, 2 * i + 1])
        values.append(G.edges[key]["tension"])

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
    ps_net.add_color_quantity(
        "Surface tensions", colors_tensions, defined_on="edges", enabled=True
    )

    Centroids_for_plotting = np.array([a[1] for a in G.nodes.data("centroid")])[
        np.array(G.nodes.data("volume"))[:, 1] > 0
    ]  # [1:]
    ps_cloud = ps.register_point_cloud(
        "Graph Nodes", Centroids_for_plotting, radius=0.02, color=[0, 0, 0]
    )  # )
    ps_ext = ps.register_point_cloud(
        "Graph Exterior Nodes", v_p0_list, color=cm.magma(1.0)[:3], radius=0.015
    )

    if Plot_pressures:
        Pressure_for_plotting = np.array([a[1] for a in G.nodes.data("pressure")])[
            np.array(G.nodes.data("volume"))[:, 1] > 0
        ]  # [1:]
        values_p = Pressure_for_plotting.copy()
        values_p -= P0

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


def plot_valid_junctions(Mesh, dict_tensions=None):
    if dict_tensions == None:
        _, dict_tensions, _ = infer_tension(Mesh, mean_tension=1)
    dict_length = Mesh.compute_length_trijunctions()
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
    ps.register_surface_mesh(
        "mesh", Mesh.v, Mesh.f[:, :3], color=(1, 1, 1), transparency=0.1
    )
    if np.amax(list(dict_validity.values())) == 0:
        plot_trijunctions_topo_change_viewer(
            Mesh,
            Dict_trijunctional_values=dict_validity,
            color="uniform",
            value_color=(0, 1, 0),
            clean_before=False,
        )
    else:
        plot_trijunctions_topo_change_viewer(
            Mesh, Dict_trijunctional_values=dict_validity, color=1, clean_before=False
        )


def plot_residual_junctions(Mesh, dict_tensions=None, alpha=0.05):
    if dict_tensions == None:
        _, dict_tensions, _ = infer_tension(Mesh, mean_tension=1)
    dict_residuals = compute_residual_junctions_dict(Mesh, dict_tensions, alpha=alpha)

    print(
        "Extremas of the residuals plotted : ",
        np.amin(list(dict_residuals.values())),
        np.amax(list(dict_residuals.values())),
    )

    ps.set_ground_plane_mode("none")
    ps.remove_all_structures()
    ps.register_surface_mesh(
        "mesh", Mesh.v, Mesh.f[:, :3], color=(1, 1, 1), transparency=0.1
    )

    plot_trijunctions(
        Mesh, Dict_trijunctional_values=dict_residuals, clean_before=False, cmap=cm.jet
    )


"""
def plot_force_inference_with_lt(Mesh,force_inference_dicts=None,alpha = 0.05, scalar_quantities = False):
    
    if force_inference_dicts == None : 
        _,dict_tensions,dict_line_tensions,dict_pressure,_=infer_forces_variational_lt(Mesh,mean_tension=1) 
    else : 
        dict_tensions,dict_line_tensions,dict_pressure = force_inference_dicts

    G=Mesh.compute_networkx_graph()
    nx.set_edge_attributes(G, dict_tensions, "tension")
    nx.set_node_attributes(G, dict_pressure, "pressure")

    ps_mesh = view_dict_values_on_mesh(Mesh,dict_tensions,alpha = alpha,ps_mesh = None, clean_before = False, clean_after=False,show=False,name_values = "Surface Tensions")
    ps_mesh = view_pressures_on_mesh(Mesh,dict_pressure,ps_mesh = ps_mesh, alpha = alpha, clean_before=False, clean_after=False, show = False)
    ps_mesh = plot_trijunctions(Mesh,Dict_trijunctional_values=dict_line_tensions,clean_before=False, clean_after=False, show = False,cmap = cm.jet)
    
    plot_stress_tensor(Mesh,G,dict_tensions, dict_pressure, clean_before = False, clean_after = False,show=False)
    display_embryo_graph_forces(G,alpha = alpha, clean_before = False, clean_after = False,show=False,P0 = 0, Plot_pressures = True)
    if scalar_quantities : 
        view_area_derivatives(Mesh, alpha = alpha,ps_mesh = ps_mesh, clean_before = False, clean_after = False,show=False)
        view_volume_derivatives(Mesh, alpha = alpha,ps_mesh = ps_mesh, clean_before = False, clean_after = False,show=False)
        view_mean_curvature_cotan(Mesh, alpha = alpha,ps_mesh = ps_mesh, clean_before = False, clean_after = False,show=False)
        view_mean_curvature_robust(Mesh, alpha = alpha,ps_mesh = ps_mesh, clean_before = False, clean_after = False,show=False)
        view_gaussian_curvature(Mesh, alpha = alpha,ps_mesh = ps_mesh, clean_before = False, clean_after = False,show=False)
        view_discrepancy_of_principal_curvatures(Mesh, alpha = alpha,ps_mesh = ps_mesh, clean_before = False, clean_after = False,show=False)

    ps.show()

from dw3d.Curvature import compute_curvature_vertices_robust_laplacian

def view_mean_curvature_robust(Mesh,alpha = 0.05,ps_mesh = None, remove_trijunctions=True,clean_before = True, clean_after = True,show=True,scattered = False):
    
    H,_,_=compute_curvature_vertices_robust_laplacian(Mesh)
    
    vmin, vmax = np.quantile(H,alpha),np.quantile(H,1-alpha)
    print("Extremas of mean curvature (robust) plotted: ",vmin,vmax)
    H = H.clip(vmin,vmax)

    ps_mesh = view_vertex_values_on_embryo(Mesh,H,ps_mesh = ps_mesh,name_values="Mean Curvature Robust",remove_trijunctions=remove_trijunctions,clean_before = clean_before, clean_after = clean_after,show=show, scattered = scattered)
        
    return(ps_mesh)

"""
