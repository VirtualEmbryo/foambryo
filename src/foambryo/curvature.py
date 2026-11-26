"""Mean curvature of a mesh's interfaces, from the cotangent Laplacian, as the Laplace pressure methods read it.

Curvatures are in inverse mesh length units. An interface all of whose vertices lie on junction
lines has no curvature (`nan`); see `foambryo.pressure_inference.UndeterminedPressureWarning`.

A 2-D mesh (`foambryo.PlanarMesh`, delaunay-watershed-2d's `DCEL_Data`, a `(points, segments, labels)` triple or a
`(vertices, edges)` pair) is read in 2D by `compute_curvature_interfaces` and `compute_curvature_vertices_cotan`: the
curvature of each interface (a curve has one curvature, which takes the place of the surface's `2 H` in Laplace's law),
from the circle the 2-D pressures read or from the vertex curvatures, and the curvature at each vertex from the
cotangent Laplacian's 1-D form; definitions, signs and derivation in `foambryo._planar_curvature_and_stress`. The
Gaussian curvature has no 2-D counterpart.

Sacha Ichbiah, 2021.
Matthieu Perez, 2024.
"""
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import trimesh
from networkx import Graph
from numpy.typing import NDArray

if TYPE_CHECKING:
    from foambryo.dcel import DcelData

# TODO: IMPLEMENT THE EDGE_BASED CURVATURE FORMULA (Paper from Julicher)


def _is_two_dimensional(mesh: object) -> bool:
    """Whether `mesh` is a 2-D mesh (`foambryo.dimension_of`); a mesh with faces is 3-D without further check."""
    if hasattr(mesh, "f"):
        return False
    from foambryo.planar_mesh import dimension_of

    return dimension_of(mesh) == 2


def compute_laplacian_cotan(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Laplacian computation.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN matrix such that LV gives a matrix of vectors:
    LV[i] gives the normal scaled by the discrete mean curvature.
    For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.
    .. code-block:: python
               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij
        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.
    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.
    .. code-block:: python
               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C
        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have
        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a
        Putting these together, we get:
        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH
    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.
    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """
    ### Traditional cotan laplacian : from
    # from "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds", Meyer et al. 2003

    # Implementation and explanation from : pytorch3d/loss/mesh_laplacian_smoothing.py
    lcot, inv_areas = laplacian_cot(points, triangles)
    inv_areas = inv_areas.reshape(-1)
    sum_cols = np.array(lcot.sum(axis=1))
    laplacian = lcot @ points - points * sum_cols
    norm = (0.75 * inv_areas).reshape(-1, 1)
    return (laplacian * norm, inv_areas)


def compute_gaussian_curvature_vertices(mesh: "DcelData") -> NDArray[np.float64]:
    """Return the discrete gaussian curvarture at every vertex of the mesh.

    Args:
        mesh (DcelData): Mesh.

    Returns:
        NDArray[np.float64]: Discrete gaussian curvature at every vertex of the mesh.

    Raises:
        ValueError: for a 2-D mesh: a curve has one curvature, and no Gaussian curvature.
    """
    if _is_two_dimensional(mesh):
        message = (
            "compute_gaussian_curvature_vertices has no 2-D counterpart: a 2-D interface is a curve, with one "
            "curvature (compute_curvature_vertices_cotan, compute_curvature_interfaces)"
        )
        raise ValueError(message)
    mesh_trimesh = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f[:, :3])
    return trimesh.curvature.discrete_gaussian_curvature_measure(mesh_trimesh, mesh.v, 0.0)


def compute_curvature_vertices_cotan(
    mesh: "DcelData",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the mean curvature at every vertex along with other unidentified data.

    A 2-D mesh gives the 2-D form, `(|K_i|, 1 / l_i, K_i)`: the magnitude of the curvature vector of the cotangent
    Laplacian's 1-D form (`K_i = (t_2 - t_1) / l_i`, exact on a circle sampled at equal steps), the inverse dual length
    and the vector `(n, 2)` in `(row, column)`, pointing to the centre of curvature; `nan` at the junctions, the ends of
    the interfaces and the image border's vertices (`foambryo._planar_curvature_and_stress.vertex_curvatures`).
    """
    if _is_two_dimensional(mesh):
        from foambryo._planar_curvature_and_stress import vertex_curvatures

        return vertex_curvatures(mesh)
    points: NDArray[np.float64] = mesh.v
    triangles = mesh.f[:, :3]
    lcot, inv_areas = laplacian_cot(points, triangles)
    inv_areas = inv_areas.reshape(-1)
    sum_cols: NDArray[np.float64] = np.array(lcot.sum(axis=1))
    first_term: float = np.dot(lcot.toarray(), points)
    second_term: NDArray[np.float64] = points * sum_cols
    laplacian: NDArray[np.float64] = (first_term - second_term) / 2
    mean_curvature: NDArray[np.float64] = np.linalg.norm(laplacian, axis=1) * 3 * inv_areas / 2
    return (
        mean_curvature,
        inv_areas,
        laplacian * 3 * (np.array([inv_areas] * 3).transpose()) / 2,
    )


def compute_curvature_interfaces(
    mesh: "DcelData",
    weighted: bool = True,
    *,
    reading: str | None = None,
) -> dict[tuple[int, int], float]:
    """Compute the mean curvature on the interfaces of the mesh.

    The vertex mean curvature of the cotangent Laplacian, signed by the vertex normal, averaged over
    each interface's vertices that are not on a junction line. Positive when the centre of curvature
    lies on the side of the interface's second label: `P_a - P_b = -2 H_ab T_ab` (a sphere of label 1
    in the medium 0 has `H_01 = +1/R`).

    A 2-D mesh gives each interface's curvature `kappa_ab` in the same sense, with `p_a - p_b = -gamma_ab
    kappa_ab` (a disk of label 1 in the medium has `kappa_01 = +1/R`), read from the circle the 2-D
    pressures fit (`reading="arc"`, the default) or from the vertex curvatures (`"local"`); the module
    `foambryo._planar_curvature_and_stress` gives the definitions and the derivation.

    Args:
        mesh (DcelData): the mesh, 3-D or 2-D.
        weighted (bool, optional): weight each vertex by its area (`PressureComputationMethod.
            WeightedLaplace`) rather than equally (`Laplace`); in 2D, each vertex by its dual length
            (`"local"`) or each piece of an interface by its length (`"arc"`). Defaults to True.
        reading (str | None, optional): how the curvature is read: in 3D `"cotan"` (the only reading),
            in 2D `"arc"` or `"local"`. Defaults to None: `"cotan"` in 3D, `"arc"` in 2D. Keyword-only.

    Returns:
        dict[tuple[int, int], float]: interface `(label_1, label_2)` -> mean curvature, in inverse
            mesh length units; `nan` for an interface all of whose vertices lie on junction lines (in 2D,
            one the reading gives no curvature).

    Raises:
        ValueError: for a reading the mesh's dimension does not have.
    """
    if _is_two_dimensional(mesh):
        from foambryo._planar_curvature_and_stress import DEFAULT_CURVATURE_READING_2D, interface_curvatures

        chosen = DEFAULT_CURVATURE_READING_2D if reading is None else reading
        return interface_curvatures(mesh, weighted, reading=chosen)
    if reading not in (None, "cotan"):
        message = f'reading must be "cotan" (or None) for a 3-D mesh, got {reading!r}; "arc" and "local" are 2-D ones'
        raise ValueError(message)
    lcot, inv_areas = compute_laplacian_cotan(points=mesh.v, triangles=mesh.f[:, :3])

    vertex_normals = mesh.compute_vertex_normals()
    mean_curvature = np.sign(np.sum(np.multiply(lcot, vertex_normals), axis=1)) * np.linalg.norm(lcot, axis=1)

    vertices_on_interfaces = {}
    for edge in mesh.half_edges:
        # pass trijunctions

        materials = (edge.incident_face.material_1, edge.incident_face.material_2)
        interface_key = (min(materials), max(materials))

        vertices_on_interfaces[interface_key] = vertices_on_interfaces.get(interface_key, [])
        vertices_on_interfaces[interface_key].append(edge.origin.key)
        vertices_on_interfaces[interface_key].append(edge.destination.key)

    verts_idx = {}
    for key in vertices_on_interfaces:
        verts_idx[key] = np.unique(np.array(vertices_on_interfaces[key]))

    interfaces_curvatures = {}
    for key in verts_idx:
        curvature = 0
        weights = 0
        for vert_idx in verts_idx[key]:
            v = mesh.vertices[vert_idx]
            if v.on_trijunction:
                continue
            else:
                if weighted:
                    curvature += mean_curvature[vert_idx] / inv_areas[vert_idx]
                    weights += 1 / inv_areas[vert_idx]
                else:
                    curvature += mean_curvature[vert_idx]
                    weights += 1

        # An interface all of whose vertices lie on junction lines has no vertex to read a
        # curvature from: it gets nan. The Laplace pressure leaves its equation out, and reports
        # as nan the cells that the remaining equations do not connect to the exterior
        # (`pressure_inference.UndeterminedPressureWarning`). On coarse meshes of small cells this
        # is common: up to about a third of the interfaces.
        if weights == 0:
            interfaces_curvatures[key] = np.nan
        else:
            interfaces_curvatures[key] = curvature / weights

    return interfaces_curvatures


def laplacian_cot(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns the Laplacian matrix with cotangent weights and the inverse of the face areas.

    Args:
        points: vertices of the mesh.
        triangles: triangles of the mesh (topology).

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: the Laplacian matrix `L`, of shape
        `(V, V)`, with `L[i, j] = cot a_ij + cot b_ij` if `(i, j)` is an edge of the mesh; and the
        inverse of the sum of the areas of the faces around each vertex, of shape `(V,)`.
    """
    nbpts, nbtri = len(points), len(triangles)

    face_verts = points[triangles]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    len_a = np.linalg.norm((v1 - v2), axis=1)
    len_b = np.linalg.norm((v0 - v2), axis=1)
    len_c = np.linalg.norm((v0 - v1), axis=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (len_a + len_b + len_c)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = np.sqrt(s * (s - len_a) * (s - len_b) * (s - len_c))  # .clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    sq_a, sq_b, sq_c = len_a * len_a, len_b * len_b, len_c * len_c
    cota = (sq_b + sq_c - sq_a) / area
    cotb = (sq_a + sq_c - sq_b) / area
    cotc = (sq_a + sq_b - sq_c) / area
    cot = np.stack([cota, cotb, cotc], axis=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = triangles[:, [1, 2, 0]]
    jj = triangles[:, [2, 0, 1]]
    idx = np.stack([ii, jj], axis=0).reshape(2, nbtri * 3)

    laplacian_matrix: NDArray[np.float64] = sp.coo_matrix((cot.reshape(-1), (idx[1], idx[0])), shape=(nbpts, nbpts))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    laplacian_matrix += laplacian_matrix.transpose()

    # For each vertex, compute the sum of areas for triangles containing it.
    inv_areas = np.zeros(nbpts)
    idx = triangles.reshape(-1)
    val = np.stack([area] * 3, axis=1).reshape(-1)
    np.add.at(inv_areas, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.reshape(-1, 1)

    return laplacian_matrix, inv_areas


def sphere_fit_residue(points: NDArray[np.float64]) -> float:
    """Get residue from the fit of a sphere from a sample of points in 3D using least squares method.

    From https://jekel.me/2015/Least-Squares-Sphere-Fit/ .

    Args:
        points (NDArray[np.float64]): sample of points in 3D

    Returns:
        float: reside from least squares method.
    """
    #   Assemble the A matrix
    sp_x = points[:, 0]
    sp_y = points[:, 1]
    sp_z = points[:, 2]
    mat_a = np.zeros((len(sp_x), 4))
    mat_a[:, 0] = sp_x * 2
    mat_a[:, 1] = sp_y * 2
    mat_a[:, 2] = sp_z * 2
    mat_a[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(sp_x), 1))
    f[:, 0] = (sp_x * sp_x) + (sp_y * sp_y) + (sp_z * sp_z)
    _, residuals, _, _ = np.linalg.lstsq(mat_a, f)
    if len(residuals) > 0:
        return residuals[0] / len(points)
    else:
        return 0


def compute_sphere_fit_residues_dict(graph: Graph) -> dict:
    """Compute a map edge of the graph -> residue from sphere fitting.

    Args:
        graph (Graph): Graph constructed from the DcelData.

    Returns:
        dict: map an edge of the graph to a fit residue.
    """
    sphere_fit_residues_faces = {}
    for key in graph.edges:
        vn = graph.edges[key]["verts"]
        sphere_fit_residues_faces[key] = sphere_fit_residue(vn)
    return sphere_fit_residues_faces


def compute_areas_interfaces(mesh: "DcelData") -> dict[tuple[int, int], float]:
    """Compute the area of each interface.

    Args:
        mesh (DcelData): Mesh

    Returns:
        dict[tuple[int, int], float]: map interface defined by two labels -> area.
    """
    # Duplicate of a function present in Geometry (with the same name), but computed in a different manner
    points = mesh.v
    triangles = mesh.f[:, :3]
    labels = mesh.f[:, 3:]
    areas = compute_triangles_areas(points, triangles)
    interfaces_areas: dict[tuple[int, int], float] = {}
    for i, double_label in enumerate(labels):
        label1, label2 = double_label
        table = interfaces_areas.get((label1, label2), 0)
        interfaces_areas[(label1, label2)] = table + areas[i]
    return interfaces_areas


def compute_triangles_areas(points: NDArray[np.float64], triangles: NDArray[np.int64]) -> NDArray[np.float64]:
    """Compute the area of every triangles in a mesh.

    Args:
        points (NDArray[np.float64]): points of a mesh.
        triangles (NDArray[np.float64]): triangles (topology) of the mesh.

    Returns:
        (NDArray[np.float64]): the areas for each triangle.
    """
    positions = points[triangles]
    sides = positions - positions[:, [2, 0, 1]]
    lengths_sides: NDArray[np.float64] = np.linalg.norm(sides, axis=2)
    half_perimeters: NDArray[np.float64] = np.sum(lengths_sides, axis=1) / 2

    diffs = np.array([half_perimeters] * 3).transpose() - lengths_sides
    areas = (half_perimeters * diffs[:, 0] * diffs[:, 1] * diffs[:, 2]) ** (0.5)
    return areas


# import robust-laplacian
# def compute_laplacian_robust(Mesh: "DcelData" ):
#     ### Robust Laplacian using implicit triangulations :
#     # from "A Laplacian for Nonmanifold Triangle Meshes", N.Sharp, K.Crane, 2020
#     verts = Mesh.v
#     faces = Mesh.f[:,[0,1,2]]
#     L, M=robust_laplacian.mesh_laplacian(Mesh.v,Mesh.f[:,[0,1,2]])
#     inv_areas = 1/M.diagonal().reshape(-1)/3
#     Sum_cols = np.array(L.sum(axis=1)) #Useless as it is already 0 (sum comprised in the central term) see http://rodolphe-vaillant.fr/entry/101/definition-laplacian-matrix-for-triangle-meshes
#     first_term = np.dot(L.toarray(),verts)
#     second_term = verts*Sum_cols
#     Laplacian = (first_term-second_term)
#     norm = (1.5*inv_areas).reshape(-1,1)
#     return(-Laplacian*norm,inv_areas)


# def compute_curvature_vertices_robust_laplacian(Mesh: "DcelData" ):
#     verts = Mesh.v
#     faces = Mesh.f[:,[0,1,2]]
#     L, M=robust_laplacian.mesh_laplacian(Mesh.v,Mesh.f[:,[0,1,2]])
#     inv_areas = 1/M.diagonal().reshape(-1)/3
#     Sum_cols = np.array(L.sum(axis=1))
#     first_term = np.dot(L.toarray(),verts)
#     second_term = verts*Sum_cols
#     Laplacian = (first_term-second_term)
#     H = np.linalg.norm(Laplacian,axis=1)*3*inv_areas/2
#     return(H,inv_areas,Laplacian*3*(np.array([inv_areas]*3).transpose())/2)
