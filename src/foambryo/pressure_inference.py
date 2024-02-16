import numpy as np
from scipy import linalg


def change_idx_cells(faces, mapping):
    # mapping : {key_init:key_end} has to be a bijection
    new_faces = faces.copy()
    for key in mapping:
        new_faces[faces[:, 3] == key][:, 3] = mapping[key]
        new_faces[faces[:, 4] == key][:, 4] = mapping[key]
    return new_faces


def infer_pressure_laplace(Mesh, dict_tensions, P0=0, weighted=False):
    dict_areas = Mesh.compute_areas_interfaces()
    dict_curvature = Mesh.compute_curvatures_interfaces(weighted=weighted)

    M, B = build_matrix_laplace(
        dict_curvature, dict_areas, dict_tensions, Mesh.n_materials, Mesh.materials, P0
    )
    x, resid, rank, sigma = linalg.lstsq(M, B)

    nc = Mesh.n_materials

    dict_pressures = {0: P0}
    for i, key in enumerate(Mesh.materials[1:]):
        dict_pressures[key] = x[i + 1] + P0

    return (x, dict_pressures, resid)


def infer_pressure_variational(Mesh, dict_tensions, P0=0, torch=True, prints=False):
    DA, DV = Mesh.compute_area_derivatives(), Mesh.compute_volume_derivatives()

    M, B, nm = build_matrix_discrete(DA, DV, Mesh.materials)

    F_g, F_p, G_g, G_p = extract_submatrices(M, nm)

    Mat_t = F_g - F_p @ (np.linalg.inv(G_p)) @ G_g
    if prints:
        print(F_g.shape, F_p.shape, G_g.shape, G_p.shape)
        print("Det of F_g:", np.linalg.det(F_g))
        print("Det of G_p:", np.linalg.det(G_p))

    Mat_inference = np.zeros((nm + 1, nm))
    Mat_inference[:nm, :nm] = Mat_t
    Mat_inference[nm:] = 1
    B_inf = np.zeros(nm + 1)
    B_inf[-1] = nm

    dict_pressures = {0: P0}

    x = np.zeros(len(dict_tensions.keys()))
    for i, key in enumerate(sorted(list(dict_tensions.keys()))):
        x[i] = dict_tensions[key]

    P = np.linalg.inv(G_p) @ G_g @ x

    for i, key in enumerate(Mesh.materials[1:]):
        dict_pressures[key] = P[i] + P0

    return (x, dict_pressures, 0)


def infer_pressure(Mesh, dict_tensions, mode="Variational", P0=0, weighted=False):
    if mode == "Laplace":
        return infer_pressure_laplace(Mesh, dict_tensions, P0, weighted=weighted)
    elif mode == "Variational":
        return infer_pressure_variational(Mesh, dict_tensions, P0)
    else:
        print("Unimplemented method")
        return None


def build_matrix_laplace(
    dict_curvature, dict_areas, dict_tensions, n_materials, materials, P0=0
):
    Total_area_mesh = sum(
        dict_areas.values()
    )  # sum of the areas of all the surfaces of the mesh
    T = np.array(list(dict_curvature.keys()))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t))))

    # MX = B
    # x : structure : [0,nc[ : pressions
    nm = len(T)
    nc = n_materials - 1

    reverse_map_materials = dict(zip(materials, np.arange(len(materials))))

    B = np.zeros(nm + 1)
    B[-1] = P0  # Po = Pb with Pb = 1
    M = np.zeros((nm + 1, nc + 1))

    # CLASSICAL
    # LAPLACE
    for i, key in enumerate(Keys_t):
        a = key // kt
        b = key % kt
        curvature_k = dict_curvature[(a, b)]
        tension_k = dict_tensions[(a, b)]

        if np.isnan(curvature_k):
            continue

        area_k = dict_areas[(a, b)]
        factor = area_k / Total_area_mesh
        idx_tk = reverse_map_t[key]

        B[i] = -2 * curvature_k * tension_k * factor  # +2Hk*Tk
        #  -> Sign is negative because we choose the following convention :
        # If 1 goes inside 2 the interface viewed from 1 is concave, viewed from 2 it is convex.
        # The curvature is noted negatively : Hk<0. Besides,  P1 > P2 and Tk > 0 thus laplace relation writes :
        # P1 - P2 = -2*Hk*Tk  <=> P1 - P2 + 2*Hk*Tk = 0
        am = reverse_map_materials[a]
        bm = reverse_map_materials[b]
        M[i, am] = 1 * factor  # Pi
        M[i, bm] = -1 * factor  # Pj

    M[-1, 0] = 1  # Enforces Po = Pb (see B[-2])
    # print(B)
    """
    pM = np.linalg.pinv(M)
    x = pM@B"""

    return (M, B)


def build_matrix_discrete(DA, DV, materials, mean_tension=1):
    T = np.array(list(DA.keys()))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] + T[:, 1] * kt
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t))))

    # MX = B
    # x : structure : ([0,nm[ : tensions) ([nm:nm+nc] : pressions) : (ya,yb,yc,yd...p0,p1,p2..pnc)
    nm = len(T)
    nc = len(materials) - 1
    size = (nm + nc + 1) * (nm + nc + 1)
    B = np.zeros(nm + nc + 1)
    B[-1] = (
        nm * mean_tension
    )  # Sum of tensions = number of membrane <=> mean of tensions = mean_tensions
    M = np.zeros((nm + nc + 1, nm + nc))
    # print("n_j",nj,"\nn_m",nm,"\nn_c",nc,"\nNumber of unknowns ",nm+nc+1)

    ###DISCRETE
    ###YOUNG-DUPRÉ
    ###EQUATIONS

    for i, tup_l in enumerate(DA.keys()):
        a_l, b_l = tup_l
        key_l = a_l + b_l * kt

        for tup_m in DA.keys():
            a_m, b_m = tup_m
            key_m = a_m + b_m * kt
            idx_tm = reverse_map_t[key_m]
            M[i, idx_tm] = np.sum(DA[tup_m] * DA[tup_l])

        for n, key_n in enumerate(materials[1:]):
            M[i, nm + n] = -np.sum(DV[key_n] * DA[tup_l])

    off = nm

    ###DISCRETE
    ###LAPLACE
    ###EQUATIONS
    for i, key_i in enumerate(materials[1:]):
        for tup in DA.keys():
            a, b = tup
            key = a + b * kt
            idx_tk = reverse_map_t[key]
            M[off + i, idx_tk] = np.sum(DA[(a, b)] * DV[key_i])

        for n, key_n in enumerate(materials[1:]):
            M[off + i, nm + n] = -np.sum(DV[key_n] * DV[key_i])

    off += nc

    Lines_sum = np.sum(M, axis=1)
    for i in range(len(M) - 1):
        if Lines_sum[i] != 0:
            M[i] /= Lines_sum[i]

    M[off, :nm] = 1  # Enforces mean of tensions = mean_tensions (see B[-1])

    return (M, B, nm)


def extract_submatrices(M, nm):
    F_g = M[:nm, :nm]
    F_p = -M[:nm, nm:]
    G_g = M[nm:-1, :nm]
    G_p = -M[nm:-1, nm:]
    return (F_g, F_p, G_g, G_p)
