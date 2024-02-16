import numpy as np
from scipy import linalg

from .pressure_inference import infer_pressure


def infer_forces(Mesh, mean_tension=1, P0=0, mode_tension="Young-Dupré", mode_pressure="Variational"):
    _, dict_tensions, _ = infer_tension(Mesh, mean_tension=mean_tension, mode=mode_tension)
    _, dict_pressures, _ = infer_pressure(Mesh, dict_tensions, mode=mode_pressure, P0=P0)
    return (dict_tensions, dict_pressures)


def infer_forces_variational_lt(Mesh):
    # TODO
    return None


def change_idx_cells(faces, mapping):
    # mapping : {key_init:key_end} has to be a bijection
    new_faces = faces.copy()
    for key in mapping:
        new_faces[faces[:, 3] == key][x:, 3] = mapping[key]
        new_faces[faces[:, 4] == key][:, 4] = mapping[key]
    return new_faces


def cot(x):
    return np.cos(x) / np.sin(x)


def compute_z(phi1, phi2, phi3):
    z1 = np.sin(phi1) / (1 - np.cos(phi1))
    z2 = np.sin(phi2) / (1 - np.cos(phi2))
    z3 = np.sin(phi3) / (1 - np.cos(phi3))
    return (z1, z2, z3)


def factors_z(z1, z2, z3):
    f1 = z2 * z3
    f2 = z1 * z3
    f3 = z1 * z2
    s1 = z1 * (z2 + z3) / 2
    s2 = z2 * (z1 + z3) / 2
    s3 = z3 * (z1 + z2) / 2
    return (f1, f2, f3, s1, s2, s3)


"""
FORCE INFERENCE ONLY TAKING INTO ACCOUNT TRIJUNCTIONS :
THERE ARE A LOT OF QUADRIJUNCTIONS IN THE MESH
HOWEVER, TAKING ONLY THE TRIJUNCTIONS INTO ACCOUNT SHOULD SUFFICE IN MOST CASES, ESPECIALLY WITH A FEW CELLS
BECAUSE :
THE PROBLEM IS HIGHLY OVERDETERMINED
AND THE QUADRIJUNCTIONS ARE NOT VERY PRESENT (THEY OCCUPY A LITTLE LENGTH)
"""


def infer_tension(Mesh, mean_tension=1, mode="Young-Dupré"):
    if mode == "Young-Dupré":
        return infer_tension_symmetrical_yd(Mesh, mean_tension)
    elif mode == "Eq":
        return infer_tension_equilibrium(Mesh, mean_tension)
    elif mode == "Projection Young-Dupré":
        return infer_tension_projection_yd(Mesh, mean_tension)
    elif mode == "cotan":
        return infer_tension_cotan(Mesh, mean_tension)
    elif mode == "inv_cotan":
        return infer_tension_inv_cotan(Mesh, mean_tension)
    elif mode == "Lami":
        return infer_tension_lamy(Mesh, mean_tension)
    elif mode == "Lami inverse":
        return infer_tension_inv_lamy(Mesh, mean_tension)
    elif mode == "Lami logarithm":
        return infer_tension_lamy_log(Mesh, mean_tension)
    elif mode == "Variational":
        return infer_tension_variational_yd(Mesh, mean_tension)

    else:
        print("Unimplemented method")
        return None


def build_matrix_tension_inv_lamy(dict_rad, dict_areas, dict_length_tri, n_materials, mean_tension=1):
    Total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # Index angles :
    A = np.array(list(dict_rad.keys()))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    nc = n_materials - 1
    nj = len(A)
    size = (3 * nj + 1) * (nm)
    B = np.zeros(3 * nj + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        O_b = dict_rad[(a, b, c)]  # O_b = abc angle
        O_a = dict_rad[(b, a, c)]  # O_a = cab angle
        O_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * kt + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * kt + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * kt + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / Total_length_mesh
        M[3 * i, idx_tab] = np.sin(O_a) * factor
        M[3 * i, idx_tbc] = -np.sin(O_c) * factor

        M[3 * i + 1, idx_tab] = np.sin(O_b) * factor
        M[3 * i + 1, idx_tac] = -np.sin(O_c) * factor

        M[3 * i + 2, idx_tbc] = np.sin(O_b) * factor
        M[3 * i + 2, idx_tac] = -np.sin(O_a) * factor

    M[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (M, B)


def build_matrix_tension_lamy(dict_rad, dict_areas, dict_length_tri, n_materials, mean_tension=1):
    Total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # Index angles :
    A = np.array(list(dict_rad.keys()))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    nc = n_materials - 1
    nj = len(A)
    size = (3 * nj + 1) * (nm)
    B = np.zeros(3 * nj + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        O_b = dict_rad[(a, b, c)]  # O_b = abc angle
        O_a = dict_rad[(b, a, c)]  # O_a = cab angle
        O_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * kt + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * kt + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * kt + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / Total_length_mesh
        M[3 * i, idx_tab] = 1 / np.sin(O_c) * factor
        M[3 * i, idx_tbc] = -1 / np.sin(O_a) * factor

        M[3 * i + 1, idx_tab] = 1 / np.sin(O_c) * factor
        M[3 * i + 1, idx_tac] = -1 / np.sin(O_b) * factor

        M[3 * i + 2, idx_tbc] = 1 / np.sin(O_a) * factor
        M[3 * i + 2, idx_tac] = -1 / np.sin(O_b) * factor

    M[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (M, B)


def build_matrix_tension_projection_yd(dict_rad, dict_areas, dict_length_tri, n_materials, mean_tension=1):
    Total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # Index angles :
    A = np.array(list(dict_rad.keys()))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    nc = n_materials - 1
    nj = len(A)
    size = (2 * nj + 1) * (nm)
    B = np.zeros(2 * nj + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((2 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka
        O_b = dict_rad[(a, b, c)]  # O_b = abc angle
        O_a = dict_rad[(b, a, c)]  # O_a = cab angle
        idx_tab = reverse_map_t[min(a, b) * kt + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * kt + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * kt + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / Total_length_mesh
        M[2 * i, idx_tab] = 1 * factor
        M[2 * i, idx_tbc] = np.cos(O_b) * factor
        M[2 * i, idx_tac] = np.cos(O_a) * factor
        M[2 * i + 1, idx_tac] = -np.sin(O_a) * factor
        M[2 * i + 1, idx_tbc] = np.sin(O_b) * factor

    M[2 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (M, B)


def build_matrix_tension_lamy_log(dict_rad, dict_areas, dict_length_tri, n_materials, mean_tension=1):
    Total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # Index angles :
    A = np.array(list(dict_rad.keys()))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    nc = n_materials - 1
    nj = len(A)
    size = (3 * nj + 1) * (nm)
    B = np.zeros(3 * nj + 1)
    B[-1] = nm * np.log(mean_tension)  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        O_b = dict_rad[(a, b, c)]  # O_b = abc angle
        O_a = dict_rad[(b, a, c)]  # O_a = cab angle
        O_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * kt + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * kt + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * kt + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / Total_length_mesh
        M[3 * i, idx_tab] = 1 * factor
        M[3 * i, idx_tbc] = -1 * factor
        B[3 * i] = (np.log(np.sin(O_c)) - np.log(np.sin(O_a))) * factor

        M[3 * i + 1, idx_tab] = 1 * factor
        M[3 * i + 1, idx_tac] = -1 * factor
        B[3 * i + 1] = (np.log(np.sin(O_c)) - np.log(np.sin(O_b))) * factor

        M[3 * i + 2, idx_tbc] = 1 * factor
        M[3 * i + 2, idx_tac] = -1 * factor
        B[3 * i + 2] = (np.log(np.sin(O_a)) - np.log(np.sin(O_b))) * factor

    M[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (M, B)


def build_matrix_tension_cotan(dict_rad, dict_areas, dict_length_tri, n_materials, mean_tension=1):
    Total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # Index angles :
    A = np.array(list(dict_rad.keys()))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    nc = n_materials - 1
    nj = len(A)
    size = (3 * nj + 1) * (nm)
    B = np.zeros(3 * nj + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        O_b = dict_rad[(a, b, c)]  # O_b = abc angle
        O_a = dict_rad[(b, a, c)]  # O_a = cab angle
        O_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * kt + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * kt + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * kt + max(a, c)]

        z1, z2, z3 = compute_z(O_a, O_b, O_c)
        f1, f2, f3, s1, s2, s3 = factors_z(z1, z2, z3)

        factor = dict_length_tri[(a, b, c)] / Total_length_mesh
        M[3 * i, idx_tab] = -s3 - f3 * factor
        M[3 * i, idx_tbc] = s3 * factor
        M[3 * i, idx_tac] = s3 * factor

        M[3 * i + 1, idx_tab] = s1 * factor
        M[3 * i + 1, idx_tbc] = -s1 - f1 * factor
        M[3 * i + 1, idx_tac] = s1 * factor

        M[3 * i + 2, idx_tab] = s2 * factor
        M[3 * i + 2, idx_tbc] = s2 * factor
        M[3 * i + 2, idx_tac] = -s2 - f2 * factor

    M[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (M, B)


def build_matrix_tension_inv_cotan(dict_rad, dict_areas, dict_length_tri, n_materials, mean_tension=1):
    Total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # Index angles :
    A = np.array(list(dict_rad.keys()))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    nc = n_materials - 1
    nj = len(A)
    size = (3 * nj + 1) * (nm)
    B = np.zeros(3 * nj + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        O_b = dict_rad[(a, b, c)]  # O_b = abc angle
        O_a = dict_rad[(b, a, c)]  # O_a = cab angle
        O_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * kt + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * kt + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * kt + max(a, c)]

        z1, z2, z3 = compute_z(O_a, O_b, O_c)
        f1, f2, f3, s1, s2, s3 = factors_z(z1, z2, z3)

        factor = dict_length_tri[(a, b, c)] / Total_length_mesh
        M[3 * i, idx_tab] = -1 / s3 - 1 / f3 * factor
        M[3 * i, idx_tbc] = 1 / f3 * factor
        M[3 * i, idx_tac] = 1 / f3 * factor

        M[3 * i + 1, idx_tab] = 1 / f1 * factor
        M[3 * i + 1, idx_tbc] = -1 / s1 - 1 / f1 * factor
        M[3 * i + 1, idx_tac] = 1 / f1 * factor

        M[3 * i + 2, idx_tab] = 1 / f2 * factor
        M[3 * i + 2, idx_tbc] = 1 / f2 * factor
        M[3 * i + 2, idx_tac] = -1 / f2 - 1 / s2 * factor

    M[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (M, B)


def build_matrix_equilibrium_tensions(Mesh, dict_areas, mean_tension=1):
    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    table = []
    for edge in Mesh.half_edges:
        if len(edge.twin) > 1:
            table.append(edge.key)
    ne = len(table)

    B = np.zeros(3 * ne + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((3 * ne + 1, nm))
    M[-1] = 1

    ne_i = 0
    for edge in Mesh.half_edges:
        if len(edge.twin) == 1:
            continue

        list_edges = [edge.key] + edge.twin
        vectors_tension = {}
        for current_edge_idx in list_edges:
            # print(current_edge_idx)
            current_edge = Mesh.half_edges[current_edge_idx]
            # face_attached = current_edge.incident_face
            # Find the edges of the faces
            e = current_edge.incident_face.outer_component  # current_edge
            edges_face = []
            for i in range(3):
                # print(e.incident_face,e)
                edges_face.append([e.origin.key, e.destination.key])
                e = e.next
            edges_face = np.array(edges_face)

            # find the direction of the force vector
            verts_face = Mesh.v[np.unique(edges_face)]
            verts_edge = Mesh.v[[current_edge.origin.key, current_edge.destination.key]]

            index_edge = []
            index_not = []
            for index in np.unique(edges_face):
                if index in [current_edge.origin.key, current_edge.destination.key]:
                    index_edge.append(index)
                else:
                    index_not.append(index)

            # np.unique(edges_face),index,[edge.origin.key,edge.destination.key]
            # index_edge,index_not
            vector_tension = Mesh.v[index_not[0]] - Mesh.v[index_edge[0]]
            edge_vect = verts_edge[1] - verts_edge[0]
            edge_vect /= np.linalg.norm(edge_vect)
            vector_tension -= np.dot(vector_tension, edge_vect) * edge_vect
            vector_tension /= np.linalg.norm(vector_tension)

            a, b = (
                current_edge.incident_face.material_1,
                current_edge.incident_face.material_2,
            )
            idx_tension = reverse_map_t[min(a, b) * kt + max(a, b)]

            M[3 * ne_i, idx_tension] = vector_tension[0]
            M[3 * ne_i + 1, idx_tension] = vector_tension[1]
            M[3 * ne_i + 2, idx_tension] = vector_tension[2]

        ne_i += 1

    return (M, B)


def cot(x):
    return np.cos(x) / np.sin(x)


def tensions_equilibrium(phi1, phi2, phi3):
    t1 = cot(phi1 / 2) * (cot(phi2 / 2) + cot(phi3 / 2))
    t2 = cot(phi2 / 2) * (cot(phi1 / 2) + cot(phi3 / 2))
    t3 = cot(phi3 / 2) * (cot(phi1 / 2) + cot(phi2 / 2))
    return (t1, t2, t3)


def build_matrix_tension_symmetrical_yd(dict_rad, dict_areas, dict_length_tri, n_materials, mean_tension=1):
    Total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # Index angles :
    A = np.array(list(dict_rad.keys()))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    nc = n_materials - 1
    nj = len(A)
    size = (3 * nj + 1) * (nm)
    B = np.zeros(3 * nj + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        O_b = dict_rad[(a, b, c)]  # O_b = abc angle
        O_a = dict_rad[(b, a, c)]  # O_a = cab angle
        O_c = dict_rad[(a, c, b)]  # O_c = acb angle

        idx_tab = reverse_map_t[min(a, b) * kt + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * kt + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * kt + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / Total_length_mesh
        M[3 * i, idx_tab] = 1 * factor
        M[3 * i, idx_tbc] = np.cos(O_b) * factor
        M[3 * i, idx_tac] = np.cos(O_a) * factor

        M[3 * i + 1, idx_tab] = np.cos(O_b) * factor
        M[3 * i + 1, idx_tbc] = 1 * factor
        M[3 * i + 1, idx_tac] = np.cos(O_c) * factor

        M[3 * i + 2, idx_tab] = np.cos(O_a) * factor
        M[3 * i + 2, idx_tbc] = np.cos(O_c) * factor
        M[3 * i + 2, idx_tac] = 1 * factor

    M[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (M, B)


def build_matrix_discrete(DA, DV, materials, mean_tension=1):
    T = np.array(list(DA.keys()))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] + T[:, 1] * kt
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # MX = B
    # x : structure : ([0,nm[ : tensions) ([nm:nm+nc] : pressions) : (ya,yb,yc,yd...p0,p1,p2..pnc)
    nm = len(T)
    nc = len(materials) - 1
    size = (nm + nc + 1) * (nm + nc + 1)
    B = np.zeros(nm + nc + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = mean_tensions
    M = np.zeros((nm + nc + 1, nm + nc))
    # print("n_j",nj,"\nn_m",nm,"\nn_c",nc,"\nNumber of unknowns ",nm+nc+1)

    ###VARIATIONAL
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

    ###VARIATIONAL
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


def infer_tension_projection_yd(Mesh, mean_tension=1):
    dict_rad, _, dict_length = Mesh.compute_angles_tri(unique=False)
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_tension_projection_yd(dict_rad, dict_areas, dict_length, Mesh.n_materials, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(M, B)
    dict_tensions = {}
    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    return (x, dict_tensions, resid)


def infer_tension_symmetrical_yd(Mesh, mean_tension=1):
    dict_rad, _, dict_length = Mesh.compute_angles_tri(unique=False)
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_tension_symmetrical_yd(dict_rad, dict_areas, dict_length, Mesh.n_materials, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(M, B)
    dict_tensions = {}
    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    return (x, dict_tensions, resid)


def infer_tension_cotan(Mesh, mean_tension=1):
    dict_rad, _, dict_length = Mesh.compute_angles_tri(unique=False)
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_tension_cotan(dict_rad, dict_areas, dict_length, Mesh.n_materials, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(M, B)
    dict_tensions = {}
    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    return (x, dict_tensions, resid)


def infer_tension_inv_cotan(Mesh, mean_tension=1):
    dict_rad, _, dict_length = Mesh.compute_angles_tri(unique=False)
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_tension_inv_cotan(dict_rad, dict_areas, dict_length, Mesh.n_materials, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(M, B)
    dict_tensions = {}
    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    return (x, dict_tensions, resid)


def infer_tension_lamy(Mesh, mean_tension=1):
    dict_rad, _, dict_length = Mesh.compute_angles_tri(unique=False)
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_tension_lamy(dict_rad, dict_areas, dict_length, Mesh.n_materials, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(M, B)
    dict_tensions = {}
    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    return (x, dict_tensions, resid)


def infer_tension_inv_lamy(Mesh, mean_tension=1):
    dict_rad, _, dict_length = Mesh.compute_angles_tri(unique=False)
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_tension_inv_lamy(dict_rad, dict_areas, dict_length, Mesh.n_materials, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(M, B)
    dict_tensions = {}
    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    return (x, dict_tensions, resid)


def infer_tension_lamy_log(Mesh, mean_tension=1):
    dict_rad, _, dict_length = Mesh.compute_angles_tri(unique=False)
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_tension_lamy_log(dict_rad, dict_areas, dict_length, Mesh.n_materials, mean_tension)
    x_log, resid, rank, sigma = linalg.lstsq(M, B)
    x = np.exp(x_log)
    dict_tensions = {}
    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    return (x, dict_tensions, resid)


def infer_tension_variational_yd(Mesh, mean_tension=1, prints=False):
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
    x, resid, rank, sigma = linalg.lstsq(Mat_inference, B_inf)

    dict_tensions = {}
    dict_pressures = {0: 0}

    T = np.array(list(DA.keys()))
    nm = len(T)

    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    P = np.linalg.inv(G_p) @ G_g @ x

    return (x, dict_tensions, resid)


def infer_tension_equilibrium(Mesh, mean_tension=1):
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_equilibrium_tensions(Mesh, dict_areas, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(M, B)
    dict_tensions = {}
    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    for i in range(nm):
        dict_tensions[tuple(T[i])] = x[i]

    return (x, dict_tensions, resid)


###RESIDUAL STUFF


def build_matrix_tension_validity(dict_rad, dict_areas, dict_length_tri, n_materials, mean_tension=1):
    Total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    T = np.array(sorted(list(dict_areas.keys())))
    kt = np.amax(T) + 1
    Keys_t = T[:, 0] * kt + T[:, 1]
    reverse_map_t = dict(zip(Keys_t, np.arange(len(Keys_t)), strict=False))

    # Index angles :
    A = np.array(list(dict_rad.keys()))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(T)
    nc = n_materials - 1
    nj = len(A)
    size = (3 * nj + 1) * (nm)
    B = np.zeros(3 * nj + 1)
    B[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    M = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        O_b = dict_rad[(a, b, c)]  # O_b = abc angle
        O_a = dict_rad[(b, a, c)]  # O_a = cab angle
        O_c = dict_rad[(a, c, b)]  # O_c = acb angle

        idx_tab = reverse_map_t[min(a, b) * kt + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * kt + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * kt + max(a, c)]

        M[3 * i, idx_tab] = 1
        M[3 * i, idx_tbc] = np.cos(O_b)
        M[3 * i, idx_tac] = np.cos(O_a)

        M[3 * i + 1, idx_tab] = np.cos(O_b)
        M[3 * i + 1, idx_tbc] = 1
        M[3 * i + 1, idx_tac] = np.cos(O_c)

        M[3 * i + 2, idx_tab] = np.cos(O_a)
        M[3 * i + 2, idx_tbc] = np.cos(O_c)
        M[3 * i + 2, idx_tac] = 1

    M[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (M, B)


def compute_residual_junctions_dict(Mesh, dict_tensions, alpha=0.05):
    dict_rad, _, dict_length = Mesh.compute_angles_tri(unique=False)
    dict_areas = Mesh.compute_areas_interfaces()
    M, B = build_matrix_tension_validity(dict_rad, dict_areas, dict_length, Mesh.n_materials, mean_tension=1)

    T = np.array(sorted(list(dict_areas.keys())))
    nm = len(T)
    x = np.zeros(nm)
    for i in range(nm):
        x[i] = dict_tensions[tuple(T[i])]

    A = np.array(sorted(list(dict_rad.keys())))
    A = -np.sort(-A, axis=1)
    ka = np.amax(A)
    Keys_a = A[:, 0] * ka**2 + A[:, 1] * ka + A[:, 2]
    Keys_a, index = np.unique(Keys_a, return_index=True)
    A = A[index]
    dict_residuals = {}
    array_resid = (np.abs(M @ x - B) ** 2)[:-1]

    array_resid = array_resid.clip(np.quantile(array_resid, alpha), np.quantile(array_resid, 1 - alpha))

    for i, key in enumerate(Keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        dict_residuals[(a, b, c)] = array_resid[3 * i + 0] + array_resid[3 * i + 1] + array_resid[3 * i + 2]

    return dict_residuals
