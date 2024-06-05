import numpy as np
from trimesh import grouping
from trimesh.geometry import faces_to_edges


def subdivide(vertices,
              faces,
              face_index=None,
              vertex_attributes=None):
    """
    Adapted from trimesh, because trimesh apparently treats vertex_attributes wrongly in subdivision.

    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
    vertex_attributes : dict
      Contains (n, d) attribute data
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    index_dict : dict
      Only returned if `return_index`, {index of
      original face : index of new faces}.
    """
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            interpolated_attributes = values[edges[unique]].mean(axis=1)
            new_attributes[key] = np.vstack((values, interpolated_attributes))
        return new_vertices, new_faces, new_attributes

    return new_vertices, new_faces
