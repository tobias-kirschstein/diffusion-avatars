from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh
from dreifus.matrix import Pose, Intrinsics
from dreifus.render import project
from scipy.spatial import KDTree
from trimesh import Trimesh

from diffusion_avatars.data_manager.nersemble_data_manager import NeRSembleSequenceDataManager
# from diffusion_avatars.renderer.provider.expression_animation_provider import ExpressionAnimationManager
from diffusion_avatars.renderer.provider.mesh_provider import MeshProvider, MeshProviderConfig
from diffusion_avatars.util.mesh import subdivide
from diffusion_avatars.util.quantization import CanonicalCoordinatesQuantizer, to_spherical


class NPHMProvider(MeshProvider):

    def __init__(self,
                 participant_id: int,
                 sequence: str,
                 mesh_provider_config: MeshProviderConfig = MeshProviderConfig()
                 ):
        self._data_manager = NeRSembleSequenceDataManager(participant_id, sequence)
        self._use_subdivision = mesh_provider_config.use_subdivision
        self._mesh_cache = dict()
        self._nphm_tracking_version = mesh_provider_config.nphm_tracking
        self._cut_throat = mesh_provider_config.cut_throat
        self._cut_throat_margin = mesh_provider_config.cut_throat_margin
        if mesh_provider_config.cut_throat:
            self._nphm_flame_reference_mesh = self._data_manager.load_NPHM_FLAME_reference_mesh()

        tracker_name = mesh_provider_config.flame_tracking.get_version()
        flame_params = self._data_manager.load_3DMM_tracking(tracker_name)

        # We need the rigid transformation from the FLAME fitting, because NPHM is fitted in FLAME space
        self._timestep_mapping = {timestep: i for i, timestep in enumerate(self._data_manager.get_timesteps())}
        self._flame_rotation = flame_params['rotation']  # [T, 3]
        self._flame_translation = flame_params['translation']  # [T, 3]
        self._flame_scale = flame_params['scale']  # [1, 3]

        T = len(self._flame_rotation)
        self._nphm_corrective_rotation = np.stack([np.eye(3) for _ in range(T)])
        self._nphm_corrective_translation = np.stack([np.array([0, 0, 0]) for _ in range(T)])
        self._nphm_corrective_scale = np.ones(T)

        self._expression_animation_manager = None

    def get_n_timesteps(self) -> int:
        return len(self._timestep_mapping)

    def get_uvs_per_vertex(self, mesh: trimesh.Trimesh) -> np.ndarray:
        canonical_coordinates = self.get_canonical_coordinates_per_vertex(mesh)
        return canonical_coordinates[..., :3]

    def get_ambient_coordinates_per_vertex(self, mesh: trimesh.Trimesh) -> np.ndarray:
        canonical_coordinates = self.get_canonical_coordinates_per_vertex(mesh)
        return canonical_coordinates[..., 3:]

    def get_canonical_coordinates_per_vertex(self, mesh: trimesh.Trimesh) -> np.ndarray:
        quantizer = CanonicalCoordinatesQuantizer()
        canonical_coordinates = mesh.vertex_attributes["canonical_coordinates"]
        # normalize each dimension into [0, 1]
        canonical_coordinates = (canonical_coordinates - quantizer._min_values) \
                                / (quantizer._max_values - quantizer._min_values)

        return canonical_coordinates

    def get_expression_code(self, timestep: int) -> np.ndarray:
        expression_code = self._data_manager.load_NPHM_expression_code(timestep,
                                                                       self._nphm_tracking_version.get_version())
        return expression_code

    @staticmethod
    def canonical_coordinates_to_uv(rendering: np.ndarray) -> np.ndarray:
        # NPHM "uvs" are 3-dim canonical coordinates
        # Create a "fake" uv space here by transforming them into spherical coordinates
        canonical_coordinates = rendering.copy()
        background_mask = np.all(rendering == -1, axis=-1)

        # Roughly move center of head into the center of the sphere (Or even slightly further away).
        # That way, more of the sphere surface will be allocated to the face and less for the back of the head
        canonical_coordinates[..., 2] -= 0.4

        # Spherical coordinates have a seam.
        # Per default, it is horizontal on the right cheek
        # We want to have it vertically at the back of the head
        canonical_coordinates = canonical_coordinates[..., [2, 0, 1]]  # x,y,z -> y,z,x
        spherical_coordinates = to_spherical(canonical_coordinates)
        thetas = spherical_coordinates[..., 1]  # [0, pi]
        phis = spherical_coordinates[..., 2]  # [-pi, pi]

        thetas = thetas / np.pi  # [0, pi] -> [0, 1]
        phis = (phis + np.pi) / (2 * np.pi)  # [-pi, pi] -> [0, 1]

        # [0,1] -> [-1, 1]
        thetas = 2 * thetas - 1
        phis = 2 * phis - 1
        rendering = np.stack([phis, -thetas], axis=-1)
        rendering[background_mask] = -1

        return rendering

    def get_vertices(self, timestep: int) -> np.ndarray:
        mesh = self._load_mesh(timestep)

        # Simon fits like: 4 x FLAME_2_NPHM x MVS_2_FLAME
        #
        # flame_rotation/translation/scale is FLAME_2_WORLD
        # NPHM corrective is FLAME_2_NPHM
        i = self._timestep_mapping[timestep]
        vertices = np.asarray(mesh.vertices)  # [V, 3]

        # NPHM_scaled -> NPHM
        vertices = vertices / 4

        # NPHM -> FLAME
        # Technically, the corrective_transform.npz contains values per frame
        # But Simon only uses the corrective transform of the first timestep throughout the whole sequence
        # vertices = 1 / self._nphm_corrective_scale[i] * (vertices - self._nphm_corrective_translation[i]) @ \
        #            self._nphm_corrective_rotation[i]
        vertices = 1 / self._nphm_corrective_scale[0] * (vertices - self._nphm_corrective_translation[0]) @ \
                   self._nphm_corrective_rotation[0]
        # vertices = self._nphm_corrective_scale[i] * vertices @ self._nphm_corrective_rotation[i].T + self._nphm_corrective_translation[i]

        # FLAME -> WORLD
        V = vertices.shape[0]
        model_transformations = Pose.from_euler(self._flame_rotation[i], self._flame_translation[i], 'XYZ')
        model_transformations[:3, :3] *= self._flame_scale[0]
        vertices = np.concatenate([vertices, np.ones((V, 1))], axis=-1)  # [V, 4]
        vertices = vertices @ model_transformations.T

        vertices = vertices[..., :3]

        return vertices

    def get_flame_2_nphm_pose(self) -> np.ndarray:
        flame_2_nphm_pose = Pose(self._nphm_corrective_rotation[0], self._nphm_corrective_translation[0])
        flame_2_nphm_pose[:3, :3] *= self._nphm_corrective_scale[0]

        return flame_2_nphm_pose.numpy()

    def get_flame_2_world_pose(self, timestep: int) -> np.ndarray:
        flame_rotation = self._flame_rotation[self._timestep_mapping[timestep]]
        flame_translation = self._flame_translation[self._timestep_mapping[timestep]]
        flame_scale = self._flame_scale[0]
        flame_2_world_pose = Pose.from_euler(flame_rotation, flame_translation, 'XYZ')
        flame_2_world_pose[:3, :3] *= flame_scale

        return flame_2_world_pose.numpy()

    def get_world_2_nphm_pose(self, timestep: int) -> np.ndarray:
        flame_2_world_pose = self.get_flame_2_world_pose(timestep)
        flame_2_nphm_pose = self.get_flame_2_nphm_pose()
        world_2_flame_pose = np.linalg.inv(flame_2_world_pose)
        world_2_nphm_pose = flame_2_nphm_pose @ world_2_flame_pose
        return world_2_nphm_pose

    def get_flame_pose_params(self, timestep: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        flame_rotation = self._flame_rotation[self._timestep_mapping[timestep]]
        flame_translation = self._flame_translation[self._timestep_mapping[timestep]]
        flame_scale = self._flame_scale[0]
        return flame_scale, flame_rotation, flame_translation

    def create_mesh(self, vertices: np.ndarray, timestep: int) -> trimesh.Trimesh:
        reference_mesh = self._load_mesh(timestep)
        mesh = trimesh.Trimesh(vertices,
                               reference_mesh.faces,
                               vertex_attributes=reference_mesh.vertex_attributes,
                               process=False)

        return mesh

    def has_mesh(self, timestep: int) -> bool:
        nphm_mesh_path = self._data_manager.get_NPHM_mesh_path(timestep,
                                                               mm_name=self._nphm_tracking_version.get_version())
        return Path(nphm_mesh_path).exists()

    def _load_mesh(self, timestep: int) -> trimesh.Trimesh:
        if timestep not in self._mesh_cache:
            mesh = self._data_manager.load_NPHM_mesh(timestep,
                                                     include_canonical_coordinates=True,
                                                     mm_name=self._nphm_tracking_version.get_version())

            if self._cut_throat:
                self.cut_throat_and_move_vertices(mesh, margin=self._cut_throat_margin)

            if self._use_subdivision > 0:
                vertices_subdiv, faces_subdiv, vertex_attributes_subdiv = subdivide(
                    mesh.vertices, mesh.faces, vertex_attributes=mesh.vertex_attributes)
                mesh = trimesh.Trimesh(vertices_subdiv,
                                       faces_subdiv,
                                       vertex_attributes=vertex_attributes_subdiv,
                                       process=False)

            self._mesh_cache[timestep] = mesh
            return mesh
        else:
            return self._mesh_cache[timestep]

    @classmethod
    def compute_vertex_optical_flow(cls,
                                    mesh_1: Trimesh,
                                    mesh_2: Trimesh,
                                    world_to_cam_pose_1: Pose,
                                    world_to_cam_pose_2: Pose,
                                    intrinsics: Intrinsics,
                                    use_hyper: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        canonical_coordinates_1 = mesh_1.vertex_attributes["canonical_coordinates"]
        canonical_coordinates_2 = mesh_2.vertex_attributes["canonical_coordinates"]

        if use_hyper:
            kd_tree_1 = KDTree(canonical_coordinates_1)
            kd_tree_2 = KDTree(canonical_coordinates_2)

            nns_1 = kd_tree_2.query(canonical_coordinates_1)
            nns_2 = kd_tree_1.query(canonical_coordinates_2)
        else:
            kd_tree_1 = KDTree(canonical_coordinates_1[..., :3])
            kd_tree_2 = KDTree(canonical_coordinates_2[..., :3])

            nns_1 = kd_tree_2.query(canonical_coordinates_1[..., :3])
            nns_2 = kd_tree_1.query(canonical_coordinates_2[..., :3])

        screen_points_1 = project(mesh_1.vertices, world_to_cam_pose_1, intrinsics)
        screen_points_2 = project(mesh_2.vertices, world_to_cam_pose_2, intrinsics)

        of_1_to_2 = screen_points_2[nns_1[1]] - screen_points_1
        of_2_to_1 = screen_points_1[nns_2[1]] - screen_points_2

        return of_1_to_2, of_2_to_1

    def cut_throat(
            self,
            points: np.ndarray,
            margin: float = 0) -> np.ndarray:
        idv1 = 3276  # 3281 # 3278 # 3276
        idv2 = 3207
        idv3 = 3310
        v1 = self._nphm_flame_reference_mesh.vertices[idv1, :]
        v2 = self._nphm_flame_reference_mesh.vertices[idv2, :]
        v3 = self._nphm_flame_reference_mesh.vertices[idv3, :]
        origin = v1
        line1 = v2 - v1
        line2 = v3 - v1
        normal = np.cross(line1, line2)
        normal = normal / np.linalg.norm(normal)

        direc = points - origin
        distance_from_plane = np.sum(normal * direc, axis=-1)
        above = distance_from_plane > margin
        return above

    def cut_throat_and_move_vertices(
            self,
            mesh: trimesh.Trimesh,
            margin: float = 0,
            move_vertices: bool = True) -> np.ndarray:
        above = self.cut_throat(mesh.vertex_attributes["canonical_coordinates"][..., :3], margin=margin)

        if move_vertices:
            original_vertices = mesh.vertices
            original_faces = mesh.faces

            # Remove vertices below canonical plane cut in temporary mesh copy
            # mesh_temp = trimesh.Trimesh(original_vertices.copy(), faces=original_faces.copy(), process=False)
            mesh_temp = trimesh.Trimesh(original_vertices, faces=original_faces, process=False)
            faces_to_delete = np.any(~above[mesh_temp.faces], axis=1)  # Delete faces that contain any deleted vertex
            mesh_temp.update_faces(~faces_to_delete)
            mesh_temp.update_vertices(above)

            # Find plane in posed space that fits through boundary vertices of canonical cut
            unique_edges = mesh_temp.edges[trimesh.grouping.group_rows(mesh_temp.edges_sorted, require_count=1)]
            boundary_vertex_ids = np.unique(unique_edges.flatten())
            boundary_vertices = mesh_temp.vertices[boundary_vertex_ids]
            boundary_vertices = boundary_vertices[boundary_vertices[..., 2] > -0.9]  # Discard potential boundary vertices at boundary of cube
            plane_normal, plane_offset, _ = fit_plane(boundary_vertices)
            if plane_normal[1] < 0:
                # In rare cases it can happen that the plane normal shows away from the head
                # We detect this by looking at the y coordinate because we know that in NPHM's posed space (which does not yet include head rotation) up is y axis
                plane_normal = -plane_normal
                plane_offset = -plane_offset
            plane_point = -plane_offset * plane_normal

            # Remove vertices below estimated posed plane in actual mesh
            original_vertices_to_plane = original_vertices - plane_point
            distance_from_plane = np.sum(plane_normal * original_vertices_to_plane, axis=-1)
            above = distance_from_plane > 0
            faces_to_delete = np.any(~above[original_faces], axis=1)  # Delete faces that contain any deleted vertex
            mesh.update_faces(~faces_to_delete)
            mesh.update_vertices(above)

            # Move boundary vertices onto posed plane
            unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
            boundary_vertex_ids = np.unique(unique_edges.flatten())
            boundary_vertices = mesh.vertices[boundary_vertex_ids]
            boundary_vertex_ids = boundary_vertex_ids[boundary_vertices[..., 2] > -0.9]
            boundary_vertices = boundary_vertices[boundary_vertices[..., 2] > -0.9]  # Discard potential boundary vertices at boundary of cube
            boundary_vertices_to_plane = boundary_vertices - plane_point
            distance_from_plane = np.sum(plane_normal * boundary_vertices_to_plane, axis=-1)
            boundary_vertices -= distance_from_plane[..., None] * plane_normal
            mesh.vertices[boundary_vertex_ids] = boundary_vertices

            # mesh.vertices[~above] -= distance_from_plane[~above][..., None] * normal
            # center_point_plane = np.mean(mesh.vertices[~above], axis=0)
            # mesh.vertices[~above] = center_point_plane
        else:
            faces_to_delete = np.any(~above[mesh.faces], axis=1)  # Delete faces that contain any deleted vertex
            mesh.update_faces(~faces_to_delete)
            mesh.update_vertices(above)

        return above


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    assert points.shape[1] == 3
    centroid = points.mean(axis=0)
    x = points - centroid[None, :]
    U, S, Vt = np.linalg.svd(x.T @ x)
    normal = U[:, -1]
    origin_distance = normal @ centroid
    rmse = np.sqrt(S[-1] / len(points))
    return normal, -origin_distance, rmse
