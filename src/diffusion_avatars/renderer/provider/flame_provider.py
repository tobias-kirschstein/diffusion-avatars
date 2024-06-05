from collections import defaultdict
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import trimesh
from dreifus.matrix import Pose, Intrinsics
from dreifus.render import project
from diffusion_avatars.config.flame import FlameConfig
from diffusion_avatars.model.flame import FLAME, FLAMETex
from scipy.spatial import KDTree
from trimesh import Trimesh

from diffusion_avatars.data_manager.nersemble_data_manager import NeRSembleSequenceDataManager
from diffusion_avatars.renderer.provider.mesh_provider import MeshProvider, MeshProviderConfig


class FlameProvider(MeshProvider):

    def __init__(self,
                 shape_params: Optional[np.ndarray] = None,  # [1, 300]
                 expr_params: Optional[np.ndarray] = None,  # [T, 100]
                 translation: Optional[np.ndarray] = None,  # [T, 3]
                 rotation: Optional[np.ndarray] = None,  # [T, 3]
                 jaw_pose: Optional[np.ndarray] = None,  # [T, 3]
                 neck_pose: Optional[np.ndarray] = None,  # [T, 3]
                 eye_pose: Optional[np.ndarray] = None,  # [T, 6]
                 scale: Optional[np.ndarray] = None,  # [1, 3]
                 separate_transformation: bool = False,
                 timestep_mapping: Optional[Dict[int, int]] = None,
                 config: MeshProviderConfig = MeshProviderConfig(),
                 ):

        if expr_params is not None:
            T = expr_params.shape[0]
        elif translation is not None:
            T = translation.shape[0]
        elif rotation is not None:
            T = rotation.shape[0]
        elif jaw_pose is not None:
            T = jaw_pose.shape[0]
        elif neck_pose is not None:
            T = neck_pose.shape[0]
        elif eye_pose is not None:
            T = eye_pose.shape[0]
        else:
            T = 1

        if shape_params is None:
            shape_params = np.zeros((1, 300), dtype=np.float32)

        if expr_params is None:
            expr_params = np.zeros((T, 100), dtype=np.float32)
        elif 'expression' in config.temporal_smoothing_config.param_groups:
            expr_params = config.temporal_smoothing_config.smooth(expr_params)

        if translation is None:
            translation = np.zeros((T, 3), dtype=np.float32)
        elif 'translation' in config.temporal_smoothing_config.param_groups:
            translation = config.temporal_smoothing_config.smooth(translation)

        if rotation is None:
            rotation = np.zeros((T, 3), dtype=np.float32)
        elif 'rotation' in config.temporal_smoothing_config.param_groups:
            rotation = config.temporal_smoothing_config.smooth(rotation)

        if jaw_pose is None:
            jaw_pose = np.zeros((T, 3), dtype=np.float32)
        elif 'jaw_pose' in config.temporal_smoothing_config.param_groups:
            jaw_pose = config.temporal_smoothing_config.smooth(jaw_pose)

        if neck_pose is None:
            neck_pose = np.zeros((T, 3), dtype=np.float32)
        elif 'neck_pose' in config.temporal_smoothing_config.param_groups:
            neck_pose = config.temporal_smoothing_config.smooth(neck_pose)

        if eye_pose is None:
            eye_pose = np.zeros((T, 6), dtype=np.float32)
        elif 'eye_pose' in config.temporal_smoothing_config.param_groups:
            eye_pose = config.temporal_smoothing_config.smooth(eye_pose)

        if scale is None:
            scale = np.ones((1, 3), dtype=np.float32)

        flame_config = FlameConfig(
            shape_params=300,
            expression_params=100,
            batch_size=1,
            flame_version='flame2023.pkl' if config.flame_tracking.is_flame_2023() else 'generic_model.pkl'
        )
        self.flame_model = FLAME(flame_config)
        self.flame_tex = FLAMETex()
        self.uv_coords_by_vertex = self.flame_tex.get_uv_coords_by_vertex(self.flame_model)

        self.shape_params = torch.tensor(shape_params)  # [1, 300]
        assert self.shape_params.shape[
                   0] == 1, "There should only be a single set of shape params for the whole sequence"
        self.expr_params = torch.tensor(expr_params)  # [T, 100]
        self.pose_params = torch.cat(
            [torch.tensor(np.zeros_like(rotation)) if separate_transformation else torch.tensor(rotation),
             torch.tensor(jaw_pose)], dim=-1)  # [T, 6]
        self.neck_pose = torch.tensor(neck_pose)
        self.eyes_pose = torch.tensor(eye_pose)
        self.rotation = torch.tensor(rotation)
        self.translation = torch.tensor(translation)  # [T, 3]
        self.scale = torch.tensor(scale)  # [1, 3]

        self._separate_transformation = separate_transformation
        self._timestep_mapping = {t: t for t in range(T)} if timestep_mapping is None else timestep_mapping

        self._use_uv_faces = config.use_uv_faces
        self._close_mouth = config.close_mouth
        if config.close_mouth:
            self._new_mouth_faces = np.array([
                [1666, 3514, 2783],
                [1665, 1666, 2783],
                [1665, 2783, 2782],
                [1665, 2782, 1739],
                [1739, 2782, 2854],
                [1739, 2854, 1742],
                [1742, 2854, 2857],
                [1742, 2857, 1747],
                [1747, 2857, 2862],
                [1747, 2862, 1746],
                [1746, 2862, 2861],
                [1746, 2861, 1595],
                [1595, 2861, 2731],
                [1595, 2731, 1594],
                [1594, 2731, 2730],
                [1594, 2730, 1572],
                [1572, 2730, 2708],
                [1572, 2708, 1573],
                [1573, 2708, 2709],
                [1573, 2709, 1860],
                [1860, 2709, 2943],
                [1860, 2943, 1862],
                [1862, 2943, 2945],
                [1862, 2945, 1830],
                [1830, 2945, 2930],
                [1830, 2930, 1835],
                [1835, 2930, 2933],
                [1835, 2933, 1852],
                [1852, 2933, 2941],
                [1852, 2941, 3497]
            ])
            self.flame_model.faces = np.concatenate([self.flame_model.faces, self._new_mouth_faces])

            if config.use_uv_faces:
                # We also need to add the new mouth faces to uv_faces. However, uv_faces uses different vertex IDs
                # than the regular faces
                # Hence, we need to find the mapping of regular vertices -> uv vertices
                # Usually, this mapping is ambiguous because one vertex can have multiple uv coords
                # However, in case of the mouth region in FLAME, we now that the vertices are not at a seem, so it is
                # safe to assume that the mapping is well-defined
                # We establish this mapping by exploiting this fact.
                # We first assign the single uv coordinate to the original FLAME mouth vertices.
                # Then, we find the closest uv coordinate in the uv_coords list, the ID will be the corresponding
                # UV vertex ID that we need
                # Finally, we can just replace the vertex IDs in new_mouth_faces with their corresponding UV vertex IDs
                new_mouth_vertex_ids = self._new_mouth_faces.flatten()
                uvs_per_vertex = self.get_uvs_per_vertex(None)
                uv_coords = self.get_uv_coords(None)
                uv_coords_kdtree = KDTree(uv_coords)
                new_mouth_vertex_ids_uv = uv_coords_kdtree.query(uvs_per_vertex[new_mouth_vertex_ids])[1]
                vertex_to_vertex_uv_mapping = dict(zip(new_mouth_vertex_ids, new_mouth_vertex_ids_uv))
                self._new_mouth_faces_uv = np.array([[vertex_to_vertex_uv_mapping[v_id] for v_id in face]
                                                     for face in self._new_mouth_faces])

    @staticmethod
    def from_dataset(participant_id: int,
                     sequence: str,
                     config: MeshProviderConfig = MeshProviderConfig(),
                     remote: bool = False) -> 'FlameProvider':
        data_manager = NeRSembleSequenceDataManager(participant_id, sequence)

        tracker_name = config.flame_tracking.get_version()

        flame_params = data_manager.load_3DMM_tracking(tracker_name)
        # Tracking params are stored incrementally by neglecting the original timesteps
        # For example, a sequence with frame_0, frame_3, frame_6 will be stored as [0, 1, 2]
        # _timestep_mapping takes the actual timestep and gives the incremental id for querying flame_params
        timestep_mapping = {timestep: i for i, timestep in enumerate(data_manager.get_timesteps())}

        rotation = flame_params['rotation']  # [T, 3]
        translation = flame_params['translation']  # [T, 3]
        shape_params = flame_params['shape']  # [1, 300]
        expr_params = flame_params['expression']  # [T, 100]
        jaw_pose = flame_params['jaw']  # [T, 3]
        scale = flame_params['scale']  # [1, 3]
        neck_pose = None
        eye_pose = None
        if 'neck' in flame_params:
            neck_pose = flame_params['neck']  # [T, 3]
        if 'eyes' in flame_params:
            eye_pose = flame_params['eyes']   # [T, 6]

        flame_provider = FlameProvider(shape_params=shape_params,
                                       expr_params=expr_params,
                                       translation=translation,
                                       rotation=rotation,
                                       jaw_pose=jaw_pose,
                                       neck_pose=neck_pose,
                                       eye_pose=eye_pose,
                                       scale=scale,
                                       separate_transformation=True,
                                       timestep_mapping=timestep_mapping,
                                       config=config)

        return flame_provider

    def get_n_timesteps(self) -> int:
        return len(self._timestep_mapping)

    def get_uvs_per_vertex(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return self.uv_coords_by_vertex

    def has_uv_faces(self) -> bool:
        return self._use_uv_faces

    def get_uv_faces(self, mesh: trimesh.Trimesh) -> np.ndarray:
        uv_faces = self.flame_tex.get_uv_faces()

        uv_faces = np.concatenate([uv_faces, self._new_mouth_faces_uv])

        return uv_faces

    def get_uv_coords(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return self.flame_tex.get_uv_coords()

    def get_vertices(self, timestep: int) -> np.ndarray:
        i = self._timestep_mapping[timestep]

        # FLAME forward
        flame_vertices, flame_lms = self.flame_model.forward(
            shape_params=self.shape_params[[0]],  # We always assume the same shape params for all timesteps
            expression_params=self.expr_params[[i]],
            pose_params=self.pose_params[[i]],
            neck_pose=None if self.neck_pose is None else self.neck_pose[[i]],
            eye_pose=None if self.eyes_pose is None else self.eyes_pose[[i]],
            transl=None if self._separate_transformation else self.translation[[i]])

        if self._separate_transformation:
            B = flame_vertices.shape[0]
            V = flame_vertices.shape[1]
            model_transformations = torch.stack([torch.from_numpy(
                Pose.from_euler(self.rotation[i].numpy(), self.translation[i].numpy(), 'XYZ'))])
            model_transformations[:, :3, :3] *= self.scale[0]
            flame_vertices = torch.cat([flame_vertices, torch.ones((B, V, 1))], dim=-1)
            flame_vertices = torch.bmm(flame_vertices, model_transformations.permute(0, 2, 1))
            flame_vertices = flame_vertices[..., :3]

        return flame_vertices

    def has_mesh(self, timestep: int) -> bool:
        return timestep in self._timestep_mapping

    def create_mesh(self, vertices: np.ndarray, timestep: Optional[int] = None) -> trimesh.Trimesh:
        flame_mesh = trimesh.Trimesh(vertices[0].detach().cpu().numpy(),
                                     self.flame_model.faces,
                                     process=False)

        if self._close_mouth:
            # Issue: the newly introduced faces in the closed mouth region are quite large
            # Since normals are specified per vertex, those faces will have mostly wrong normals due to interpolation
            # The cornering vertices' normals are computed by interpolation of neighboring faces which in this
            # case is undesired, we actually want kind of a hard jump in normals here
            # To alleviate, we "overwrite" the normals for the cornering vertices to all look into the direction
            # of the closed mouth faces instead of interpolating
            # This comes at the cost of slightly wrong normals for adjacent faces in the inner lip region
            # Since those faces are quite small, however, this can be neglected
            new_mouth_vertex_ids = np.unique(self._new_mouth_faces.flatten())
            new_mouth_face_normals = flame_mesh.face_normals[-len(self._new_mouth_faces):]
            new_mouth_vertex_to_face_id = defaultdict(list)
            for new_mouth_vertex_id in new_mouth_vertex_ids:
                for face_id, new_mouth_face in enumerate(self._new_mouth_faces):
                    if new_mouth_vertex_id in new_mouth_face:
                        new_mouth_vertex_to_face_id[new_mouth_vertex_id].append(face_id)

            # Each cornering mouth vertex is assigned the average normal over all closed mouth faces to which it belongs
            new_mouth_vertex_normals = []
            for v_id in new_mouth_vertex_ids:
                vertex_adjacent_normals = new_mouth_face_normals[new_mouth_vertex_to_face_id[v_id]]
                new_mouth_vertex_normal = np.mean(vertex_adjacent_normals, axis=0)
                new_mouth_vertex_normal = new_mouth_vertex_normal / np.linalg.norm(
                    new_mouth_vertex_normal)  # Re-normalize
                new_mouth_vertex_normals.append(new_mouth_vertex_normal)
            vertex_normals = np.array(flame_mesh.vertex_normals)
            vertex_normals[new_mouth_vertex_ids] = new_mouth_vertex_normals
            flame_mesh.vertex_normals = vertex_normals

        return flame_mesh

    @classmethod
    def compute_vertex_optical_flow(cls,
                                    mesh_1: Trimesh,
                                    mesh_2: Trimesh,
                                    world_to_cam_pose_1: Pose,
                                    world_to_cam_pose_2: Pose,
                                    intrinsics: Intrinsics,
                                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        screen_points_1 = project(mesh_1.vertices, world_to_cam_pose_1, intrinsics)
        screen_points_2 = project(mesh_2.vertices, world_to_cam_pose_2, intrinsics)

        of_1_to_2 = screen_points_2 - screen_points_1
        of_2_to_1 = screen_points_1 - screen_points_2

        return of_1_to_2, of_2_to_1
