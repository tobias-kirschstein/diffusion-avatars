from abc import abstractmethod
from dataclasses import dataclass, field
from enum import auto
from typing import Optional, Tuple, List, Literal, Dict

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from dreifus.matrix import Pose, Intrinsics
from elias.config import Config, StringEnum
from numpy import deprecate
from scipy.ndimage import gaussian_filter1d
from trimesh import Trimesh

ParamGroup3DMM = Literal['translation', 'rotation', 'jaw_pose', 'shape', 'expression']


@dataclass
class TemporalSmoothingConfig(Config):
    temporal_smoothing: int = 0
    use_gaussian: bool = False
    param_groups: List[ParamGroup3DMM] = field(default_factory=lambda: ['rotation', 'jaw_pose'])

    def smooth(self, values: np.ndarray) -> np.ndarray:
        if self.temporal_smoothing == 0:
            return values
        else:
            if self.use_gaussian:
                values_smoothed = gaussian_filter1d(values, self.temporal_smoothing, axis=0, mode="nearest")
            else:
                values = torch.from_numpy(values)
                temporal_kernel_size = self.temporal_smoothing * 2 + 1
                values = values.unsqueeze(1).permute(2, 1, 0)
                values_smoothed = F.conv1d(
                    torch.concat(
                        [values[..., :self.temporal_smoothing], values, values[..., -self.temporal_smoothing:]],
                        axis=-1),
                    torch.ones(1, 1, temporal_kernel_size) / temporal_kernel_size,
                    padding='valid').permute(2, 1, 0).squeeze(1)
                values_smoothed = values_smoothed.numpy()
            return values_smoothed


class FlameTrackingVersion(StringEnum):
    FLAME = auto()
    FLAME_2023 = auto()
    FLAME_2023_V2 = auto()

    def is_flame_2023(self) -> bool:
        return self in {self.FLAME_2023, self.FLAME_2023_V2}

    def get_version(self) -> str:
        if self == self.FLAME:
            return 'FLAME'
        elif self == self.FLAME_2023:
            return 'FLAME2023'
        elif self == self.FLAME_2023_V2:
            return 'FLAME2023_v2'
        else:
            raise NotImplementedError(f"get_version() not implemented for {self}")


class NPHMTrackingVersion(StringEnum):
    NPHM = auto()
    NPHM_V2 = auto()
    NPHM_V3 = auto()
    NPHM_TEMPORAL_NECK = auto()
    NPHM_TEMPORAL_NECK_LMS = auto()
    NPHM_TEMPORAL_NECK_LMS_6 = auto()
    NPHM_TEMPORAL_NECK_LMS_12 = auto()

    def get_version(self) -> str:
        if self == self.NPHM:
            return 'NPHM'
        elif self == self.NPHM_V2:
            return 'NPHM_v2'
        elif self == self.NPHM_V3:
            return 'NPHM_v3'
        elif self == self.NPHM_TEMPORAL_NECK:
            return 'NPHM_temp_wNeck'
        elif self == self.NPHM_TEMPORAL_NECK_LMS:
            return 'NPHM_temp_wNeck_wLMs'
        elif self == self.NPHM_TEMPORAL_NECK_LMS_6:
            return 'NPHM_temp_wNeck_wLMs_6'
        elif self == self.NPHM_TEMPORAL_NECK_LMS_12:
            return 'NPHM_temp_wNeck_wLMs_12'
        else:
            raise NotImplementedError(f"get_version() not implemented for {self}")


@dataclass
class MeshProviderConfig(Config):
    flame_tracking: FlameTrackingVersion = FlameTrackingVersion.FLAME

    # FLAME only
    close_mouth: bool = False
    use_uv_faces: bool = False
    temporal_smoothing_config: TemporalSmoothingConfig = TemporalSmoothingConfig()

    # NPHM only
    use_subdivision: bool = False
    nphm_tracking: NPHMTrackingVersion = NPHMTrackingVersion.NPHM
    include_ambient_dimensions: bool = False  # deprecated
    cut_throat: bool = False
    cut_throat_margin: float = 0

    @classmethod
    def _backward_compatibility(cls, loaded_config: Dict):
        if "use_flame_2023" in loaded_config:
            loaded_config["flame_tracking"] = 'FLAME_2023'
            del loaded_config["use_flame_2023"]

        super()._backward_compatibility(loaded_config)


class MeshProvider:

    @abstractmethod
    def get_n_timesteps(self) -> int:
        pass

    @deprecate
    @abstractmethod
    def get_uvs_per_vertex(self, mesh: trimesh.Trimesh) -> np.ndarray:
        pass

    def has_uv_faces(self) -> bool:
        return False

    @abstractmethod
    def get_uv_faces(self, mesh: trimesh.Trimesh) -> np.ndarray:
        pass

    @abstractmethod
    def get_uv_coords(self, mesh: trimesh.Trimesh) -> np.ndarray:
        pass

    @abstractmethod
    def get_vertices(self, timestep: int) -> np.ndarray:
        pass

    @abstractmethod
    def create_mesh(self, vertices: np.ndarray, timestep: Optional[int] = None) -> trimesh.Trimesh:
        pass

    def get_mesh(self, timestep: int) -> trimesh.Trimesh:
        vertices = self.get_vertices(timestep)
        mesh = self.create_mesh(vertices, timestep)
        return mesh

    @abstractmethod
    def has_mesh(self, timestep: int) -> bool:
        pass

    @classmethod
    @abstractmethod
    def compute_vertex_optical_flow(cls,
                                    mesh_1: Trimesh,
                                    mesh_2: Trimesh,
                                    world_to_cam_pose_1: Pose,
                                    world_to_cam_pose_2: Pose,
                                    intrinsics: Intrinsics,
                                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @classmethod
    def prepare_optical_flow(cls,
                             mesh_1: Trimesh,
                             mesh_2: Trimesh,
                             world_to_cam_pose_1: Pose,
                             world_to_cam_pose_2: Pose,
                             intrinsics: Intrinsics,
                             **kwargs):
        of_1_to_2, of_2_to_1 = cls.compute_vertex_optical_flow(
            mesh_1, mesh_2, world_to_cam_pose_1, world_to_cam_pose_2, intrinsics, **kwargs)

        mesh_1.vertex_attributes["forward_flow"] = of_1_to_2[..., :2]
        mesh_2.vertex_attributes["backward_flow"] = of_2_to_1[..., :2]
