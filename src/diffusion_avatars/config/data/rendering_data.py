from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

from elias.config import Config

from diffusion_avatars.renderer.provider.mesh_provider import MeshProviderConfig
from diffusion_avatars.constants import SERIALS
from dataclasses import field


@dataclass
class RenderingDataConfig(Config):
    name: Optional[str]

    downscale_factor: int
    sequences: List[Tuple[int, str]]  # [(17, "EXP-1-head"), (30, "EXP-2-eyes"), ... ]
    n_timesteps: int = -1
    use_mipmapping: bool = False
    use_antialiasing: bool = False
    use_ambient_dimensions: bool = False
    use_nphm: bool = False

    mesh_provider_config: MeshProviderConfig = MeshProviderConfig()

    serials: List[str] = field(default_factory=lambda: SERIALS)

    @classmethod
    def _backward_compatibility(cls, loaded_config: Dict):
        mesh_provider_config = dict()

        if "use_flame2023" in loaded_config:
            mesh_provider_config["use_flame_2023"] = loaded_config["use_flame2023"]
            del loaded_config["use_flame2023"]

        if "use_uv_faces" in loaded_config:
            mesh_provider_config["use_uv_faces"] = loaded_config["use_uv_faces"]
            del loaded_config["use_uv_faces"]

        if "close_mouth" in loaded_config:
            mesh_provider_config["close_mouth"] = loaded_config["close_mouth"]
            del loaded_config["close_mouth"]

        if "temporal_smoothing_config" in loaded_config:
            mesh_provider_config["temporal_smoothing_config"] = loaded_config["temporal_smoothing_config"]
            del loaded_config["temporal_smoothing_config"]

        if mesh_provider_config:
            loaded_config["mesh_provider_config"] = mesh_provider_config

        super()._backward_compatibility(loaded_config)


@dataclass
class RenderingDataStatistics(Config):
    total_n_timesteps: int  # Total number of different timesteps in dataset
    total_n_frames: int  # Total number of individual views in dataset (usually timesteps * 16)
    available_timesteps: List[List[int]]  # for each sequence (same order as in config), what timesteps are available
    available_sequences: Dict[str, Dict[str, List[int]]]  # { "p_id": {seq_1: [0,3,6], seq_2: [10,13,16]}}
    # Note: available_sequences stores participant_ids as str, because dacite cannot deal with dicts having int keys
