from typing import Any, Iterator

import cv2
import numpy as np
from elias.config import StringEnum
from elias.folder import DataFolder
from elias.manager.data import BaseDataManager, _SampleType
from elias.util import load_img, save_img, ensure_directory_exists_for_file, save_json, load_json

from diffusion_avatars.config.data.rendering_data import RenderingDataConfig, RenderingDataStatistics
from diffusion_avatars.env import DIFFUSION_AVATARS_DATA_PATH
from diffusion_avatars.util.quantization import DepthQuantizer, Quantizer, QuantizerConfig
from diffusion_avatars.util.quantization import UVQuantizer, FlameNormalsQuantizer


class RenderingName(StringEnum):
    UV = "uv"
    UV_DA = "uv_da"  # du/dx uv space derivatives. Needed for nvdiffrast's mipmapping
    UVW_DA = "uvw_da"  # du/dx uv space derivatives. Needed for nvdiffrast's mipmapping
    UV_AMBIENT = "uv_ambient"  # ambient dimensions of NPHM canonical space
    NORMALS = "normals"
    DEPTHS = "depths"
    MASK = "mask"
    FORWARD_FLOW = "forward_flow"
    FORWARD_FLOW_0 = "forward_flow_0"
    FORWARD_FLOW_1 = "forward_flow_1"
    FORWARD_FLOW_2 = "forward_flow_2"
    FORWARD_FLOW_3 = "forward_flow_3"
    FORWARD_FLOW_4 = "forward_flow_4"
    FORWARD_FLOW_5 = "forward_flow_5"
    FORWARD_FLOW_6 = "forward_flow_6"
    FORWARD_FLOW_7 = "forward_flow_7"
    BACKWARD_FLOW = "backward_flow"
    BACKWARD_FLOW_0 = "backward_flow_0"
    BACKWARD_FLOW_1 = "backward_flow_1"
    BACKWARD_FLOW_2 = "backward_flow_2"
    BACKWARD_FLOW_3 = "backward_flow_3"
    BACKWARD_FLOW_4 = "backward_flow_4"
    BACKWARD_FLOW_5 = "backward_flow_5"
    BACKWARD_FLOW_6 = "backward_flow_6"
    BACKWARD_FLOW_7 = "backward_flow_7"

    def get_n_channels(self,
                       nvdiffrast: bool = False,
                       use_canonical_coordinates: bool = False):
        if self == self.UV:
            if nvdiffrast and not use_canonical_coordinates:
                # nvdiffrast can perfectly handle arbitrary numbers of channels
                return 2
            else:
                return 3
        elif self == self.UV_AMBIENT:
            return 2
        elif self == self.NORMALS:
            return 3
        elif self == self.DEPTHS:
            return 1
        elif self == self.MASK:
            return 1
        elif self == self.FORWARD_FLOW:
            return 2
        elif self == self.BACKWARD_FLOW:
            return 2
        else:
            raise ValueError(f"Unknown rendering name: {self}")

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def forward_flow(t: int) -> 'RenderingName':
        forward_flow_names = [
            RenderingName.FORWARD_FLOW_0,
            RenderingName.FORWARD_FLOW_1,
            RenderingName.FORWARD_FLOW_2,
            RenderingName.FORWARD_FLOW_3,
            RenderingName.FORWARD_FLOW_4,
            RenderingName.FORWARD_FLOW_5,
            RenderingName.FORWARD_FLOW_6,
            RenderingName.FORWARD_FLOW_7,
        ]
        return forward_flow_names[t]

    @staticmethod
    def backward_flow(t: int) -> 'RenderingName':
        backward_flow_names = [
            RenderingName.BACKWARD_FLOW_0,
            RenderingName.BACKWARD_FLOW_1,
            RenderingName.BACKWARD_FLOW_2,
            RenderingName.BACKWARD_FLOW_3,
            RenderingName.BACKWARD_FLOW_4,
            RenderingName.BACKWARD_FLOW_5,
            RenderingName.BACKWARD_FLOW_6,
            RenderingName.BACKWARD_FLOW_7,
        ]
        return backward_flow_names[t]

    def is_flow(self) -> bool:
        return self.startswith(self.FORWARD_FLOW) or self.startswith(self.BACKWARD_FLOW)

    def is_forward_flow(self) -> bool:
        return self.startswith(self.FORWARD_FLOW)

    def is_backward_flow(self) -> bool:
        return self.startswith(self.BACKWARD_FLOW)


class RenderingDataManager(BaseDataManager[None, RenderingDataConfig, RenderingDataStatistics]):

    def __init__(self, version: str):
        super(RenderingDataManager, self).__init__(f"{DIFFUSION_AVATARS_DATA_PATH}/rendering_data", version, None)

        self._normals_quantizer = FlameNormalsQuantizer()
        self._depth_quantizer = DepthQuantizer()
        self._uv_quantizer = UVQuantizer()
        self._uv_ambient_quantizer = UVQuantizer(bits=8)  # Don't need crazy resolution for ambient dimensions

        self._config_cache = None

    @property
    def _use_nvdiffrast(self) -> bool:
        return self._config.use_nvdiffrast

    @property
    def _config(self):
        if self._config_cache is None:
            self._config_cache = self.load_config()
        return self._config_cache

    # ----------------------------------------------------------
    # Saving
    # ----------------------------------------------------------

    def save_rendering(self,
                       rendering: np.ndarray,
                       render_name: RenderingName,
                       participant_id: int,
                       sequence_name: str,
                       timestep: int,
                       serial: str):
        rendering_path = self.get_rendering_path(render_name, participant_id, sequence_name, timestep, serial)

        if self._use_nvdiffrast:
            ensure_directory_exists_for_file(rendering_path)
            if len(rendering.shape) > 2:
                rendering = rendering[..., ::-1]
            cv2.imwrite(rendering_path, rendering)
        else:
            save_img(rendering, rendering_path)

    def save_normals_rendering(self, rendering: np.ndarray, participant_id: int, sequence_name: str, timestep: int,
                               serial: str):
        if self._use_nvdiffrast:
            rendering = self._normals_quantizer.encode(rendering)

        self.save_rendering(rendering, RenderingName.NORMALS, participant_id, sequence_name, timestep, serial)

    def save_uv_rendering(self, rendering: np.ndarray, participant_id: int, sequence_name: str, timestep: int,
                          serial: str):
        if self._use_nvdiffrast:
            rendering = self._uv_quantizer.encode(rendering)
            if rendering.shape[-1] == 2:
                # Add artificial 3rd channel for .png storage
                rendering = np.concatenate([rendering,
                                            np.zeros(rendering.shape[:-1] + (1,), dtype=rendering.dtype)],
                                           axis=-1)

        self.save_rendering(rendering, RenderingName.UV, participant_id, sequence_name, timestep, serial)

    def save_uv_ambient_rendering(self, rendering: np.ndarray, participant_id: int, sequence_name: str, timestep: int,
                                  serial: str):
        rendering = self._uv_ambient_quantizer.encode(rendering)
        if rendering.shape[-1] == 2:
            # Add artificial 3rd channel for .png storage
            rendering = np.concatenate([rendering,
                                        np.zeros(rendering.shape[:-1] + (1,), dtype=rendering.dtype)],
                                       axis=-1)

        self.save_rendering(rendering, RenderingName.UV_AMBIENT, participant_id, sequence_name, timestep, serial)

    def save_depth_rendering(self, rendering: np.ndarray, participant_id: int, sequence_name: str, timestep: int,
                             serial: str):
        if self._use_nvdiffrast:
            rendering = self._depth_quantizer.encode(rendering)

        self.save_rendering(rendering, RenderingName.DEPTHS, participant_id, sequence_name, timestep, serial)

    def save_mask(self, mask: np.ndarray, participant_id: int, sequence_name: str, timestep: int, serial: str):
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        if len(mask.shape) == 2:
            mask = mask[..., None]

        self.save_rendering(mask, RenderingName.MASK, participant_id, sequence_name, timestep, serial)

    def save_rast(self, rast: np.ndarray, participant_id: int, sequence_name: str, timestep: int, serial: str):
        rast_path = self.get_rast_path(participant_id, sequence_name, timestep, serial)
        ensure_directory_exists_for_file(rast_path)

        # bary_uv is not needed for antialiasing. Therefore, we drop it here to save storage
        # quantizer_uv = Quantizer(0, 1, 8)
        # bary_u = quantizer_uv.encode(rast[..., 0])
        # bary_v = quantizer_uv.encode(rast[..., 1])

        # save_img(bary_u, f"{self.get_renderings_folder('rast', participant_id, sequence_name, serial)}/bary_u.png")
        # save_img(bary_v, f"{self.get_renderings_folder('rast', participant_id, sequence_name, serial)}/bary_v.png")

        disparity = rast[..., 2]
        min_disparity = disparity[disparity > 0].min()
        max_disparity = disparity[disparity > 0].max()
        quantizer_disparity = Quantizer(min_disparity, max_disparity, 8)
        disparity = quantizer_disparity.encode(disparity)

        triangle_ids = rast[..., 3].astype(np.uint16)

        quantizer_config_path = self.get_rast_disparity_quantizer_config_path(participant_id, sequence_name, timestep,
                                                                              serial)
        rast_disparity_path = self.get_rast_disparity_path(participant_id, sequence_name, timestep, serial)
        rast_triangle_ids_path = self.get_rast_triangle_id_path(participant_id, sequence_name, timestep, serial)

        save_img(disparity, rast_disparity_path)
        save_json(quantizer_disparity.to_config().to_json(), quantizer_config_path)
        cv2.imwrite(rast_triangle_ids_path, triangle_ids)

    def save_uvw_da(self, uvw_da: np.ndarray, participant_id: int, sequence_name: str, timestep: int, serial: str):
        x_da = uvw_da[..., [0, 2, 4]]  # du / dx, dv / dx, dw / dx
        y_da = uvw_da[..., [1, 3, 5]]  # du /dy, dv / dy, dw / dy
        x_da_path = self.get_uvw_x_da_path(participant_id, sequence_name, timestep, serial)
        y_da_path = self.get_uvw_y_da_path(participant_id, sequence_name, timestep, serial)
        x_da_quantizer_config_path = self.get_uvw_x_da_quantizer_config_path(participant_id, sequence_name, timestep, serial)
        y_da_quantizer_config_path = self.get_uvw_y_da_quantizer_config_path(participant_id, sequence_name, timestep, serial)
        ensure_directory_exists_for_file(x_da_path)
        ensure_directory_exists_for_file(y_da_path)

        x_da = np.clip(x_da, a_min=-10, a_max=10)
        y_da = np.clip(y_da, a_min=-10, a_max=10)

        # Quantize
        x_min_per_channel = x_da.min(axis=0).min(axis=0)  # [3]
        x_max_per_channel = x_da.max(axis=0).max(axis=0)  # [3]
        x_quantizer = Quantizer(x_min_per_channel, x_max_per_channel, 16)
        x_da_quantized = x_quantizer.encode(x_da)

        y_min_per_channel = y_da.min(axis=0).min(axis=0)  # [3]
        y_max_per_channel = y_da.max(axis=0).max(axis=0)  # [3]
        y_quantizer = Quantizer(y_min_per_channel, y_max_per_channel, 16)
        y_da_quantized = y_quantizer.encode(y_da)

        cv2.imwrite(x_da_path, x_da_quantized)
        cv2.imwrite(y_da_path, y_da_quantized)
        save_json(x_quantizer.to_config().to_json(), x_da_quantizer_config_path)
        save_json(y_quantizer.to_config().to_json(), y_da_quantizer_config_path)

    def save_uv_da(self, uv_da: np.ndarray, participant_id: int, sequence_name: str, timestep: int, serial: str):
        uv_da_path = self.get_uv_da_path(participant_id, sequence_name, timestep, serial)
        uv_da_quantizer_config_path = self.get_uv_da_quantizer_config_path(participant_id, sequence_name, timestep,
                                                                           serial)
        ensure_directory_exists_for_file(uv_da_path)

        # Quantize
        min_per_channel = uv_da.min(axis=0).min(axis=0)  # [4]
        max_per_channel = uv_da.max(axis=0).max(axis=0)  # [4]
        quantizer = Quantizer(min_per_channel, max_per_channel, 16)
        uv_da_quantized = quantizer.encode(uv_da)

        cv2.imwrite(uv_da_path, uv_da_quantized)
        save_json(quantizer.to_config().to_json(), uv_da_quantizer_config_path)

    def save_latent_image(self, latent_image: np.ndarray, participant_id: int, sequence_name: str, timestep: int,
                          serial: str):
        latent_image_path = self.get_latent_image_path(participant_id, sequence_name, timestep, serial)
        ensure_directory_exists_for_file(latent_image_path)
        np.save(latent_image_path, latent_image)

    # ----------------------------------------------------------
    # Loading
    # ----------------------------------------------------------

    def _load_rendering(self,
                        render_name: RenderingName,
                        participant_id: int,
                        sequence_name: str,
                        timestep: int,
                        serial: str) -> np.ndarray:
        rendering_path = self.get_rendering_path(render_name, participant_id, sequence_name, timestep, serial)

        if self._use_nvdiffrast:
            flags = cv2.IMREAD_UNCHANGED
            rendering = cv2.imread(rendering_path, flags=flags)
            if rendering is None:
                raise ValueError(f"Could not find rendering at path {rendering}")
            if len(rendering.shape) > 2:
                rendering = rendering[..., ::-1]
            return rendering
        else:
            return load_img(rendering_path)

    def load_rendering(self,
                       render_name: RenderingName,
                       participant_id: int,
                       sequence_name: str,
                       timestep: int,
                       serial: str) -> np.ndarray:
        if render_name == RenderingName.NORMALS:
            return self.load_normals_rendering(participant_id, sequence_name, timestep, serial)
        elif render_name == RenderingName.UV:
            return self.load_uv_rendering(participant_id, sequence_name, timestep, serial)
        elif render_name == RenderingName.UV_AMBIENT:
            return self.load_uv_ambient_rendering(participant_id, sequence_name, timestep, serial)
        elif render_name == RenderingName.DEPTHS:
            return self.load_depth_rendering(participant_id, sequence_name, timestep, serial)
        elif render_name == RenderingName.MASK:
            return self.load_mask(participant_id, sequence_name, timestep, serial)
        else:
            raise ValueError(f"Unknown rendering name: {render_name}")

    def load_normals_rendering(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        normals_rendering = self._load_rendering(RenderingName.NORMALS, participant_id, sequence_name, timestep, serial)

        if self._use_nvdiffrast:
            normals_rendering = self._normals_quantizer.decode(normals_rendering)
        else:
            normals_rendering = normals_rendering / 255.

        return normals_rendering

    def load_uv_rendering(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        uv_rendering = self._load_rendering(RenderingName.UV, participant_id, sequence_name, timestep, serial)

        if self._use_nvdiffrast:
            if not self._config.use_nphm:
                uv_rendering = uv_rendering[..., :2]

            uv_rendering = self._uv_quantizer.decode(uv_rendering)
        else:
            uv_rendering = uv_rendering / 255.

        return uv_rendering

    def load_uv_ambient_rendering(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        uv_ambient_rendering = self._load_rendering(RenderingName.UV_AMBIENT, participant_id, sequence_name, timestep, serial)
        uv_ambient_rendering = uv_ambient_rendering[..., :2]
        uv_ambient_rendering = self._uv_ambient_quantizer.decode(uv_ambient_rendering)
        return uv_ambient_rendering

    def load_depth_rendering(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        depth_rendering = self._load_rendering(RenderingName.DEPTHS, participant_id, sequence_name, timestep, serial)

        if self._use_nvdiffrast:
            depth_rendering = self._depth_quantizer.decode(depth_rendering)
        else:
            depth_rendering = depth_rendering / 255.

        return depth_rendering

    def load_mask(self, participant_id: int, sequence_name: str, timestep: int, serial: str):
        mask = self._load_rendering(RenderingName.MASK, participant_id, sequence_name, timestep, serial)

        return mask.astype(bool)

    def load_rast(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        rast_disparity_path = self.get_rast_disparity_path(participant_id, sequence_name, timestep, serial)
        quantizer_config_path = self.get_rast_disparity_quantizer_config_path(participant_id, sequence_name, timestep,
                                                                              serial)

        quantizer_config = QuantizerConfig.from_json(load_json(quantizer_config_path))
        quantizer_disparity = Quantizer.from_config(quantizer_config)
        disparity = load_img(rast_disparity_path)
        disparity = quantizer_disparity.decode(disparity)

        rast_triangle_ids_path = self.get_rast_triangle_id_path(participant_id, sequence_name, timestep, serial)
        flags = cv2.IMREAD_UNCHANGED
        triangle_ids = cv2.imread(rast_triangle_ids_path, flags=flags)

        rast = np.zeros((triangle_ids.shape[0], triangle_ids.shape[1], 4), dtype=np.float32)
        rast[..., 2] = disparity
        rast[..., 3] = triangle_ids

        return rast

    def load_uvw_da(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        x_da_quantizer_config_path = self.get_uvw_x_da_quantizer_config_path(participant_id, sequence_name, timestep, serial)
        y_da_quantizer_config_path = self.get_uvw_y_da_quantizer_config_path(participant_id, sequence_name, timestep, serial)

        x_da_path = self.get_uvw_x_da_path(participant_id, sequence_name, timestep, serial)
        y_da_path = self.get_uvw_y_da_path(participant_id, sequence_name, timestep, serial)

        x_quantizer_config = QuantizerConfig.from_json(load_json(x_da_quantizer_config_path))
        y_quantizer_config = QuantizerConfig.from_json(load_json(y_da_quantizer_config_path))
        x_quantizer = Quantizer.from_config(x_quantizer_config)
        y_quantizer = Quantizer.from_config(y_quantizer_config)

        x_da = cv2.imread(x_da_path, flags=cv2.IMREAD_UNCHANGED)
        y_da = cv2.imread(y_da_path, flags=cv2.IMREAD_UNCHANGED)
        x_da = x_quantizer.decode(x_da)
        y_da = y_quantizer.decode(y_da)

        # uv_da = np.zeros((x_da.shape[0], x_da.shape[1], 6), dtype=np.float32)
        # uv_da[..., [0, 2, 4]] = x_da
        # uv_da[..., [1, 3, 5]] = y_da
        uvw_da = np.concatenate([x_da[..., None], y_da[..., None]], axis=-1)  # [H, W, 3, 2]
        uvw_da = uvw_da.reshape((uvw_da.shape[0], uvw_da.shape[1], -1))  # [H, W, 6]

        return uvw_da

    def load_uv_da(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        uv_da_quantizer_config_path = self.get_uv_da_quantizer_config_path(participant_id, sequence_name, timestep,
                                                                           serial)
        uv_da_path = self.get_uv_da_path(participant_id, sequence_name, timestep, serial)

        quantizer_config = QuantizerConfig.from_json(load_json(uv_da_quantizer_config_path))
        quantizer = Quantizer.from_config(quantizer_config)
        uv_da = cv2.imread(uv_da_path, flags=cv2.IMREAD_UNCHANGED)
        uv_da = quantizer.decode(uv_da)

        return uv_da

    def load_latent_image(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        return np.load(self.get_latent_image_path(participant_id, sequence_name, timestep, serial))

    # ==========================================================
    # Paths
    # ==========================================================

    def get_rendering_path(self,
                           render_name: RenderingName,
                           participant_id: int,
                           sequence_name: str,
                           timestep: int,
                           serial: str):
        return f"{self.get_renderings_folder(render_name, participant_id, sequence_name, serial)}/frame_{timestep:05d}.png"

    def get_renderings_folder(self, render_name: RenderingName,
                              participant_id: int,
                              sequence_name: str,
                              serial: str) -> str:
        return f"{self._location}/{participant_id:03d}/{sequence_name}/cam_{serial}/{render_name}"

    def get_normals_rendering_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return self.get_rendering_path(RenderingName.NORMALS, participant_id, sequence_name, timestep, serial)

    def get_uv_rendering_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return self.get_rendering_path(RenderingName.UV, participant_id, sequence_name, timestep, serial)

    def get_depth_rendering_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return self.get_rendering_path(RenderingName.DEPTHS, participant_id, sequence_name, timestep, serial)

    def get_rast_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return f"{self.get_renderings_folder('rast', participant_id, sequence_name, serial)}/frame_{timestep:05d}.png"

    def get_rast_disparity_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return f"{self.get_renderings_folder('rast', participant_id, sequence_name, serial)}/disparity_{timestep:05d}.png"

    def get_rast_disparity_quantizer_config_path(self, participant_id: int, sequence_name: str, timestep: int,
                                                 serial: str) -> str:
        return f"{self.get_renderings_folder('rast', participant_id, sequence_name, serial)}/disparity_quantizer_config_{timestep:05d}.json"

    def get_rast_triangle_id_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return f"{self.get_renderings_folder('rast', participant_id, sequence_name, serial)}/triangle_id_{timestep:05d}.png"

    def get_uv_da_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return f"{self.get_renderings_folder('uv_da', participant_id, sequence_name, serial)}/frame_{timestep:05d}.png"

    def get_uvw_x_da_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return f"{self.get_renderings_folder('uvw_da', participant_id, sequence_name, serial)}/uvw_x_frame_{timestep:05d}.png"

    def get_uvw_y_da_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return f"{self.get_renderings_folder('uvw_da', participant_id, sequence_name, serial)}/uvw_y_frame_{timestep:05d}.png"

    def get_uv_da_quantizer_config_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str):
        return f"{self.get_renderings_folder('uv_da', participant_id, sequence_name, serial)}/quantizer_config_{timestep:05d}.json"

    def get_uvw_x_da_quantizer_config_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str):
        return f"{self.get_renderings_folder('uvw_da', participant_id, sequence_name, serial)}/uvw_x_quantizer_config_{timestep:05d}.json"

    def get_uvw_y_da_quantizer_config_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str):
        return f"{self.get_renderings_folder('uvw_da', participant_id, sequence_name, serial)}/uvw_y_quantizer_config_{timestep:05d}.json"

    def get_latent_image_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> str:
        return f"{self._location}/{participant_id:03d}/{sequence_name}/cam_{serial}/latent_images/frame_{timestep:05d}.npy"

    def __iter__(self) -> Iterator[_SampleType]:
        raise NotImplementedError()

    def _save(self, data: Any):
        raise NotImplementedError()


class RenderingDataFolder(DataFolder[RenderingDataManager]):
    def __init__(self):
        super(RenderingDataFolder, self).__init__(f"{DIFFUSION_AVATARS_DATA_PATH}/rendering_data", localize_via_run_name=True)
