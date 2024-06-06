import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import auto
from math import floor, ceil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Literal

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
from dreifus.graphics import Dimensions
from elias.config import Config, StringEnum
from elias.util.io import resize_img
from diffusion_avatars.constants import SERIALS
from diffusion_avatars.data_manager.nersemble_data_manager import NeRSembleSequenceDataManager, NeRSembleParticipantDataManager
from diffusion_avatars.util.quantization import DepthQuantizer
from scipy.ndimage import binary_dilation, label, gaussian_filter
from torch.utils.data import Dataset

from diffusion_avatars.config.data.rendering_data import RenderingDataConfig, RenderingDataStatistics
from diffusion_avatars.constants import TEST_SEQUENCES
from diffusion_avatars.data_manager.rendering_data_manager import RenderingDataFolder, RenderingName, RenderingDataManager
# from diffusion_avatars.model.frequency_encoding import FrequencyEncoding
from diffusion_avatars.renderer.provider.mesh_provider import TemporalSmoothingConfig
from diffusion_avatars.renderer.provider.nphm_provider import NPHMProvider
# from diffusion_avatars.util.flow import warp_image
from diffusion_avatars.util.render import get_mesh_provider, get_renderer, prepare_intrinsics
from diffusion_avatars.util.types import to_shallow_dict, default_dict_to_dict


# ==========================================================
# DatasetSplit
# ==========================================================

class DatasetSplit(StringEnum):
    TRAIN = auto()
    VALID_HOLD_OUT_VIEW = auto()  # Other views of the same person/sequence/expression have been seen
    VALID_HOLD_OUT_EXP = auto()  # Other expressions of the same person/sequence have been seen
    VALID_HOLD_OUT_SEQ = auto()  # Other sequences of the same person have been seen
    VALID_HOLD_OUT_PERSON = auto()  # Other persons have been seen

    def get_log_name(self):
        if self == self.TRAIN:
            return 'train'
        elif self == self.VALID_HOLD_OUT_VIEW:
            return 'validation/hold-out_view'
        elif self == self.VALID_HOLD_OUT_EXP:
            return 'validation/hold-out_expression'
        elif self == self.VALID_HOLD_OUT_SEQ:
            return 'validation/hold-out_sequence'
        elif self == self.VALID_HOLD_OUT_PERSON:
            return 'validation/hold-out_person'

        raise NotImplementedError(f"get_name() for split {self} not implemented")


# ==========================================================
# RenderingDatasetConfig
# ==========================================================

@dataclass
class SampleMetadata:
    participant_id: int
    sequence_name: str
    timestep: int
    serial: str

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class TextureFieldConfig(Config):
    n_levels: int = 8
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 2.0

    def create_encoding(self, dimensions: int = 2) -> 'tcnn.Encoding':
        import tinycudann as tcnn

        texture_field = tcnn.Encoding(dimensions, {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": self.n_levels,  # from 16 -> 4096
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
            "interpolation": "Linear"
        })
        return texture_field


SegmentationType = Literal['bisenet', 'facer']


@dataclass
class RenderingDatasetConfig(Config):
    dataset_version: str
    rendering_names: List[RenderingName]
    resolution: int
    n_participants: int = -1  # How many participants to use
    normalize: bool = False  # Whether to normalize images between [-1, 1]
    seed: int = 181998
    use_color_correction: bool = False

    # Splits
    split_ratio: float = 1.0  # Percentage of samples used for training
    hold_out_cameras: Optional[
        List[str]] = None  # If specified, the HOLD_OUT_VIEW will exclude all frames from these cameras
    supported_splits: List[DatasetSplit] = field(default_factory=lambda: [DatasetSplit.TRAIN])
    use_predefined_test_split: bool = False  # If set, will put the sequences defined in constants.py into VALID_HOLD_OUT_SEQ
    hold_out_sequences: Optional[List[str]] = None  # Only relevant, if use_predefined_test_split=False
    exclude_sequences: Optional[List[str]] = None  # If specified, the listed sequences won't be considered for ANY split

    # Cropping
    use_random_cropping: bool = False
    min_crop_size: float = 0.7
    crop_selection_aspect_ratio: float = 1.2  # Aspect ratio of face bounding box in which crops are chosen (h/w)
    interpolate_target_images: bool = False

    # Neural Textures
    use_neural_textures: bool = False
    use_background_texture: bool = False  # If set, the background mask will be used to assign background pixels a learnable color
    use_ambient_textures: bool = False
    dim_neural_textures: int = 8
    res_neural_textures: int = 32
    use_mipmapping: bool = False

    use_texture_fields: bool = False
    use_texture_triplanes: bool = False
    use_texture_hierarchy: bool = False
    n_texture_hierarchy_levels: int = 4
    texture_field_config: TextureFieldConfig = TextureFieldConfig()
    use_canonical_coordinates: bool = False  # Only relevant for NPHM, skip uv map and directly embed 3d canonical coordinates with texture field

    # Autoregressive training
    use_autoregressive: bool = False
    use_teacher_forcing: bool = True
    use_random_previous_viewpoints: bool = False  # If True, simulate viewpoint changes during training by sampling random cameras
    warp_previous_outputs: bool = False
    occlusion_mask_threshold: Optional[float] = None
    use_nphm_subdivision: bool = False  # 4x the number of vertices for NPHM. Can yield better uv correspondences
    n_previous_frames: int = 0
    prob_autoregressive_dropout: float = 0.0  # Probability that a dataset sample should be the start of a sequence
    use_crop_sweep: bool = False  # If set, previous frames won't use the same random crop but a moving one to simulate a camera trajectory

    # Frequency Encodings
    use_freq_encoding_uv: bool = False
    use_freq_encoding_depth: bool = False
    n_frequencies_uv: int = 4
    n_frequencies_depth: int = 4

    # Temporal Block Training
    temporal_batch_size: int = 0  # If >= 1, then special temporal convolutions (and attention) blocks will be added. A batch will consist of a set of sequential frames [B * T]
    use_temporal_batch_random_viewpoints: bool = False

    # Additional expression condition
    use_expression_condition: bool = False
    force_flame_expression_condition: bool = False  # Only relevant for NPHM. If set, always FLAME expression codes will be used as condition
    expression_condition_smoothing: Optional[TemporalSmoothingConfig] = None
    include_eye_condition: bool = False

    # Segmentation masks
    remove_torso: bool = False
    remove_neck_in_mask: bool = False
    remove_background_in_mask: bool = False
    remove_torso_in_mask: bool = False
    include_mouth_mask: bool = False
    include_foreground_mask: bool = False  # Useful for evaluation: samples will have a mask that excludes background and torso
    segmentation_type: SegmentationType = 'bisenet'
    smooth_neck_boundary: int = 0

    remap_noise: bool = False  # Whether noise should be re-mapped to mash to avoid screen space texture stitching

    include_view_directions: bool = False  # Whether to also output view directions
    use_spherical_harmonics: bool = False  # If set, will multiply neural textured rendering with SH coefficients

    def split_renderings(self, renderings: np.ndarray) -> List[np.ndarray]:
        separate_renderings = []
        current_channel = 0
        data_config = self.get_data_config()
        for rendering_name in self.rendering_names:
            n_channels = rendering_name.get_n_channels(
                use_canonical_coordinates=self.use_canonical_coordinates)

            # Apply frequency encodings
            if self.use_freq_encoding_uv and rendering_name == RenderingName.UV:
                n_channels += 2 * self.n_frequencies_uv * n_channels
            elif self.use_freq_encoding_depth and rendering_name == RenderingName.DEPTHS:
                n_channels += 2 * self.n_frequencies_depth * n_channels

            rendering = renderings[..., current_channel: current_channel + n_channels]
            separate_renderings.append(rendering)
            current_channel += n_channels

        # Neural Textures
        if self.use_neural_textures:
            if self.use_texture_fields:
                n_channels = self.texture_field_config.n_levels * self.texture_field_config.n_features_per_level
            elif self.use_texture_triplanes:
                n_channels = 3 * self.dim_neural_textures
            else:
                n_channels = self.dim_neural_textures
            rendering = renderings[..., current_channel: current_channel + n_channels]
            separate_renderings.append(rendering)
            current_channel += n_channels

        if self.use_ambient_textures:
            n_channels = self.dim_neural_textures
            rendering = renderings[..., current_channel: current_channel + n_channels]
            separate_renderings.append(rendering)
            current_channel += n_channels

        # Previous outputs
        if self.use_autoregressive:
            n_channels = 3
            for _ in range(self.n_previous_frames):
                rendering = renderings[..., current_channel: current_channel + n_channels]
                separate_renderings.append(rendering)
                current_channel += n_channels

        return separate_renderings

    def concat_conditions(self, batch: 'RenderingDatasetInferenceBatch',
                          n_previous_frames: Optional[int] = None) -> torch.tensor:
        conditions = []
        for rendering_name in self.rendering_names:
            condition = batch.conditions[rendering_name]

            # Apply frequency encodings
            if self.use_freq_encoding_uv and rendering_name == RenderingName.UV:
                freq_encoding = FrequencyEncoding(
                    condition.shape[1],
                    self.n_frequencies_uv,
                    min_freq_exp=0,
                    max_freq_exp=self.n_frequencies_uv - 2,
                    include_input=True)
                condition = freq_encoding(condition, dim=1)
            elif self.use_freq_encoding_depth and rendering_name == RenderingName.DEPTHS:
                freq_encoding = FrequencyEncoding(
                    condition.shape[1],
                    self.n_frequencies_depth,
                    min_freq_exp=0,
                    max_freq_exp=self.n_frequencies_depth - 2,
                    include_input=True)
                condition = freq_encoding(condition, dim=1)

            conditions.append(condition)

        if batch.neural_textured_rendering is not None:
            conditions += [batch.neural_textured_rendering]

        if batch.ambient_textured_rendering is not None:
            conditions += [batch.ambient_textured_rendering]

        if batch.previous_outputs is not None:
            B, N, C, H, W = batch.previous_outputs.shape
            previous_outputs = batch.previous_outputs.reshape((B, N * C, H, W))
            conditions += [previous_outputs]

        # [batched_cond1, batched_cond2, ..., batched_cond2]  [[B, C, H, W]]
        conditions = torch.cat(conditions, dim=1)

        # Merge previous batches into a sequential temporal block
        if self.temporal_batch_size > 0:
            #                                               b1_t1 b1_t2 ... b1_tn     b1_tnn
            #  b1_t1 b1_t2 ... b1_tn b2_t1 ... bm_tn   ->   b2_t1 b2_t2 ... b2_tn  +  b2_tnn    -> flatten
            #                                               ...                       ...
            #                                               bm_t1 bm_t2 ... bm_tn     bm_tnn
            # TODO: This is problematic if all temporal streams are shorter than temporal_batch_size
            #   In this case, the returned conditions tensor won't have the correct shape because T is too small
            n_previous_frames = self.temporal_batch_size - 1 if n_previous_frames is None else n_previous_frames
            if n_previous_frames > 0:
                if batch.previous_batch is None:
                    # Fill remaining timesteps with 0s
                    B, C, H, W = conditions.shape
                    t = n_previous_frames
                    previous_conditions = torch.zeros((B, t, C, H, W), device=conditions.device)
                    conditions = conditions.unsqueeze(1)  # [B, 1, C, H, W]
                    conditions = torch.concatenate([previous_conditions, conditions], dim=1)  # [B, t+1, C, H, W]
                    conditions = conditions.reshape((B * (t + 1), C, H, W))
                else:
                    previous_conditions = self.concat_conditions(batch.previous_batch,
                                                                 n_previous_frames=n_previous_frames - 1)  # [B*t, C, H, W]
                    B = len(batch)
                    B_prev = len(batch.previous_batch)
                    Bt, C, H, W = previous_conditions.shape
                    t = int(Bt / B_prev)
                    previous_conditions = previous_conditions.reshape((B_prev, t, C, H, W))
                    merged_conditions = torch.zeros((B, t + 1, C, H, W), device=conditions.device)
                    merged_conditions[batch.previous_sample_ids, :t] = previous_conditions
                    merged_conditions[:, -1] = conditions
                    conditions = merged_conditions.reshape((B * (t + 1), C, H, W))

        return conditions

    def concat_target_images(self, batch: 'RenderingDatasetBatch',
                             n_previous_frames: Optional[int] = None) -> torch.Tensor:
        target_images = batch.target_images

        if self.temporal_batch_size > 0:
            n_previous_frames = self.temporal_batch_size - 1 if n_previous_frames is None else n_previous_frames
            if n_previous_frames > 0:
                if batch.previous_batch is None:
                    # Fill remaining timesteps with 0s
                    B, C, H, W = target_images.shape
                    t = n_previous_frames
                    previous_target_images = torch.zeros((B, t, C, H, W), device=target_images.device)
                    target_images = target_images.unsqueeze(1)  # [B, 1, C, H, W]
                    target_images = torch.concatenate([previous_target_images, target_images],
                                                      dim=1)  # [B, t+1, C, H, W]
                    target_images = target_images.reshape((B * (t + 1), C, H, W))
                else:
                    B, C, H, W = batch.target_images.shape
                    B_prev = len(batch.previous_batch)
                    previous_target_images = self.concat_target_images(batch.previous_batch,
                                                                       n_previous_frames=n_previous_frames - 1)  # [B*t, C, H, W]
                    Bt, C, H, W = previous_target_images.shape
                    t = int(Bt / B_prev)
                    previous_target_images = previous_target_images.reshape((B_prev, t, C, H, W))
                    merged_target_images = torch.zeros((B, t + 1, C, H, W), device=batch.target_images.device)
                    merged_target_images[batch.previous_sample_ids, :t] = previous_target_images
                    merged_target_images[:, -1] = batch.target_images
                    target_images = merged_target_images.reshape((B * (t + 1), C, H, W))

        return target_images

    def prepare_batch(self,
                      batch: 'RenderingDatasetBatch',
                      device: torch.device,
                      latent_noise: Optional[torch.Tensor] = None,
                      neural_textures: Optional[torch.Tensor] = None,
                      neural_texture_fields: Optional[List['tcnn.Encoding']] = None,
                      neural_texture_triplanes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                      neural_texture_hierarchy: Optional[List[torch.Tensor]] = None,
                      ambient_textures: Optional[torch.Tensor] = None,
                      background_texture: Optional[torch.Tensor] = None,
                      n_previous_frames: Optional[int] = None) -> 'RenderingDatasetBatch':
        """
        Put tensors to device.
        Render guiding mesh with given neural texture.
        Run this preparation for "previous batches" as well

        Parameters
        ----------
            batch
            neural_textures
            n_previous_frames

        Returns
        -------

        """

        B = len(batch)
        batch.to(device)

        if latent_noise is not None:
            if len(latent_noise.shape) == 3:
                # Broadcast noise across batch dimension
                latent_noise = torch.stack([latent_noise for _ in range(B)])
            batch.latent_noise = latent_noise.to(device)

        if self.use_neural_textures:
            if neural_textures is not None:
                neural_textures = neural_textures.to(device)
            batch.add_neural_textures(neural_textures, neural_texture_fields=neural_texture_fields, neural_texture_triplanes=neural_texture_triplanes,
                                      neural_texture_hierarchy=neural_texture_hierarchy,
                                      use_spherical_harmonics=self.use_spherical_harmonics)

        if self.use_ambient_textures:
            if ambient_textures is not None:
                ambient_textures = ambient_textures.to(device)
                batch.add_ambient_textures(ambient_textures)

        if self.use_autoregressive and not self.use_teacher_forcing or self.temporal_batch_size > 0:
            if n_previous_frames is None:
                if self.use_autoregressive:
                    n_previous_frames = self.n_previous_frames
                elif self.temporal_batch_size > 0:
                    n_previous_frames = self.temporal_batch_size

            if n_previous_frames > 0 and batch.previous_batch is not None:
                assert batch.previous_sample_ids is not None
                self.prepare_batch(batch.previous_batch,
                                   device,
                                   latent_noise=latent_noise,
                                   neural_textures=neural_textures[
                                       batch.previous_sample_ids] if neural_textures is not None else None,
                                   neural_texture_fields=[neural_texture_fields[i] for i in
                                                          batch.previous_sample_ids] if neural_texture_fields is not None else None,
                                   neural_texture_triplanes=(neural_texture_triplanes[0][batch.previous_sample_ids],
                                                             neural_texture_triplanes[1][batch.previous_sample_ids],
                                                             neural_texture_triplanes[2][
                                                                 batch.previous_sample_ids]) if neural_texture_triplanes is not None else None,
                                   neural_texture_hierarchy=[neural_texture[batch.previous_sample_ids] for neural_texture in
                                                             neural_texture_hierarchy] if neural_texture_hierarchy is not None else None,
                                   background_texture=background_texture,
                                   n_previous_frames=n_previous_frames - 1)

        if self.use_background_texture:
            if batch.neural_textured_rendering is None:
                batch.neural_textured_rendering = torch.zeros(
                    (B,
                     self.dim_neural_textures,
                     self.resolution,
                     self.resolution), device=device)

            background_mask = ~batch.conditions[RenderingName.MASK].permute(0, 2, 3, 1)[..., 0]
            batch.neural_textured_rendering.permute(0, 2, 3, 1)[background_mask] = background_texture

        return batch

    def get_data_manager(self) -> RenderingDataManager:
        return RenderingDataFolder().open_dataset(self.dataset_version)

    def get_data_config(self) -> RenderingDataConfig:
        return self.get_data_manager().load_config()

    def get_data_statistics(self) -> RenderingDataStatistics:
        return self.get_data_manager().load_stats()

    def eval(self, remove_neck: bool = False, remove_background: bool = False, remove_torso: bool = False) -> 'RenderingDatasetConfig':
        dataset_config_eval = replace(self,
                                      min_crop_size=1.0,
                                      crop_selection_aspect_ratio=1.0,
                                      prob_autoregressive_dropout=0,
                                      use_temporal_batch_random_viewpoints=False,
                                      use_random_previous_viewpoints=False,
                                      use_crop_sweep=False,
                                      include_foreground_mask=True,
                                      remove_neck_in_mask=remove_neck,
                                      remove_background_in_mask=remove_background,
                                      remove_torso_in_mask=remove_torso)

        return dataset_config_eval

    def load_target_image(self, sample_metadata: SampleMetadata) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[bool]]:
        sequence_data_manager = NeRSembleSequenceDataManager(
            sample_metadata.participant_id,
            sample_metadata.sequence_name,
            downscale_factor=self.get_data_config().downscale_factor)

        # Can only load target image, if it exists
        if not sequence_data_manager.has_image(sample_metadata.timestep, sample_metadata.serial):
            image_path = sequence_data_manager.get_image_path(sample_metadata.timestep, sample_metadata.serial)
            print(f"[WARNING] Cannot load target image for {sample_metadata}. File {image_path} does not exist!")
            return None, None, None

        target_image = sequence_data_manager.load_image(
            sample_metadata.timestep,
            sample_metadata.serial,
            use_color_correction=self.use_color_correction,
            use_robust_matting_mask=True)

        segmentation_mask = None
        is_facer_mask = None
        if self.remove_torso or self.include_mouth_mask or self.include_foreground_mask:
            is_facer_mask = False
            if self.segmentation_type == 'facer':
                segmentation_path = sequence_data_manager.get_facer_segmentation_path(sample_metadata.timestep,
                                                                                      sample_metadata.serial)
                if Path(segmentation_path).exists():
                    segmentation_mask = sequence_data_manager.load_facer_segmentation_mask(sample_metadata.timestep,
                                                                                           sample_metadata.serial)
                    segmentation_mask = resize_img(segmentation_mask,
                                                   (target_image.shape[1] / segmentation_mask.shape[1],
                                                    target_image.shape[0] / segmentation_mask.shape[0]),
                                                   interpolation='nearest')

                    remove_ids = [3]  # torso
                    if segmentation_mask.max() > 14:
                        # TODO: Simon is currently re-running the segmentation. Once it is done, this needs to be adapted
                        #   There will be a proper class ID for out-of-bounding-box (instead of just the max)
                        remove_ids.append(segmentation_mask.max())  # out-of-bounding-box
                    is_facer_mask = True

            if self.segmentation_type == 'bisenet' \
                    or self.segmentation_type == 'facer' and segmentation_mask is None:
                segmentation_mask = sequence_data_manager.load_bisenet_segmentation_mask(sample_metadata.timestep,
                                                                                         sample_metadata.serial)
                segmentation_mask = resize_img(segmentation_mask,
                                               (target_image.shape[1] / segmentation_mask.shape[1],
                                                target_image.shape[0] / segmentation_mask.shape[0]),
                                               interpolation='nearest')
                remove_ids = [16]  # torso

            if self.segmentation_type not in {'facer', 'bisenet'}:
                raise ValueError(f"Unknown segmentation type: {self.segmentation_type}")

            torso_mask = np.zeros_like(segmentation_mask, dtype=bool)
            for remove_id in remove_ids:
                torso_mask = torso_mask | (segmentation_mask == remove_id)

            if is_facer_mask:
                # Islands need to have a median color difference larger than that to be considered outside neck region
                neck_color_threshold = 0.15

                # "Grow" torso mask to include some background pixels as well
                # This addresses the issue that the torso region is often estimated too small resulting in a thin
                # line of t-shirt colored pixel being left over after the torso removal
                background_mask = segmentation_mask == 0
                torso_mask_dilated = binary_dilation(torso_mask, iterations=50)

                # Only grow into background region if it is not an island that borders the "neck" region
                # It sometimes happens that part of the neck region is detected as background which is wrong
                # background_mask_islands, n_islands = label(background_mask)
                background_mask_islands, n_islands = get_sorted_islands(background_mask)
                neck_mask = segmentation_mask == 1
                neck_color = np.median(target_image[neck_mask], axis=0) / 255
                for i_island in range(2, n_islands + 1):
                    island_mask = background_mask_islands[i_island]
                    island_color = np.median(target_image[island_mask], axis=0) / 255.
                    if np.linalg.norm(neck_color - island_color) < neck_color_threshold:
                        # Color is very close to neck color, this island is probably inside neck region and should
                        # not be considered proper background
                        island_mask_dilated = binary_dilation(island_mask, iterations=2)
                        island_mask_boundary = island_mask_dilated & ~island_mask
                        island_mask_intersection_neck = island_mask_boundary & (neck_mask)
                        if np.any(island_mask_intersection_neck):
                            background_mask &= ~island_mask

                torso_mask_extension = torso_mask_dilated & background_mask
                torso_mask = torso_mask | torso_mask_extension

                # "Absorb" islands of background inside the torso region
                # This addresses the issue that the torso region is sometimes not good and parts of the actual
                # torso will be predicted as background. As such we manually post-process the segmentation mask to
                # mark background islands that are completely surrounded by torso pixels as torso as well
                background_mask = background_mask & ~torso_mask  # Update background mask to excluded grown torso region
                non_torso_ids = [i for i in range(segmentation_mask.max() + 1) if
                                 i not in remove_ids and i not in {0, 1}]
                islands, n_islands = get_sorted_islands(background_mask)
                for i_island in range(2, n_islands + 1):
                    island_mask = islands[i_island]
                    island_mask_dilated = binary_dilation(island_mask, iterations=2)
                    island_mask_boundary = island_mask_dilated & ~background_mask
                    absorb_island = all([~np.any(island_mask_boundary & (segmentation_mask == non_torso_id))
                                         for non_torso_id in non_torso_ids])
                    absorb_island &= ~np.any(
                        island_mask_boundary & background_mask)  # Island should also not touch updated background region
                    if np.any(island_mask_boundary & neck_mask):
                        # Absorb islands touching the neck region if the color of the island is significantly different
                        # from the neck region
                        island_color = np.median(target_image[island_mask], axis=0) / 255.
                        if np.linalg.norm(island_color - neck_color) <= neck_color_threshold:
                            absorb_island = False
                            # But, we can mark the island as "neck" region instead
                            segmentation_mask[island_mask] = 1

                        # absorb_island &= np.linalg.norm(island_color - neck_color) > neck_color_threshold

                    if absorb_island:
                        torso_mask = torso_mask | island_mask

            # Finally, there can still be islands of color if they were connected to the main background region
            # Here, we exploit the fact, that we have a reliable background region and use the alpha map to find
            # this visual islands. Include islands in the torso mask if they touch the torso region
            alpha_map = sequence_data_manager.load_robust_matting_alpha_image(sample_metadata.timestep,
                                                                              sample_metadata.serial)
            foreground_regions = ~(torso_mask | (alpha_map == 0))
            sorted_foreground_islands, n_islands = get_sorted_islands(foreground_regions)
            for i_island in range(2, n_islands + 1):
                island_mask = sorted_foreground_islands[i_island]
                island_mask_dilated = binary_dilation(island_mask, iterations=2)
                island_mask_boundary = island_mask_dilated & ~island_mask
                if np.any(island_mask_boundary & torso_mask):
                    # Only include islands that touch the torso region
                    torso_mask = torso_mask | island_mask

            # Update segmentation mask with improved torso mask
            segmentation_mask[torso_mask] = 3 if is_facer_mask else 16

            if self.remove_torso:
                # Remove torso from image
                target_image[torso_mask] = 255

                if self.smooth_neck_boundary > 0:
                    # Smoothly grow torso into neck
                    smooth_torso_mask = torso_mask.astype(float)
                    smooth_torso_mask = gaussian_filter(smooth_torso_mask, sigma=self.smooth_neck_boundary)
                    neck_mask = segmentation_mask == 1 if is_facer_mask else segmentation_mask == 14
                    min_torso_value = smooth_torso_mask[torso_mask].min()
                    smooth_torso_mask[neck_mask] /= min_torso_value  # normalize smoothed outside region such that it approaches 1 at the boundary to torso
                    # smooth_torso_mask[torso_mask] = 1  # Undo the smoothing inside the torso region
                    smooth_torso_mask[~neck_mask] = 0  # Undo the smoothing outside the neck region
                    smooth_torso_mask = np.clip(smooth_torso_mask, a_min=0, a_max=1)
                    target_image = (((1 - smooth_torso_mask[..., None]) * target_image / 255 + smooth_torso_mask[..., None]) * 255).round().astype(np.uint8)

        return target_image, segmentation_mask, is_facer_mask


@dataclass
class Crop:
    x: int
    y: int
    w: int
    h: int

    def get_point_1(self) -> Tuple[int, int]:
        return self.x, self.y

    def get_point_2(self) -> Tuple[int, int]:
        return self.x + self.w, self.y + self.h

    def scale(self, scale_factor: float):
        self.x = int(scale_factor * self.x)
        self.y = int(scale_factor * self.y)
        self.w = int(scale_factor * self.w)
        self.h = int(scale_factor * self.h)


# ==========================================================
# Samples & Batches
# ==========================================================

@dataclass
class RenderingDatasetInferenceSample:
    renderings: Dict[RenderingName, torch.Tensor]  # [C, H, W]
    previous_outputs: Optional[List[torch.Tensor]] = None  # N x [3, H, W]

    # For autoregressive training w/o teacher forcing
    previous_sample: Optional['RenderingDatasetInferenceSample'] = None
    # Relevant for autoregressive training w/o teacher forcing: This sample is too far in the past, we do not have
    # conditions. Store a flag here so that during the autoregressive loop, we can easily skip it
    is_empty: bool = False

    expression_code: Optional[torch.Tensor] = None

    # needed for nvdiffrast's antialiasing and mipmapping operations
    rast: Optional[torch.Tensor] = None
    uv_da: Optional[torch.Tensor] = None

    # Needed for Deferred Neural Rendering
    view_direction: Optional[torch.Tensor] = None


@dataclass
class RenderingDatasetSample(RenderingDatasetInferenceSample):
    target_image: torch.Tensor = None
    i_participant: int = -1
    mouth_mask: Optional[torch.Tensor] = None
    foreground_mask: Optional[torch.Tensor] = None


@dataclass
class RenderingDatasetInferenceBatch:
    conditions: Dict[RenderingName, torch.Tensor]  # [B, C, H, W]
    neural_textured_rendering: Optional[torch.Tensor] = None  # [B, C, H, W]
    ambient_textured_rendering: Optional[torch.Tensor] = None  # [B, C, H, W]
    prompts: Optional[Union[str, List[str]]] = None  # Optionally, overwrite global prompt used during training
    latent_noise: Optional[torch.Tensor] = None  # [B, C, H_l, W_l] Optionally, specify manual noise

    # Autoregressive Training
    previous_outputs: Optional[torch.Tensor] = None  # [B, N, C, H, W]
    previous_batch: Optional['RenderingDatasetInferenceBatch'] = None  # For autoregressive training w/o teacher forcing

    # For autoregressive training w/o teacher forcing: len(previous_batch) != len(self)
    # since some samples in self might not have any previous timesteps.
    # In this case, previous_batch will only contain the subset for which we actually have conditions.
    # To properly map back to the shape of self, we need to keep track of which of the B samples in self actually
    # have a previous sample.
    # self[previous_sample_ids] <-> previous_batch...
    previous_sample_ids: Optional[torch.Tensor] = None  # [B_p]

    expression_codes: Optional[torch.Tensor] = None  # [B, 100]

    # needed for nvdiffrast's antialiasing and mipmapping operations
    rast: Optional[torch.Tensor] = None
    uv_da: Optional[torch.Tensor] = None

    # Needed for Deferred Neural Rendering
    view_directions: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return list(self.conditions.values())[0].shape[0]

    def __getitem__(self, item) -> 'RenderingDatasetInferenceBatch':
        if isinstance(item, slice):
            item = list(range(len(self))[item])
        elif not isinstance(item, list):
            item = [item]

        neural_textured_rendering = None
        if self.neural_textured_rendering is not None:
            neural_textured_rendering = self.neural_textured_rendering[item]

        ambient_textured_rendering = None
        if self.ambient_textured_rendering is not None:
            ambient_textured_rendering = self.ambient_textured_rendering[item]

        prompt = None
        if self.prompts is not None:
            prompt = [self.prompts[i] for i in item]

        latent_noise = None
        if self.latent_noise is not None:
            latent_noise = self.latent_noise[item]

        previous_outputs = None
        if self.previous_outputs is not None:
            previous_outputs = self.previous_outputs[item]

        previous_batch = None
        previous_sample_ids = None
        if self.previous_batch is not None:
            items_with_previous = [i for i in item if i in self.previous_sample_ids]
            if len(items_with_previous) > 0:
                idx_with_previous = [list(self.previous_sample_ids).index(i) for i in items_with_previous]
                previous_batch = self.previous_batch[idx_with_previous]

                # previous_sample_ids: 0, 1, 4
                # items: 2, 3, 5, 1
                # new previous_sample_ids: 3 (pos 3 in items which is idx 1)
                previous_sample_ids = [item.index(i) for i in items_with_previous]

        expression_codes = None
        if self.expression_codes is not None:
            expression_codes = self.expression_codes[item]

        rast = None
        if self.rast is not None:
            rast = self.rast[item]

        uv_da = None
        if self.uv_da is not None:
            uv_da = self.uv_da[item]

        view_directions = None
        if self.view_directions is not None:
            view_directions = self.view_directions[item]

        return RenderingDatasetInferenceBatch(
            conditions={rendering_name: renderings[item] for rendering_name, renderings in self.conditions.items()},
            neural_textured_rendering=neural_textured_rendering,
            ambient_textured_rendering=ambient_textured_rendering,
            prompts=prompt,
            latent_noise=latent_noise,
            previous_outputs=previous_outputs,
            previous_batch=previous_batch,
            previous_sample_ids=previous_sample_ids,
            expression_codes=expression_codes,
            rast=rast,
            uv_da=uv_da,
            view_directions=view_directions,
        )

    def warp_previous_outputs(self, occlusion_mask_threshold: Optional[float] = None):
        n_previous_outputs = self.previous_outputs.shape[1]
        for t in range(n_previous_outputs):
            image_1_torch = self.previous_outputs[:, t]  # [B, 3, H, W]
            flow_1_to_2_torch = self.conditions[RenderingName.forward_flow(t)].float()  # [B, 2, H, W]
            flow_2_to_1_torch = self.conditions[RenderingName.backward_flow(t)].float()  # [B, 2, H, W]
            mask_2 = self.conditions[RenderingName.MASK][:, 0]  # [B, H, W]

            self.previous_outputs[:, t] = warp_image(
                image_1_torch,
                flow_1_to_2_torch,
                flow_2_to_1_torch,
                mask_2,
                occlusion_mask_threshold=occlusion_mask_threshold
            )

    def add_neural_textures(self,
                            neural_textures: torch.Tensor,
                            neural_texture_fields: Optional[List['tcnn.Encoding']] = None,
                            neural_texture_triplanes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                            neural_texture_hierarchy: Optional[List[torch.Tensor]] = None,
                            use_spherical_harmonics: bool = False):
        uvs = self.conditions[RenderingName.UV].permute(0, 2, 3,
                                                        1).contiguous()  # [B, H, W, 2], nvdiffrast requires contiguous()
        uvs = (uvs + 1) / 2  # uvs should be in [0, 1]
        masks = self.conditions[RenderingName.MASK]

        if neural_texture_fields is not None:
            neural_textured_rendering = torch.zeros((*uvs.shape[:-1], neural_texture_fields[0].n_output_dims),
                                                    dtype=uvs.dtype,
                                                    device=uvs.device)
            for b, (neural_texture_field, mask) in enumerate(zip(neural_texture_fields, masks[:, 0])):
                neural_textured_rendering[b, mask] = neural_texture_field(uvs[b, mask]).to(uvs.dtype)
        elif neural_texture_triplanes is not None:
            uvw_da = None
            if RenderingName.UVW_DA in self.conditions:
                uvw_da = self.conditions[RenderingName.UVW_DA]  # [B, H, W, 4]

            # uvw_da is
            #   du / dx, du / dy,
            #   dv / dx, dv / dy,
            #   dw / dx, dw / dy
            uv_da_triplane = uvw_da[..., [0, 1, 2, 3]].contiguous() if uvw_da is not None else None
            uw_da_triplane = uvw_da[..., [0, 1, 4, 5]].contiguous() if uvw_da is not None else None
            vw_da_triplane = uvw_da[..., [2, 3, 4, 5]].contiguous() if uvw_da is not None else None
            neural_textured_rendering_uv = dr.texture(neural_texture_triplanes[0], uvs[..., [0, 1]].contiguous(), uv_da=uv_da_triplane)
            neural_textured_rendering_uw = dr.texture(neural_texture_triplanes[1], uvs[..., [0, 2]].contiguous(), uv_da=uw_da_triplane)
            neural_textured_rendering_vw = dr.texture(neural_texture_triplanes[2], uvs[..., [1, 2]].contiguous(), uv_da=vw_da_triplane)

            if use_spherical_harmonics:
                self._apply_spherical_harmonics(neural_textured_rendering_uv)
                self._apply_spherical_harmonics(neural_textured_rendering_uw)
                self._apply_spherical_harmonics(neural_textured_rendering_vw)

            neural_textured_rendering = torch.concat([neural_textured_rendering_uv, neural_textured_rendering_uw, neural_textured_rendering_vw], dim=-1)
            neural_textured_rendering[~masks[:, 0]] = 0
        elif neural_texture_hierarchy is not None:
            uv_da = None
            if RenderingName.UV_DA in self.conditions:
                uv_da = self.conditions[RenderingName.UV_DA]  # [B, H, W, 4]
            neural_textured_renderings = [dr.texture(neural_texture, uvs, uv_da=uv_da) for neural_texture in neural_texture_hierarchy]
            neural_textured_rendering = sum(neural_textured_renderings)

            # Spherical Harmonics
            if use_spherical_harmonics:
                self._apply_spherical_harmonics(neural_textured_rendering)
        else:
            uv_da = None
            if RenderingName.UV_DA in self.conditions:
                uv_da = self.conditions[RenderingName.UV_DA]  # [B, H, W, 4]
            neural_textured_rendering = dr.texture(neural_textures, uvs, uv_da=uv_da)

            # Spherical Harmonics
            if use_spherical_harmonics:
                self._apply_spherical_harmonics(neural_textured_rendering)

            # TODO: antialiasing is never used
            neural_textured_rendering[~masks[:, 0]] = 0
        neural_textured_rendering = neural_textured_rendering.permute(0, 3, 1, 2)  # [B, C, H, W]

        self.neural_textured_rendering = neural_textured_rendering

    def _apply_spherical_harmonics(self, neural_textured_rendering: torch.Tensor):
        D = neural_textured_rendering.shape[-1]
        if D >= 12:
            sh_levels = 3
        elif D >= 7:
            sh_levels = 2
        elif D >= 4:
            sh_levels = 1
        else:
            sh_levels = None

        if sh_levels is not None:
            from nerfstudio.field_components.encodings import SHEncoding
            sh_encoding = SHEncoding(sh_levels)
            n_sh_dims = sh_encoding.get_out_dim()
            sh_coefficients = sh_encoding(self.view_directions)
            neural_textured_rendering[..., 3: 3 + n_sh_dims] *= sh_coefficients[:, None, None, :]

    def add_ambient_textures(self, ambient_textures: torch.Tensor):
        uvs = self.conditions[RenderingName.UV_AMBIENT].permute(0, 2, 3, 1).contiguous()  # [B, H, W, 2], nvdiffrast requires contiguous()
        uvs = (uvs + 1) / 2  # uvs should be in [0, 1]
        masks = self.conditions[RenderingName.MASK]

        ambient_textured_rendering = dr.texture(ambient_textures, uvs)
        ambient_textured_rendering[~masks[:, 0]] = 0
        ambient_textured_rendering = ambient_textured_rendering.permute(0, 3, 1, 2)  # [B, C, H, W]

        self.ambient_textured_rendering = ambient_textured_rendering

    def to(self, device: torch.device) -> 'RenderingDatasetInferenceBatch':
        for rendering_name, condition in self.conditions.items():
            self.conditions[rendering_name] = condition.to(device)

        if self.neural_textured_rendering is not None:
            self.neural_textured_rendering = self.neural_textured_rendering.to(device)

        if self.ambient_textured_rendering is not None:
            self.ambient_textured_rendering = self.ambient_textured_rendering.to(device)

        if self.latent_noise is not None:
            self.latent_noise = self.latent_noise.to(device)

        if self.previous_outputs is not None:
            self.previous_outputs = self.previous_outputs.to(device)

        if self.previous_batch is not None:
            self.previous_batch = self.previous_batch.to(device)

        if self.expression_codes is not None:
            self.expression_codes = self.expression_codes.to(device)

        if self.rast is not None:
            self.rast = self.rast.to(device)

        if self.uv_da is not None:
            self.uv_da = self.uv_da.to(device)

        if self.view_directions is not None:
            self.view_directions = self.view_directions.to(device)

        return self

    def get_previous_batch(self, previous_timestep: int) -> 'RenderingDatasetInferenceBatch':
        """
        Parameters
        ----------
            previous_timestep: How many steps to follow the "chain" into the past
        """

        batch = self
        for _ in range(previous_timestep):
            batch = self.previous_batch

        return batch


@dataclass
class RenderingDatasetBatch(RenderingDatasetInferenceBatch):
    target_images: torch.Tensor = None  # [B, 3, H, W]
    i_participant: torch.Tensor = None  # [B,]
    mouth_masks: torch.Tensor = None  # [B, H, W]
    foreground_masks: torch.Tensor = None  # [B, H, W]

    def __getitem__(self, item) -> 'RenderingDatasetBatch':
        return RenderingDatasetBatch(
            **to_shallow_dict(super(RenderingDatasetBatch, self).__getitem__(item)),
            target_images=self.target_images[item],
            i_participant=self.i_participant[item],
            mouth_masks=self.mouth_masks[item] if self.mouth_masks is not None else None,
            foreground_masks=self.foreground_masks[item] if self.foreground_masks is not None else None,
        )

    def to(self, device: torch.device) -> 'RenderingDatasetBatch':
        super(RenderingDatasetBatch, self).to(device)

        self.target_images = self.target_images.to(device)
        if self.i_participant is not None:
            self.i_participant = self.i_participant.to(device)

        if self.mouth_masks is not None:
            self.mouth_masks = self.mouth_masks.to(device)

        if self.foreground_masks is not None:
            self.foreground_masks = self.foreground_masks.to(device)

        return self


# ==========================================================
# Dataset class
# ==========================================================

class RenderingDataset(Dataset):
    """
    Usage:
    batch = next(iter(dataset))
    batch.add_neural_textures(neural_textures)
    conditions = dataset.concat_conditions(batch)
    """

    def __init__(self, config: RenderingDatasetConfig, split: DatasetSplit = DatasetSplit.TRAIN):
        assert split in config.supported_splits

        data_manager = RenderingDataFolder().open_dataset(config.dataset_version)

        self._config = config
        self._data_config: RenderingDataConfig = data_manager.load_config()
        self._data_stats = data_manager.load_stats()
        self._data_manager = data_manager

        available_data = self._data_stats.available_sequences  # { "p_id": {seq_1: [0,3,6], seq_2: [10,13,16]}}
        valid_split_ratio = 1 - self._config.split_ratio
        valid_rng = random.Random(config.seed)

        # hold out persons
        available_participants = sorted(list(available_data.keys()))
        hold_out_participants = []
        if DatasetSplit.VALID_HOLD_OUT_PERSON in self._config.supported_splits:
            n_valid_persons = round(len(available_participants) * valid_split_ratio)
            hold_out_participants = valid_rng.sample(available_participants, n_valid_persons)
        selected_participants = [p for p in available_participants if p not in hold_out_participants]

        # hold out sequences
        available_sequences = [(participant, sequence)
                               for participant in selected_participants
                               for sequence in available_data[participant].keys()
                               if config.exclude_sequences is None or sequence not in config.exclude_sequences]
        hold_out_sequences = []
        if DatasetSplit.VALID_HOLD_OUT_SEQ in self._config.supported_splits:
            if self._config.use_predefined_test_split:
                participant_ids = set([participant for participant, _ in available_sequences])
                for participant_id in participant_ids:
                    test_sequences = TEST_SEQUENCES[int(participant_id)]
                    for test_sequence in test_sequences:
                        assert (participant_id, test_sequence) in available_sequences, \
                            f"Expected {participant_id} - {test_sequence} to be available, but could not find in specified dataset {config.dataset_version}"
                        hold_out_sequences.append((participant_id, test_sequence))
            elif self._config.hold_out_sequences is not None:
                participant_ids = set([participant for participant, _ in available_sequences])
                for participant_id in participant_ids:
                    for test_sequence in self._config.hold_out_sequences:
                        if (participant_id, test_sequence) in available_sequences:
                            hold_out_sequences.append((participant_id, test_sequence))
            else:
                n_valid_seq = round(len(available_sequences) * valid_split_ratio)
                hold_out_sequences = valid_rng.sample(available_sequences, n_valid_seq)
        selected_sequences = [s for s in available_sequences if s not in hold_out_sequences]

        # hold out expressions
        available_expressions = [(participant, sequence, timestep)
                                 for participant, sequence in selected_sequences
                                 for timestep in available_data[participant][sequence]]
        hold_out_expressions = []
        if DatasetSplit.VALID_HOLD_OUT_EXP in self._config.supported_splits:
            n_valid_exp = round(len(available_expressions) * valid_split_ratio)
            hold_out_expressions = valid_rng.sample(available_expressions, n_valid_exp)
        selected_expressions = [e for e in available_expressions if e not in hold_out_expressions]

        # hold out views
        available_views = [(participant, sequence, timestep, serial)
                           for participant, sequence, timestep in selected_expressions
                           for serial in self._data_config.serials]
        hold_out_views = []
        if DatasetSplit.VALID_HOLD_OUT_VIEW in self._config.supported_splits:
            if self._config.hold_out_cameras is not None:
                hold_out_views = [(participant, sequence, timestep, serial)
                                  for participant, sequence, timestep, serial
                                  in available_views
                                  if serial in self._config.hold_out_cameras]
            else:
                n_valid_views = round(len(available_views) * valid_split_ratio)
                hold_out_views = valid_rng.sample(available_views, n_valid_views)
        selected_views = [v for v in available_views if v not in hold_out_views]

        if split == DatasetSplit.TRAIN:
            # selected views already contains the train views
            pass
        elif split == DatasetSplit.VALID_HOLD_OUT_VIEW:
            selected_views = hold_out_views
        elif split == DatasetSplit.VALID_HOLD_OUT_EXP:
            # Use all views for hold_out_expressions
            selected_views = [(participant, sequence, timestep, serial)
                              for participant, sequence, timestep in hold_out_expressions
                              for serial in self._data_config.serials]
        elif split == DatasetSplit.VALID_HOLD_OUT_SEQ:
            # Use all timesteps/views for hold_out_sequences
            selected_views = [(participant, sequence, timestep, serial)
                              for participant, sequence in hold_out_sequences
                              for timestep in available_data[participant][sequence]
                              for serial in self._data_config.serials]
        elif split == DatasetSplit.VALID_HOLD_OUT_PERSON:
            # Use all sequence/timesteps/views for hold_out_participants
            selected_views = [(participant, sequence, timestep, serial)
                              for participant in hold_out_participants
                              for sequence in available_data[participant].keys()
                              for timestep in available_data[participant][sequence]
                              for serial in self._data_config.serials]
        else:
            raise NotImplementedError()

        # Specify mapping sample_idx -> (participant_id, sequence, timestep, serial)
        samples_mapping = []

        # Specify mapping (participant_id, sequence, timestep, serial) -> sample_idx
        sample_idx_mapping = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        self.participant_ids = set()
        for participant_id, sequence_name, timestep, serial in selected_views:
            participant_id = int(participant_id)

            if config.n_participants < 0 \
                    or config.n_participants <= len(self.participant_ids) \
                    or participant_id in self.participant_ids:
                self.participant_ids.add(participant_id)

                sample_idx = len(samples_mapping)
                samples_mapping.append(SampleMetadata(participant_id, sequence_name, timestep, serial))
                sample_idx_mapping[participant_id][sequence_name][timestep][serial] = sample_idx

        # for (participant_id, sequence_name), timesteps in zip(self._data_config.sequences,
        #                                                       self._data_stats.available_timesteps):
        #     self.participant_ids.add(participant_id)
        #     for timestep in timesteps:
        #         for serial in SERIALS:
        #             sample_idx = len(samples_mapping)
        #             samples_mapping.append(SampleMetadata(participant_id, sequence_name, timestep, serial))
        #             sample_idx_mapping[participant_id][sequence_name][timestep][serial] = sample_idx
        #
        #     if 0 < config.n_participants <= len(self.participant_ids):
        #         break
        self.participant_ids = sorted(list(self.participant_ids))

        self.samples_mapping = samples_mapping

        def ddict2dict(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = ddict2dict(v)
            return dict(d)

        self.sample_idx_mapping = default_dict_to_dict(sample_idx_mapping)

        for sample_idx, sample_metadata in enumerate(self.samples_mapping):
            assert self.sample_idx_mapping[sample_metadata.participant_id][sample_metadata.sequence_name][
                       sample_metadata.timestep][sample_metadata.serial] == sample_idx

        # if config.n_participants == -1:
        #     # Ensure, we collected metadata for all frames
        #     assert len(self.samples_mapping) == self._data_stats.total_n_frames

        # Used for Optical Flow renderings
        self._renderer = None

        # Expression codes are stored per sequence, but loaded per timestep. Avoid loading the full sequence every
        # time we only need a single timestep
        self._expression_code_cache = defaultdict(dict)  # p_id -> sequence

        calibration_config = NeRSembleParticipantDataManager(self.participant_ids[0]).load_calibration_run_config()
        self._image_size = calibration_config.image_size

    def get_n_cond_channels(self) -> int:
        n_channels = 0
        for rendering_name in self._config.rendering_names:
            n_channels += rendering_name.get_n_channels(
                use_canonical_coordinates=self._config.use_canonical_coordinates)

        if self._config.use_neural_textures:
            if self._config.use_texture_fields:
                n_channels += self._config.texture_field_config.n_levels * self._config.texture_field_config.n_features_per_level
            elif self._config.use_texture_triplanes:
                n_channels += 3 * self._config.dim_neural_textures
            else:
                n_channels += self._config.dim_neural_textures

        if self._config.use_ambient_textures:
            n_channels += self._config.dim_neural_textures

        if self._config.use_autoregressive:
            n_channels += 3 * self._config.n_previous_frames  # n previous RGB frames will be additional conditions

        if self._config.use_freq_encoding_uv:
            n_channels += 2 * self._config.n_frequencies_uv * RenderingName.UV.get_n_channels(
                use_canonical_coordinates=self._config.use_canonical_coordinates)

        if self._config.use_freq_encoding_depth:
            n_channels += 2 * self._config.n_frequencies_depth * RenderingName.DEPTHS.get_n_channels(
                use_canonical_coordinates=self._config.use_canonical_coordinates)

        return n_channels

    def get_random_crop(self, sample_metadata: SampleMetadata) -> Crop:
        sequence_data_manager = NeRSembleSequenceDataManager(sample_metadata.participant_id, sample_metadata.sequence_name)
        # TODO: Caching of bounding boxes?
        bounding_boxes = sequence_data_manager.load_bounding_boxes(sample_metadata.serial)
        timestep_to_i_timestep = {timestep: i for i, timestep in enumerate(sequence_data_manager.get_timesteps())}
        i_timestep = timestep_to_i_timestep[sample_metadata.timestep]
        bounding_box = bounding_boxes[i_timestep]

        image_w = self._image_size.w / self._data_config.downscale_factor
        image_h = self._image_size.h / self._data_config.downscale_factor
        center_x = int(image_w * (bounding_box.x + bounding_box.width / 2))  # [0, 1100]
        center_y = int(image_h * (bounding_box.y + bounding_box.height / 2))  # [0, 1604]

        # Find the 1100x1100 square which contains the face
        # crop_y_min = int(round(center_y - image_w / 2))
        # crop_y_max = int(round(center_y + image_w / 2))
        # NB: Python's round() does round-to-even! Regular round can be implemented by int(x + 0.5)
        crop_y_min = int(center_y - image_w / 2 + 0.5)
        crop_y_max = int(center_y + image_w / 2 + 0.5)

        # In principle, the bbox preselection does not have to be a square. Faces are typically higher than wide
        crop_y_min -= int((self._config.crop_selection_aspect_ratio - 1) * image_w / 2)
        crop_y_max += int((self._config.crop_selection_aspect_ratio - 1) * image_w / 2)

        if crop_y_min < 0:
            overshoot_top = max(-crop_y_min, 0)  # If crop_y_min is negative, we have to move the selection down
            crop_y_min = crop_y_min + overshoot_top
            crop_y_max = crop_y_max + overshoot_top
        elif crop_y_max >= image_h:
            overshoot_bottom = max(crop_y_max - image_h, 0)  # If crop_y_max > image_h, we have to move the selection up
            crop_y_min = crop_y_min - overshoot_bottom
            crop_y_max = crop_y_max - overshoot_bottom

        assert crop_y_min >= 0
        assert crop_y_max <= image_h

        min_crop_size = int(self._config.min_crop_size * image_w)
        crop_size = random.randint(min_crop_size, image_w)  # random choice in [min_crop_size, image_w]
        crop_x = random.randint(0, image_w - crop_size)
        assert crop_y_min <= crop_y_max - crop_size
        crop_y = random.randint(crop_y_min, crop_y_max - crop_size)

        return Crop(crop_x, crop_y, crop_size, crop_size)

    def _crop_and_resize(self,
                         img: np.ndarray,
                         crop: Optional[Crop] = None,
                         return_scale_factors: bool = False,
                         use_interpolation: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dimensions]]:

        if crop is not None:
            img = img[crop.y: crop.y + crop.h, crop.x: crop.x + crop.w]

        was_bool = False
        if img.dtype == bool:
            was_bool = True
            img = img.astype(np.uint8)
            use_interpolation = False

        new_size = (self._config.resolution, self._config.resolution)
        downscale_factors = Dimensions(new_size[0] / img.shape[1], new_size[1] / img.shape[0])

        if use_interpolation:
            img = cv2.resize(img, (self._config.resolution, self._config.resolution),
                             interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            img = cv2.resize(img, (self._config.resolution, self._config.resolution),
                             interpolation=cv2.INTER_NEAREST_EXACT)

        if was_bool:
            img = img.astype(bool)

        if return_scale_factors:
            return img, downscale_factors

        return img

    def __len__(self):
        return len(self.samples_mapping)

    def __getitem__(self, index) -> RenderingDatasetSample:
        return self.load_sample(index)

    def load_sample(self, index: int, n_previous_frames: Optional[int] = None,
                    crop: Optional[Crop] = None) -> RenderingDatasetSample:
        assert not (self._config.use_autoregressive and self._config.temporal_batch_size > 0)

        if n_previous_frames is None:
            if self._config.use_autoregressive:
                n_previous_frames = self._config.n_previous_frames
            elif self._config.temporal_batch_size > 0:
                n_previous_frames = self._config.temporal_batch_size - 1
            else:
                n_previous_frames = 0

        return self._load_sample(index, n_previous_frames=n_previous_frames, crop=crop)

    def _load_sample(self, index: int,
                     n_previous_frames: int = 0,
                     crop: Optional[Crop] = None,
                     crop_end: Optional[Crop] = None) -> RenderingDatasetSample:
        sample_metadata = self.samples_mapping[index]

        if crop is None:
            if self._config.use_random_cropping:
                crop = self.get_random_crop(sample_metadata)
            else:
                crop = Crop(0, 0, self._image_size.w, self._image_size.w)

        renderings = dict()
        for rendering_name in self._config.rendering_names:
            if rendering_name not in {RenderingName.FORWARD_FLOW, RenderingName.BACKWARD_FLOW}:
                rendering = self._data_manager.load_rendering(
                    rendering_name,
                    sample_metadata.participant_id,
                    sample_metadata.sequence_name,
                    sample_metadata.timestep,
                    sample_metadata.serial
                )
                renderings[rendering_name] = rendering

        # Neural Textures
        if self._config.use_neural_textures or self._config.remap_noise:
            # Temporally add UV and mask if they are not already there
            if RenderingName.UV not in renderings:
                renderings[RenderingName.UV] = self._data_manager.load_uv_rendering(
                    sample_metadata.participant_id,
                    sample_metadata.sequence_name,
                    sample_metadata.timestep,
                    sample_metadata.serial)

            if self._data_config.use_mipmapping and self._config.use_mipmapping:
                if RenderingName.UVW_DA not in renderings and self._config.use_texture_triplanes:
                    renderings[RenderingName.UVW_DA] = self._data_manager.load_uvw_da(
                        sample_metadata.participant_id,
                        sample_metadata.sequence_name,
                        sample_metadata.timestep,
                        sample_metadata.serial)
                elif RenderingName.UV_DA not in renderings:
                    renderings[RenderingName.UV_DA] = self._data_manager.load_uv_da(
                        sample_metadata.participant_id,
                        sample_metadata.sequence_name,
                        sample_metadata.timestep,
                        sample_metadata.serial)

        # Ambient Textures
        if self._config.use_ambient_textures and RenderingName.UV_AMBIENT not in renderings:
            renderings[RenderingName.UV_AMBIENT] = self._data_manager.load_uv_ambient_rendering(
                sample_metadata.participant_id,
                sample_metadata.sequence_name,
                sample_metadata.timestep,
                sample_metadata.serial)

        if self._config.use_neural_textures or self._config.use_background_texture or self._config.use_autoregressive:
            if RenderingName.MASK not in renderings:
                renderings[RenderingName.MASK] = self._data_manager.load_mask(
                    sample_metadata.participant_id,
                    sample_metadata.sequence_name,
                    sample_metadata.timestep,
                    sample_metadata.serial)

        renderings = self.normalize_renderings(renderings)
        inference_sample = self.process_conditions(renderings, crop)

        # Target image
        target_image, mouth_mask, foreground_mask = self._load_target_image(sample_metadata, crop)

        # Previous frames for autoregressive training/inference
        self._load_autoregressive_sample(inference_sample, sample_metadata, crop, n_previous_frames=n_previous_frames,
                                         crop_end=crop_end)

        i_participant = self.participant_ids.index(sample_metadata.participant_id)

        # Expression code
        inference_sample.expression_code = self._load_expression_code(sample_metadata)

        # View direction
        if self._config.include_view_directions:
            participant_data_manager = FamudyParticipantDataManager(sample_metadata.participant_id)
            calibration_result = participant_data_manager.load_calibration_result().params_result
            world_2_cam_pose = calibration_result.get_pose(SERIALS.index(sample_metadata.serial))
            view_direction = world_2_cam_pose.invert().get_look_direction()
            view_direction = torch.from_numpy(view_direction)
            inference_sample.view_direction = view_direction

        sample = RenderingDatasetSample(
            **to_shallow_dict(inference_sample),
            target_image=target_image,
            i_participant=i_participant,
            mouth_mask=mouth_mask,
            foreground_mask=foreground_mask,
        )

        return sample

    def _load_target_image(self, sample_metadata: SampleMetadata,
                           crop: Optional[Crop] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        target_image, segmentation_mask, is_facer_mask = self._config.load_target_image(sample_metadata)

        mouth_mask = None
        if self._config.include_mouth_mask:
            segmentation_mask = self._crop_and_resize(segmentation_mask, crop)
            segmentation_mask = torch.from_numpy(segmentation_mask)  # [H, W]

            if is_facer_mask:
                mouth_mask = (segmentation_mask == 11) | (segmentation_mask == 12) | (segmentation_mask == 13)
            else:
                # Bisenet
                mouth_mask = (segmentation_mask == 11) | (segmentation_mask == 12) | (segmentation_mask == 13)

        foreground_mask = None
        if self._config.include_foreground_mask:
            sequence_data_manager = NeRSembleSequenceDataManager(sample_metadata.participant_id, sample_metadata.sequence_name)
            alpha_map = sequence_data_manager.load_robust_matting_alpha_image(sample_metadata.timestep,
                                                                              sample_metadata.serial)
            alpha_map = alpha_map.copy()
            if self._config.remove_torso or self._config.remove_torso_in_mask:
                torso_mask = segmentation_mask == 3 if is_facer_mask else segmentation_mask == 16
                alpha_map[torso_mask] = 0
            if self._config.remove_neck_in_mask:
                neck_mask = segmentation_mask == 1 if is_facer_mask else segmentation_mask == 14
                alpha_map[neck_mask] = 0
            if self._config.remove_background_in_mask:
                background_mask = segmentation_mask == 0
                alpha_map[background_mask] = 0
            foreground_mask = self._crop_and_resize(alpha_map, crop, use_interpolation=self._config.interpolate_target_images)
            foreground_mask = torch.from_numpy(foreground_mask)  # [H, W]

        target_image = self._crop_and_resize(target_image, crop, use_interpolation=self._config.interpolate_target_images)
        target_image = target_image.astype(np.float32) / 255.  # [0-255] -> [0-1]
        if self._config.normalize:
            target_image = target_image * 2 - 1
        target_image = torch.from_numpy(target_image).permute(2, 0, 1)  # [3, H, W]

        return target_image, mouth_mask, foreground_mask

    def _load_autoregressive_sample(self,
                                    inference_sample: RenderingDatasetInferenceSample,
                                    sample_metadata: SampleMetadata,
                                    crop: Crop,
                                    n_previous_frames: int = 0,
                                    crop_end: Optional[Crop] = None,
                                    n_timesteps_skip: int = 1):

        if self._config.use_crop_sweep:
            start_crop = crop
            end_crop = self.get_random_crop(sample_metadata) if crop_end is None else crop_end

            crops = []
            for i in range(n_previous_frames):
                alpha = (i + 1) / n_previous_frames
                x = int((1 - alpha) * start_crop.x + alpha * end_crop.x)
                y = int((1 - alpha) * start_crop.y + alpha * end_crop.y)
                w = int((1 - alpha) * start_crop.w + alpha * end_crop.w)
                h = int((1 - alpha) * start_crop.h + alpha * end_crop.h)
                interpolated_crop = Crop(x, y, w, h)
                crops.append(interpolated_crop)
        else:
            crops = [crop for _ in range(n_previous_frames)]

        if self._config.use_autoregressive or self._config.temporal_batch_size > 0:
            if self._config.use_teacher_forcing and self._config.temporal_batch_size == 0:
                previous_outputs = []

                # if self._config.prob_autoregressive_dropout > 0 and np.random.random() < self._config.prob_autoregressive_dropout:
                #     # This sample was randomly chosen to be the start of the sequence.
                #     # Hence, previous_outputs is all None
                #     previous_outputs = [None for _ in range(self._config.n_previous_frames)]
                #
                #     if self._config.warp_previous_outputs:
                #         for rendering_name in {RenderingName.FORWARD_FLOW, RenderingName.BACKWARD_FLOW}:
                #             # Just use 0s for Optical flow because we don't have a previous output
                #             inference_sample.renderings[rendering_name] = \
                #                 torch.zeros(2, self._config.resolution, self._config.resolution)
                # else:
                n_previous_frames = self._config.n_previous_frames if n_previous_frames is None else n_previous_frames
                if self._config.prob_autoregressive_dropout > 0 and np.random.random() < self._config.prob_autoregressive_dropout:
                    n_previous_frames_before_dropout = np.random.randint(0, n_previous_frames)
                else:
                    n_previous_frames_before_dropout = n_previous_frames

                for previous_timestep in range(n_previous_frames_before_dropout):
                    # NB: We only processed every 3rd timestep (24.3 fps)!
                    # Hence, we need to multiply the timestep delta by 3 here
                    timestep_delta = 3 * n_timesteps_skip * (-previous_timestep - 1)

                    timestep = sample_metadata.timestep + timestep_delta
                    if self._config.use_random_previous_viewpoints:
                        # Find random serial among available cameras for previous frame
                        # This simulates viewpoint changes during training and hopefully helps with the issue
                        #  that teacher forcing causes the model to ignore all other conditions except previous img
                        available_timesteps = self.sample_idx_mapping[sample_metadata.participant_id][
                            sample_metadata.sequence_name]
                        if timestep in available_timesteps:
                            available_serials = available_timesteps[timestep]
                            available_serials = list(available_serials.keys())
                            serial = random.choice(available_serials)
                        else:
                            # Just use any serial, _load_target_image() will return None anyways
                            serial = "222200037"
                    else:
                        # Otherwise, previous frames should come from same camera
                        serial = sample_metadata.serial

                    previous_sample_metadata = SampleMetadata(
                        sample_metadata.participant_id,
                        sample_metadata.sequence_name,
                        timestep,
                        serial)

                    crop_previous = crops[previous_timestep]
                    if self._config.use_random_previous_viewpoints:
                        # crop_previous = self.get_random_crop(previous_sample_metadata)
                        # TODO: For this to work, we would have to adapt the
                        #  self.process_sample() method with a previous_crop attribute
                        crop_previous = crop_previous
                    else:
                        crop_previous = crop_previous

                    if timestep in self.sample_idx_mapping[sample_metadata.participant_id][
                        sample_metadata.sequence_name]:
                        previous_outputs.append(self._load_target_image(previous_sample_metadata, crop_previous)[0])
                    else:
                        previous_outputs.append(None)

                    if self._config.warp_previous_outputs \
                            or RenderingName.BACKWARD_FLOW in self._config.rendering_names \
                            or RenderingName.FORWARD_FLOW in self._config.rendering_names:
                        if self._renderer is None:
                            # Use None here, to just initialize the renderer and swap the mesh provider later
                            # TODO: This is sub-optimal. The renderer only needs the mesh provider for obtaining
                            #   the uv maps. Hence, here we can provide a single mesh provider for all
                            #   kind of samples from multiple people
                            #   It would be better if the mesh_provider just created meshes with a "uv"
                            #   vertex attribute. Then we could completely decouple mesh provider from renderer

                            mesh_provider = get_mesh_provider(self._data_config,
                                                              sample_metadata.participant_id,
                                                              sample_metadata.sequence_name,
                                                              use_subdivision=self._config.use_nphm_subdivision)
                            self._renderer = get_renderer(self._data_config, mesh_provider)
                        else:
                            mesh_provider = self._renderer.mesh_provider

                        if not mesh_provider.has_mesh(previous_sample_metadata.timestep) \
                                or not mesh_provider.has_mesh(sample_metadata.timestep):
                            for rendering_name in {RenderingName.FORWARD_FLOW, RenderingName.BACKWARD_FLOW}:
                                # Just use 0s for Optical flow because we don't have a mesh
                                inference_sample.renderings[rendering_name] = \
                                    torch.zeros(2, self._config.resolution, self._config.resolution)

                            inference_sample.renderings[
                                RenderingName.forward_flow(previous_timestep)] = \
                                torch.zeros(2, self._config.resolution, self._config.resolution)
                            inference_sample.renderings[
                                RenderingName.backward_flow(previous_timestep)] = \
                                torch.zeros(2, self._config.resolution, self._config.resolution)
                        else:
                            mesh_1 = mesh_provider.get_mesh(previous_sample_metadata.timestep)
                            mesh_2 = mesh_provider.get_mesh(sample_metadata.timestep)

                            data_manager = NeRSembleSequenceDataManager(sample_metadata.participant_id,
                                                                     sample_metadata.sequence_name)
                            calibration_result = data_manager.load_calibration_result().params_result
                            cam_id_1 = data_manager.serial_to_cam_id(previous_sample_metadata.serial)
                            cam_id_2 = data_manager.serial_to_cam_id(sample_metadata.serial)
                            pose_1 = calibration_result.get_pose(cam_id_1)
                            pose_2 = calibration_result.get_pose(cam_id_2)
                            intrinsics = calibration_result.get_intrinsics()
                            intrinsics = prepare_intrinsics(self._data_config, intrinsics)
                            mesh_provider.prepare_optical_flow(mesh_1, mesh_2, pose_1, pose_2, intrinsics)

                            flow_renderings = dict()
                            renderings_1 = self._renderer.render(mesh_1, pose_1, intrinsics,
                                                                 [RenderingName.FORWARD_FLOW])
                            flow_renderings.update(renderings_1)

                            renderings_2 = self._renderer.render(mesh_2, pose_2, intrinsics,
                                                                 [RenderingName.BACKWARD_FLOW])

                            flow_renderings.update(renderings_2)

                            # flow_renderings = self.normalize_renderings(flow_renderings)
                            flow_renderings_sample = self.process_conditions(flow_renderings,
                                                                             crop=crop_previous,
                                                                             crop_context=crop)

                            if previous_timestep == 0:
                                # TODO: When multiple previous frames are used, we can only return the
                                #   optical flow for one of them
                                #   Ideally, we would also create previous_sample attributes for the
                                #   autoregressive case. Then optical flow for all previous frames could be
                                #   returned
                                inference_sample.renderings[RenderingName.FORWARD_FLOW] = \
                                    flow_renderings_sample.renderings[RenderingName.FORWARD_FLOW]
                                inference_sample.renderings[RenderingName.BACKWARD_FLOW] = \
                                    flow_renderings_sample.renderings[RenderingName.BACKWARD_FLOW]

                            inference_sample.renderings[
                                RenderingName.forward_flow(previous_timestep)] = flow_renderings_sample.renderings[
                                RenderingName.FORWARD_FLOW]
                            inference_sample.renderings[
                                RenderingName.backward_flow(previous_timestep)] = flow_renderings_sample.renderings[
                                RenderingName.BACKWARD_FLOW]

                            if self._config.warp_previous_outputs:
                                warped_previous_output = warp_image(
                                    previous_outputs[previous_timestep],
                                    flow_renderings_sample.renderings[RenderingName.FORWARD_FLOW],
                                    flow_renderings_sample.renderings[RenderingName.BACKWARD_FLOW],
                                    inference_sample.renderings[RenderingName.MASK],
                                    occlusion_mask_threshold=self._config.occlusion_mask_threshold)  # None if self._config.occlusion_mask_threshold is None else (self._config.occlusion_mask_threshold + abs(crop.w - crop_previous.w) / 15))
                                previous_outputs[previous_timestep] = warped_previous_output[0]

                for t in range(n_previous_frames - n_previous_frames_before_dropout):
                    # This sample was randomly chosen to be the start of the sequence.
                    # Hence, previous_outputs is all None
                    previous_outputs.append(None)

                    if self._config.warp_previous_outputs:
                        for rendering_name in {RenderingName.FORWARD_FLOW, RenderingName.BACKWARD_FLOW}:
                            if rendering_name not in inference_sample.renderings:
                                # Just use 0s for Optical flow because we don't have a previous output
                                inference_sample.renderings[rendering_name] = \
                                    torch.zeros(2, self._config.resolution, self._config.resolution)

                        inference_sample.renderings[RenderingName.forward_flow(n_previous_frames_before_dropout + t)] = \
                            torch.zeros(2, self._config.resolution, self._config.resolution)
                        inference_sample.renderings[RenderingName.backward_flow(n_previous_frames_before_dropout + t)] = \
                            torch.zeros(2, self._config.resolution, self._config.resolution)

                inference_sample.previous_outputs = previous_outputs

            else:
                if n_previous_frames > 0:
                    # Have to load sample from this camera at previous timestep as well
                    # _load_sample() is called recursively here.
                    # Hence, n_previous_frames is passed to keep track of the recursion depth
                    # Once, it reaches 0, it means we have loaded enough previous frames and do not need to continue
                    # NB: We only processed every 3rd timestep (24.3 fps)!
                    # Hence, we the previous timestep will have -3 time delta
                    previous_timestep = sample_metadata.timestep - 3
                    available_timesteps = self.sample_idx_mapping[sample_metadata.participant_id][
                        sample_metadata.sequence_name]
                    if previous_timestep in available_timesteps \
                            and sample_metadata.serial in available_timesteps[previous_timestep]:
                        # The frame from the correct camera at previous timestep indeed exists
                        if self._config.use_temporal_batch_random_viewpoints:
                            available_serials = list(available_timesteps[previous_timestep].keys())
                            serial = random.choice(available_serials)
                            previous_sample_idx = available_timesteps[previous_timestep][serial]
                        else:
                            previous_sample_idx = available_timesteps[previous_timestep][sample_metadata.serial]

                        previous_crop = crops[0]  # "Pop" the first crop
                        previous_sample = self._load_sample(previous_sample_idx,
                                                            n_previous_frames=n_previous_frames - 1,
                                                            crop=previous_crop,
                                                            crop_end=crop_end)  # TODO: Using crop here will be wrong if we use autoregressive without teacher forcing but with random_viewpoint_changes

                        inference_sample.previous_sample = previous_sample

                # previous outputs is still used because the model still expects these 3 RGB channels
                # However, without teacher forcing, we cannot know the previous output yet because we did not
                # generate it yet (only the model itself can do it). Hence, we set it to None for now.
                # Later, the training_step() of the model will sequentially set the previous outputs
                # (expect for the first inference in the chain, there it will be all 0s)
                # TODO: Should we use teacher forcing for the first input in the chain?
                previous_outputs = [None for _ in range(self._config.n_previous_frames)]
                inference_sample.previous_outputs = previous_outputs

    def _load_expression_code(self, sample_metadata: SampleMetadata) -> Optional[torch.Tensor]:
        if not self._config.use_expression_condition:
            return None
        else:
            participant_id = sample_metadata.participant_id
            sequence_name = sample_metadata.sequence_name
            if not participant_id in self._expression_code_cache \
                    or not sequence_name in self._expression_code_cache[participant_id]:

                sequence_data_manager = NeRSembleSequenceDataManager(
                    participant_id,
                    sequence_name,
                    downscale_factor=self._data_config.downscale_factor)
                if self._data_config.use_nphm and not self._config.force_flame_expression_condition:
                    timesteps = sequence_data_manager.get_timesteps()
                    nphm_version = self._data_config.mesh_provider_config.nphm_tracking.get_version()
                    all_expr_params = [sequence_data_manager.load_NPHM_expression_code(timestep, nphm_version)
                                       for timestep in timesteps]
                    all_expr_params = np.stack(all_expr_params)  # [T, 200]

                    if self._config.include_eye_condition:
                        # Concat eye condition to other expression codes
                        tracker_name = self._data_config.mesh_provider_config.flame_tracking.get_version()
                        flame_params = sequence_data_manager.load_3DMM_tracking(tracker_name)
                        eye_params = flame_params['eyes']  # [T, 6]
                        all_expr_params = np.concatenate([all_expr_params, eye_params], axis=1)
                else:
                    tracker_name = self._data_config.mesh_provider_config.flame_tracking.get_version()
                    flame_params = sequence_data_manager.load_3DMM_tracking(tracker_name)
                    all_expr_params = flame_params['expression']  # [T, 100]

                # Potentially, smooth 3DMM expression params
                if self._config.expression_condition_smoothing is None:
                    # If nothing specific was specified, we use the same smoothing as for generate the renderings
                    temporal_smoothing_config = self._data_config.mesh_provider_config.temporal_smoothing_config
                else:
                    temporal_smoothing_config = self._config.expression_condition_smoothing

                if 'expression' in temporal_smoothing_config.param_groups:
                    all_expr_params = temporal_smoothing_config.smooth(all_expr_params)

                all_expr_params = {timestep: all_expr_params[i]
                                   for i, timestep in enumerate(sequence_data_manager.get_timesteps())}
                self._expression_code_cache[participant_id][sequence_name] = all_expr_params

            expression_code = self._expression_code_cache[participant_id][sequence_name][sample_metadata.timestep]
            expression_code = torch.from_numpy(expression_code)

            return expression_code

    def collate_fn_inference(self,
                             samples: List[RenderingDatasetInferenceSample],
                             n_previous_frames: Optional[int] = None) -> RenderingDatasetInferenceBatch:
        batched_input_condition = dict()
        rendering_names = samples[0].renderings.keys()  # Assume that all samples have the same renderings
        for rendering_name in rendering_names:
            concatenated_renderings = torch.stack([sample.renderings[rendering_name] for sample in samples], dim=0)
            batched_input_condition[rendering_name] = concatenated_renderings

        batched_previous_outputs = None
        previous_batch = None
        previous_sample_ids = None
        if self._config.use_autoregressive:
            # Batch previous outputs
            batched_previous_outputs = []
            for sample in samples:
                previous_outputs = []
                for previous_output in sample.previous_outputs:
                    if previous_output is None:
                        previous_output = torch.zeros((3, self._config.resolution, self._config.resolution))
                    previous_outputs.append(previous_output)
                batched_previous_outputs.append(torch.stack(previous_outputs))  # [N, 3, H, W]
            batched_previous_outputs = torch.stack(batched_previous_outputs)  # [B, N, 3, H, W]

        # Batch previous samples
        if self._config.use_autoregressive and not self._config.use_teacher_forcing \
                or self._config.temporal_batch_size > 0:
            if n_previous_frames is None:
                if self._config.use_autoregressive:
                    n_previous_frames = self._config.n_previous_frames
                elif self._config.temporal_batch_size > 0:
                    n_previous_frames = self._config.temporal_batch_size - 1

            if n_previous_frames > 0:
                previous_samples = []
                previous_sample_ids = []
                for i_sample, sample in enumerate(samples):
                    if sample.previous_sample is not None:
                        previous_samples.append(sample.previous_sample)
                        previous_sample_ids.append(torch.tensor(i_sample, dtype=torch.int32))

                # previous_samples = [sample.previous_sample for sample in samples]
                if len(previous_samples) > 0:
                    previous_batch = self.collate_fn_inference(previous_samples,
                                                               n_previous_frames=n_previous_frames - 1)
                    previous_sample_ids = torch.stack(previous_sample_ids)

        # Expression Codes
        batched_expression_codes = None
        if self._config.use_expression_condition:
            batched_expression_codes = torch.stack([sample.expression_code for sample in samples], dim=0)

        # View Directions
        batched_view_directions = None
        if self._config.include_view_directions:
            batched_view_directions = torch.stack([sample.view_direction for sample in samples], dim=0)

        return RenderingDatasetInferenceBatch(
            conditions=batched_input_condition,
            previous_outputs=batched_previous_outputs,
            previous_batch=previous_batch,
            previous_sample_ids=previous_sample_ids,
            expression_codes=batched_expression_codes,
            view_directions=batched_view_directions,
        )

    def collate_fn(self, samples: List[RenderingDatasetSample]) -> RenderingDatasetBatch:
        inference_batch = self.collate_fn_inference(samples)

        batched_target_image = torch.stack([sample.target_image for sample in samples])
        batched_i_participant = torch.tensor([sample.i_participant for sample in samples],
                                             dtype=torch.int32)
        batched_mouth_mask = None
        if self._config.include_mouth_mask:
            batched_mouth_mask = torch.stack([sample.mouth_mask for sample in samples])

        batched_foreground_mask = None
        if self._config.include_foreground_mask:
            batched_foreground_mask = torch.stack([sample.foreground_mask for sample in samples])

        rendering_dataset_batch = RenderingDatasetBatch(
            **to_shallow_dict(inference_batch),
            target_images=batched_target_image,
            i_participant=batched_i_participant,
            mouth_masks=batched_mouth_mask,
            foreground_masks=batched_foreground_mask,
        )

        if self._config.temporal_batch_size > 0:
            previous_samples = [sample.previous_sample for sample in samples if sample.previous_sample is not None]
            if rendering_dataset_batch.previous_batch is not None:
                assert len(previous_samples) == len(rendering_dataset_batch.previous_batch)
                previous_batch = self.collate_fn(previous_samples)
                rendering_dataset_batch.previous_batch = previous_batch

        return rendering_dataset_batch

    def concat_conditions(self, batch: RenderingDatasetInferenceBatch) -> torch.tensor:
        return self._config.concat_conditions(batch)

    @staticmethod
    def normalize_renderings(renderings: Dict[RenderingName, np.ndarray]) -> Dict[RenderingName, np.ndarray]:
        processed_renderings = dict()
        for rendering_name, rendering in renderings.items():
            if rendering_name == RenderingName.DEPTHS:
                rendering = rendering / DepthQuantizer()._max_values  # [0, 2] -> [0, 1]
            elif rendering_name == RenderingName.NORMALS:
                rendering = (rendering + 1) / 2  # [-1, 1] -> [0, 1]

            if rendering_name not in {RenderingName.MASK, RenderingName.UV_DA, RenderingName.UVW_DA,
                                      RenderingName.FORWARD_FLOW, RenderingName.BACKWARD_FLOW} \
                    and not rendering_name.is_flow():
                rendering = rendering * 2 - 1  # [0, 1] -> [-1, 1]

            processed_renderings[rendering_name] = rendering

        return processed_renderings

    def process_conditions(self,
                           renderings: Dict[RenderingName, np.ndarray],
                           crop: Optional[Crop] = None,
                           crop_context: Optional[Crop] = None) -> RenderingDatasetInferenceSample:

        processed_renderings = dict()

        for rendering_name, rendering in renderings.items():
            if crop_context is not None:
                rendering_crop_self = rendering[crop.y: crop.y + crop.h, crop.x: crop.x + crop.w]
                rendering_crop_context = rendering[crop_context.y: crop_context.y + crop_context.h,
                                         crop_context.x: crop_context.x + crop_context.w]
                downscale_factors_self = Dimensions(self._config.resolution / rendering_crop_self.shape[1],
                                                    self._config.resolution / rendering_crop_self.shape[0])
                downscale_factors_context = Dimensions(self._config.resolution / rendering_crop_context.shape[1],
                                                       self._config.resolution / rendering_crop_context.shape[0])

            if rendering_name.is_backward_flow() and crop_context is not None:
                rendering, downscale_factors = self._crop_and_resize(rendering, crop_context, return_scale_factors=True,
                                                                     use_interpolation=True)
            elif rendering_name.is_flow():
                rendering, downscale_factors = self._crop_and_resize(rendering, crop, return_scale_factors=True,
                                                                     use_interpolation=True)
            elif rendering_name == RenderingName.MASK:
                rendering, downscale_factors = self._crop_and_resize(rendering, crop, return_scale_factors=True)
            else:
                rendering, downscale_factors = self._crop_and_resize(rendering, crop, return_scale_factors=True,
                                                                     use_interpolation=self._config.interpolate_target_images)

            if len(rendering.shape) == 2:
                # unsqueeze() for single-channel images
                rendering = rendering[..., None]

            if rendering_name == RenderingName.UV and self._data_config.use_nphm and not self._config.use_canonical_coordinates:
                # NPHM UV renderings are 3-channel XYZ canonical coordinates
                # Create an improvised uv mapping here by projecting XYZ points onto sphere surface
                rendering = NPHMProvider.canonical_coordinates_to_uv(rendering)
            elif rendering_name.is_flow():
                # Cropping and resizing has an effect on the already computed optical flow values
                # They need to be adjusted here

                if crop_context is not None:
                    x_diff = crop_context.x - crop.x
                    y_diff = crop_context.y - crop.y
                    w_diff = (crop_context.w - crop.w)
                    h_diff = (crop_context.h - crop.h)

                    if rendering_name.is_forward_flow():
                        w_diff = w_diff * downscale_factors_context.w / downscale_factors.w
                        h_diff = h_diff * downscale_factors_context.h / downscale_factors.h
                        x_diff = x_diff * downscale_factors_context.w / downscale_factors.w
                        y_diff = y_diff * downscale_factors_context.h / downscale_factors.h
                    elif rendering_name.is_backward_flow():
                        w_diff = w_diff * downscale_factors_self.w / downscale_factors.w
                        h_diff = h_diff * downscale_factors_self.h / downscale_factors.h
                        x_diff = x_diff * downscale_factors_self.w / downscale_factors.w
                        y_diff = y_diff * downscale_factors_self.h / downscale_factors.h

                    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, rendering.shape[1]),
                                                 np.linspace(0, 1, rendering.shape[0]))
                    grid = np.stack([grid_x * w_diff, grid_y * h_diff], axis=-1)

                    if rendering_name.is_forward_flow():
                        rendering = rendering - np.array([x_diff, y_diff])
                        rendering = rendering - grid
                    elif rendering_name.is_backward_flow():
                        rendering = rendering + np.array([x_diff, y_diff])
                        rendering = rendering + grid

                rendering = rendering * np.array([downscale_factors.w, downscale_factors.h])

            rendering = torch.from_numpy(rendering)
            if rendering_name not in {RenderingName.UV_DA, RenderingName.UVW_DA}:
                # UV_DA is not meant to be used as an input condition. Hence, it does not need to swap the channel
                # dimension because nvdiffrast expects the 4 channels of uv_da to be last in the tensor
                rendering = rendering.permute(2, 0, 1)  # [C, H, W]
            processed_renderings[rendering_name] = rendering

        return RenderingDatasetInferenceSample(
            renderings=processed_renderings
        )


class TemporalBatchRenderingDatasetView(Dataset):
    """
    In case temporal batches are used, we do not want to keep all samples in the dataset for evaluation.
    The reason is that we just want to generate the full video sequence once.
    However, with temporal batches, each sample contains also a set of previous frames, meaning that we would have
    overlapping temporal batches during generation.
    Hence, we keep only every n-th consecutive sample if temporal_batch_size = n
    """

    def __init__(self,
                 rendering_dataset: RenderingDataset,
                 serials: Optional[List[str]] = None,
                 sequences: Optional[List[str]] = None,
                 timesteps: Optional[List[int]] = None,
                 max_samples: int = -1):
        self._rendering_dataset = rendering_dataset

        # Only keep the last sample in a temporal batch
        # temporal batch size: 8
        # 1 2 3 4 5 6 7 8
        # - - - - - - - 1
        samples_mapping_view = []
        for idx, sample_metadata in enumerate(rendering_dataset.samples_mapping):
            # Keep if timestep is last in temporal batch
            if rendering_dataset._config.temporal_batch_size == 0 \
                    or (int(sample_metadata.timestep / 3) + 1) % rendering_dataset._config.temporal_batch_size == 0:

                # Keep if serial, sequence and timesteps was selected
                if (serials is None or sample_metadata.serial in serials) \
                        and (sequences is None or sample_metadata.sequence_name in sequences) \
                        and (timesteps is None or sample_metadata.timestep in timesteps):
                    samples_mapping_view.append(idx)

            if 0 < max_samples <= len(samples_mapping_view):
                break

        self.samples_mapping_view = samples_mapping_view

    def __len__(self) -> int:
        return len(self.samples_mapping_view)

    def __getitem__(self, item) -> RenderingDatasetSample:
        idx_dataset = self.samples_mapping_view[item]
        return self._rendering_dataset[idx_dataset]

    def get_sample_metadata(self, idx: int) -> SampleMetadata:
        idx_dataset = self.samples_mapping_view[idx]
        return self._rendering_dataset.samples_mapping[idx_dataset]

    def collate_fn(self, samples: List[RenderingDatasetSample]) -> RenderingDatasetBatch:
        return self._rendering_dataset.collate_fn(samples)

    def collate_fn_inference(self, samples: List[RenderingDatasetInferenceSample]) -> RenderingDatasetInferenceBatch:
        return self._rendering_dataset.collate_fn_inference(samples)


# ==========================================================
# Target images with proper masks
# ==========================================================


def get_sorted_islands(mask: np.ndarray) -> Tuple[List[np.ndarray], int]:
    island_map, n_islands = label(mask)
    island_masks = [island_map == i_island for i_island in range(n_islands + 1)]
    sorted_island_masks = sorted(
        list(zip(island_masks, [island.sum() for island in island_masks])),
        key=lambda x: x[1],
        reverse=True)
    sorted_island_masks = [island[0] for island in sorted_island_masks]
    return sorted_island_masks, n_islands
