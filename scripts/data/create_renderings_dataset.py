from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Dict, Optional

import numpy as np
import torch
import tyro
from dreifus.graphics import Dimensions
from tqdm import tqdm

from diffusion_avatars.config.data.rendering_data import RenderingDataConfig, RenderingDataStatistics
from diffusion_avatars.constants import SERIALS
from diffusion_avatars.data_manager.nersemble_data_manager import NeRSembleSequenceDataManager
from diffusion_avatars.data_manager.rendering_data_manager import RenderingDataFolder, RenderingName, RenderingDataManager
from diffusion_avatars.renderer.nvdiffrast_renderer import NvDiffrastRenderer
from diffusion_avatars.renderer.provider.flame_provider import FlameProvider
from diffusion_avatars.renderer.provider.mesh_provider import MeshProviderConfig, FlameTrackingVersion, TemporalSmoothingConfig, \
    NPHMTrackingVersion
from diffusion_avatars.renderer.provider.nphm_provider import NPHMProvider


def save_renderings(rendering_data_manager: RenderingDataManager,
                    renderings: Dict[RenderingName, np.ndarray],
                    rast: Optional[torch.Tensor],
                    uv_da: Optional[torch.Tensor],
                    uvw_da: Optional[torch.Tensor],
                    participant_id: int,
                    sequence_name: str,
                    timestep: int,
                    serial: str):
    rendering_data_manager.save_uv_rendering(
        renderings[RenderingName.UV], participant_id, sequence_name, timestep, serial)
    rendering_data_manager.save_normals_rendering(
        renderings[RenderingName.NORMALS], participant_id, sequence_name, timestep, serial)
    rendering_data_manager.save_depth_rendering(
        renderings[RenderingName.DEPTHS], participant_id, sequence_name, timestep, serial)
    rendering_data_manager.save_mask(
        renderings[RenderingName.MASK], participant_id, sequence_name, timestep, serial)
    if rast is not None:
        rendering_data_manager.save_rast(rast.cpu().numpy(), participant_id, sequence_name, timestep, serial)
    if uv_da is not None:
        rendering_data_manager.save_uv_da(uv_da.cpu().numpy(), participant_id, sequence_name, timestep, serial)
    if uvw_da is not None:
        rendering_data_manager.save_uvw_da(uvw_da.cpu().numpy(), participant_id, sequence_name, timestep, serial)
    if RenderingName.UV_AMBIENT in renderings:
        rendering_data_manager.save_uv_ambient_rendering(
            renderings[RenderingName.UV_AMBIENT], participant_id, sequence_name, timestep, serial
        )


def create_renderings_dataset(config: RenderingDataConfig):
    rendering_data_folder = RenderingDataFolder()
    rendering_data_manager = rendering_data_folder.create_dataset(name=config.name)
    rendering_data_manager.save_config(config)

    rendering_names = [RenderingName.NORMALS, RenderingName.UV, RenderingName.DEPTHS]
    if config.use_nphm and config.use_ambient_dimensions:
        rendering_names.append(RenderingName.UV_AMBIENT)

    pool = ThreadPool(16)
    futures = []

    available_timesteps = []
    available_sequences = defaultdict(dict)
    total_n_timesteps = 0
    total_n_frames = 0
    for participant_id, sequence_name in config.sequences:
        data_manager = NeRSembleSequenceDataManager(participant_id,
                                                    sequence_name,
                                                    downscale_factor=config.downscale_factor)

        world_2_cam_poses = data_manager.load_calibration_result().params_result.get_poses()
        # intrinsics = data_manager.load_calibration_result().params_result.get_intrinsics()
        # intrinsics.rescale(1 / config.downscale_factor, inplace=True)
        all_intrinsics = data_manager.load_calibration_result().params_result.get_all_intrinsics()
        all_intrinsics = [intr.rescale(1 / config.downscale_factor, inplace=True) for intr in all_intrinsics]

        # Only use views for selected serials
        selected_cam_ids = [SERIALS.index(serial) for serial in config.serials]
        world_2_cam_poses = [world_2_cam_poses[cam_id] for cam_id in selected_cam_ids]
        all_intrinsics = [all_intrinsics[cam_id] for cam_id in selected_cam_ids]

        image_size = data_manager.load_calibration_run_config().image_size

        render_size = Dimensions(int(image_size.w / config.downscale_factor), int(image_size.h / config.downscale_factor))
        if config.use_nphm:
            mesh_provider = NPHMProvider(participant_id, sequence_name,
                                         mesh_provider_config=config.mesh_provider_config)
            renderer = NvDiffrastRenderer(mesh_provider, render_size)
        else:
            mesh_provider = FlameProvider.from_dataset(participant_id,
                                                       sequence_name,
                                                       config=config.mesh_provider_config)
            renderer = NvDiffrastRenderer(mesh_provider, render_size)

        timesteps = data_manager.get_timesteps()
        if config.n_timesteps > 0:
            # Only use the first n timesteps
            timesteps = timesteps[:config.n_timesteps]

        available_timesteps.append(timesteps)
        available_sequences[f"{participant_id}"][sequence_name] = timesteps
        for timestep in tqdm(timesteps):
            with torch.no_grad():
                vertices = mesh_provider.get_vertices(timestep)
            mesh = mesh_provider.create_mesh(vertices, timestep)

            total_n_timesteps += 1

            if config.use_mipmapping or config.use_antialiasing:
                all_renderings, rast, uv_da, uvw_da = renderer.render(mesh, world_2_cam_poses, all_intrinsics, rendering_names,
                                                                      return_rast_and_derivatives=True)
                if not config.use_antialiasing:
                    rast = [None for _ in range(len(all_renderings))]
                if not config.use_mipmapping:
                    uv_da = [None for _ in range(len(all_renderings))]
                if not config.use_mipmapping or uvw_da is None:
                    uvw_da = [None for _ in range(len(all_renderings))]

            else:
                all_renderings = renderer.render(mesh, world_2_cam_poses, all_intrinsics, rendering_names)
                rast = [None for _ in range(len(all_renderings))]
                uv_da = [None for _ in range(len(all_renderings))]
                uvw_da = [None for _ in range(len(all_renderings))]

            for cam_id, (renderings, single_rast, single_uv_da, single_uvw_da) in enumerate(zip(all_renderings, rast, uv_da, uvw_da)):
                serial = config.serials[cam_id]

                # Very important to use the args argument of apply_async instead of passing a lambda
                # Otherwise, not all functions will be executed? Or their parameters might change randomly?
                futures.append(pool.apply_async(save_renderings, (
                    rendering_data_manager, renderings, single_rast, single_uv_da, single_uvw_da, participant_id, sequence_name, timestep,
                    serial)))

                total_n_frames += 1

    for future in futures:
        future.wait()

    stats = RenderingDataStatistics(
        total_n_timesteps=total_n_timesteps,
        total_n_frames=total_n_frames,
        available_timesteps=available_timesteps,
        available_sequences=available_sequences
    )
    rendering_data_manager.save_stats(stats)


def main(participant_id: int,
         sequence_name: str,
         /,
         name: str = None,
         n_timesteps: int = -1,
         close_mouth: bool = True,
         use_uv_faces: bool = True,
         use_nphm: bool = True,
         use_mipmapping: bool = False,
         use_ambient_dimensions: bool = True,
         use_antialiasing: bool = False,
         cut_throat: bool = True,
         cut_throat_margin: float = -0.05):
    downscale_factor = 2
    flame_tracking = FlameTrackingVersion.FLAME_2023_V2
    if participant_id in {18, 37}:
        nphm_tracking = NPHMTrackingVersion.NPHM_TEMPORAL_NECK
    else:
        nphm_tracking = NPHMTrackingVersion.NPHM_TEMPORAL_NECK_LMS
    temporal_smoothing = 0
    use_gaussian_smoothing = False

    participant_ids = [participant_id]
    sequence_names = sequence_name.split(',')

    sequences = []
    for participant_id in participant_ids:
        for sequence_name in sequence_names:
            sequences.append((participant_id, sequence_name))

    serials = SERIALS

    temporal_smoothing_config = TemporalSmoothingConfig(
        temporal_smoothing=temporal_smoothing,
        use_gaussian=use_gaussian_smoothing
    )

    mesh_provider_config = MeshProviderConfig(
        temporal_smoothing_config=temporal_smoothing_config,
        use_uv_faces=use_uv_faces,
        close_mouth=close_mouth,
        flame_tracking=flame_tracking,
        nphm_tracking=nphm_tracking,
        cut_throat=cut_throat,
        cut_throat_margin=cut_throat_margin,
    )
    config = RenderingDataConfig(name=name,
                                 downscale_factor=downscale_factor,
                                 sequences=sequences,
                                 n_timesteps=n_timesteps,
                                 use_mipmapping=use_mipmapping,
                                 use_antialiasing=use_antialiasing,
                                 use_ambient_dimensions=use_ambient_dimensions,
                                 use_nphm=use_nphm,
                                 mesh_provider_config=mesh_provider_config,
                                 serials=serials
                                 )
    create_renderings_dataset(config)


if __name__ == '__main__':
    tyro.cli(main)
