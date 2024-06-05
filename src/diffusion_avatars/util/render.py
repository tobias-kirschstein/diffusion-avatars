from typing import Optional

from dreifus.graphics import Dimensions
from dreifus.matrix import Intrinsics

from diffusion_avatars.config.data.rendering_data import RenderingDataConfig
# from diffusion_avatars.renderer.flame_pyvista import FlamePyVistaRenderer
from diffusion_avatars.renderer.nvdiffrast_renderer import NvDiffrastRenderer
from diffusion_avatars.renderer.provider.flame_provider import FlameProvider
from diffusion_avatars.renderer.provider.mesh_provider import MeshProvider
from diffusion_avatars.renderer.provider.nphm_provider import NPHMProvider


def get_mesh_provider(config: RenderingDataConfig,
                      participant_id: int,
                      sequence: str,
                      **kwargs) -> 'MeshProvider':
    if config.use_nphm:
        mesh_provider = NPHMProvider(participant_id, sequence,
                                     use_flame_2023=config.mesh_provider_config.flame_tracking.is_flame_2023(),
                                     **kwargs)
    else:
        mesh_provider = FlameProvider.from_dataset(participant_id, sequence, config=config.mesh_provider_config,)

    return mesh_provider


def get_renderer(config: RenderingDataConfig,
                 mesh_provider: MeshProvider,
                 render_size: Optional[Dimensions] = None) -> NvDiffrastRenderer:
    if render_size is None:
        render_size = Dimensions(int(2200 / config.downscale_factor), int(3208 / config.downscale_factor))

    if config.use_nvdiffrast:
        return NvDiffrastRenderer(mesh_provider, render_size)
    else:
        return FlamePyVistaRenderer(mesh_provider, render_size)


def prepare_intrinsics(config: RenderingDataConfig,
                       intrinsics: Intrinsics,
                       render_size: Optional[Dimensions] = None) -> Intrinsics:
    render_size_given = True
    if render_size is None:
        render_size = Dimensions(int(2200 / config.downscale_factor), int(3208 / config.downscale_factor))
        render_size_given = False

    intrinsics = intrinsics.rescale(render_size.w / 2200, render_size.h / 3208, inplace=False)

    if render_size_given:
        # Assume, that viewpoint is created artificially
        # Ensure that principle point is still in image center
        intrinsics.cx = render_size.w / 2
        intrinsics.cy = render_size.h / 2

    return intrinsics
