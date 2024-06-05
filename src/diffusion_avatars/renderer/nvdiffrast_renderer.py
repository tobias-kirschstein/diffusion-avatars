from typing import List, Dict, Union, Tuple, Optional

import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.graphics import Dimensions
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector.vector_numpy import to_homogeneous

from diffusion_avatars.data_manager.rendering_data_manager import RenderingName
from diffusion_avatars.renderer.provider.mesh_provider import MeshProvider
from diffusion_avatars.renderer.provider.nphm_provider import NPHMProvider


def intrinsics2projection(intrinsics: Intrinsics, znear: float, zfar: float, render_size: Dimensions):
    width = render_size.w
    height = render_size.h
    x0 = 0
    y0 = 0
    # @formatter:off
    return np.array([
        [2 * intrinsics.fx / width, -2 * intrinsics.s / width,      (width - 2 * intrinsics.cx + 2 * x0) / width,   0],
        [0,                         -2 * intrinsics.fy / height,    (height - 2 * intrinsics.cy + 2 * y0) / height, 0],
        [0,                         0,                              (-zfar - znear) / (zfar - znear),               -2 * zfar * znear / (zfar - znear)],
        [0,                         0,                              -1,                                             0]])
    # @formatter:on


class NvDiffrastRenderer:

    def __init__(self,
                 mesh_provider: MeshProvider,
                 render_size: Dimensions
                 ):
        self.mesh_provider = mesh_provider
        self._glctx = dr.RasterizeCudaContext()

        if render_size.w > 2048 or render_size.h > 2048:
            raise ValueError(f"RasterizeCudaContext can only deal with at most 2048 pixels in each dimension. "
                             f"Got {render_size}")

        # nvdiffrast RasterizeCudaContext can only render at resolutions that are a multiple of 8
        # Hence, we have to artificially increase the resolution to match that
        # and then later always deduct those pixels when returning the renderings
        render_offset_w = (8 - (render_size.w % 8)) % 8
        render_offset_h = (8 - (render_size.h % 8)) % 8

        self._render_size = render_size
        self._render_offset = Dimensions(render_offset_w, render_offset_h)

    def render(self,
               mesh: trimesh.Trimesh,
               poses: Union[Pose, List[Pose]],
               intrinsics: Union[Intrinsics, List[Intrinsics]],
               rendering_names: List[RenderingName],
               return_rast_and_derivatives: bool = False) -> \
            Union[
                Dict[RenderingName, np.ndarray], List[Dict[RenderingName, np.ndarray]],
                Tuple[Dict[RenderingName, np.ndarray], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:

        render_size = self._render_size + self._render_offset
        is_batched = True
        if isinstance(poses, Pose):
            poses = [poses]
            is_batched = False

        # Nvdiffrast expects OpenGL
        poses = [pose.copy() for pose in poses]
        for pose in poses:
            if pose.pose_type != PoseType.WORLD_2_CAM \
                    or pose.camera_coordinate_convention != CameraCoordinateConvention.OPEN_GL:
                pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=True)
                pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL, inplace=True)
                pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=True)
        poses = np.stack(poses)  # [B, 4, 4]
        B = len(poses)
        V = len(mesh.vertices)

        if not isinstance(intrinsics, list):
            intrinsics = [intrinsics for _ in range(len(poses))]

        # intrinsics -> OpenGL projection matrix
        z_near = 0.1
        z_far = 2  # Heads should never be further away than that
        projection = np.stack([intrinsics2projection(intr, z_near, z_far, render_size) for intr in intrinsics])  # [B, 4, 4]
        # projection = np.repeat(projection[None], B, axis=0)  # [B, 4, 4]

        # Rendering from a specific camera view point is done by already applying WORLD_2_CAM and projection matrix
        # The vertices will be in clip space afterwards
        mesh_vertices = mesh.vertices
        mesh_vertices = np.repeat(mesh_vertices[None], B, axis=0)  # [B, V, 3]
        mesh_vertices = to_homogeneous(mesh_vertices)
        mesh_vertices_cam = mesh_vertices @ np.transpose(poses, (0, 2, 1))
        mesh_vertices_clip = mesh_vertices_cam @ np.transpose(projection, (0, 2, 1))  # [B, V, 3]

        device = torch.device('cuda')
        vertices = torch.tensor(mesh_vertices_clip, dtype=torch.float32, device=device)  # [B, V, 3]
        faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)  # [F, 3]

        # Prepare vertex attributes (normals, UV, depths)
        attrs = []
        attrs_uv = []
        attr_nphm_uv = None
        rendering_channels = dict()
        rendering_channels_is_uv = dict()
        uv_attr_idxs = None
        uv_face_attr_idxs = None
        for rendering_name in rendering_names:
            attr = None  # Attribute per vertex
            attr_uv = None  # Attribute per UV vertex
            if rendering_name == RenderingName.UV:

                if self.mesh_provider.has_uv_faces():
                    attr_uv = torch.tensor(self.mesh_provider.get_uv_coords(mesh), dtype=torch.float32, device=device)
                    attr_uv = attr_uv[None].repeat(B, 1, 1)  # [B, VT, 2]
                    n_current_channels = sum(rendering_channels_is_uv.values())
                    uv_face_attr_idxs = list(range(n_current_channels, n_current_channels + attr_uv.shape[-1]))
                else:
                    attr = torch.tensor(self.mesh_provider.get_uvs_per_vertex(mesh), dtype=torch.float32, device=device)
                    attr = attr[None].repeat(B, 1, 1)  # [B, V, 2]
                    n_current_channels = sum(rendering_channels.values())
                    uv_attr_idxs = list(range(n_current_channels, n_current_channels + attr.shape[-1]))

                    if isinstance(self.mesh_provider, NPHMProvider):
                        attr_nphm_uv = torch.from_numpy(
                            self.mesh_provider.canonical_coordinates_to_uv(attr.cpu().numpy())).cuda()
            elif rendering_name == RenderingName.UV_AMBIENT:
                attr = torch.tensor(self.mesh_provider.get_ambient_coordinates_per_vertex(mesh), dtype=torch.float32, device=device)
                attr = attr[None].repeat(B, 1, 1)  # [B, V, 2]
            elif rendering_name == RenderingName.NORMALS:
                attr = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
                attr = attr[None].repeat(B, 1, 1)  # [B, V, 3]
            elif rendering_name == RenderingName.DEPTHS:
                attr = vertices[..., [2]]
            elif rendering_name == RenderingName.FORWARD_FLOW:
                assert B == 1
                attr = torch.tensor(mesh.vertex_attributes["forward_flow"], dtype=torch.float32, device=device)[None]
            elif rendering_name == RenderingName.BACKWARD_FLOW:
                assert B == 1
                attr = torch.tensor(mesh.vertex_attributes["backward_flow"], dtype=torch.float32, device=device)[None]
            else:
                raise ValueError(f"Unknown rendering name: {rendering_name}")

            if attr is not None:
                attrs.append(attr)
                rendering_channels[rendering_name] = attr.shape[-1]  # number of channels in this rendering
                rendering_channels_is_uv[rendering_name] = False

            elif attr_uv is not None:
                attrs_uv.append(attr_uv)
                rendering_channels[rendering_name] = attr_uv.shape[-1]  # number of channels in this rendering
                rendering_channels_is_uv[rendering_name] = True

        attrs = torch.cat(attrs, dim=-1)
        if self.mesh_provider.has_uv_faces() and len(attrs_uv) > 0:
            attrs_uv = torch.cat(attrs_uv, dim=-1)

        # Rendering + screen-space attribute interpolation
        # Note: For antialiasing, it may be necessary to write out texture derivatives as well
        rast, rast_db = dr.rasterize(self._glctx, vertices, faces, resolution=[render_size.h, render_size.w])
        rendered_images, uv_da = dr.interpolate(attrs, rast, faces, rast_db=rast_db, diff_attrs=uv_attr_idxs)

        if attr_nphm_uv is not None:
            # For NPHM, we compute 2 texture derivatives:
            #   - uv_da are the derivates wrt the uv spaced induced by the spherical uv mapping
            #   - uvw_da are the derivates wrt to the 3D canonical coordinates (i.e., texture volume)
            uvw_da = uv_da
            _, uv_da = dr.interpolate(attr_nphm_uv, rast, faces, rast_db=rast_db, diff_attrs=[0, 1])
        else:
            uvw_da = None

        rendered_images = rendered_images.cpu().numpy()
        # Drop offset pixels at bottom and right
        rendered_images = rendered_images[..., :self._render_size.h, :self._render_size.w, :]

        rendered_images_uv = None
        if self.mesh_provider.has_uv_faces() and len(attrs_uv) > 0:
            uv_faces = torch.tensor(self.mesh_provider.get_uv_faces(mesh).astype(int), dtype=torch.int32,
                                    device=device)  # [FT, 3]
            rendered_images_uv, uv_da = dr.interpolate(attrs_uv, rast, uv_faces,
                                                       rast_db=rast_db, diff_attrs=uv_face_attr_idxs)
            rendered_images_uv = rendered_images_uv.cpu().numpy()
            # Drop offset pixels at bottom and right
            rendered_images_uv = rendered_images_uv[..., :self._render_size.h, :self._render_size.w, :]

        all_renderings = dict()

        render_masks = (rast[..., :self._render_size.h, :self._render_size.w, 3] > 0).cpu().numpy()
        all_renderings[RenderingName.MASK] = list(render_masks)

        # Collect attribute renderings by slicing the respective channels
        channel = 0
        channel_uv = 0
        for rendering_name in rendering_names:
            # n_channels = rendering_name.get_n_channels(nvdiffrast=True)
            n_channels = rendering_channels[rendering_name]
            is_uv = rendering_channels_is_uv[rendering_name]

            if is_uv:
                rendering = rendered_images_uv[..., channel_uv: channel_uv + n_channels]
            else:
                rendering = rendered_images[..., channel: channel + n_channels]

            if rendering_name == RenderingName.NORMALS:
                # Ensure normals have unit norm
                # This is probably a side effect of doing the normal interpolation in euclidean space
                # Should maybe use spherical coordinates for more accurate normals?
                rendering[render_masks] /= np.linalg.norm(rendering[render_masks], axis=-1)[..., None]

                # Explicitly set all values outside to 0.
                # This is necessary as the dr.interpolate() can produce both +0 and -0 values...
                # This would case ambiguities later when encoding the normals as spherical coordinates
                rendering[~render_masks] = 0

            all_renderings[rendering_name] = list(rendering)
            if is_uv:
                channel_uv += n_channels
            else:
                channel += n_channels

        # dict-of-lists -> list-of-dicts
        all_renderings = [dict(zip(all_renderings, t)) for t in zip(*all_renderings.values())]

        if not is_batched:
            all_renderings = all_renderings[0]

        if return_rast_and_derivatives:
            rast = rast[:, :self._render_size.h, : self._render_size.w]
            uv_da = uv_da[:, :self._render_size.h, :self._render_size.w]
            if uvw_da is not None:
                uvw_da = uvw_da[:, :self._render_size.h, :self._render_size.w]
            return all_renderings, rast, uv_da, uvw_da
        else:
            return all_renderings
