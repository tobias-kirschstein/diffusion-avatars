"""
FLAME Layer: Implementation of the 3D Statistical Face model in PyTorch

It is designed in a way to directly plug in as a decoder layer in a 
Deep learning framework for training and testing

It can also be used for 2D or 3D optimisation applications

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
# Modified from smplx code [https://github.com/vchoutas/smplx] for FLAME

import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler

from diffusion_avatars.config.flame import FlameConfig
from diffusion_avatars.env import DIFFUSION_AVATARS_FLAME_MODEL_PATH


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """

    def __init__(self, config: FlameConfig):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")

        flame_model_path = f"{DIFFUSION_AVATARS_FLAME_MODEL_PATH}/{config.flame_version}"
        static_landmark_embedding_path = f"{DIFFUSION_AVATARS_FLAME_MODEL_PATH}/flame_static_embedding.pkl"
        dynamic_landmark_embedding_path = f"{DIFFUSION_AVATARS_FLAME_MODEL_PATH}/flame_dynamic_embedding.npy"

        with open(flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))

        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAME; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project 
        default_shape = torch.zeros([self.batch_size, 300 - config.shape_params],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape_betas', nn.Parameter(default_shape,
                                                            requires_grad=False))

        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAME; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project 
        default_exp = torch.zeros([self.batch_size, 100 - config.expression_params],
                                  dtype=self.dtype, requires_grad=False)
        self.register_parameter('expression_betas', nn.Parameter(default_exp,
                                                                 requires_grad=False))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros([self.batch_size, 6],
                                          dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))

        default_neck_pose = torch.zeros([self.batch_size, 3],
                                        dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = config.use_3D_translation

        default_transl = torch.zeros([self.batch_size, 3],
                                     dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            'transl',
            nn.Parameter(default_transl, requires_grad=False))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

        # Static and Dynamic Landmark embeddings for FLAME

        with open(static_landmark_embedding_path, 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))

        lmk_faces_idx = (static_embeddings.lmk_face_idx).astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=self.dtype))

        if self.use_face_contour:
            conture_embeddings = np.load(dynamic_landmark_embedding_path,
                                         allow_pickle=True, encoding='latin1')
            conture_embeddings = conture_embeddings[()]
            dynamic_lmk_faces_idx = np.array(conture_embeddings['lmk_face_idx']).astype(np.int64)
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = conture_embeddings['lmk_b_coords']
            dynamic_lmk_bary_coords = torch.tensor(
                dynamic_lmk_bary_coords, dtype=self.dtype)
            self.register_buffer('dynamic_lmk_bary_coords',
                                 dynamic_lmk_bary_coords)

            neck_kin_chain = []
            curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.register_buffer('neck_kin_chain',
                                 torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(self, vertices, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
            Source: Modified for batches from https://github.com/vchoutas/smplx
        """

        batch_size = vertices.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_vecs = aa_pose.view(-1, 3)

        assert rot_vecs.dtype == dtype

        rot_mats = batch_rodrigues(
            rot_vecs).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=vertices.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None, eye_pose=None,
                transl=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters. Typically, the first 3 will be global rotation, and the last 3 the jaw pose
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        B = shape_params.shape[0]
        betas = torch.cat(
            [shape_params, self.shape_betas.repeat(B, 1), expression_params, self.expression_betas.repeat(B, 1)],
            dim=1)
        neck_pose = (neck_pose if neck_pose is not None else self.neck_pose)
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose)
        transl = (transl if transl is not None else self.transl)
        full_pose = torch.cat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)

        assert betas.dtype == self.dtype

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(
            self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, dtype=self.dtype)

            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx.repeat(B, 1)], 1)
            lmk_bary_coords = torch.cat(
                [dyn_lmk_bary_coords, lmk_bary_coords.repeat(B, 1, 1)], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        if self.use_3D_translation:
            landmarks += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        return vertices, landmarks


@dataclass
class FlameTexConfig:
    tex_space_path: str = f"{DIFFUSION_AVATARS_FLAME_MODEL_PATH}/FLAME_texture.npz"
    tex_params: int = 200
    tex_size: int = 256  # number of pixels in width and height (quadratic). If 512, no interpolation is used


class FLAMETex(nn.Module):
    """
    current FLAME texture are adapted from BFM Texture Model
    """

    def __init__(self, config: FlameTexConfig = FlameTexConfig()):
        super(FLAMETex, self).__init__()
        tex_params = config.tex_params
        tex_space = np.load(config.tex_space_path)
        self._tex_size = config.tex_size

        texture_mean = tex_space['mean'].reshape(1, -1)
        texture_basis = tex_space['tex_dir'].reshape(-1, 200)
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :tex_params]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

        self._vt = tex_space['vt']
        self._ft = tex_space['ft']

    def get_uv_coords_by_vertex(self, flame: FLAME, use_mean: bool = False) -> np.ndarray:
        texture_coordinates_by_vertex = [[] for _ in range(len(flame.v_template))]
        for i, face in enumerate(flame.faces):
            for j in [0, 1, 2]:
                texture_coordinates_by_vertex[face[j]].append(self._vt[self._ft[i][j]])

        # Essentially, there are two topologies: one used for mapping the texture and one for controlling the
        # actual geometry
        # 'ft' translates between the two topologies by mapping 1 geometry face onto the 3 corresp. texture vertices
        # It can happen that the same geometry vertex v which is part of several geometry faces is mapped to different
        # texture vertices in 'ft'.
        # Consequently, texture_coordinates_by_vertex contains an arbitrary long list of uv candidates for every vertex.
        # To resolve this, we can either only use the first uv candidate in the list or average over all of them.
        if use_mean:
            texture_coordinates_by_vertex = np.stack([np.mean(uvs, axis=0) for uvs in texture_coordinates_by_vertex])
        else:
            # Use first
            # texture_coordinates_by_vertex = np.stack([uvs[0] for uvs in texture_coordinates_by_vertex])
            # Use left-most
            texture_coordinates_by_vertex = np.stack([sorted(uvs, key=lambda uv: uv[0])[0]
                                                      for uvs
                                                      in texture_coordinates_by_vertex])

        return texture_coordinates_by_vertex

    def get_uv_coords_by_edge(self, flame: FLAME) -> np.ndarray:
        edges = set()
        for face in flame.faces:
            for v_start, v_end in zip(face, face[1:] + [face[0]]):
                v_start = v_start if v_start < v_end else v_end
                v_end = v_end if v_start < v_end else v_start
                edges.add
        pass

    def get_uv_coords_by_face(self, flame: FLAME) \
            -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        uv_coords_by_face = []
        for i_face, face in enumerate(flame.faces):
            uv_coords_by_face.append(tuple([self._vt[i_tex_vertex] for i_tex_vertex in self._ft[i_face]]))
        return uv_coords_by_face

    def get_uv_faces(self) -> np.ndarray:
        """
        UV attributes per vertex must be handled differently than 3D position attributes.
        The same 3D position can have different uv coordinates.
        Hence, the 3D position per vertex list is shorter (because vertices with same 3D position are combined).
        The UV per vertex list is slightly longer (see get_uv_coords()).
        As a result, we also need a different list of faces, since they must index into a different vertex list for UV.

        Returns
        -------
            9976 x 3
        """

        return self._ft

    def get_uv_coords(self) -> np.ndarray:
        """
        The UV per vertex list is longer than the 3D position per vertex list.
        This reflects the fact that the same 3D position in the mesh can have multiple UV coords attached to it.
        Usually, this happens at texture seems, where the same 3D vertex will have 2 UV coords, one to the
        outer left and one to the outer right of the texture.

        Returns
        -------
            5118 x 2
        """

        return self._vt

    def forward(self, texcode):
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)

        if self._tex_size < 512:
            texture = F.interpolate(texture, [self._tex_size, self._tex_size])
        elif self._tex_size > 512:
            raise ValueError("Cannot upscale texture beyond 512 x 512")

        texture = texture[:, [2, 1, 0], :, :]
        return texture
