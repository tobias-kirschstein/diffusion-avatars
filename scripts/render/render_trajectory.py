from collections import defaultdict
from math import sin, cos
from typing import Optional, List

import numpy as np
import torch
import tyro
from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.camera_bundle import align_poses
from dreifus.graphics import Dimensions, homogenize
from dreifus.matrix import Pose
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from elias.util.batch import batchify_sliced
from elias.util.random import make_deterministic
from tqdm import tqdm

from diffusion_avatars.constants import TEST_SEQUENCES, SERIALS
from diffusion_avatars.data_manager.nersemble_data_manager import NeRSembleSequenceDataManager
from diffusion_avatars.data_manager.rendering_data_manager import RenderingName
from diffusion_avatars.dataset.rendering_dataset import DatasetSplit, RenderingDataset, SampleMetadata, \
    RenderingDatasetInferenceBatch
from diffusion_avatars.model.diffusion_avatars import DiffusionAvatarsModel
from diffusion_avatars.model.inference_output import SchedulerType, DiffusionAvatarsInferenceOutput
from diffusion_avatars.model_manager.finder import find_model_folder
from diffusion_avatars.render_manager.diffusion_avatars_render_manager import TrajectoryType, DiffusionAvatarsRenderManager
from diffusion_avatars.renderer.nvdiffrast_renderer import NvDiffrastRenderer
from diffusion_avatars.renderer.provider.flame_provider import FlameProvider
from diffusion_avatars.renderer.provider.nphm_provider import NPHMProvider


def main(run_name: str,
         /,
         checkpoint: int = -1,
         trajectory_type: TrajectoryType = "camera_shake",
         batch_size: int = 1,
         fix_latents: bool = True,
         fix_timestep: bool = False,
         seed: Optional[int] = None,
         prompt: Optional[str] = None,
         disable_neural_texture: bool = False,
         n_inference_steps: int = 20,
         resolution: Optional[int] = None,
         scheduler: SchedulerType = 'ddpm',

         # plot_type specific settings
         n_steps_per_camera: int = 10,
         person: Optional[int] = None,
         camera_distance: float = 1,
         serial: str = '222200037',
         cfg_weight: float = 1,
         shake: float = 0.3,
         move_y: float = 0.02,
         look_y: Optional[float] = None,
         curve_z: float = 0.02,
         test_sequence: Optional[int] = None,
         frequency: int = 40,
         ):
    model_folder = find_model_folder(run_name)
    model_manager = model_folder.open_run(run_name)
    train_config = model_manager.load_train_setup()
    dataset_config = model_manager.load_dataset_config()
    data_config = dataset_config.get_data_config()
    data_stats = dataset_config.get_data_statistics()
    if prompt is not None:
        train_config.global_prompt = prompt

    checkpoint = model_manager._resolve_checkpoint_id(checkpoint)
    model = model_manager.load_checkpoint(checkpoint)
    model.cuda()

    if resolution is not None:
        dataset_config.resolution = resolution
        data_config.resolution = resolution
        model._dataset_config.resolution = resolution
    else:
        resolution = dataset_config.resolution

    if run_name.startswith('DA-'):
        model.change_inference_scheduler(scheduler)

    render_manager = DiffusionAvatarsRenderManager(run_name,
                                                   checkpoint=checkpoint,
                                                   plot_type=trajectory_type,
                                                   disable_neural_texture=disable_neural_texture,
                                                   n_inference_steps=n_inference_steps,
                                                   resolution=resolution,
                                                   camera_distance=camera_distance,
                                                   cfg_weight=cfg_weight,
                                                   prompt=prompt,
                                                   scheduler=scheduler,
                                                   shake=shake,
                                                   move_y=move_y,
                                                   look_y=look_y,
                                                   curve_z=curve_z,
                                                   test_sequence=test_sequence,
                                                   frequency=frequency)

    if trajectory_type == 'valid_hold_out_view':
        split = DatasetSplit.VALID_HOLD_OUT_VIEW
    elif trajectory_type == 'valid_hold_out_expression':
        split = DatasetSplit.VALID_HOLD_OUT_EXP
    elif trajectory_type == 'valid_hold_out_sequence':
        split = DatasetSplit.VALID_HOLD_OUT_SEQ
    else:
        split = DatasetSplit.TRAIN
    valid_dataset = RenderingDataset(dataset_config, split=split)

    if seed is None:
        seed = train_config.seed
    make_deterministic(seed)

    # Latent noise
    use_latent_noise = run_name.startswith('DA-')
    if use_latent_noise and fix_latents:
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=model.device).manual_seed(seed)

        latent_shape = model.get_latent_shape()
        latent_noise = torch.randn(latent_shape, device=model.device, generator=generator)
    else:
        latent_noise = None

    if trajectory_type in {'camera_moveX', 'camera_shake', 'camera'}:
        seconds = 4
        fps = 24

        p_id = valid_dataset.participant_ids[0 if person is None else person]
        person_metadata = data_stats.available_sequences[str(p_id)]
        p_id = int(p_id)
        if test_sequence is None:
            sequence, timesteps = next(iter(person_metadata.items()))
        else:
            sequence = TEST_SEQUENCES[p_id][test_sequence]
            data_manager = NeRSembleSequenceDataManager(p_id, sequence)
            timesteps = data_manager.get_timesteps()

        resolution = dataset_config.resolution
        data_manager = NeRSembleSequenceDataManager(p_id, sequence)
        world_2_cam_poses = data_manager.load_calibration_result().params_result.get_poses()

        # Create trajectory
        trajectory = []

        if trajectory_type == 'camera_moveX':
            cam_2_world_poses, transform = align_poses(world_2_cam_poses, return_transformation=True)

            intrinsics = data_manager.load_calibration_result().params_result.get_intrinsics()

            intrinsics.rescale(resolution / 3208)
            # Ensure that principle point is still in image center
            intrinsics.cx = resolution / 2
            intrinsics.cy = resolution / 2

            trajectory = circle_around_axis(int(seconds * fps),
                                            axis=Vec3(0, 0, 1),
                                            up=Vec3(0, 1, 0),
                                            move=Vec3(0, 0, camera_distance),
                                            distance=0.3)

            trajectory = [transform.invert() @ pose for pose in
                          trajectory]  # Move trajectory cameras from aligned -> person-specific calibration space

        elif trajectory_type == 'camera_shake':
            data_manager = NeRSembleSequenceDataManager(p_id, sequence)
            timesteps = data_manager.get_timesteps()

            seconds = len(timesteps) / fps  # We want to render the full sequence actually

            intrinsics = data_manager.load_calibration_result().params_result.get_intrinsics()
            intrinsics.rescale(resolution / 2200)
            # Ensure that principle point is still in image center
            intrinsics.cx = resolution / 2
            intrinsics.cy = resolution / 2

            nphm_provider = NPHMProvider(p_id, sequence, mesh_provider_config=data_config.mesh_provider_config)
            world_2_nphm = nphm_provider.get_world_2_nphm_pose(0)

            trajectory = []
            for i in range(int(seconds * fps)):
                x = sin(2 * (i / frequency) * np.pi) * shake  # [-0.1, 0.1]
                y = move_y
                z = cos(2 * 2 * (i / frequency) * np.pi) * curve_z + camera_distance
                camera_position = Vec3(x, y, z)

                pose = Pose(np.eye(3), camera_position, pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                pose.look_at(Vec3(0, y if look_y is None else look_y, 0), up=Vec3(0, 1, 0))  # Look at head
                pose = np.linalg.inv(world_2_nphm) @ pose.numpy()
                pose_2 = Pose(pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                pose_2[:] = pose
                trajectory.append(pose_2)

        elif trajectory_type == 'camera':
            intrinsics = data_manager.load_calibration_result().params_result.get_intrinsics()

            intrinsics.rescale(resolution / 2200)
            # Ensure that principle point is still in image center
            intrinsics.cx = resolution / 2
            intrinsics.cy = resolution / 2

            trajectory = [world_2_cam_poses[SERIALS.index(serial)].invert() for _ in range(int(seconds * fps))]
            up_direction = trajectory[0].get_up_direction()
            for pose in trajectory:
                pose.move(up_direction * move_y)
        else:
            raise NotImplementedError(f"Unknown plot type: {trajectory_type}")

        idx_timesteps = np.linspace(0, len(timesteps) - 1, len(trajectory), dtype=int)
        chosen_timesteps = [timesteps[0] if fix_timestep else timesteps[i] for i in idx_timesteps]

        # Nvdiffrast rendering
        render_size = Dimensions(resolution, resolution)
        if data_config.use_nphm:
            mesh_provider = NPHMProvider(p_id, sequence, mesh_provider_config=data_config.mesh_provider_config)
            renderer = NvDiffrastRenderer(mesh_provider, render_size)
        else:
            mesh_provider = FlameProvider.from_dataset(p_id, sequence, config=data_config.mesh_provider_config)
            renderer = NvDiffrastRenderer(mesh_provider, render_size)

        source_world_2_target_world = None

        # Generate renderings for all frames of trajectory
        all_renderings = []
        conditions = defaultdict(list)
        previous_meshes = []
        previous_poses = []
        expression_codes = []
        view_directions = []
        for timestep, pose in tqdm(zip(chosen_timesteps, trajectory), desc="Generating renderings"):

            with torch.no_grad():
                vertices = mesh_provider.get_vertices(timestep)

            if source_world_2_target_world is not None:
                vertices = (homogenize(vertices) @ source_world_2_target_world.T)[..., :3]
            mesh = mesh_provider.create_mesh(vertices, timestep)

            rendering_names = dataset_config.rendering_names
            if dataset_config.use_ambient_textures and RenderingName.UV_AMBIENT not in rendering_names:
                rendering_names.append(RenderingName.UV_AMBIENT)

            pose_2 = Pose(pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
            pose_2[:] = pose
            pose_2 = pose_2.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
            pose = pose_2.invert()
            renderings = renderer.render(mesh, pose, intrinsics, rendering_names)
            renderings = valid_dataset.normalize_renderings(renderings)

            for rendering_name, rendering in renderings.items():
                if rendering_name in dataset_config.rendering_names \
                        or dataset_config.use_neural_textures \
                        and (rendering_name == RenderingName.UV or rendering_name == RenderingName.MASK):

                    if len(rendering.shape) > 2 and rendering.shape[-1] == 2:
                        rendering = np.concatenate([rendering,
                                                    -1 * np.ones(rendering.shape[:-1] + (1,),
                                                                 dtype=rendering.dtype)],
                                                   axis=-1)  # Add artificial 3rd channel for .png storage
                    elif len(rendering.shape) > 2 and rendering.shape[-1] > 3:
                        rendering = rendering[..., :3]

                    conditions[rendering_name].append((np.squeeze(rendering) + 1) / 2)

            all_renderings.append(renderings)
            previous_meshes.append(mesh)
            previous_poses.append(pose)
            expression_codes.append(valid_dataset._load_expression_code(SampleMetadata(p_id, sequence, timestep, None)))
            if pose.pose_type != PoseType.CAM_2_WORLD:
                pose = pose.invert()
            view_directions.append(torch.from_numpy(pose.get_look_direction()))

        i_participant = valid_dataset.participant_ids.index(p_id)
        inference_samples = [valid_dataset.process_conditions(rendering, crop=None) for rendering in all_renderings]
        for inference_sample, expression_code, view_direction in zip(inference_samples, expression_codes, view_directions):
            inference_sample.expression_code = expression_code
            inference_sample.view_direction = view_direction

        inference_batch = valid_dataset.collate_fn_inference(inference_samples)
        if use_latent_noise:
            if fix_latents:
                inference_batch.latent_noise = torch.stack([latent_noise for _ in range(len(inference_batch))])
            else:
                latent_shape = model.get_latent_shape()
                latent_noise = torch.randn((len(inference_batch), *latent_shape), device=model.device)
                inference_batch.latent_noise = latent_noise

        if prompt is not None:
            inference_batch.prompts = [prompt for _ in range(len(inference_batch))]

        # Forward all frames
        all_images = []
        inference_outputs = generate_images(model, inference_batch, batch_size, i_participant=i_participant,
                                            disable_neural_texture=disable_neural_texture,
                                            n_inference_steps=n_inference_steps, cfg_weight=cfg_weight)
        for inference_output in inference_outputs:
            all_images.append(inference_output.prediction)

            if inference_output.neural_textured_rendering is not None:
                conditions["neural_textured_rendering"].append(inference_output.neural_textured_rendering[..., -3:])

            if inference_output.previous_outputs is not None and dataset_config.warp_previous_outputs:
                for prev, previous_output in enumerate(inference_output.previous_outputs):
                    conditions[f"previous_output_{prev}"].append(previous_output)

        for rendering_name, single_condition_images in conditions.items():
            render_manager.save_conditions_video(single_condition_images, rendering_name, fps=fps,
                                                 fix_timestep=fix_timestep,
                                                 person=person)
        render_manager.save_generations_video(all_images,
                                              fps=fps,
                                              fix_latents=fix_latents,
                                              fix_timestep=fix_timestep,
                                              seed=seed,
                                              prompt=prompt,
                                              person=person)

        if trajectory_type == 'camera_moveX' or trajectory_type == 'camera_shake':
            render_manager.save_generation_frames(all_images)


def generate_images(diffusion_avatars_model: DiffusionAvatarsModel,
                    inference_batch: RenderingDatasetInferenceBatch,
                    batch_size: int,
                    i_participant: Optional[int] = None,
                    neural_textures: Optional[torch.Tensor] = None,
                    disable_neural_texture: bool = False,
                    n_inference_steps: int = 20,
                    cfg_weight: float = 1) -> List[DiffusionAvatarsInferenceOutput]:

    if neural_textures is not None:
        iter_neural_textures = batchify_sliced(neural_textures, batch_size)

    # Forward all frames
    all_inference_outputs = []
    for batch in tqdm(batchify_sliced(inference_batch, batch_size)):
        if isinstance(batch, list):
            assert len(batch) == 1
            batch = batch[0]

        batch_neural_textures = None
        if neural_textures is not None:
            batch_neural_textures = next(iter_neural_textures)

        if i_participant is not None:
            batch.i_participant = torch.tensor([i_participant for _ in range(len(batch))], dtype=torch.int32,
                                               device=diffusion_avatars_model.device)

        diffusion_avatars_model.prepare_batch(batch, neural_textures=batch_neural_textures)
        if disable_neural_texture and batch.neural_textured_rendering is not None:
            batch.neural_textured_rendering = torch.zeros_like(batch.neural_textured_rendering)
        inference_outputs = diffusion_avatars_model.inference(batch, n_inference_steps=n_inference_steps, cfg_weight=cfg_weight)

        all_inference_outputs.extend(inference_outputs)

    return all_inference_outputs

if __name__ == '__main__':
    tyro.cli(main)
