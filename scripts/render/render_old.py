from collections import defaultdict
from itertools import islice
from math import sqrt, ceil, sin, cos
from typing import List, Optional, Dict, Tuple

import numpy as np
import tinycudann as tcnn
import torch
import tyro
from PIL import Image
from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.camera_bundle import align_poses
from dreifus.graphics import Dimensions, homogenize
from dreifus.image import Img
from dreifus.matrix import Pose, Intrinsics
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from elias.util import save_img
from elias.util.batch import batchify_sliced
from elias.util.random import make_deterministic
from famudy.constants import SERIALS
from famudy.data import FamudySequenceDataManager, FamudyParticipantDataManager
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from diff_vp.config.data.rendering_data import RenderingDataConfig
from diff_vp.constants import TEST_SEQUENCES
from diff_vp.data_manager.rendering_data_manager import RenderingName
from diff_vp.dataset.rendering_dataset import RenderingDataset, Crop, RenderingDatasetInferenceSample, \
    RenderingDatasetInferenceBatch, RenderingDatasetConfig, DatasetSplit, TemporalBatchRenderingDatasetView, \
    SampleMetadata
from diff_vp.model.diff_vp import DiffVP
from diff_vp.model.inference_output import DiffVPInferenceOutput, SchedulerType
from diff_vp.model_manager.finder import find_model_folder
from diff_vp.model_manager.unseen_person_fitting_manager import UnseenPersonFittingManager
from diff_vp.render_manager.diff_vp_render_manager import DiffVPRenderManager, PlotType, DiffVPTrajectory
from diff_vp.renderer.nvdiffrast_renderer import NvDiffrastRenderer
from diff_vp.renderer.provider.expression_animation_provider import ExpressionAnimationManager
from diff_vp.renderer.provider.flame_provider import FlameProvider
from diff_vp.renderer.provider.mesh_provider import TemporalSmoothingConfig, NPHMTrackingVersion
from diff_vp.renderer.provider.nphm_provider import NPHMProvider
from diff_vp.util.flow import colorize_flow
from diff_vp.util.render import get_mesh_provider, get_renderer


def main(run_name: str,
         /,
         checkpoint: int = -1,
         plot_type: PlotType = "valid_hold_out_view",
         batch_size: int = 16,
         fix_latents: bool = True,
         fix_timestep: bool = False,
         seed: Optional[int] = None,
         prompt: Optional[str] = None,
         disable_previous_output: bool = False,
         disable_neural_texture: bool = False,
         use_no_variance_scheduler: bool = False,
         use_ddpm_scheduler: bool = False,
         n_inference_steps: int = 20,
         visualize_controlnet_cond_embeddings: bool = False,
         use_unseen_persons: bool = False,
         enable_extended_temporal_attention: bool = False,
         single_batch: bool = False,
         smooth_expression_condition: bool = False,
         resolution: Optional[int] = None,
         remap_noise: bool = False,
         use_consistency_decoder: bool = False,
         scheduler: SchedulerType = 'ddpm',

         # plot_type specific settings
         n_steps_per_camera: int = 10,
         person: Optional[int] = None,
         camera_distance: float = 1,
         source_actor: Optional[int] = None,
         source_sequence: Optional[str] = None,
         source_animation: Optional[str] = None,
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
    model_config = model_manager.load_model_config()
    train_config = model_manager.load_train_setup()
    dataset_config = model_manager.load_dataset_config()
    data_config = dataset_config.get_data_config()
    data_stats = dataset_config.get_data_statistics()
    if prompt is not None:
        train_config.global_prompt = prompt

    checkpoint = model_manager._resolve_checkpoint_id(checkpoint)
    model = model_manager.load_checkpoint(checkpoint,
                                          enable_extended_temporal_attention=enable_extended_temporal_attention,
                                          use_consistency_decoder=use_consistency_decoder)
    model.cuda()

    if resolution is not None:
        dataset_config.resolution = resolution
        data_config.resolution = resolution
        model._dataset_config.resolution = resolution
    else:
        resolution = dataset_config.resolution

    if remap_noise:
        dataset_config.remap_noise = True
        model._dataset_config.remap_noise = True

    if use_unseen_persons:
        fitting_manager = UnseenPersonFittingManager(run_name, checkpoint)
        neural_texture_lookup = fitting_manager.load_neural_texture_lookup()
        neural_texture_lookup.to(model.device)
        model.neural_texture_lookup = neural_texture_lookup

    controlnet_cond_embeddings = None
    if visualize_controlnet_cond_embeddings and isinstance(model, DiffVP):
        controlnet_cond_embeddings = []

        def collect_controlnet_cond_embedding(module, args: Tuple[torch.Tensor], output: torch.Tensor):
            if not controlnet_cond_embeddings or (
                    torch.stack(controlnet_cond_embeddings[-len(output):]) != output).any():
                controlnet_cond_embeddings.extend(output)

        model.controlnet.controlnet_cond_embedding.register_forward_hook(collect_controlnet_cond_embedding)

    if run_name.startswith('DVP-'):
        model.change_inference_scheduler(scheduler)
    # if use_no_variance_scheduler:
    #     model.inference_pipeline.scheduler = NoVarianceDDPMScheduler.from_config(
    #         model.inference_pipeline.scheduler.config)
    # if use_ddpm_scheduler:
    #     model.inference_pipeline.scheduler = DDPMScheduler.from_config(model.inference_pipeline.scheduler.config)

    if smooth_expression_condition:
        dataset_config.expression_condition_smoothing = TemporalSmoothingConfig(
            temporal_smoothing=5,
            use_gaussian=True,
            param_groups=['expression'])

    render_manager = DiffVPRenderManager(run_name,
                                         checkpoint=checkpoint,
                                         plot_type=plot_type,
                                         disable_previous_output=disable_previous_output,
                                         disable_neural_texture=disable_neural_texture,
                                         n_inference_steps=n_inference_steps,
                                         use_ddpm_scheduler=use_ddpm_scheduler,
                                         use_no_variance_scheduler=use_no_variance_scheduler,
                                         use_consistency_decoder=use_consistency_decoder,
                                         use_unseen_persons=use_unseen_persons,
                                         enable_extended_temporal_attention=enable_extended_temporal_attention,
                                         smooth_expression_condition=smooth_expression_condition,
                                         resolution=resolution,
                                         camera_distance=camera_distance,
                                         source_actor=source_actor,
                                         source_sequence=source_sequence,
                                         source_animation=source_animation,
                                         cfg_weight=cfg_weight,
                                         prompt=prompt,
                                         scheduler=scheduler,
                                         shake=shake,
                                         move_y=move_y,
                                         look_y=look_y,
                                         curve_z=curve_z,
                                         test_sequence=test_sequence,
                                         frequency=frequency)

    if plot_type == 'valid_hold_out_view':
        split = DatasetSplit.VALID_HOLD_OUT_VIEW
    elif plot_type == 'valid_hold_out_expression':
        split = DatasetSplit.VALID_HOLD_OUT_EXP
    elif plot_type == 'valid_hold_out_sequence':
        split = DatasetSplit.VALID_HOLD_OUT_SEQ
    elif plot_type == 'valid_hold_out_person':
        split = DatasetSplit.VALID_HOLD_OUT_PERSON
    elif use_unseen_persons:
        split = DatasetSplit.VALID_HOLD_OUT_PERSON
    elif plot_type == 'noise_predictions':
        split = DatasetSplit.VALID_HOLD_OUT_SEQ
    else:
        split = DatasetSplit.TRAIN
    valid_dataset = RenderingDataset(dataset_config, split=split)

    if single_batch:
        valid_dataset = TemporalBatchRenderingDatasetView(valid_dataset, serials=["222200037", "222200047"],
                                                          max_samples=1)
    elif plot_type == 'noise_predictions':
        valid_dataset = TemporalBatchRenderingDatasetView(valid_dataset, serials=["222200037"])

    if seed is None:
        seed = train_config.seed
    make_deterministic(seed)

    # Latent noise
    use_latent_noise = run_name.startswith('DVP-')
    if use_latent_noise and fix_latents:
        if use_unseen_persons:
            latent_noise = fitting_manager.load_latent_noise()
            latent_noise.to(model.device)
        else:
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=model.device).manual_seed(seed)

            if dataset_config.remap_noise:
                latent_noise = torch.randn(model.get_noise_texture_shape(), device=model.device, generator=generator)
            else:
                latent_shape = model.get_latent_shape()
                latent_noise = torch.randn(latent_shape, device=model.device, generator=generator)
    else:
        latent_noise = None

    if plot_type == 'mosaic_crops':
        crops = []
        for _ in range(16):
            size = np.random.randint(800, 1100)
            offset_x = np.random.randint(0, 1100 - size)
            offset_y = np.random.randint(0, 1604 - size)

            crops.append(Crop(offset_x, offset_y, size, size))

        # TODO: sequence is hardcoded
        renderings = {
            rendering_name: valid_dataset._data_manager.load_rendering(rendering_name, 17, "EXP-1-head", 0, "222200037")
            for rendering_name in dataset_config.rendering_names}

        inference_samples = [valid_dataset.process_conditions(renderings, crop=crop) for crop in crops]
        inference_batch = valid_dataset.collate_fn_inference(inference_samples)
        if fix_latents:
            inference_batch.latent_noise = torch.stack([latent_noise for _ in range(len(inference_batch))])

        all_images = []
        with torch.autocast("cuda"):
            for batch in tqdm(batchify_sliced(inference_batch, batch_size)):
                model.prepare_batch(batch)
                inference_outputs = model.inference(batch, n_inference_steps=n_inference_steps)

                for inference_output in inference_outputs:
                    all_images.append(inference_output.prediction)

        plot_mosaic(all_images, conditions=inference_samples)
    elif plot_type in {'train', 'valid_hold_out_view', 'valid_hold_out_expression', 'valid_hold_out_sequence',
                       'valid_hold_out_person'}:
        max_elements = 16
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      collate_fn=valid_dataset.collate_fn,
                                      shuffle=True)

        all_images = []
        for batch in tqdm(islice(valid_dataloader, int(max_elements / batch_size))):
            if fix_latents:
                batch.latent_noise = torch.stack([latent_noise for _ in range(len(batch))])
            if disable_previous_output:
                batch.previous_outputs = torch.zeros_like(batch.previous_outputs)

            model.prepare_batch(batch)

            inference_outputs = model.inference(batch, n_inference_steps=n_inference_steps)
            for inference_output in inference_outputs:
                img = np.concatenate([inference_output.prediction, inference_output.target_image], axis=1)
                all_images.append(img)

        figure = plot_mosaic(all_images)
        render_manager.save_figure(figure, prompt=prompt)
    elif plot_type == 'noise_predictions':
        n_timesteps = 10
        sample_idx = 24

        inference_sample = valid_dataset[sample_idx]
        inference_batch = valid_dataset.collate_fn([inference_sample])
        model.prepare_batch(inference_batch)

        with torch.no_grad():
            encoded_target_images = model.vae.encode(
                dataset_config.concat_target_images(inference_batch).to(dtype=model._weight_dtype))
            if train_config.use_vae_mean:
                latents = encoded_target_images.latent_dist.mean
            else:
                latents = encoded_target_images.latent_dist.sample()
            latents = latents * model.vae.config.scaling_factor

            noise = torch.randn_like(latents[[0]])
            noise = noise.repeat(1 if dataset_config.temporal_batch_size == 0 else dataset_config.temporal_batch_size, 1, 1, 1)

            n_train_timesteps = model.scheduler.config.num_train_timesteps
            timesteps = torch.linspace(0, n_train_timesteps - 1, n_timesteps, dtype=torch.long,
                                       device=model.device)

            if single_batch:
                all_noisy_latents = [model.scheduler.add_noise(latents, noise, timesteps[i]) for i in
                                     range(n_timesteps)]
                all_timesteps = [timestep.repeat(latents.shape[0]) for timestep in timesteps]
                batch_size = latents.shape[0]
            else:
                all_noisy_latents = [model.scheduler.add_noise(latents, noise, timesteps)]
                all_timesteps = [timesteps]

            conditions = model._dataset_config.concat_conditions(inference_batch)
            controlnet_image = conditions.to(dtype=model._weight_dtype)

            for t in range(n_timesteps):
                noisy_latents = all_noisy_latents[t]
                timesteps = all_timesteps[t]
                all_model_pred = []
                decoded_predictions = []
                for noisy_latents_batch, timesteps_batch in tqdm(zip(batchify_sliced(noisy_latents, batch_size),
                                                                     batchify_sliced(timesteps, batch_size))):
                    model_pred = model.predict_noise(noisy_latents_batch, timesteps_batch, controlnet_image, expression_codes=inference_batch.expression_codes)
                    all_model_pred.extend(model_pred)

                if not single_batch:
                    last_timestep = torch.tensor([n_train_timesteps - 1], device=model.device, dtype=torch.long)
                    pred_from_full_noise = model.predict_noise(
                        noise,
                        last_timestep,
                        controlnet_image,
                        expression_codes=inference_batch.expression_codes)
                    all_model_pred.extend(pred_from_full_noise)
                    n_timesteps += 1
                    timesteps = torch.cat([timesteps, last_timestep])
                    noisy_latents = torch.cat([noisy_latents, noise])

                nrows = 5  # noisy latent, predicted noise, GT noise, noise error, denoised latent, decoded prediction
                ncols = len(noisy_latents)
                fontsize = 20

                plt.figure(figsize=(ncols * 4, nrows * 4))
                for i in range(len(noisy_latents)):
                    row = 0

                    # noisy latents (input)
                    plt.subplot(nrows, ncols, 1 + i + row * ncols)
                    if i < len(noisy_latents) - 1:
                        plt.title(f"{timesteps[i].item()}", fontsize=fontsize)
                    else:
                        plt.title("Full noise", fontsize=fontsize)
                    plt.imshow(Img.from_torch(noisy_latents[i][:3]).to_numpy().img)
                    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                    if i == 0:
                        plt.ylabel("Noisy Latents", fontsize=fontsize)
                    row += 1

                    # # prediction
                    # plt.subplot(nrows, ncols, 1 + i + row * ncols)
                    # plt.imshow(Img.from_torch(all_model_pred[i][:3]).to_numpy().img)
                    # row += 1
                    #
                    # # GT noise
                    # plt.subplot(nrows, ncols, 1 + i + row * ncols)
                    # plt.imshow(Img.from_torch(noise[0][:3]).to_numpy().img)
                    # row += 1

                    # Imitate scheduling
                    alphas_cumprod = model.scheduler.alphas_cumprod.to(device=model.device,
                                                                       dtype=noisy_latents[i].dtype)

                    sqrt_alpha_prod = alphas_cumprod[timesteps[[i]]] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(noisy_latents[i].shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps[[i]]]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latents[i].shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

                    # Noise Error
                    if model.scheduler.config.prediction_type == 'v_prediction':
                        # velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
                        pred_noise = (all_model_pred[i] + sqrt_one_minus_alpha_prod * noisy_latents[
                            i]) / sqrt_alpha_prod

                        pred_noise = sqrt_alpha_prod * all_model_pred[i] + sqrt_one_minus_alpha_prod * noisy_latents[i]
                    else:
                        pred_noise = all_model_pred[i]
                    error = (pred_noise - noise[0]).abs().mean(dim=0)
                    plt.subplot(nrows, ncols, 1 + i + row * ncols)
                    plt.imshow(error.cpu().numpy(), vmax=0.3, vmin=0)
                    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                    if i == 0:
                        plt.ylabel("Noise Error", fontsize=fontsize)
                    row += 1

                    # Denoised latent

                    if model.scheduler.config.prediction_type == 'v_prediction':
                        # velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
                        # noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
                        denoised_latent = (all_model_pred[i] ** 2 + noisy_latents[i] ** 2 - noise[0] ** 2).sqrt()

                        denoised_latent = sqrt_alpha_prod * noisy_latents[i] - sqrt_one_minus_alpha_prod * \
                                          all_model_pred[i]
                    else:
                        denoised_latent = (noisy_latents[i] - sqrt_one_minus_alpha_prod * all_model_pred[
                            i]) / sqrt_alpha_prod

                    plt.subplot(nrows, ncols, 1 + i + row * ncols)
                    plt.imshow(Img.from_torch(denoised_latent[:3]).to_numpy().img)
                    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                    if i == 0:
                        plt.ylabel("Denoised latent", fontsize=fontsize)
                    row += 1

                    # Latent Error
                    error = (denoised_latent - latents[0]).abs().mean(dim=0)
                    plt.subplot(nrows, ncols, 1 + i + row * ncols)
                    plt.imshow(error.cpu().numpy(), vmax=0.3, vmin=0)
                    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                    if i == 0:
                        plt.ylabel("Latent Error", fontsize=fontsize)
                    row += 1

                    # Decoded prediction
                    decoded_prediction = model.vae.decode(
                        denoised_latent[None] / model.vae.config.scaling_factor)
                    # TODO: Is there random noise involved in decoding?
                    decoded_prediction = Img.from_normalized_torch(
                        decoded_prediction.sample[0].clamp(-1, 1)).to_numpy().img
                    decoded_predictions.append(decoded_prediction)

                    plt.subplot(nrows, ncols, 1 + i + row * ncols)
                    plt.imshow(decoded_prediction)
                    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                    if i == 0:
                        plt.ylabel("Decoded prediction", fontsize=fontsize)
                    row += 1
                    save_img(decoded_prediction, f"{render_manager._location}/{plot_type}/{run_name}/decoded_prediction_{timesteps[i]}.png")

                plt.tight_layout()
                figure = fig2img(plt.gcf())
                plt.show()

                render_manager.save_generations_video(decoded_predictions, fps=12, fix_latents=fix_latents,
                                                      fix_timestep=fix_timestep, seed=seed,
                                                      prompt=f"denoising-step-{all_timesteps[t][0].item()}")
                render_manager.save_figure(figure, prompt=f"denoising-step-{all_timesteps[t][0].item()}")
                save_img(Img.from_normalized_torch(inference_batch.target_images[0]).to_numpy().img,
                         f"{render_manager._location}/{plot_type}/{run_name}/target.png")

    elif plot_type == 'random_latents':
        USE_TRAIN_VIEW_AND_EXPRESSION = False
        # Currently, random_latents does not give good renderings. The neural textures apparently are too specific
        n_latents = 16

        resolution = dataset_config.resolution

        iter_sequences = iter(data_stats.available_sequences.items())
        p_id, person_metadata = next(iter_sequences)
        ref_sequence, ref_timesteps = next(iter(person_metadata.items()))
        ref_p_id = int(p_id)

        if USE_TRAIN_VIEW_AND_EXPRESSION:
            mesh_provider = get_mesh_provider(data_config, ref_p_id, ref_sequence)
            renderer = get_renderer(data_config, mesh_provider, Dimensions(resolution, resolution))

            vertices = mesh_provider.get_vertices(0)
            mesh = mesh_provider.create_mesh(vertices, 0)

            data_manager = FamudyParticipantDataManager(ref_p_id)
            calibration_result = data_manager.load_calibration_result().params_result
            pose = calibration_result.get_pose(8)
        else:
            # Using the default neutral FLAME mesh
            flame_provider = FlameProvider(
                config=data_config.mesh_provider_config,
            )
            renderer = NvDiffrastRenderer(
                flame_provider,
                Dimensions(resolution, resolution)
            )
            vertices = flame_provider.get_vertices(0)
            mesh = flame_provider.create_mesh(vertices)

            pose = Pose(np.eye(3),
                        translation=(0, 0, 1),
                        pose_type=PoseType.CAM_2_WORLD,
                        camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)

        intrinsics = Intrinsics(8193, 8193, 1099.5, 1099.5)
        intrinsics.rescale(resolution / 2200)
        renderings = renderer.render(mesh, pose, intrinsics, dataset_config.rendering_names)
        renderings = valid_dataset.normalize_renderings(renderings)

        inference_sample = valid_dataset.process_conditions(renderings, crop=None)
        inference_sample.previous_outputs = [None for _ in range(dataset_config.n_previous_frames)]

        inference_batch = valid_dataset.collate_fn_inference([inference_sample for _ in range(n_latents)])

        all_images = []
        for batch in tqdm(batchify_sliced(inference_batch, batch_size)):
            if dataset_config.use_neural_textures:
                # neural_textures = torch.randn((len(batch), dataset_config.res_neural_textures,
                #                                dataset_config.res_neural_textures, dataset_config.dim_neural_textures))
                neural_textures = model.neural_texture_lookup(torch.tensor(0, device=model.device))
                neural_textures = neural_textures.reshape((1, dataset_config.res_neural_textures,
                                                           dataset_config.res_neural_textures,
                                                           dataset_config.dim_neural_textures))
            else:
                neural_textures = None
            model.prepare_batch(batch, neural_textures=neural_textures)
            inference_outputs = model.inference(batch, n_inference_steps=n_inference_steps)
            for inference_output in inference_outputs:
                all_images.append(inference_output.prediction)

        figure = plot_mosaic(all_images, shared_conditions=inference_sample)
        render_manager.save_figure(figure, prompt=prompt)
        for i, prediction in enumerate(all_images):
            save_img(prediction, f"{render_manager._location}/{plot_type}/{run_name}/image_{i:02d}.png")

    elif plot_type == 'random_flame_expressions':
        resolution = dataset_config.resolution

        fps = 24
        n_keyframes = 10
        n_frames_per_keyframe = 10

        iter_sequences = iter(data_stats.available_sequences.items())
        p_id, person_metadata = next(iter_sequences)
        ref_sequence, ref_timesteps = next(iter(person_metadata.items()))
        ref_p_id = int(p_id)
        ref_flame_provider = FlameProvider.from_dataset(ref_p_id, ref_sequence,
                                                        config=data_config.mesh_provider_config, )

        T = n_keyframes * n_frames_per_keyframe
        # expr_params_keyframes = ref_flame_provider.expr_params[np.linspace(0, len(ref_flame_provider.expr_params) -1, n_keyframes)]
        expr_params_keyframes = np.random.randn(n_keyframes, 100)
        expr_params_keyframes = np.concatenate([expr_params_keyframes, expr_params_keyframes[[0]]],
                                               axis=0)  # Make it loop
        # jaw_params_keyframes = ref_flame_provider.pose_params[:, -3:][np.linspace(0, len(ref_flame_provider.expr_params) -1, n_keyframes)]
        jaw_params_keyframes = np.random.random((n_keyframes, 3)) / 2
        jaw_params_keyframes[:, 0] *= 0.5  # [0, 0.5]
        jaw_params_keyframes[:, 1:] *= 0.05  # [0, 0.05]
        jaw_params_keyframes = np.concatenate([jaw_params_keyframes, jaw_params_keyframes[[0]]], axis=0)  # Make it loop

        expr_params = []
        jaw_params = []
        for i in range(T):
            keyframe_1 = int(i / n_frames_per_keyframe)
            keyframe_2 = keyframe_1 + 1
            alpha = (i % n_frames_per_keyframe) / (n_frames_per_keyframe - 1)

            expr_param = (1 - alpha) * expr_params_keyframes[keyframe_1] + alpha * expr_params_keyframes[keyframe_2]
            expr_params.append(expr_param)

            jaw_param = (1 - alpha) * jaw_params_keyframes[keyframe_1] + alpha * jaw_params_keyframes[keyframe_2]
            jaw_params.append(jaw_param)

        # Using the default neutral FLAME mesh
        flame_provider = FlameProvider(
            shape_params=ref_flame_provider.shape_params,
            rotation=ref_flame_provider.rotation[[0 for _ in range(T)]],
            translation=ref_flame_provider.translation[[0 for _ in range(T)]],
            expr_params=np.stack(expr_params).astype(np.float32),
            jaw_pose=np.stack(jaw_params).astype(np.float32),
            config=data_config.mesh_provider_config,
            separate_transformation=True
        )
        renderer = NvDiffrastRenderer(
            flame_provider,
            Dimensions(resolution, resolution)
        )

        world_2_cam_poses = FamudyParticipantDataManager(ref_p_id).load_calibration_result().params_result.get_poses()
        _, transform = align_poses(world_2_cam_poses, return_transformation=True)

        pose = Pose(np.eye(3),
                    translation=(0, 0, 1),
                    pose_type=PoseType.CAM_2_WORLD,
                    camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)

        transform = Pose.from_euler(ref_flame_provider.rotation[0].numpy(),
                                    ref_flame_provider.translation[0].numpy(), 'XYZ',
                                    pose_type=PoseType.CAM_2_CAM)
        transform[:3, :3] *= ref_flame_provider.scale[0].numpy()

        # It is super important that transform and the pose are in the same camera coordinate convention!
        # transform is OPEN_CV and pose is OPEN_GL
        # Since transform is a CAM_2_CAM matrix, dreifus currently doesn't allow to change the coordinate convention
        # Hence, we have to change pose to OPEN_CV -> transform @ pose -> pose to OPEN_GL
        pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV)
        pose = transform @ pose
        pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)

        # pose to OPEN_GL
        # The hacky pose[:] and setting pose_type are necessary, because we cannot call invert() on pose anymore
        # since it includes a scale. dreifus doesn't support scale on Poses yet, but for us to undo the model
        # transformation in the flame provider, we need to add the scale to the render pose
        pose[:] = np.linalg.inv(pose)
        pose.pose_type = PoseType.WORLD_2_CAM

        intrinsics = Intrinsics(8193, 8193, 1099.5, 1099.5)
        intrinsics.rescale(resolution / 2200)

        inference_samples = []
        conditions = defaultdict(list)
        for t in tqdm(range(T)):
            vertices = flame_provider.get_vertices(t)
            mesh = flame_provider.create_mesh(vertices)
            renderings = renderer.render(mesh, pose, intrinsics, dataset_config.rendering_names)
            renderings = valid_dataset.normalize_renderings(renderings)
            collect_conditions(renderings, data_config, dataset_config, conditions)

            inference_sample = valid_dataset.process_conditions(renderings, crop=None)
            # inference_sample.previous_outputs = [None for _ in range(dataset_config.n_previous_frames)]

            inference_samples.append(inference_sample)

        inference_batch = valid_dataset.collate_fn_inference(inference_samples)
        if fix_latents:
            inference_batch.latent_noise = torch.stack([latent_noise for _ in range(T)])
        i_participant = 0
        # i_participant = torch.zeros((T,), dtype=torch.int, device=model.device)
        # neural_textures = model.neural_texture_lookup(i_participant)
        # neural_textures = neural_textures.reshape(
        #     (T,
        #      dataset_config.res_neural_textures,
        #      dataset_config.res_neural_textures,
        #      dataset_config.dim_neural_textures))

        inference_outputs = generate_images(model, inference_batch, batch_size,
                                            i_participant=i_participant,
                                            # neural_textures=neural_textures,
                                            disable_previous_output=disable_previous_output,
                                            n_inference_steps=n_inference_steps)

        all_images = []
        for inference_output in inference_outputs:
            all_images.append(inference_output.prediction)

            if inference_output.neural_textured_rendering is not None:
                conditions["neural_textured_rendering"].append(inference_output.neural_textured_rendering[..., -3:])

            if inference_output.previous_outputs is not None and dataset_config.warp_previous_outputs:
                for prev, previous_output in enumerate(inference_output.previous_outputs):
                    conditions[f"previous_output_{prev}"].append(previous_output)

        for rendering_name, single_condition_images in conditions.items():
            render_manager.save_conditions_video(single_condition_images, rendering_name, fps=fps,
                                                 fix_timestep=fix_timestep)

        render_manager.save_generations_video(all_images,
                                              fps=fps,
                                              fix_latents=fix_latents,
                                              fix_timestep=fix_timestep,
                                              seed=seed,
                                              prompt=prompt,
                                              person=person)

    elif plot_type == 'shape_interpolation':
        seconds = 4
        fps = 24
        n_interpolations = 5
        interpolate_textures = False

        interpolation_target_frames = np.linspace(0, fps * seconds - 1, n_interpolations, dtype=int)
        interpolation_participants = [int(p_id) for p_id in
                                      list(data_stats.available_sequences.keys())[:n_interpolations]]

        serial = "222200037"

        # data_manager = RenderingDataManager(dataset_config.dataset_version)

        iter_sequences = iter(data_stats.available_sequences.items())
        p_id, person_metadata = next(iter_sequences)
        ref_sequence, ref_timesteps = next(iter(person_metadata.items()))
        ref_p_id = int(p_id)

        famudy_data_manager = FamudyParticipantDataManager(ref_p_id)
        pose = famudy_data_manager.load_calibration_result().params_result.get_pose(SERIALS.index(serial))
        intrinsics = famudy_data_manager.load_calibration_result().params_result.get_intrinsics()
        intrinsics.rescale(resolution / 2200)
        intrinsics.cx = resolution / 2
        intrinsics.cy = resolution / 2

        ref_flame_provider = FlameProvider.from_dataset(
            ref_p_id,
            ref_sequence,
            config=data_config.mesh_provider_config, )

        ref_renderer = NvDiffrastRenderer(ref_flame_provider,
                                          Dimensions(dataset_config.resolution, dataset_config.resolution))

        all_renderings = []
        conditions = defaultdict(list)
        interpolated_neural_textures = []
        shape_params_1 = None
        shape_params_2 = None
        previous_interpolation_phase = None
        for frame_id in tqdm(range(seconds * fps), desc="Generate conditions"):
            # interpolation_target_frames:  0| 11| 22| 33|
            # interpolation_phase:           | 0 | 1 | 2 |
            #
            interpolation_phase = (frame_id > interpolation_target_frames).sum() - 1

            if frame_id == 0:
                participant_id_1 = interpolation_participants[0]
                participant_id_2 = interpolation_participants[0]
                alpha = 1
            else:
                frame_id_1 = interpolation_target_frames[interpolation_phase]
                frame_id_2 = interpolation_target_frames[interpolation_phase + 1]
                participant_id_1 = interpolation_participants[interpolation_phase]
                participant_id_2 = interpolation_participants[interpolation_phase + 1]
                alpha = (frame_id - frame_id_1) / (frame_id_2 - frame_id_1)

            sequence_dict_1 = data_stats.available_sequences[f"{participant_id_1}"]
            sequence_dict_2 = data_stats.available_sequences[f"{participant_id_2}"]
            sequence_name_1 = list(sequence_dict_1.keys())[0]
            sequence_name_2 = list(sequence_dict_2.keys())[0]

            if previous_interpolation_phase is None or previous_interpolation_phase != interpolation_phase:
                # Need to interpolate between different people now
                flame_provider_1 = FlameProvider.from_dataset(
                    participant_id_1,
                    sequence_name_1,
                    config=data_config.mesh_provider_config, )
                flame_provider_2 = FlameProvider.from_dataset(
                    participant_id_2,
                    sequence_name_2,
                    config=data_config.mesh_provider_config, )

                shape_params_1 = flame_provider_1.shape_params
                shape_params_2 = flame_provider_2.shape_params
                previous_interpolation_phase = interpolation_phase

            interpolated_shape_params = (1 - alpha) * shape_params_1 + alpha * shape_params_2

            ref_flame_provider.shape_params = interpolated_shape_params
            vertices = ref_flame_provider.get_vertices(ref_timesteps[0])
            mesh = ref_flame_provider.create_mesh(vertices)

            renderings = ref_renderer.render(mesh, pose, intrinsics, dataset_config.rendering_names)
            renderings = valid_dataset.normalize_renderings(renderings)

            for rendering_name, rendering in renderings.items():
                if rendering_name in dataset_config.rendering_names \
                        or dataset_config.use_neural_textures \
                        and (rendering_name == RenderingName.UV or rendering_name == RenderingName.MASK):
                    if rendering_name == RenderingName.NORMALS:
                        if dataset_config.dataset_version == 'v0.6':
                            # TODO: Issue in v0.6: normals were rendered in [0, 1] and then normalized which changes them
                            #   Now, we render them correctly in [-1, 1] but to still inference old models, we have
                            #   have to imitate the normalization here
                            render_mask = np.any(rendering != 0, axis=-1)
                            rendering = (rendering + 1) / 2
                            rendering[~render_mask] = 0
                            rendering[render_mask] /= np.linalg.norm(rendering[render_mask], axis=-1)[..., None]
                            rendering[~render_mask] = 0
                            rendering = rendering * 2 - 1
                            renderings[rendering_name] = rendering

                    if len(rendering.shape) > 2 and rendering.shape[-1] == 2:
                        rendering = np.concatenate([rendering,
                                                    -1 * np.ones(rendering.shape[:-1] + (1,),
                                                                 dtype=rendering.dtype)],
                                                   axis=-1)  # Add artificial 3rd channel for .png storage
                    elif len(rendering.shape) > 2 and rendering.shape[-1] > 3:
                        rendering = rendering[..., :3]

                    conditions[rendering_name].append((np.squeeze(rendering) + 1) / 2)

            all_renderings.append(renderings)

            if dataset_config.use_neural_textures:
                if interpolate_textures:
                    i_participant_1 = valid_dataset.participant_ids.index(participant_id_1)
                    i_participant_2 = valid_dataset.participant_ids.index(participant_id_2)
                    neural_texture_1 = model.neural_texture_lookup(
                        torch.tensor(i_participant_1, device=model.device))
                    neural_texture_2 = model.neural_texture_lookup(
                        torch.tensor(i_participant_2, device=model.device))

                    interpolated_neural_texture = (1 - alpha) * neural_texture_1 + alpha * neural_texture_2
                else:
                    i_participant = valid_dataset.participant_ids.index(ref_p_id)
                    interpolated_neural_texture = model.neural_texture_lookup(
                        torch.tensor(i_participant, device=model.device))

                interpolated_neural_texture = interpolated_neural_texture.reshape(
                    (dataset_config.res_neural_textures, dataset_config.res_neural_textures,
                     dataset_config.dim_neural_textures))
                interpolated_neural_textures.append(interpolated_neural_texture)

        inference_samples = [valid_dataset.process_conditions(rendering, crop=None) for rendering in all_renderings]
        inference_batch = valid_dataset.collate_fn_inference(inference_samples)
        if use_latent_noise and fix_latents:
            inference_batch.latent_noise = torch.stack([latent_noise for _ in range(len(inference_batch))])

        if interpolated_neural_textures:
            interpolated_neural_textures = torch.stack(interpolated_neural_textures)
            # iter_neural_textures = batchify_sliced(interpolated_neural_textures, batch_size)

        # Forward all frames
        all_images = []
        inference_outputs = generate_images(model, inference_batch, batch_size,
                                            neural_textures=interpolated_neural_textures,
                                            disable_previous_output=disable_previous_output,
                                            n_inference_steps=n_inference_steps)
        for inference_output in inference_outputs:
            all_images.append(inference_output.prediction)

            if inference_output.neural_textured_rendering is not None:
                conditions["neural_textured_rendering"].append(inference_output.neural_textured_rendering[..., -3:])

        # Forward all frames
        # all_images = []
        # for batch in tqdm(batchify_sliced(inference_batch, batch_size)):
        #     neural_textures = None
        #     if dataset_config.use_neural_textures:
        #         neural_textures = next(iter_neural_textures)
        #
        #     diff_vp_model.prepare_batch(batch, neural_textures=neural_textures)
        #
        #     inference_outputs = diff_vp_model.inference(batch)
        #     for inference_output in inference_outputs:
        #         all_images.append(inference_output.prediction)
        #
        #         if inference_output.neural_textured_rendering is not None:
        #             conditions["neural_textured_rendering"].append(inference_output.neural_textured_rendering[..., :3])

        for rendering_name, single_condition_images in conditions.items():
            render_manager.save_conditions_video(single_condition_images, rendering_name, fps=fps,
                                                 fix_timestep=fix_timestep)
        render_manager.save_generations_video(all_images,
                                              fps=fps,
                                              fix_latents=fix_latents,
                                              fix_timestep=fix_timestep,
                                              seed=seed,
                                              prompt=prompt)

    elif plot_type in {'camera_moveX', 'camera_shake', 'camera_interpolation', 'camera'}:
        seconds = 4
        fps = 24

        p_id = valid_dataset.participant_ids[0 if person is None else person]
        # iter_sequences = iter(data_stats.available_sequences.items())
        # for _ in range(1 if person is None else person + 1):
        #     # Take the n-th person
        #     p_id, person_metadata = next(iter_sequences)
        person_metadata = data_stats.available_sequences[str(p_id)]
        p_id = int(p_id)
        if test_sequence is None:
            sequence, timesteps = next(iter(person_metadata.items()))
        else:
            sequence = TEST_SEQUENCES[p_id][test_sequence]
            data_manager = FamudySequenceDataManager(p_id, sequence)
            timesteps = data_manager.get_timesteps()

        expression_animation_manager = None
        resolution = dataset_config.resolution
        if source_actor is not None:
            data_manager = FamudySequenceDataManager(source_actor, source_sequence)
            timesteps = data_manager.get_timesteps()
            # TODO: Initial cross-reenactment done with NPHM_temp_wNeck_wLMs. This might change
            if source_actor in {18, 37}:
                # 18 and 37 still use old tracking
                valid_dataset._data_config.mesh_provider_config.nphm_tracking = NPHMTrackingVersion.NPHM_TEMPORAL_NECK
            else:
                valid_dataset._data_config.mesh_provider_config.nphm_tracking = NPHMTrackingVersion.NPHM_TEMPORAL_NECK_LMS
            seconds = len(timesteps) / fps  # We want to render the full sequence actually
        elif source_animation is not None:
            expression_animation_manager = ExpressionAnimationManager(source_animation, skip=2)
            timesteps = expression_animation_manager.get_timesteps()
            seconds = len(timesteps) / fps
            data_manager = None
        else:
            data_manager = FamudySequenceDataManager(p_id, sequence)

        if data_manager is not None:
            world_2_cam_poses = data_manager.load_calibration_result().params_result.get_poses()

        # Create trajectory
        intrinsics = None
        trajectory = []
        if plot_type == 'camera_interpolation':
            intrinsics = data_manager.load_calibration_result().params_result.get_intrinsics()
            # That way, the bottom part will be cropped (because we only render a square image)
            # This is very similar to the current cropping in the RenderingDataset
            intrinsics.rescale(resolution / 2200)

            for pose_1, pose_2 in zip(world_2_cam_poses[:-1], world_2_cam_poses[1:]):
                t1 = pose_1.get_translation()
                t2 = pose_2.get_translation()

                r1 = pose_1.get_rodriguez_vector()
                r2 = pose_2.get_rodriguez_vector()

                for i in range(n_steps_per_camera):
                    alpha = i / n_steps_per_camera
                    t = (1 - alpha) * t1 + alpha * t2
                    r = (1 - alpha) * r1 + alpha * r2

                    trajectory.append(
                        Pose.from_rodriguez(r, t,
                                            camera_coordinate_convention=pose_1.camera_coordinate_convention,
                                            pose_type=pose_1.pose_type))

        elif plot_type == 'camera_moveX':
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

        elif plot_type == 'camera_shake':
            if source_actor is None and source_animation is None:
                data_manager = FamudySequenceDataManager(p_id, sequence)
                timesteps = data_manager.get_timesteps()

            seconds = len(timesteps) / fps  # We want to render the full sequence actually

            if source_animation is None:
                intrinsics = data_manager.load_calibration_result().params_result.get_intrinsics()
            else:
                intrinsics = FamudyParticipantDataManager(p_id).load_calibration_result().params_result.get_intrinsics()
            intrinsics.rescale(resolution / 2200)
            # Ensure that principle point is still in image center
            intrinsics.cx = resolution / 2
            intrinsics.cy = resolution / 2

            if source_actor is not None:
                nphm_provider = NPHMProvider(source_actor, source_sequence, mesh_provider_config=data_config.mesh_provider_config, target_actor=p_id)
            elif source_animation is not None:
                nphm_provider = NPHMProvider(p_id, sequence, mesh_provider_config=data_config.mesh_provider_config, source_animation=source_animation)
            else:
                nphm_provider = NPHMProvider(p_id, sequence, mesh_provider_config=data_config.mesh_provider_config)
            world_2_nphm = nphm_provider.get_world_2_nphm_pose(0)

            trajectory = []
            for i in range(int(seconds * fps)):
                x = sin(2 * (i / frequency) * np.pi) * shake  # [-0.1, 0.1]
                y = move_y
                z = cos(2 * 2 * (i / frequency) * np.pi) * curve_z + camera_distance
                # z = camera_distance
                camera_position = Vec3(x, y, z)

                pose = Pose(np.eye(3), camera_position, pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                pose.look_at(Vec3(0, y if look_y is None else look_y, 0), up=Vec3(0, 1, 0))  # Look at head
                pose = np.linalg.inv(world_2_nphm) @ pose.numpy()
                pose_2 = Pose(pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                pose_2[:] = pose
                # pose_2 = pose_2.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
                # pose_2 = pose_2.invert()
                trajectory.append(pose_2)

        elif plot_type == 'camera':
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
            raise NotImplementedError(f"Unknown plot type: {plot_type}")

        idx_timesteps = np.linspace(0, len(timesteps) - 1, len(trajectory), dtype=int)
        chosen_timesteps = [timesteps[0] if fix_timestep else timesteps[i] for i in idx_timesteps]

        # TODO: Also incorporate PyVista Flame rendering?

        # Nvdiffrast rendering
        render_size = Dimensions(resolution, resolution)
        if data_config.use_nphm:
            if source_actor is not None:
                mesh_provider = NPHMProvider(source_actor, source_sequence, mesh_provider_config=data_config.mesh_provider_config, target_actor=p_id)
            elif source_animation is not None:
                mesh_provider = NPHMProvider(p_id, sequence,
                                             mesh_provider_config=data_config.mesh_provider_config, source_animation=source_animation)
            else:
                mesh_provider = NPHMProvider(p_id, sequence, mesh_provider_config=data_config.mesh_provider_config)
            renderer = NvDiffrastRenderer(mesh_provider, render_size)
        else:
            mesh_provider = FlameProvider.from_dataset(p_id, sequence, config=data_config.mesh_provider_config, )
            renderer = NvDiffrastRenderer(mesh_provider, render_size)

        # Important to save trajectory before source_world_2_target_world is applied. This is only done for fixing the normals
        # But we do not care about the normals when we use the trajectory config
        trajectory_config = DiffVPTrajectory(trajectory, intrinsics, sequence, chosen_timesteps)
        render_manager.save_trajectory(trajectory_config)

        source_world_2_target_world = None
        if source_actor is not None:
            source_world_2_cam_poses = data_manager.load_calibration_result().params_result.get_poses()
            target_world_2_cam_poses = FamudyParticipantDataManager(p_id).load_calibration_result().params_result.get_poses()
            _, source_world_2_aligned = align_poses(source_world_2_cam_poses, return_transformation=True)
            _, target_world_2_aligned = align_poses(target_world_2_cam_poses, return_transformation=True)
            source_world_2_target_world = target_world_2_aligned.invert() @ source_world_2_aligned

            trajectory = [source_world_2_target_world @ pose for pose in trajectory]

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
                #pose = source_world_2_target_world @ pose
                vertices = (homogenize(vertices) @ source_world_2_target_world.T)[..., :3]
            mesh = mesh_provider.create_mesh(vertices, timestep)
            # if source_world_2_target_world is not None:
            #     mesh.vertex_normals = mesh.vertex_normals @ source_world_2_target_world[:3, :3].T
            # unchanged_normals = flame_provider.create_flame_mesh(flame_vertices).vertex_normals
            # flame_vertices = torch.hstack([flame_vertices[0], torch.ones(flame_vertices[0].shape[0], 1)]) @ transform.T
            # flame_vertices = flame_vertices[None, ..., :3]
            # flame_mesh = flame_provider.create_flame_mesh(flame_vertices)
            # flame_mesh.vertex_normals = unchanged_normals

            rendering_names = dataset_config.rendering_names
            if dataset_config.use_ambient_textures and RenderingName.UV_AMBIENT not in rendering_names:
                rendering_names.append(RenderingName.UV_AMBIENT)

            pose_2 = Pose(pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
            pose_2[:] = pose
            pose_2 = pose_2.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
            pose = pose_2.invert()
            renderings, rast, uv_da, uvw_da = renderer.render(mesh, pose, intrinsics, rendering_names, return_rast_and_derivatives=True)

            if dataset_config.use_mipmapping:
                renderings[RenderingName.UV_DA] = uv_da.cpu().numpy()[0]
                if uvw_da is not None:
                    renderings[RenderingName.UVW_DA] = uvw_da.cpu().numpy()[0]

            if dataset_config.warp_previous_outputs and not disable_previous_output:
                for t, (previous_mesh, previous_pose) in enumerate(
                        zip(reversed(previous_meshes[-dataset_config.n_previous_frames:]),
                            reversed(previous_poses[-dataset_config.n_previous_frames:]))):
                    mesh_provider.prepare_optical_flow(previous_mesh, mesh, previous_pose, pose, intrinsics)
                    backward_flow_renderings = renderer.render(mesh, pose, intrinsics, [RenderingName.BACKWARD_FLOW])
                    renderings[RenderingName.backward_flow(t)] = backward_flow_renderings[RenderingName.BACKWARD_FLOW]

            if dataset_config.warp_previous_outputs and not disable_previous_output:
                for t, (previous_mesh, previous_pose) in enumerate(
                        zip(reversed(previous_meshes[-dataset_config.n_previous_frames:]),
                            reversed(previous_poses[-dataset_config.n_previous_frames:]))):
                    # if previous_mesh is None:
                    #     # Use 0 flow, otherwise samples cannot be packed into a batch
                    #     renderings[RenderingName.FORWARD_FLOW] = np.zeros(
                    #         (dataset_config.resolution, dataset_config.resolution, 2))
                    #     renderings[RenderingName.BACKWARD_FLOW] = np.zeros(
                    #         (dataset_config.resolution, dataset_config.resolution, 2))
                    # else:
                    #     renderings_previous = renderer.render(previous_mesh, previous_pose, intrinsics,
                    #                                           [RenderingName.FORWARD_FLOW])
                    #     renderings[RenderingName.FORWARD_FLOW] = renderings_previous[RenderingName.FORWARD_FLOW]

                    renderings_previous = renderer.render(previous_mesh, previous_pose, intrinsics,
                                                          [RenderingName.FORWARD_FLOW])
                    renderings[RenderingName.forward_flow(t)] = renderings_previous[RenderingName.FORWARD_FLOW]

                for t in range(dataset_config.n_previous_frames - len(previous_meshes)):
                    # Use 0 flow, otherwise samples cannot be packed into a batch
                    renderings[RenderingName.forward_flow(len(previous_meshes) + t)] = np.zeros(
                        (dataset_config.resolution, dataset_config.resolution, 2))
                    renderings[RenderingName.backward_flow(len(previous_meshes) + t)] = np.zeros(
                        (dataset_config.resolution, dataset_config.resolution, 2))

            renderings = valid_dataset.normalize_renderings(renderings)

            for rendering_name, rendering in renderings.items():
                if rendering_name in dataset_config.rendering_names \
                        or dataset_config.use_neural_textures \
                        and (rendering_name == RenderingName.UV or rendering_name == RenderingName.MASK):
                    if rendering_name == RenderingName.NORMALS:
                        if dataset_config.dataset_version == 'v0.6':
                            # TODO: Issue in v0.6: normals were rendered in [0, 1] and then normalized which changes them
                            #   Now, we render them correctly in [-1, 1] but to still inference old models, we have
                            #   have to imitate the normalization here
                            render_mask = np.any(rendering != 0, axis=-1)
                            rendering = (rendering + 1) / 2
                            rendering[~render_mask] = 0
                            rendering[render_mask] /= np.linalg.norm(rendering[render_mask], axis=-1)[..., None]
                            rendering[~render_mask] = 0
                            rendering = rendering * 2 - 1
                            renderings[rendering_name] = rendering

                    elif rendering_name == RenderingName.UV:
                        if not data_config.use_nvdiffrast:
                            # In older datasets, we used 3 channels for UV instead of just two
                            renderings[rendering_name] = np.concatenate(
                                [rendering, np.zeros((rendering.shape[0], rendering.shape[1], 1))], axis=-1)

                    elif rendering_name.is_flow():
                        rendering = colorize_flow(rendering)

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
            if source_actor is not None:
                expression_codes.append(valid_dataset._load_expression_code(SampleMetadata(source_actor, source_sequence, timestep, None)))
            elif source_animation is not None:
                expression_code = expression_animation_manager.load_expression_code(timestep)
                if dataset_config.include_eye_condition:
                    # We don't have any eyes for NPHM animations, because they come from FLAME
                    expression_code = np.concatenate([expression_code, np.zeros(6)])
                expression_codes.append(torch.tensor(expression_code, dtype=torch.float32))
            else:
                expression_codes.append(valid_dataset._load_expression_code(SampleMetadata(p_id, sequence, timestep, None)))
            if pose.pose_type != PoseType.CAM_2_WORLD:
                pose = pose.invert()
            view_directions.append(torch.from_numpy(pose.get_look_direction()))

        # import pyvista as pv
        # p = pv.Plotter()
        # add_coordinate_axes(p, scale=0.1)
        # for pose in trajectory:
        #     add_camera_frustum(p, pose, intrinsics, img_w=resolution, img_h=resolution)
        # p.add_mesh(flame_mesh)
        #
        # p.show()

        i_participant = valid_dataset.participant_ids.index(p_id)
        inference_samples = [valid_dataset.process_conditions(rendering, crop=None) for rendering in all_renderings]
        for inference_sample, expression_code, view_direction in zip(inference_samples, expression_codes, view_directions):
            inference_sample.expression_code = expression_code
            inference_sample.view_direction = view_direction

        if dataset_config.use_autoregressive:
            # In the beginning, we do not have any previous outputs for autoregressive inference => None
            for inference_sample in inference_samples:
                inference_sample.previous_outputs = [None for _ in range(dataset_config.n_previous_frames)]

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
                                            disable_previous_output=disable_previous_output,
                                            disable_neural_texture=disable_neural_texture,
                                            n_inference_steps=n_inference_steps, cfg_weight=cfg_weight)
        for inference_output in inference_outputs:
            all_images.append(inference_output.prediction)

            if inference_output.neural_textured_rendering is not None:
                conditions["neural_textured_rendering"].append(inference_output.neural_textured_rendering[..., -3:])

            if inference_output.previous_outputs is not None and dataset_config.warp_previous_outputs:
                for prev, previous_output in enumerate(inference_output.previous_outputs):
                    conditions[f"previous_output_{prev}"].append(previous_output)

            if remap_noise:
                conditions["noise_input"].append(inference_output.latent_noise[..., :3])

        if visualize_controlnet_cond_embeddings and controlnet_cond_embeddings is not None:
            controlnet_cond_embeddings_flattened = torch.stack(controlnet_cond_embeddings).permute(0, 2, 3, 1).reshape(
                -1, 320).cpu().numpy()
            pca = PCA(3)
            controlnet_cond_embeddings_pca_flattened = pca.fit_transform(controlnet_cond_embeddings_flattened)
            controlnet_cond_embeddings_pca = controlnet_cond_embeddings_pca_flattened.reshape(
                len(controlnet_cond_embeddings), 64, 64, 3)
            controlnet_cond_embeddings_pca = (controlnet_cond_embeddings_pca + 2.5) / 5
            conditions[f"controlnet_cond_embedding_pca"] = controlnet_cond_embeddings_pca
            for controlnet_cond_embedding in controlnet_cond_embeddings:
                controlnet_cond_embedding = (controlnet_cond_embedding.permute(1, 2, 0)[...,
                                             :3].cpu().numpy() + 2.5) / 5
                conditions[f"controlnet_cond_embedding_1"].append(controlnet_cond_embedding)

        # all_images = []
        # for batch in tqdm(batchify_sliced(inference_batch, batch_size)):
        #     batch.i_participant = torch.tensor([i_participant for _ in range(len(batch))], dtype=torch.int32,
        #                                        device=diff_vp_model.device)
        #     diff_vp_model.prepare_batch(batch)
        #
        #     inference_outputs = diff_vp_model.inference(batch)
        #     for inference_output in inference_outputs:
        #         all_images.append(inference_output.prediction)
        #
        #         if inference_output.neural_textured_rendering is not None:
        #             conditions["neural_textured_rendering"].append(inference_output.neural_textured_rendering[..., :3])

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

        if plot_type == 'camera' and source_actor is not None:
            source_data_manager = FamudySequenceDataManager(source_actor, source_sequence)
            source_images = [source_data_manager.load_image(timestep, serial) for timestep in chosen_timesteps]
            render_manager.save_conditions_video(source_images, "DRIVER", fps=fps,
                                                 fix_timestep=fix_timestep,
                                                 person=person)

            render_manager.save_generation_frames(all_images)
        elif plot_type == 'camera_moveX' or plot_type == 'camera_interpolation' or plot_type == 'camera_shake':
            render_manager.save_generation_frames(all_images)


def generate_images(diff_vp_model: DiffVP,
                    inference_batch: RenderingDatasetInferenceBatch,
                    batch_size: int,
                    i_participant: Optional[int] = None,
                    neural_textures: Optional[torch.Tensor] = None,
                    neural_texture_fields: Optional[List[tcnn.Encoding]] = None,
                    disable_previous_output: bool = False,
                    disable_neural_texture: bool = False,
                    n_inference_steps: int = 20,
                    cfg_weight: float = 1) -> List[DiffVPInferenceOutput]:
    dataset_config = diff_vp_model._dataset_config

    if neural_textures is not None:
        iter_neural_textures = batchify_sliced(neural_textures, batch_size)

    if neural_texture_fields is not None:
        iter_neural_texture_fields = batchify_sliced(neural_texture_fields, batch_size)

    if dataset_config.temporal_batch_size > 0:
        n_packed_batches = int(ceil(len(inference_batch) / dataset_config.temporal_batch_size))
        packed_batches = []

        for i_batch in range(n_packed_batches):
            idx = (i_batch + 1) * dataset_config.temporal_batch_size - 1
            packed_batch = inference_batch[idx]
            current_packed_batch = packed_batch
            for i in range(1, dataset_config.temporal_batch_size):
                current_packed_batch.previous_batch = inference_batch[idx - i]
                current_packed_batch.previous_sample_ids = torch.tensor([0], dtype=torch.int)
                current_packed_batch = current_packed_batch.previous_batch
            packed_batches.append(packed_batch)

        inference_batch = packed_batches
        batch_size = 1

    # Forward all frames
    all_inference_outputs = []
    for batch in tqdm(batchify_sliced(inference_batch, batch_size)):
        if isinstance(batch, list):
            assert len(batch) == 1
            batch = batch[0]

        batch_neural_textures = None
        if neural_textures is not None:
            batch_neural_textures = next(iter_neural_textures)

        batch_neural_texture_fields = None
        if neural_texture_fields is not None:
            batch_neural_texture_fields = next(iter_neural_texture_fields)

        if i_participant is not None:
            batch.i_participant = torch.tensor([i_participant for _ in range(len(batch))], dtype=torch.int32,
                                               device=diff_vp_model.device)

        if dataset_config.use_autoregressive:
            assert batch_size == 1, "Autoregressive inference is only implemented with batch_size=1"
            previous_outputs = []
            for t in range(dataset_config.n_previous_frames):
                if len(all_inference_outputs) > t and not disable_previous_output:
                    previous_output = all_inference_outputs[-t - 1].prediction
                    previous_output = Img.from_numpy(previous_output).to_normalized_torch().img
                else:
                    previous_output = torch.zeros((3, dataset_config.resolution, dataset_config.resolution))
                previous_outputs.append(previous_output)
            batch.previous_outputs = torch.stack(previous_outputs)[None]  # [1, N, C, H, W]
            if dataset_config.warp_previous_outputs and not disable_previous_output:
                batch.warp_previous_outputs(occlusion_mask_threshold=dataset_config.occlusion_mask_threshold)

        # TODO: originally, n_previous_frames=0 is necessary to prevent prepare_batch() from trying to add neural textures to previous_batch (which will be None here
        diff_vp_model.prepare_batch(batch, neural_textures=batch_neural_textures,
                                    neural_texture_fields=batch_neural_texture_fields)
        if disable_neural_texture and batch.neural_textured_rendering is not None:
            batch.neural_textured_rendering = torch.zeros_like(batch.neural_textured_rendering)
        inference_outputs = diff_vp_model.inference(batch, n_inference_steps=n_inference_steps, cfg_weight=cfg_weight)

        # with torch.no_grad():
        #     conditions = diff_vp_model._dataset_config.concat_conditions(batch)
        #     controlnet_image = conditions.to(dtype=diff_vp_model._weight_dtype)
        #
        #     n_train_timesteps = diff_vp_model.scheduler.config.num_train_timesteps
        #     timesteps = torch.tensor([n_train_timesteps - 1 for _ in range(controlnet_image.shape[0])], dtype=torch.long,
        #                              device=diff_vp_model.device)
        #     latent_noise = batch.latent_noise.repeat_interleave(diff_vp_model._dataset_config.temporal_batch_size, dim=0)
        #
        #     model_pred = diff_vp_model.predict_noise(latent_noise, timesteps, controlnet_image)
        #
        #     sqrt_alpha_prod = diff_vp_model.scheduler.alphas_cumprod.cuda()[timesteps[[0]]] ** 0.5
        #     sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        #     while len(sqrt_alpha_prod.shape) < len(latent_noise.shape):
        #         sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        #
        #     sqrt_one_minus_alpha_prod = (1 - diff_vp_model.scheduler.alphas_cumprod.cuda()[timesteps[[0]]]) ** 0.5
        #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        #     while len(sqrt_one_minus_alpha_prod.shape) < len(latent_noise.shape):
        #         sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        #     denoised_latent = sqrt_alpha_prod * latent_noise - sqrt_one_minus_alpha_prod * model_pred
        #     decoded_predictions = [diff_vp_model.vae.decode(denoised_latent[[i]] / diff_vp_model.vae.config.scaling_factor) for i in range(len(denoised_latent))]
        #     decoded_predictions = [Img.from_normalized_torch(decoded_prediction.sample[0].clamp(-1, 1)).to_numpy().img for decoded_prediction in decoded_predictions]
        #     inference_outputs = []
        #     for i in range(len(decoded_predictions)):
        #         inference_outputs.append(DiffVPInferenceOutput(prediction=decoded_predictions[i], conditions=dict()))

        all_inference_outputs.extend(inference_outputs)

    return all_inference_outputs


def concat_conditions(conditions: RenderingDatasetInferenceSample) -> np.ndarray:
    image_row = []
    for rendering_name, condition in conditions.renderings.items():
        condition = ((condition.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        if condition.shape[-1] == 1:
            condition = np.repeat(condition, 3, axis=-1)
        if condition.shape[-1] == 2:
            condition = np.concatenate(
                [condition, np.zeros((condition.shape[0], condition.shape[1], 1), dtype=np.uint8)], axis=-1)
        image_row.append(condition)

    img = np.concatenate(image_row, axis=1)
    return img


def fig2img(fig: Optional = None) -> np.ndarray:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    if fig is None:
        fig = plt.gcf()

    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img = np.asarray(Image.open(buf))
    return img[..., :3]


def plot_mosaic(images,
                conditions: Optional[List[RenderingDatasetInferenceSample]] = None,
                shared_conditions: Optional[RenderingDatasetInferenceSample] = None) -> np.ndarray:
    ncols = ceil(sqrt(len(images)))
    nrows = ceil(len(images) / ncols)
    aspect_ratio = images[0].shape[0] / images[0].shape[1]  # h/w

    if shared_conditions is not None:
        nrows += 1

    plt.figure(figsize=(ncols * 2, int(nrows * 2 * aspect_ratio)), dpi=500)

    if shared_conditions is not None:
        img = concat_conditions(shared_conditions)
        plt.subplot2grid((nrows, ncols), (0, 0), colspan=ncols)
        plt.imshow(img)
        plt.axis('off')

    for i, img in enumerate(images):
        if shared_conditions is not None:
            i += ncols

        plt.subplot(nrows, ncols, i + 1)
        if conditions is not None:
            image_row = []
            for rendering_name, condition in conditions[i].renderings.items():
                condition = ((condition.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                if condition.shape[-1] == 1:
                    condition = np.repeat(condition, 3, axis=-1)
                image_row.append(condition)
            image_row.append(img)

            img = np.concatenate(image_row, axis=1)
        plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()

    figure = fig2img(plt.gcf())
    plt.show()

    return figure


def collect_conditions(renderings: Dict[RenderingName, np.ndarray],
                       data_config: RenderingDataConfig,
                       dataset_config: RenderingDatasetConfig,
                       conditions: Dict[RenderingName, List[np.ndarray]]):
    for rendering_name, rendering in renderings.items():
        if rendering_name in dataset_config.rendering_names \
                or dataset_config.use_neural_textures \
                and (rendering_name == RenderingName.UV or rendering_name == RenderingName.MASK):
            if rendering_name == RenderingName.NORMALS:
                if dataset_config.dataset_version == 'v0.6':
                    # TODO: Issue in v0.6: normals were rendered in [0, 1] and then normalized which changes them
                    #   Now, we render them correctly in [-1, 1] but to still inference old models, we have
                    #   have to imitate the normalization here
                    render_mask = np.any(rendering != 0, axis=-1)
                    rendering = (rendering + 1) / 2
                    rendering[~render_mask] = 0
                    rendering[render_mask] /= np.linalg.norm(rendering[render_mask], axis=-1)[..., None]
                    rendering[~render_mask] = 0
                    rendering = rendering * 2 - 1
                    renderings[rendering_name] = rendering

            elif rendering_name == RenderingName.UV:
                if not data_config.use_nvdiffrast:
                    # In older datasets, we used 3 channels for UV instead of just two
                    renderings[rendering_name] = np.concatenate(
                        [rendering, np.zeros((rendering.shape[0], rendering.shape[1], 1))], axis=-1)

            elif rendering_name in {RenderingName.FORWARD_FLOW, RenderingName.BACKWARD_FLOW}:
                rendering = colorize_flow(rendering)

            if len(rendering.shape) > 2 and rendering.shape[-1] == 2:
                rendering = np.concatenate([rendering,
                                            -1 * np.ones(rendering.shape[:-1] + (1,),
                                                         dtype=rendering.dtype)],
                                           axis=-1)  # Add artificial 3rd channel for .png storage
            elif len(rendering.shape) > 2 and rendering.shape[-1] > 3:
                rendering = rendering[..., :3]

            conditions[rendering_name].append((np.squeeze(rendering) + 1) / 2)


if __name__ == '__main__':
    tyro.cli(main)
