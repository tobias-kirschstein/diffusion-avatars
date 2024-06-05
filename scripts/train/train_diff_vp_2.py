import logging
import sys
from collections import defaultdict
from logging import getLogger
from statistics import mean
from typing import Optional

import diffusers
import numpy as np
import torch
import transformers
import tyro
from accelerate.utils import set_seed, PrecisionType, ProjectConfiguration
from elias.util import ensure_directory_exists
from elias.util.batch import batchify_sliced
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import wandb
from diff_vp.data_manager.rendering_data_manager import RenderingName, RenderingDataFolder
from diff_vp.dataset.rendering_dataset import RenderingDatasetConfig, RenderingDataset, DatasetSplit, \
    TemporalBatchRenderingDatasetView, TextureFieldConfig, SegmentationType
from diff_vp.model.diff_vp import DiffVP, DiffVPExperimentConfig
from diff_vp.config.diff_vp import DiffVPOptimizerConfig, DiffVPTrainConfig, DiffVPModelConfig
from diff_vp.model.temporal.temporal_config import VideoControlNetConfig, PositionalEncodingType
from diff_vp.model_manager.diff_vp_model_manager import DiffVPModelManager, DiffVPModelFolder
from diff_vp.renderer.provider.mesh_provider import TemporalSmoothingConfig
from diff_vp.util.log import log_to_wandb

logger = getLogger(__name__)


def main(name: Optional[str] = None,

         # Dataset config
         dataset_version: str = "v0.1",
         rendering_names: str = "uv",
         resolution: int = 512,
         hold_out_sequences: Optional[str] = None,
         exclude_sequences: Optional[str] = None,
         hold_out_cameras: str = '222200037',

         # Images
         use_color_correction: bool = True,
         use_random_cropping: bool = True,
         interpolate_target_images: bool = False,

         # Model config
         diffusion_model_name: str = "stabilityai/stable-diffusion-2-1",
         use_original_scheduler: bool = True,

         # Train config
         batch_size: int = 8,
         validation_batch_size: Optional[int] = None,
         dataloader_num_workers: int = 8,
         gradient_accumulation: int = 1,
         validate_every: int = 5000,
         save_model_every: int = 10000,
         save_only_latest: int = -1,  # To save space, delete older checkpoints and only keep the save_only_latest
         num_train_epochs: int = 10000,
         num_train_iterations: int = 100001,
         n_validation_images: int = 16,
         global_prompt: str = "",
         mixed_precision: PrecisionType = PrecisionType.NO,
         use_full_noise: bool = False,
         use_vae_mean: bool = False,

         # Optimizer
         use_8_bit_adam: bool = False,
         set_grads_to_none: bool = False,
         learning_rate: float = 1e-4,
         learning_rate_neural_textures: float = 1e-2,

         # Losses
         lambda_neural_texture_rgb_loss: float = 0,
         lambda_laplacian_reg: float = 0,
         lambda_mouth_loss: float = 0,

         # Neural Textures
         use_neural_textures: bool = True,
         use_texture_fields: bool = False,
         use_texture_triplanes: bool = False,
         use_texture_hierarchy: bool = False,
         use_background_texture: bool = False,
         use_ambient_textures: bool = False,
         use_mipmapping: bool = False,
         res_neural_textures: int = 512,
         dim_neural_textures: int = 16,
         texture_field_n_levels: int = 8,
         texture_field_per_level_scale: float = 2.0,
         use_canonical_coordinates: bool = False,
         use_spherical_harmonics: bool = False,
         use_adam_for_neural_textures: bool = False,
         init_neural_textures_gain: float = 1,

         # Autoregressive Training
         use_autoregressive: bool = False,
         use_teacher_forcing: bool = True,
         use_random_previous_viewpoint: bool = False,
         warp_previous_outputs: bool = False,
         use_crop_sweep: bool = False,
         n_previous_frames: int = 1,
         prob_autoregressive_dropout: float = 0.0,

         # Frequency Encoding
         use_freq_encoding_uv: bool = False,
         use_freq_encoding_depth: bool = False,
         n_frequencies_uv: int = 4,
         n_frequencies_depth: int = 4,

         # Temporal Block Training
         temporal_batch_size: int = 0,
         use_temporal_batch_random_viewpoints: bool = False,
         share_temporal_noise: bool = True,
         fix_pseudo_3d: bool = True,

         use_temporal_convolution: bool = True,
         temporal_kernel_size: int = 3,
         temporal_padding_mode: str = 'zeros',
         downscale_factor_convolution: int = 1,
         downscale_threshold_convolution: int = 32,
         downscale_mode: str = 'bilinear',

         use_temporal_attention: bool = False,
         attention_dim_factor: float = 1,
         downscale_factor_attention: int = 1,
         downscale_threshold_attention: int = 32,

         use_temporal_unet_encoder: bool = False,
         use_temporal_unet_decoder: bool = False,
         use_temporal_controlnet_encoder: bool = True,

         positional_encoding: Optional[PositionalEncodingType] = None,

         # Expression codes
         use_expression_condition: bool = False,
         smooth_expression_codes: bool = False,
         include_eye_condition: bool = False,

         # Segmentation masks
         remove_torso: bool = True,
         segmentation_type: SegmentationType = 'facer',

         # Noise
         remap_noise: bool = False,
         remap_noise_scale: int = 2,
         rescale_betas_zero_snr: bool = True,
         use_trailing_timestep_spacing: bool = True,
         predict_x0: bool = False,
         n_train_steps: Optional[int] = None,
         disable_noise: bool = False,

         # Training from scratch without ControlNet
         no_pretrained: bool = False,

         # Debugging
         single_batch: bool = False,
         ):
    # Train config
    train_config = DiffVPTrainConfig(
        name=name,
        group_name="DiffVP",
        train_batch_size=batch_size,
        validation_batch_size=validation_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=num_train_epochs,
        num_train_iterations=num_train_iterations,
        validate_every=validate_every,
        save_model_every=save_model_every,
        global_prompt=global_prompt,
        dataloader_num_workers=dataloader_num_workers,
        n_validation_images=n_validation_images,
        mixed_precision=mixed_precision,
        use_full_noise=use_full_noise,
        use_vae_mean=use_vae_mean,
        share_temporal_noise=share_temporal_noise,
    )

    # Optimizer config
    optimizer_config = DiffVPOptimizerConfig(
        set_grads_to_none=set_grads_to_none,
        use_8bit_adam=use_8_bit_adam,
        learning_rate=learning_rate,
        learning_rate_neural_textures=learning_rate_neural_textures,
        adam_epsilon=1e-8,
        lambda_neural_texture_rgb_loss=lambda_neural_texture_rgb_loss,
        lambda_laplacian_reg=lambda_laplacian_reg,
        lambda_mouth_loss=lambda_mouth_loss,
        use_adam_for_neural_textures=use_adam_for_neural_textures,
        init_neural_textures_gain=init_neural_textures_gain,
    )

    # Model config
    model_config = DiffVPModelConfig(
        use_original_scheduler=use_original_scheduler,
        diffusion_model_name=diffusion_model_name,  # "stabilityai/stable-diffusion-2-1"
        temporal_config=VideoControlNetConfig(
            fix_pseudo_3d_block=fix_pseudo_3d,
            # Convolution
            use_temporal_convolution=use_temporal_convolution,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding_mode=temporal_padding_mode,
            downscale_factor_convolution=downscale_factor_convolution,
            downscale_threshold_convolution=downscale_threshold_convolution,

            # Attention
            use_temporal_attention=use_temporal_attention,
            attention_dim_factor=attention_dim_factor,
            downscale_factor_attention=downscale_factor_attention,
            downscale_threshold_attention=downscale_threshold_attention,

            downscale_mode=downscale_mode,

            use_temporal_unet_encoder=use_temporal_unet_encoder,
            use_temporal_unet_decoder=use_temporal_unet_decoder,
            use_temporal_controlnet_encoder=use_temporal_controlnet_encoder,

            positional_encoding=positional_encoding,
        ),
        remap_noise_scale=remap_noise_scale,
        rescale_betas_zero_snr=rescale_betas_zero_snr,
        use_trailing_timestep_spacing=use_trailing_timestep_spacing,
        n_train_steps=n_train_steps,
        predict_x0=predict_x0,
        no_pretrained=no_pretrained,
        disable_noise=disable_noise,
    )

    rendering_data_manager = RenderingDataFolder().open_dataset(dataset_version)
    data_stats = rendering_data_manager.load_stats()
    dataset_n_persons = len(data_stats.available_sequences)
    dataset_n_sequences_per_person = mean([len(participant_metadata)
                                           for participant_metadata in data_stats.available_sequences.values()])

    dataset_splits = [DatasetSplit.TRAIN,
                      DatasetSplit.VALID_HOLD_OUT_VIEW]
    if dataset_n_sequences_per_person > 4:
        dataset_splits.append(DatasetSplit.VALID_HOLD_OUT_SEQ)
    if dataset_n_persons > 4:
        dataset_splits.append(DatasetSplit.VALID_HOLD_OUT_PERSON)

    # Dataset config
    dataset_config = RenderingDatasetConfig(
        dataset_version=dataset_version,
        rendering_names=[RenderingName(n) for n in rendering_names.split(',')],
        resolution=resolution,
        normalize=True,
        seed=train_config.seed,

        # Splits
        supported_splits=dataset_splits,
        split_ratio=0.9,
        hold_out_cameras=hold_out_cameras.split(','),
        use_predefined_test_split=hold_out_sequences is None,
        hold_out_sequences=hold_out_sequences.split(',') if hold_out_sequences is not None else None,
        exclude_sequences=exclude_sequences.split(',') if exclude_sequences is not None else None,

        # Images
        use_color_correction=use_color_correction,
        use_random_cropping=use_random_cropping,
        interpolate_target_images=interpolate_target_images,

        # Neural Textures
        use_neural_textures=use_neural_textures,
        use_texture_fields=use_texture_fields,
        use_texture_triplanes=use_texture_triplanes,
        use_texture_hierarchy=use_texture_hierarchy,
        use_background_texture=use_background_texture,
        use_ambient_textures=use_ambient_textures,
        use_mipmapping=use_mipmapping,
        dim_neural_textures=dim_neural_textures,
        res_neural_textures=res_neural_textures,
        texture_field_config=TextureFieldConfig(
            n_levels=texture_field_n_levels,
            per_level_scale=texture_field_per_level_scale),
        use_canonical_coordinates=use_canonical_coordinates,
        use_spherical_harmonics=use_spherical_harmonics,
        include_view_directions=use_spherical_harmonics,

        # Autoregressive
        use_autoregressive=use_autoregressive,
        use_teacher_forcing=use_teacher_forcing,
        use_random_previous_viewpoints=use_random_previous_viewpoint,
        warp_previous_outputs=warp_previous_outputs,
        use_crop_sweep=use_crop_sweep,
        occlusion_mask_threshold=5,
        n_previous_frames=n_previous_frames,
        prob_autoregressive_dropout=prob_autoregressive_dropout,

        # Frequency Encodings
        use_freq_encoding_uv=use_freq_encoding_uv,
        use_freq_encoding_depth=use_freq_encoding_depth,
        n_frequencies_uv=n_frequencies_uv,
        n_frequencies_depth=n_frequencies_depth,

        # Temporal block training
        temporal_batch_size=temporal_batch_size,
        use_temporal_batch_random_viewpoints=use_temporal_batch_random_viewpoints,

        # Expression Codes
        use_expression_condition=use_expression_condition,
        expression_condition_smoothing=TemporalSmoothingConfig(
            temporal_smoothing=5,
            use_gaussian=True,
            param_groups=['expression']) if smooth_expression_codes else TemporalSmoothingConfig(),
        include_eye_condition=include_eye_condition,

        # Segmentation Masks
        remove_torso=remove_torso,
        segmentation_type=segmentation_type,
        include_mouth_mask=lambda_mouth_loss > 0,

        remap_noise=remap_noise,
    )

    experiment_config = DiffVPExperimentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_config=train_config,
        optimizer_config=optimizer_config
    )

    set_seed(train_config.seed)

    # ----------------------------------------------------------
    # Setup Model Manager
    # ----------------------------------------------------------

    model_manager: DiffVPModelManager = DiffVPModelFolder().new_run(train_config.name)

    # ----------------------------------------------------------
    # Create Datasets
    # ----------------------------------------------------------

    datasets = {split: RenderingDataset(dataset_config, split=split) for split in dataset_config.supported_splits}
    train_dataset = datasets[DatasetSplit.TRAIN]

    total_batch_size = train_config.train_batch_size * train_config.gradient_accumulation_steps
    model_config.n_cond_channels = train_dataset.get_n_cond_channels()
    model_config.n_participants = len(train_dataset.participant_ids)
    model_config.temporal_config.temporal_batch_size = dataset_config.temporal_batch_size
    model_config.temporal_config.use_expression_condition = dataset_config.use_expression_condition
    train_config.n_samples_per_epoch = len(train_dataset)
    max_train_steps = train_config.get_max_train_steps()

    if single_batch:
        datasets = {split: TemporalBatchRenderingDatasetView(dataset, max_samples=1, serials=["222200047", "222200037"])
                    for split, dataset in datasets.items()}
        train_dataset = datasets[DatasetSplit.TRAIN]

    for dataset in datasets.values():
        assert len(dataset) > 0

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=train_config.train_batch_size,
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=train_config.dataloader_num_workers * n_previous_frames if n_previous_frames > 1 else train_config.dataloader_num_workers,
                                  persistent_workers=train_config.dataloader_num_workers > 0,
                                  # Workers will persist across epochs
                                  multiprocessing_context="spawn" if train_config.dataloader_num_workers > 0 else None
                                  )

    # Increase maximum number of file descriptors a process can have
    # This addresses the error "RuntimeError: received 0 items of ancdata"
    # For the solution see: https://stackoverflow.com/questions/71642653/how-to-resolve-the-error-runtimeerror-received-0-items-of-ancdata
    if sys.platform == 'linux':
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # ----------------------------------------------------------
    # Setup Model
    # ----------------------------------------------------------

    diff_vp_model = DiffVP(model_config,
                           dataset_config,
                           train_config=train_config,
                           optimizer_config=optimizer_config)

    # ----------------------------------------------------------
    # Inference scheduler
    # ----------------------------------------------------------

    diff_vp_model.change_inference_scheduler('ddpm')

    # ----------------------------------------------------------
    # Setup Accelerator
    # ----------------------------------------------------------

    accelerator_project_config = ProjectConfiguration(project_dir=model_manager.get_location(),
                                                      logging_dir=model_manager.get_log_folder())

    accelerator = diff_vp_model.setup_accelerator(accelerator_project_config)

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # ----------------------------------------------------------
    # Setup Optimizer & LR Scheduler
    # ----------------------------------------------------------

    optimizers_and_lr_schedulers = diff_vp_model.configure_optimizers()

    # Prepare everything with our `accelerator`.
    train_dataloader = accelerator.prepare(train_dataloader)

    # ----------------------------------------------------------
    # Initialize Trackers
    # ----------------------------------------------------------

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {train_config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_config.gradient_accumulation_steps}")
    global_step = 1
    initial_global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # # Only show the progress bar once on each machine.
        # disable=not accelerator.is_local_main_process,
    )

    ensure_directory_exists(model_manager.get_wandb_folder())
    wandb.init(project=train_config.project_name,
               group=train_config.group_name,
               name=model_manager.get_run_name(),
               config=experiment_config.to_json(),
               dir=model_manager.get_wandb_folder())

    # ----------------------------------------------------------
    # Store configs
    # ----------------------------------------------------------
    model_manager.store_model_config(model_config)
    model_manager.store_dataset_config(dataset_config)
    model_manager.store_optimization_config(optimizer_config)
    model_manager.store_train_setup(train_config)

    # ----------------------------------------------------------
    # Train Loop
    # ----------------------------------------------------------

    summary(diff_vp_model)
    diff_vp_model.to(accelerator.device)

    # Fixed latent noise for validation batches
    if train_config.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(train_config.seed)

    latent_shape = diff_vp_model.get_latent_shape()
    validation_latent_noise = torch.randn((train_config.n_validation_images,) + latent_shape,
                                          device=diff_vp_model.device, generator=generator)

    is_first_validation = True
    for epoch in range(first_epoch, train_config.num_train_epochs):

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(diff_vp_model):
                # Autocast is only necessary if we add trainable layers into the SD backbone
                with accelerator.autocast():
                    loss_dict = diff_vp_model.get_loss_dict(batch)
                loss = diff_vp_model.combine_losses(loss_dict)
                accelerator.backward(loss)

                # Gradient clipping and gradient logging
                if accelerator.sync_gradients:
                    params_to_clip = diff_vp_model.controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, optimizer_config.max_grad_norm)

                    # Important that gradient logging is after clipping, as clipping performs gradient unscaling
                    controlnet_grad_norm = mean([p.grad.norm().item()
                                                 for p in diff_vp_model.controlnet.parameters()
                                                 if p.requires_grad and p.grad is not None])
                    wandb.log({"grad_norm/controlnet": controlnet_grad_norm}, step=global_step)

                    unet_learnable_parameters = diff_vp_model.unet.get_learnable_parameters()
                    if unet_learnable_parameters:
                        temporal_unet_grad_norm = mean([p.grad.norm().item()
                                                        for p in unet_learnable_parameters.values()
                                                        if p.requires_grad and p.grad is not None])
                        wandb.log({"grad_norm/temporal_unet": temporal_unet_grad_norm}, step=global_step)

                    # Neural Textures
                    if diff_vp_model.neural_texture_lookup is not None:
                        wandb.log({"grad_norm/neural_texture_lookup":
                                       diff_vp_model.neural_texture_lookup.weight.grad.norm().item()}, step=global_step)

                    if diff_vp_model.ambient_texture_lookup is not None:
                        wandb.log({"grad_norm/ambient_texture_lookup":
                                       diff_vp_model.ambient_texture_lookup.weight.grad.norm().item()}, step=global_step)

                    if diff_vp_model.neural_texture_fields is not None:
                        wandb.log({"grad_norm/neural_texture_fields":
                                       mean([neural_texture_field.params.grad.norm().item()
                                             for neural_texture_field in diff_vp_model.neural_texture_fields
                                             if neural_texture_field.params.grad is not None])},
                                  step=global_step)

                    if diff_vp_model.neural_texture_lookup_uv is not None:
                        wandb.log({"grad_norm/neural_texture_lookup_uv":
                                       diff_vp_model.neural_texture_lookup_uv.weight.grad.norm().item()}, step=global_step)
                        wandb.log({"grad_norm/neural_texture_lookup_uw":
                                       diff_vp_model.neural_texture_lookup_uw.weight.grad.norm().item()}, step=global_step)
                        wandb.log({"grad_norm/neural_texture_lookup_vw":
                                       diff_vp_model.neural_texture_lookup_vw.weight.grad.norm().item()}, step=global_step)

                    # Background Texture
                    if diff_vp_model.learnable_background_color is not None:
                        wandb.log({"grad_norm/background_texture":
                                       diff_vp_model.learnable_background_color.grad.norm().item()}, step=global_step)

                    if model_config.temporal_config.temporal_batch_size >= 1 and model_config.temporal_config.use_temporal_convolution:
                        temporal_convolutions_grad_norms = [p.grad.norm().item()
                                                            for n, p in diff_vp_model.controlnet.named_parameters()
                                                            if p.requires_grad and p.grad is not None
                                                            and 'temporal_convolutions' in n]
                        if temporal_convolutions_grad_norms:
                            temporal_convolutions_grad_norm = mean(temporal_convolutions_grad_norms)
                            wandb.log({"grad_norm/temporal_convolutions": temporal_convolutions_grad_norm},
                                      step=global_step)

                    if model_config.temporal_config.temporal_batch_size >= 1 and model_config.temporal_config.use_temporal_attention:
                        temporal_attentions_grad_norms = [p.grad.norm().item()
                                                          for n, p in diff_vp_model.controlnet.named_parameters()
                                                          if p.requires_grad and p.grad is not None
                                                          and 'temporal_attentions' in n]
                        if temporal_attentions_grad_norms:
                            temporal_attentions_grad_norm = mean(temporal_attentions_grad_norms)
                            wandb.log({"grad_norm/temporal_attentions": temporal_attentions_grad_norm},
                                      step=global_step)

                # Optimizer/Scheduler step()
                for optimizer_and_scheduler in optimizers_and_lr_schedulers:
                    optimizer_and_scheduler["optimizer"].step()
                    if "lr_scheduler" in optimizer_and_scheduler:
                        optimizer_and_scheduler["lr_scheduler"].step()

                for optimizer in optimizers_and_lr_schedulers:
                    optimizer["optimizer"].zero_grad(set_to_none=optimizer_config.set_grads_to_none)

                # Validation / Checkpoint
                if accelerator.sync_gradients:
                    # Checkpoint saving
                    if global_step % train_config.save_model_every == 0:
                        model_manager.store_checkpoint(diff_vp_model, global_step)
                        checkpoint_path = model_manager.get_checkpoint_path(global_step)

                        # Remove all checkpoints except the last
                        if save_only_latest > 0:
                            checkpoint_ids = sorted(model_manager.list_checkpoint_ids())
                            for checkpoint_id in checkpoint_ids[:-save_only_latest]:
                                model_manager.delete_checkpoint(checkpoint_id)

                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                    # Validation step
                    if train_config.validation_prompt is not None and global_step % train_config.validate_every == 0:
                        with torch.no_grad():
                            with accelerator.autocast():
                                # Iterate over all available splits
                                for split, dataset in datasets.items():
                                    split_name = split.get_log_name()
                                    validation_indices = np.linspace(0, len(dataset) - 1,
                                                                     train_config.n_validation_images,
                                                                     dtype=np.int)
                                    validation_samples = [dataset[i] for i in validation_indices]
                                    validation_batch = dataset.collate_fn(validation_samples)
                                    validation_batch.latent_noise = validation_latent_noise
                                    diff_vp_model.prepare_batch(validation_batch)

                                    inference_outputs = []
                                    valid_losses = defaultdict(list)
                                    for small_validation_batch in tqdm(
                                            batchify_sliced(validation_batch,
                                                            batch_size=batch_size
                                                            if validation_batch_size is None
                                                            else validation_batch_size),
                                            desc=split_name):
                                        # Potentially, generate outputs for previous timesteps first
                                        diff_vp_model.prepare_autoregressive_inference(small_validation_batch)
                                        inference_output = diff_vp_model.inference(small_validation_batch)
                                        inference_outputs.extend(inference_output)

                                        valid_loss_dict = diff_vp_model.get_loss_dict(small_validation_batch)
                                        for loss_key, loss_value in valid_loss_dict.items():
                                            valid_losses[loss_key].append(loss_value.item())

                                    valid_losses = {
                                        f"{split_name}/metrics/{loss_key}":
                                            mean(losses) for loss_key, losses in valid_losses.items()}
                                    wandb.log(valid_losses, step=global_step)
                                    log_to_wandb(inference_outputs, global_step, is_first_validation, dataset_config,
                                                 split_name)

                            is_first_validation = False

                # Log loss and learning rate
                logs = {key: loss_value.detach().item() for key, loss_value in loss_dict.items()}
                logs["loss"] = loss

                for i, optimizer_and_scheduler in enumerate(optimizers_and_lr_schedulers):
                    if "lr_scheduler" in optimizer_and_scheduler:
                        logs[f"lr_{i}"] = optimizer_and_scheduler["lr_scheduler"].get_last_lr()[0]

                # Step handling
                progress_bar.set_postfix(**logs)
                wandb.log(logs, step=global_step)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                if global_step >= max_train_steps:
                    # Jump out of batch loop
                    break

        if global_step >= max_train_steps:
            # Also jump out of epoch loop
            break

    accelerator.wait_for_everyone()
    # We do not use any trackers in accelerator, so end_training() not needed
    # accelerator.end_training()


if __name__ == '__main__':
    tyro.cli(main)
