from collections import defaultdict
from dataclasses import replace
from typing import Optional

import numpy as np
import torch
import tyro
from diffusion_avatars.evaluation.evaluation_manager import EvaluationManager, CropConfig
from matplotlib.cm import get_cmap
from tqdm import tqdm

from diffusion_avatars.constants import SERIALS
from diffusion_avatars.dataset.rendering_dataset import DatasetSplit, TemporalBatchRenderingDatasetView, RenderingDataset
from diffusion_avatars.model.inference_output import SchedulerType
from diffusion_avatars.model_manager.finder import find_model_folder


def main(run_name: str,
         split: DatasetSplit = 'VALID_HOLD_OUT_SEQ',
         /,
         checkpoint: int = -1,

         scheduler: Optional[SchedulerType] = 'ddpm',
         seed: Optional[int] = None,
         n_inference_steps: int = 20,
         name: Optional[str] = None,
         serials: Optional[str] = None,
         all_serials: bool = False,
         sequences: Optional[str] = None,
         timesteps: Optional[str] = None,

         remove_neck: bool = False,
         remove_background: bool = False,
         remove_torso: bool = False,

         save_error_maps: bool = True):
    model_folder = find_model_folder(run_name)
    model_manager = model_folder.open_run(run_name)
    evaluation_manager = EvaluationManager(run_name, checkpoint, split,
                                           scheduler=scheduler,
                                           n_inference_steps=n_inference_steps,
                                           name=name,
                                           seed=seed,
                                           remove_neck=remove_neck,
                                           remove_torso=remove_torso,
                                           remove_background=remove_background,
                                           all_serials=all_serials)

    model = model_manager.load_checkpoint(checkpoint)
    model.cuda()

    # Switch scheduler
    if run_name.startswith('DA-'):
        model.change_inference_scheduler(scheduler)

    # Seed
    if seed is None:
        seed = model_manager.load_train_setup().seed

    dataset_config = model_manager.load_dataset_config()

    # Only evaluate on the specified serials
    if all_serials:
        serials = SERIALS
    elif serials is None:
        if split == DatasetSplit.TRAIN:
            serials = ["222200047"]
        elif dataset_config.hold_out_cameras is None:
            serials = ["222200037"]
        else:
            serials = dataset_config.hold_out_cameras
    else:
        serials = serials.split(',')

    if sequences is not None:
        sequences = sequences.split(',')

    # Create dataset
    dataset_config = dataset_config.eval(remove_neck=remove_neck, remove_background=remove_background, remove_torso=remove_torso)

    dataset = RenderingDataset(dataset_config, split)
    timesteps = None if timesteps is None else [int(t) for t in timesteps.split(',')]
    dataset_view = TemporalBatchRenderingDatasetView(dataset, serials=serials, sequences=sequences, timesteps=timesteps)

    # Create latent noise
    use_latent_noise = run_name.startswith('DA-')
    if use_latent_noise:
        generator = torch.Generator(device=model.device).manual_seed(seed)
        if dataset_config.remap_noise:
            latent_noise = torch.randn(model.get_noise_texture_shape(), device=model.device, generator=generator)
        else:
            latent_shape = model.get_latent_shape()
            latent_noise = torch.randn(latent_shape, device=model.device, generator=generator)

    # ----------------------------------------------------------
    # Inference loop
    # ----------------------------------------------------------

    dataset_indices = range(len(dataset_view))

    # participant -> sequence -> serial -> timestep: (InferenceOutput, foreground_mask)
    all_inference_outputs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    crop_mapping = defaultdict(dict)  # sequence -> serial -> crop

    for idx in tqdm(dataset_indices):
        sample_metadata = dataset_view.get_sample_metadata(idx)

        # Get face crop for sequence and camera
        if not sample_metadata.sequence_name in crop_mapping or not sample_metadata.serial in crop_mapping[sample_metadata.sequence_name]:
            assert sample_metadata.timestep == 0
            crop_mapping[sample_metadata.sequence_name][sample_metadata.serial] = dataset.get_random_crop(sample_metadata)

        crop = crop_mapping[sample_metadata.sequence_name][sample_metadata.serial]

        # Get batch
        dataset_idx = dataset_view.samples_mapping_view[idx]
        sample = dataset.load_sample(dataset_idx, crop=crop)
        batch = dataset.collate_fn([sample])

        current_sequence_inference_outputs = all_inference_outputs[
            sample_metadata.participant_id][sample_metadata.sequence_name][sample_metadata.serial]

        if use_latent_noise:
            batch = model.prepare_batch(batch, latent_noise=latent_noise)
        else:
            batch = model.prepare_batch(batch)

        inference_outputs = model.inference(batch, n_inference_steps=n_inference_steps)

        timesteps = range(sample_metadata.timestep - 3 * (len(inference_outputs) - 1), sample_metadata.timestep + 1, 3)

        # Save predictions / images
        assert len(inference_outputs) == len(timesteps)
        for i, (timestep, inference_output) in enumerate(zip(timesteps, inference_outputs)):
            foreground_mask = batch.foreground_masks[i].cpu().numpy()
            current_sequence_inference_outputs[timestep] = (inference_output, foreground_mask)

            prediction_masked = (foreground_mask[..., None] / 255) * (inference_output.prediction / 255) + (1 - foreground_mask[..., None] / 255)
            prediction_masked = (255 * prediction_masked).round().astype(np.uint8)

            target_masked = (foreground_mask[..., None] / 255) * (inference_output.target_image / 255) + (1 - foreground_mask[..., None] / 255)
            target_masked = (255 * target_masked).round().astype(np.uint8)

            temporal_sample_metadata = replace(sample_metadata)
            temporal_sample_metadata.timestep = timestep
            evaluation_manager.save_prediction(inference_output.prediction, temporal_sample_metadata)
            evaluation_manager.save_target_image(inference_output.target_image, temporal_sample_metadata)
            evaluation_manager.save_prediction(prediction_masked, temporal_sample_metadata, masked=True)
            evaluation_manager.save_target_image(target_masked, temporal_sample_metadata, masked=True)
            evaluation_manager.save_mask(foreground_mask, temporal_sample_metadata)

            if save_error_maps:
                cmap = get_cmap('turbo')
                error = np.linalg.norm(inference_output.prediction / 255. - inference_output.target_image / 255., axis=-1, ord=1)
                error_map = cmap(error)
                error_masked = np.linalg.norm(prediction_masked / 255. - target_masked / 255., axis=-1, ord=1)
                error_map_masked = cmap(error_masked)
                evaluation_manager.save_error_map((error_map[..., :3] * 255).round().astype(np.uint8), temporal_sample_metadata)
                evaluation_manager.save_error_map((error_map_masked[..., :3] * 255).round().astype(np.uint8), temporal_sample_metadata, masked=True)

    # ----------------------------------------------------------
    # Evaluation loop
    # ----------------------------------------------------------

    evaluation_manager.save_crop_config(CropConfig(crop_mapping))
    evaluation_manager.evaluate(all_inference_outputs)


if __name__ == '__main__':
    tyro.cli(main)
