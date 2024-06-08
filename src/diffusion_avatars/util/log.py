from collections import defaultdict
from math import ceil, sqrt, floor
from typing import List, Dict

import cv2
import numpy as np
import wandb
from visage.evaluator.paired_image_evaluator import PairedImageEvaluator

from diffusion_avatars.dataset.rendering_dataset import RenderingDatasetConfig
from diffusion_avatars.model.inference_output import DiffusionAvatarsInferenceOutput


def log_to_wandb(inference_outputs: List[DiffusionAvatarsInferenceOutput],
                 step: int,
                 is_first_validation: bool,
                 dataset_config: RenderingDatasetConfig,
                 log_name: str):
    paired_image_evaluator = PairedImageEvaluator()

    conditioning_images = defaultdict(list)
    target_images = []
    predictions = []
    neural_textured_images = []
    ambient_textured_images = []
    previous_outputs = defaultdict(list)
    for inference_output in inference_outputs:

        for rendering_name, rendering in inference_output.conditions.items():
            if len(rendering.shape) > 2 and rendering.shape[-1] == 2:
                rendering = np.concatenate([rendering,
                                            np.zeros(rendering.shape[:-1] + (1,),
                                                     dtype=rendering.dtype)],
                                           axis=-1)  # Add artificial 3rd channel for .png storage
            elif len(rendering.shape) > 2 and rendering.shape[-1] > 3:
                rendering = rendering[..., :3]

            conditioning_images[rendering_name].append(rendering)

        target_images.append(inference_output.target_image)
        predictions.append(inference_output.prediction)

        if inference_output.neural_textured_rendering is not None:
            neural_textured_images.append(inference_output.neural_textured_rendering[..., -3:])

        if inference_output.ambient_textured_rendering is not None:
            ambient_textured_images.append(inference_output.ambient_textured_rendering[..., -3:])

        if inference_output.previous_outputs is not None:
            for i, previous_output in enumerate(inference_output.previous_outputs):
                previous_outputs[i].append(previous_output)

    log_base = log_name

    # log validation metrics
    paired_image_metrics = paired_image_evaluator.evaluate(predictions, target_images)
    wandb.log({
        f"{log_base}/metrics/psnr": paired_image_metrics.psnr,
        f"{log_base}/metrics/ssim": paired_image_metrics.ssim,
        f"{log_base}/metrics/lpips": paired_image_metrics.lpips,
    }, step=step)

    wandb.log(
        {f"{log_base}/images/prediction": wandb.Image(tile_image(predictions), caption=inference_outputs[0].prompt)},
        step=step)
    if neural_textured_images:
        wandb.log({f"{log_base}/images/condition_neural_texture": wandb.Image(tile_image(neural_textured_images))},
                  step=step)
    if ambient_textured_images:
        wandb.log({f"{log_base}/images/condition_ambient_texture": wandb.Image(tile_image(ambient_textured_images))},
                  step=step)

    if is_first_validation or dataset_config.use_random_cropping:
        # Only log targets and non-learnable conditions once to save storage
        wandb.log({f"{log_base}/images/target_image": wandb.Image(tile_image(target_images))}, step=step)

    if is_first_validation:
        # Rendered conditions
        for rendering_name, conditions in conditioning_images.items():
            wandb.log({f"{log_base}/images/condition_{rendering_name}": wandb.Image(tile_image(conditions))}, step=step)

    # Previous outputs
    if is_first_validation \
            or dataset_config.use_autoregressive and not dataset_config.use_teacher_forcing \
            or dataset_config.use_random_cropping:
        for i, previous_output_list in previous_outputs.items():
            wandb.log({f"{log_base}/images/previous_output_{i}": wandb.Image(tile_image(previous_output_list))},
                      step=step)


def tile_image(images: List[np.ndarray], scale_factor: float = 2) -> np.ndarray:
    n_images = len(images)
    nrows = int(ceil(sqrt(n_images)))
    ncols = int(ceil(n_images / nrows))
    tile_width = min(int(floor(images[0].shape[1] * scale_factor / nrows)), images[0].shape[1])
    tile_height = min(int(floor(images[0].shape[0] * scale_factor / nrows)), images[0].shape[0])

    tiled_image = np.ones((tile_height * nrows, tile_width * ncols, images[0].shape[-1]), dtype=np.uint8) * 255
    for i, image in enumerate(images):
        row = int(floor(i / ncols))
        col = i % ncols
        resized_image = cv2.resize(image, (tile_width, tile_height))
        if len(resized_image.shape) == 2:
            resized_image = resized_image[..., None]
        tiled_image[row * tile_height: (row + 1) * tile_height,
        col * tile_width: (col + 1) * tile_width] = resized_image

    return tiled_image


def make_conditions_viewable(inference_outputs: List[DiffusionAvatarsInferenceOutput]) -> Dict[str, List[np.ndarray]]:
    viewable_conditions = defaultdict(list)

    for inference_output in inference_outputs:
        if inference_output.neural_textured_rendering is not None:
            viewable_conditions['neural_textured_rendering'].append(
                inference_output.neural_textured_rendering[..., -3:])

        if inference_output.previous_outputs is not None:
            for prev, previous_output in enumerate(inference_output.previous_outputs):
                viewable_conditions[f"previous_output_{prev}"].append(previous_output)

        if inference_output.conditions is not None:
            for rendering_name, rendering in inference_output.conditions.items():
                if len(rendering.shape) > 2 and rendering.shape[-1] == 1:
                    rendering = rendering.repeat(3, axis=-1)  # Mostly for depth renderings. Just repeat the single channel
                elif len(rendering.shape) > 2 and rendering.shape[-1] == 2:
                    rendering = np.concatenate([rendering,
                                                np.zeros(rendering.shape[:-1] + (1,),
                                                         dtype=rendering.dtype)],
                                               axis=-1)  # Add artificial 3rd channel for .png storage
                elif len(rendering.shape) > 2 and rendering.shape[-1] > 3:
                    rendering = rendering[..., :3]

                viewable_conditions[rendering_name.name].append(rendering)

    return viewable_conditions
