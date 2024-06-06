import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import mediapy
import numpy as np
from elias.config import Config
from elias.util import save_img, save_json, load_json, load_img, ensure_directory_exists
from visage.evaluator.paired_face_image_evaluator import PairedFaceImageEvaluator
from visage.evaluator.paired_video_evaluator import PairedVideoEvaluator

from diffusion_avatars.dataset.rendering_dataset import DatasetSplit, SampleMetadata, Crop
from diffusion_avatars.evaluation.evaluation_result import DiffusionAvatarsEvaluationResult, PerSequenceMetric
from diffusion_avatars.model.inference_output import DiffusionAvatarsInferenceOutput, SchedulerType
from diffusion_avatars.model_manager.finder import find_model_folder
from diffusion_avatars.util.types import default_dict_to_dict


@dataclass
class CropConfig(Config):
    crop_mapping: Dict[str, Dict[str, Crop]]


class EvaluationManager:

    def __init__(self, run_name: str, checkpoint: int, split: DatasetSplit,
                 smooth_expression_condition: bool = False,
                 reproduce_train_metrics: bool = False,
                 scheduler: Optional[SchedulerType] = None,
                 n_inference_steps: int = 20,
                 name: Optional[str] = None,
                 seed: Optional[int] = None,
                 remove_neck: bool = False,
                 remove_torso: bool = False,
                 remove_background: bool = False,
                 all_serials: bool = False,
                 color_correction: bool = False,
                 source_actor: Optional[int] = None,
                 source_sequence: Optional[str] = None,
                 move_y: Optional[float] = None):
        self._evaluation_folder = EvaluationFolder(run_name,
                                                   smooth_expression_condition=smooth_expression_condition,
                                                   reproduce_train_metrics=reproduce_train_metrics,
                                                   scheduler=scheduler,
                                                   n_inference_steps=n_inference_steps,
                                                   name=name,
                                                   seed=seed,
                                                   remove_neck=remove_neck,
                                                   remove_torso=remove_torso,
                                                   remove_background=remove_background,
                                                   all_serials=all_serials,
                                                   color_correction=color_correction,
                                                   source_actor=source_actor,
                                                   move_y=move_y)
        checkpoint_id = self._evaluation_folder._model_manager._resolve_checkpoint_id(checkpoint)

        evaluation_name = self._evaluation_folder._get_evaluation_name(checkpoint_id)

        self._checkpoint_id = checkpoint_id
        self._split = split
        self._source_actor = source_actor
        self._evaluation_name = evaluation_name
        self._split_name = self._evaluation_folder._get_split_name(split)
        if source_actor is None:
            self._location = f"{self._evaluation_folder.get_evaluations_location()}/{evaluation_name}/{self._split_name}"
        else:
            self._location = f"{self._evaluation_folder.get_evaluations_location()}/{evaluation_name}/from_{source_actor:03d}_{source_sequence}"

    def save_prediction(self, prediction: np.ndarray, sample_metadata: SampleMetadata, masked: bool = False):
        prediction_path = self.get_prediction_path(sample_metadata, masked=masked)
        save_img(prediction, prediction_path)

    def save_target_image(self, target_image: np.ndarray, sample_metadata: SampleMetadata, masked: bool = False):
        target_image_path = self.get_target_image_path(sample_metadata, masked=masked)
        save_img(target_image, target_image_path)

    def save_mask(self, mask: np.ndarray, sample_metadata: SampleMetadata):
        mask_image_path = self.get_mask_image_path(sample_metadata)
        save_img(mask, mask_image_path)

    def save_evaluation_result(self, evaluation_result: DiffusionAvatarsEvaluationResult):
        evaluation_result_path = self.get_evaluation_result_path()
        save_json(evaluation_result.to_json(), evaluation_result_path)

    def save_crop_config(self, crop_config: CropConfig):
        save_json(crop_config.to_json(), self.get_crop_config_path())

    def save_error_map(self, error: np.ndarray, sample_metadata: SampleMetadata, masked: bool = False):
        error_map_path = self.get_error_map_path(sample_metadata, masked=masked)
        save_img(error, error_map_path)

    def load_evaluation_result(self) -> DiffusionAvatarsEvaluationResult:
        evaluation_result_path = self.get_evaluation_result_path()
        return DiffusionAvatarsEvaluationResult.from_json(load_json(evaluation_result_path))

    def load_prediction(self, sample_metadata: SampleMetadata) -> np.ndarray:
        prediction_path = self.get_prediction_path(sample_metadata)
        return load_img(prediction_path)

    def load_target_image(self, sample_metadata: SampleMetadata) -> np.ndarray:
        target_image_path = self.get_target_image_path(sample_metadata)
        return load_img(target_image_path)

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    def get_evaluation_images_folder(self, participant_id: int, sequence_name: str, serial: str) -> str:
        if self._source_actor is None:
            return f"{self._location}/{participant_id:03d}/{sequence_name}/{serial}"
        else:
            return f"{self._location}/{serial}"

    def get_prediction_path(self, sample_metadata: SampleMetadata, masked: bool = False) -> str:
        evaluation_images_folder = self.get_evaluation_images_folder(sample_metadata.participant_id,
                                                                     sample_metadata.sequence_name,
                                                                     sample_metadata.serial)
        if masked:
            return f"{evaluation_images_folder}/frame_{sample_metadata.timestep:05d}_prediction_masked.png"
        else:
            return f"{evaluation_images_folder}/frame_{sample_metadata.timestep:05d}_prediction.png"

    def get_target_image_path(self, sample_metadata: SampleMetadata, masked: bool = False) -> str:
        evaluation_images_folder = self.get_evaluation_images_folder(sample_metadata.participant_id,
                                                                     sample_metadata.sequence_name,
                                                                     sample_metadata.serial)
        if masked:
            return f"{evaluation_images_folder}/frame_{sample_metadata.timestep:05d}_target_masked.png"
        else:
            return f"{evaluation_images_folder}/frame_{sample_metadata.timestep:05d}_target.png"

    def get_mask_image_path(self, sample_metadata: SampleMetadata) -> str:
        evaluation_images_folder = self.get_evaluation_images_folder(sample_metadata.participant_id,
                                                                     sample_metadata.sequence_name,
                                                                     sample_metadata.serial)

        return f"{evaluation_images_folder}/frame_{sample_metadata.timestep:05d}_mask.png"

    def get_error_map_path(self, sample_metadata: SampleMetadata, masked: bool = False) -> str:
        evaluation_images_folder = self.get_evaluation_images_folder(sample_metadata.participant_id,
                                                                     sample_metadata.sequence_name,
                                                                     sample_metadata.serial)
        if masked:
            return f"{evaluation_images_folder}/frame_{sample_metadata.timestep:05d}_error_masked.png"
        else:
            return f"{evaluation_images_folder}/frame_{sample_metadata.timestep:05d}_error.png"

    def get_evaluation_result_path(self) -> str:
        return self._evaluation_folder.get_evaluation_result_path(self._checkpoint_id, self._split)

    def get_crop_config_path(self) -> str:
        return f"{self._location}/crop_config.json"

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    def get_checkpoint_id(self) -> int:
        return self._checkpoint_id

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------

    def _evaluate_inference_outputs(self, all_inference_outputs: Dict[int, Dict[str, Dict[str, Dict[int, Tuple[DiffusionAvatarsInferenceOutput, np.ndarray]]]]],
                                    apply_mask: bool = False):
        fps = 24
        paired_image_evaluator = PairedFaceImageEvaluator()
        paired_video_evaluator = PairedVideoEvaluator(fps=fps)

        per_sequence_metrics = defaultdict(lambda: defaultdict(dict))
        per_sequence_metrics_list = []
        for participant_id, sequences_dict in all_inference_outputs.items():
            for sequence, serials_dict in sequences_dict.items():
                for serial, timesteps_dict in serials_dict.items():
                    timesteps, inference_outputs_and_foreground_masks = zip(*sorted(timesteps_dict.items(), key=lambda x: x[0]))
                    inference_outputs, foreground_masks = zip(*inference_outputs_and_foreground_masks)

                    predictions = [inference_output.prediction for inference_output in inference_outputs]
                    targets = [inference_output.target_image for inference_output in inference_outputs]

                    if apply_mask:
                        predictions = [(foreground_mask[..., None] / 255) * (inference_output.prediction / 255) + (1 - foreground_mask[..., None] / 255)
                                       for (inference_output, foreground_mask) in zip(inference_outputs, foreground_masks)]
                        predictions = [(255 * prediction).round().astype(np.uint8) for prediction in predictions]

                        targets = [(foreground_mask[..., None] / 255) * (inference_output.target_image / 255) + (1 - foreground_mask[..., None] / 255)
                                   for (inference_output, foreground_mask) in zip(inference_outputs, foreground_masks)]
                        targets = [(255 * target).round().astype(np.uint8) for target in targets]

                    error_maps = paired_image_evaluator.compute_error_maps(predictions, targets)
                    paired_image_metrics = paired_image_evaluator.evaluate(predictions, targets)
                    paired_video_metric = paired_video_evaluator.evaluate(predictions, targets)

                    per_sequence_metric = PerSequenceMetric(paired_image_metrics, paired_video_metric)
                    per_sequence_metrics[participant_id][sequence][serial] = per_sequence_metric
                    per_sequence_metrics_list.append(per_sequence_metric)

                    # Store videos
                    evaluation_images_folder = self.get_evaluation_images_folder(participant_id, sequence, serial)
                    if apply_mask:
                        prediction_video_path = f"{evaluation_images_folder}/video_predictions_masked.mp4"
                        targets_video_path = f"{evaluation_images_folder}/video_targets_masked.mp4"
                        error_maps_video_path = f"{evaluation_images_folder}/video_error_maps_masked.mp4"
                    else:
                        prediction_video_path = f"{evaluation_images_folder}/video_predictions.mp4"
                        targets_video_path = f"{evaluation_images_folder}/video_targets.mp4"
                        error_maps_video_path = f"{evaluation_images_folder}/video_error_maps.mp4"

                    ensure_directory_exists(evaluation_images_folder)
                    mediapy.write_video(prediction_video_path, predictions, fps=fps)
                    mediapy.write_video(targets_video_path, targets, fps=fps)
                    mediapy.write_video(error_maps_video_path, error_maps, fps=fps)

        return per_sequence_metrics, per_sequence_metrics_list

    def evaluate(self,
                 all_inference_outputs: Dict[int, Dict[str, Dict[str, Dict[int, Tuple[DiffusionAvatarsInferenceOutput, np.ndarray]]]]],
                 all_inference_outputs_masked: Dict[int, Dict[str, Dict[str, Dict[int, Tuple[DiffusionAvatarsInferenceOutput, np.ndarray]]]]] = None,
                 skip_mask: bool = False):
        per_sequence_metrics, per_sequence_metrics_list = self._evaluate_inference_outputs(all_inference_outputs, apply_mask=False)
        if not skip_mask:
            if all_inference_outputs_masked is None:
                all_inference_outputs_masked = all_inference_outputs
            per_sequence_metrics_masked, per_sequence_metrics_masked_list = self._evaluate_inference_outputs(all_inference_outputs_masked, apply_mask=True)

        average_per_sequence_metric = sum(per_sequence_metrics_list) / len(per_sequence_metrics_list)
        per_sequence_metrics = default_dict_to_dict(per_sequence_metrics)

        if skip_mask:
            average_per_sequence_metric_masked = None
            per_sequence_metrics_masked = None
        else:
            average_per_sequence_metric_masked = sum(per_sequence_metrics_masked_list) / len(per_sequence_metrics_masked_list)
            per_sequence_metrics_masked = default_dict_to_dict(per_sequence_metrics_masked)

        evaluation_result = DiffusionAvatarsEvaluationResult(
            per_sequence_metrics=per_sequence_metrics,
            average_per_sequence_metric=average_per_sequence_metric,
            masked_per_sequence_metrics=per_sequence_metrics_masked,
            masked_average_per_sequence_metric=average_per_sequence_metric_masked)

        self.save_evaluation_result(evaluation_result)


class EvaluationFolder:

    def __init__(self, run_name: str,
                 smooth_expression_condition: bool = False,
                 reproduce_train_metrics: bool = False,
                 scheduler: Optional[SchedulerType] = None,
                 n_inference_steps: int = 20,
                 name: Optional[str] = None,
                 seed: Optional[int] = None,
                 remove_neck: bool = False,
                 remove_torso: bool = False,
                 remove_background: bool = False,
                 all_serials: bool = False,
                 color_correction: bool = False,
                 source_actor: Optional[int] = None,
                 move_y: Optional[float] = None):
        model_folder = find_model_folder(run_name)
        model_manager = model_folder.open_run(run_name)

        self._run_name = run_name
        self._source_actor = source_actor
        if source_actor is None:
            self._evaluations_location = f"{model_manager.get_location()}/evaluations"
        else:
            self._evaluations_location = f"{model_manager.get_location()}/cross_reenactments"
        self._model_manager = model_manager

        self._smooth_expression_condition = smooth_expression_condition
        self._reproduce_train_metrics = reproduce_train_metrics
        self._scheduler = scheduler
        self._n_inference_steps = n_inference_steps
        self._name = name
        self._seed = seed
        self._remove_neck = remove_neck
        self._remove_torso = remove_torso
        self._remove_background = remove_background
        self._all_serials = all_serials
        self._color_correction = color_correction
        self._move_y = move_y

    def get_evaluation_manager(self, checkpoint: int, split: DatasetSplit) -> EvaluationManager:
        if checkpoint < 0:
            evaluated_checkpoints = self.list_evaluated_checkpoint_ids(split)
            checkpoint = evaluated_checkpoints[checkpoint]

        evaluation_manager = EvaluationManager(self._run_name, checkpoint, split,
                                               smooth_expression_condition=self._smooth_expression_condition,
                                               reproduce_train_metrics=self._reproduce_train_metrics,
                                               scheduler=self._scheduler,
                                               n_inference_steps=self._n_inference_steps,
                                               name=self._name,
                                               remove_neck=self._remove_neck,
                                               remove_torso=self._remove_torso,
                                               remove_background=self._remove_background,
                                               all_serials=self._all_serials,
                                               color_correction=self._color_correction)
        return evaluation_manager

    def list_evaluated_checkpoint_ids(self, split: DatasetSplit) -> List[int]:
        pattern = re.compile(r"ckpt-(\d+)")
        checkpoint_candidates = []
        for checkpoint_folder in Path(f"{self._evaluations_location}").glob("ckpt-*"):
            if checkpoint_folder.is_dir():
                match = pattern.match(checkpoint_folder.name)
                if match:
                    checkpoint_candidates.append(int(match.group(1)))

        checkpoints = [checkpoint
                       for checkpoint in checkpoint_candidates
                       if Path(self.get_evaluation_result_path(checkpoint, split)).exists()]
        checkpoints = sorted(set(checkpoints))

        return checkpoints

    def _get_evaluation_name(self, checkpoint_id: int) -> str:
        evaluation_name = f"ckpt-{checkpoint_id}"

        if self._source_actor is None:
            if self._run_name.startswith('DA-'):
                if self._smooth_expression_condition:
                    evaluation_name += f"_smooth-expr-cond"
                if self._reproduce_train_metrics:
                    evaluation_name += f"_repr-train-metrics"
                if self._scheduler is not None:
                    evaluation_name += f"_scheduler-{self._scheduler}"
                if self._n_inference_steps != 20:
                    evaluation_name += f"_denoising-steps-{self._n_inference_steps}"
            if self._seed is not None:
                evaluation_name += f"_seed-{self._seed}"
            if self._remove_neck:
                evaluation_name += f"_remove-neck"
            if self._remove_torso:
                evaluation_name += f"_remove-torso"
            if self._remove_background:
                evaluation_name += f"_remove-background"
            if self._color_correction:
                evaluation_name += f"_color-correction"
            if self._all_serials:
                evaluation_name += f"_all-serials"
            if self._move_y is not None:
                evaluation_name += f"_move-y-{self._move_y}"
            if self._name is not None:
                evaluation_name += f"_name-{self._name}"

        return evaluation_name

    def _get_split_name(self, split: DatasetSplit) -> str:
        return split.name.lower().replace('_', '-')

    def get_evaluation_result_path(self, checkpoint_id: int, split: DatasetSplit) -> str:
        return f"{self._evaluations_location}/eval-result_{self._get_evaluation_name(checkpoint_id)}_{self._get_split_name(split)}.json"

    def get_evaluations_location(self) -> str:
        return self._evaluations_location
