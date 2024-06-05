from typing import Union, Optional

import numpy as np
import torch
from elias.folder import ModelFolder
from elias.manager import ModelManager
from elias.manager.model import _ModelType, _ModelConfigType, _OptimizationConfigType
from elias.util import ensure_directory_exists_for_file

from diffusion_avatars.dataset.rendering_dataset import RenderingDatasetConfig
from diffusion_avatars.env import DIFFUSION_AVATARS_MODELS_PATH
from diffusion_avatars.config.diffusion_avatars import DiffusionAvatarsOptimizerConfig, DiffusionAvatarsTrainConfig, DiffusionAvatarsModelConfig


class DiffusionAvatarsModelManager(ModelManager[
                             'DiffusionAvatarsModel', DiffusionAvatarsModelConfig, DiffusionAvatarsOptimizerConfig, RenderingDatasetConfig, DiffusionAvatarsTrainConfig, None, None]):

    def __init__(self, run_name: str):
        super(DiffusionAvatarsModelManager, self).__init__(f"{DIFFUSION_AVATARS_MODELS_PATH}/diffusion-avatars", run_name,
                                                           checkpoint_name_format='ckpt-$.pt',
                                                           checkpoints_sub_folder='checkpoints',
                                                           evaluation_name_format='evaluation_$')

    def get_log_folder(self) -> str:
        return f"{self._location}/logs"

    def get_wandb_folder(self) -> str:
        return f"{self._location}/wandb"

    def _build_model(self, model_config: _ModelConfigType,
                     optimization_config: Optional[_OptimizationConfigType] = None, **kwargs) -> _ModelType:
        pass

    def _store_checkpoint(self, model: 'DiffusionAvatarsModel', checkpoint_file_name: str, **kwargs):
        checkpoint_path = f"{self.get_checkpoint_folder()}/{checkpoint_file_name}"
        ensure_directory_exists_for_file(checkpoint_path)
        torch.save(model.state_dict(), f"{self.get_checkpoint_folder()}/{checkpoint_file_name}")

    def _load_checkpoint(self,
                         checkpoint_file_name: Union[str, int],
                         enable_extended_temporal_attention: bool = False,
                         use_consistency_decoder: bool = False,
                         **kwargs) -> 'DiffusionAvatarsModel':
        from diffusion_avatars.model.diffusion_avatars import DiffusionAvatarsModel

        model_config = self.load_model_config()
        if enable_extended_temporal_attention:
            model_config.temporal_config.enable_extended_temporal_attention = True
        if use_consistency_decoder:
            model_config.use_consistency_decoder = True

        model = DiffusionAvatarsModel(
            model_config=model_config,
            dataset_config=self.load_dataset_config(),
            train_config=self.load_train_setup(),
            optimizer_config=self.load_optimization_config(),
        )
        checkpoint = torch.load(f"{self.get_checkpoint_folder()}/{checkpoint_file_name}")
        incompatible_keys = model.load_state_dict(checkpoint, strict=False)
        if any(["temporal_convolutions" in missing_key for missing_key in incompatible_keys.missing_keys]):
            # Backward compatibility: Some models were trained with temporal_batch_size=1 but don't have any
            # temporal convolutions in the model.
            # Retry loading with temporal_batch_size=0

            print(f"Could not load checkpoint due to missing keys: {incompatible_keys.missing_keys}. "
                  f"Retrying with temporal_batch_size=0...")

            model_config.temporal_config.temporal_batch_size = 0
            model = DiffusionAvatarsModel(
                model_config=model_config,
                dataset_config=self.load_dataset_config(),
                train_config=self.load_train_setup(),
                optimizer_config=self.load_optimization_config(),
            )
            model.load_state_dict(checkpoint)
        else:
            assert not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys, f"missing keys: {incompatible_keys.missing_keys}, unexpected keys: {incompatible_keys.unexpected_keys}"

        return model

    @staticmethod
    def build_inference_pipeline(model_config: DiffusionAvatarsModelConfig, train_config: DiffusionAvatarsTrainConfig) \
            -> 'StableDiffusionControlNetPipeline':
        from diffusion_avatars.model.diffusion_avatars import DiffusionAvatarsModel
        from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler

        weight_dtype = train_config.get_dtype()

        unet, controlnet = DiffusionAvatarsModel.build_unet_and_controlnet(model_config)
        vae = DiffusionAvatarsModel.build_vae(model_config)
        tokenizer = DiffusionAvatarsModel.build_tokenizer(model_config)
        text_encoder = DiffusionAvatarsModel.build_text_encoder(model_config)

        vae.to(weight_dtype)
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_config.diffusion_model_name,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            revision=model_config.revision,
            torch_dtype=train_config.get_dtype(),
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=True)

        pipeline.enable_xformers_memory_efficient_attention()

        return pipeline

    def get_fixed_noise_path(self) -> str:
        return f"{self.get_location()}/fixed_noise.npz"

    def store_fixed_noise(self, noise: np.ndarray):
        np.savez(self.get_fixed_noise_path(), noise=noise)

    def load_fixed_noise(self) -> np.ndarray:
        return np.load(self.get_fixed_noise_path())['noise']

    def get_participant_id(self) -> int:
        data_stats = self.load_dataset_config().get_data_statistics()
        participant_id = int(next(iter(data_stats.available_sequences.keys())))
        return participant_id


class DiffusionAvatarsModelFolder(ModelFolder[DiffusionAvatarsModelManager]):

    def __init__(self):
        super(DiffusionAvatarsModelFolder, self).__init__(f"{DIFFUSION_AVATARS_MODELS_PATH}/diffusion-avatars", 'DA', localize_via_run_name=True)

    def open_run(self, run_name_or_id: Union[str, int]) -> DiffusionAvatarsModelManager:
        return super().open_run(run_name_or_id)

    def new_run(self, name: Optional[str] = None) -> DiffusionAvatarsModelManager:
        return super().new_run(name)
