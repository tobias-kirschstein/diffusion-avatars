import math
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict

import torch
import wandb
from accelerate.utils import PrecisionType
from diffusion_avatars.model.temporal.temporal_config import VideoControlNetConfig

from elias.config import Config, implicit


@dataclass
class DiffusionAvatarsOptimizerConfig(Config):
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8

    learning_rate_neural_textures: float = 1e-2

    lr_scheduler: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] = 'constant'  #
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1  # Number of hard resets of the lr in cosine_with_restarts scheduler
    lr_power: float = 1.0  # Power factor of the polynomial scheduler
    scale_lr: bool = False  # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size

    use_8bit_adam: bool = False  # Need to change importlib_metadata.version("bitsandbytes-windows") in transformers.modeling_utils:2159
    set_grads_to_none: bool = False  # Save more memory by using setting grads to None instead of zero
    max_grad_norm: float = 1.0
    use_adam_for_neural_textures: bool = False  # Per default, AdamW is used
    init_neural_textures_gain: float = 1

    lambda_neural_texture_rgb_loss: float = 0  # Whether to force the first 3 channels to explain the GT RGB image
    lambda_laplacian_reg: float = 0  # regularize predictions in temporal batch to be similar
    lambda_mouth_loss: float = 1  # How much mouth region should be weighted


@dataclass
class DiffusionAvatarsTrainConfig(Config):
    name: Optional[str] = None
    project_name: str = 'diffusion-avatars'
    group_name: Optional[str] = None
    seed: int = 181998

    num_train_epochs: int = 10000
    num_train_iterations: int = -1  # Training ends when either num_train_epochs or num_train_iterations is reached
    train_batch_size: int = 4
    validation_batch_size: Optional[int] = None
    n_samples_per_epoch: int = implicit()
    dataloader_num_workers: int = 0
    gradient_accumulation_steps: int = 1
    mixed_precision: PrecisionType = PrecisionType.FP16

    save_model_every: int = 5000
    validate_every: int = 1000
    global_prompt: str = ""
    validation_prompt: List[str] = field(default_factory=lambda: [""])
    n_validation_images: int = 1

    fix_noise: bool = False
    use_full_noise: bool = False  # Whether to also train on full noise images during training
    use_vae_mean: bool = False  # Whether to use deterministic mean of the VAE encoder instead of sampling GT images
    share_temporal_noise: bool = False  # If temporal_batch_size > 0, this flag will broadcast one noise image across the temporal dimension

    def init_wandb(self):
        wandb.init(project=self.project_name,
                   group=self.group_name,
                   name=self.name)

    def get_dtype(self) -> torch.dtype:
        weight_dtype = torch.float32
        if self.mixed_precision == PrecisionType.FP16:
            weight_dtype = torch.float16
        elif self.mixed_precision == PrecisionType.BF16:
            weight_dtype = torch.bfloat16
        return weight_dtype

    def get_n_batches_per_epoch(self) -> int:
        return math.ceil(self.n_samples_per_epoch / self.train_batch_size)

    def get_n_update_steps_per_epoch(self) -> int:
        return math.ceil(self.n_samples_per_epoch / self.train_batch_size / self.gradient_accumulation_steps)

    def get_max_train_steps(self) -> int:
        max_train_steps = self.num_train_epochs * self.get_n_update_steps_per_epoch()
        if self.num_train_iterations > 0:
            max_train_steps = min(self.num_train_iterations, max_train_steps)

        return max_train_steps


@dataclass
class DiffusionAvatarsModelConfig(Config):
    diffusion_model_name: str = "stabilityai/stable-diffusion-2-1-base"  # "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    n_cond_channels: int = implicit()
    n_participants: int = implicit()

    use_original_scheduler: bool = False  # Whether to use the same scheduler as in the scheduling config of pretrained
    use_ddpm_inference_scheduler: bool = False
    use_consistency_decoder: bool = False
    temporal_config: VideoControlNetConfig = VideoControlNetConfig()

    remap_noise_scale: int = 2  # How much larger the noise texture should be than the actual noise image size
    rescale_betas_zero_snr: bool = False  # Whether to fix the noise schedule to also include full noise images during training
    use_trailing_timestep_spacing: bool = False  # Whether to use trailing instead of linspace timestep spacing
    n_train_steps: Optional[int] = None
    predict_x0: bool = False
    no_pretrained: bool = False
    disable_noise: bool = False

    @classmethod
    def _backward_compatibility(cls, loaded_config: Dict):
        super()._backward_compatibility(loaded_config)

        # Previously, there was no "temporal_config" field.
        # Instead, the hyperparameters were all directly set in the model config
        # However, this is annoying for adding more parameters, as the signatures of several classes have to be changed
        # Hence, the temporal_config field was introduced
        if 'temporal_config' not in loaded_config:
            loaded_config['temporal_config'] = dict()

        if 'temporal_batch_size' in loaded_config:
            loaded_config['temporal_config']['temporal_batch_size'] = loaded_config['temporal_batch_size']
            del loaded_config['temporal_batch_size']

        if 'temporal_kernel_size' in loaded_config:
            loaded_config['temporal_config']['temporal_kernel_size'] = loaded_config['temporal_kernel_size']
            del loaded_config['temporal_kernel_size']

        if 'temporal_padding_mode' in loaded_config:
            loaded_config['temporal_config']['temporal_padding_mode'] = loaded_config['temporal_padding_mode']
            del loaded_config['temporal_padding_mode']
