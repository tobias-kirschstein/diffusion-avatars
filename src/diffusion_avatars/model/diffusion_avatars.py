from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Type, Any

import diffusers.schedulers
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
# from consistencydecoder import ConsistencyDecoder
from diffusion_avatars.config.diffusion_avatars import DiffusionAvatarsOptimizerConfig, DiffusionAvatarsTrainConfig, DiffusionAvatarsModelConfig
from diffusion_avatars.model.inference_output import DiffusionAvatarsInferenceOutput, SchedulerType
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, DDPMScheduler, AutoencoderKL, \
    ControlNetModel, get_scheduler
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from diffusers.schedulers.scheduling_ddim import rescale_zero_terminal_snr, DDIMScheduler
from dreifus.image import Img
from elias.config import Config
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig, PreTrainedModel

from diffusion_avatars.data_manager.rendering_data_manager import RenderingName
from diffusion_avatars.dataset.rendering_dataset import RenderingDatasetInferenceBatch, RenderingDatasetBatch, \
    RenderingDatasetConfig
from diffusion_avatars.model.conditional_sd_pipeline import ConditionalStableDiffusionPipeline
# from diffusion_avatars.model.no_variance_scheduler import NoVarianceDDPMScheduler
from diffusion_avatars.model.temporal.video_controlnet import VideoControlNetModel
from diffusion_avatars.model.temporal.video_unet import VideoUNet2DConditionModel
from diffusion_avatars.renderer.provider.nphm_provider import NPHMProvider


@dataclass
class DiffusionAvatarsExperimentConfig(Config):
    dataset_config: RenderingDatasetConfig
    model_config: DiffusionAvatarsModelConfig
    train_config: DiffusionAvatarsTrainConfig
    optimizer_config: DiffusionAvatarsOptimizerConfig


class DiffusionAvatarsModel(LightningModule):

    def __init__(self,
                 model_config: DiffusionAvatarsModelConfig,
                 dataset_config: RenderingDatasetConfig,
                 train_config: Optional[DiffusionAvatarsTrainConfig] = None,
                 optimizer_config: Optional[DiffusionAvatarsOptimizerConfig] = None
                 ):
        super(DiffusionAvatarsModel, self).__init__()
        pass

        self._model_config = model_config
        self._dataset_config = dataset_config
        self._train_config = train_config
        self._optimizer_config = optimizer_config

        # Load scheduler and models
        self.scheduler = self.build_scheduler(model_config)
        self.text_encoder = self.build_text_encoder(model_config)
        self.tokenizer = self.build_tokenizer(model_config)
        self.vae = self.build_vae(model_config)
        self.unet, self.controlnet = self.build_unet_and_controlnet(model_config)
        if model_config.use_consistency_decoder:
            self.consistency_decoder = ConsistencyDecoder()
        self.accelerator = None

        self.vae.requires_grad_(False)
        if not model_config.no_pretrained:
            self.unet.requires_grad_(False)
            # Enable gradients for newly introduced temporal layers. This will also automatically put them into checkpoints
            for _, param in self.unet.get_learnable_parameters().items():
                param.requires_grad = True

        self.text_encoder.requires_grad_(False)
        self.controlnet.train()

        # dtype
        self._weight_dtype = torch.float32
        if train_config is not None:
            self._weight_dtype = train_config.get_dtype()

        # Inference pipeline
        if model_config.no_pretrained:
            # In this case, controlnet is just the ControlnetConditionalEmbedding that maps 512 -> 64 latent dimension
            # inference pipeline expects already encoded conditions as input
            self.inference_pipeline = ConditionalStableDiffusionPipeline.from_pretrained(
                self._model_config.diffusion_model_name,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                safety_checker=None,
                revision=self._model_config.revision,
                torch_dtype=self._weight_dtype,
            )
        else:
            self.inference_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self._model_config.diffusion_model_name,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                controlnet=self.controlnet,
                safety_checker=None,
                revision=self._model_config.revision,
                torch_dtype=self._weight_dtype,
            )
        extra_kwargs = {
            'rescale_betas_zero_snr': model_config.rescale_betas_zero_snr
        }
        if model_config.predict_x0:
            extra_kwargs['prediction_type'] = 'sample'
        if model_config.n_train_steps is not None:
            extra_kwargs['num_train_timesteps'] = model_config.n_train_steps
        if model_config.use_trailing_timestep_spacing:
            extra_kwargs['timestep_spacing'] = 'trailing'
        if model_config.n_train_steps == 1:
            extra_kwargs['rescale_betas_zero_snr'] = False
            extra_kwargs['trained_betas'] = torch.tensor([1])

        if self._model_config.use_ddpm_inference_scheduler:
            self.inference_pipeline.scheduler = DDPMScheduler.from_config(self.inference_pipeline.scheduler.config, **extra_kwargs)
        else:
            # TODO: Cannot use zero_snr stuff for inference since UniPC gives black images due to -inf popping up in the formulas in between
            # if model_config.rescale_betas_zero_snr:
            #     scheduler_config = self.inference_pipeline.scheduler.config
            #     betas = (
            #             torch.linspace(scheduler_config.beta_start ** 0.5,
            #                            scheduler_config.beta_end ** 0.5,
            #                            scheduler_config.num_train_timesteps,
            #                            dtype=torch.float32) ** 2
            #     )
            #     betas = rescale_zero_terminal_snr(betas)
            #     extra_kwargs['trained_betas'] = betas
            self.inference_pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.inference_pipeline.scheduler.config, **extra_kwargs)
        # self.inference_pipeline.scheduler = NoVarianceDDPMScheduler.from_config(self.inference_pipeline.scheduler.config)

        self.inference_pipeline.set_progress_bar_config(disable=True)
        self.inference_pipeline.enable_xformers_memory_efficient_attention()

        # Neural Textures
        self.neural_texture_lookup = None
        self.neural_texture_fields = None
        self.neural_texture_lookup_uv = None
        self.neural_texture_lookup_uw = None
        self.neural_texture_lookup_vw = None
        self.ambient_texture_lookup = None
        assert not dataset_config.use_canonical_coordinates or dataset_config.use_texture_fields or dataset_config.use_texture_triplanes, \
            "If canonical coordinates is used, use_texture_fields or use_texture_triplanes must be true!"
        if dataset_config.use_neural_textures:
            if dataset_config.use_texture_fields:
                dimensions = 3 if dataset_config.use_canonical_coordinates else 2
                self.neural_texture_fields = nn.ModuleList([dataset_config.texture_field_config.create_encoding(dimensions=dimensions)
                                                            for _ in range(model_config.n_participants)])
            elif dataset_config.use_texture_triplanes:
                self.neural_texture_lookup_uv = nn.Embedding(
                    model_config.n_participants,
                    dataset_config.res_neural_textures * dataset_config.res_neural_textures * dataset_config.dim_neural_textures)
                self.neural_texture_lookup_uw = nn.Embedding(
                    model_config.n_participants,
                    dataset_config.res_neural_textures * dataset_config.res_neural_textures * dataset_config.dim_neural_textures)
                self.neural_texture_lookup_vw = nn.Embedding(
                    model_config.n_participants,
                    dataset_config.res_neural_textures * dataset_config.res_neural_textures * dataset_config.dim_neural_textures)
                nn.init.normal_(self.neural_texture_lookup_uv.weight, std=self._optimizer_config.init_neural_textures_gain)
                nn.init.normal_(self.neural_texture_lookup_uw.weight, std=self._optimizer_config.init_neural_textures_gain)
                nn.init.normal_(self.neural_texture_lookup_vw.weight, std=self._optimizer_config.init_neural_textures_gain)
            elif dataset_config.use_texture_hierarchy:
                self.neural_texture_hierarchy = nn.ModuleList([nn.Embedding(
                    model_config.n_participants,
                    (dataset_config.res_neural_textures // 2 ** i) * (dataset_config.res_neural_textures // 2 ** i) * dataset_config.dim_neural_textures)
                    for i in range(self._dataset_config.n_texture_hierarchy_levels)])
                for neural_texture_lookup in self.neural_texture_hierarchy:
                    nn.init.normal_(neural_texture_lookup.weight, std=self._optimizer_config.init_neural_textures_gain)
            else:
                self.neural_texture_lookup = nn.Embedding(
                    model_config.n_participants,
                    dataset_config.res_neural_textures * dataset_config.res_neural_textures * dataset_config.dim_neural_textures)
                nn.init.normal_(self.neural_texture_lookup.weight, std=self._optimizer_config.init_neural_textures_gain)

        if dataset_config.use_ambient_textures:
            self.ambient_texture_lookup = nn.Embedding(
                model_config.n_participants,
                dataset_config.res_neural_textures * dataset_config.res_neural_textures * dataset_config.dim_neural_textures)
            nn.init.normal_(self.ambient_texture_lookup.weight, std=self._optimizer_config.init_neural_textures_gain)

        if dataset_config.use_background_texture:
            self.learnable_background_color = nn.Parameter(torch.randn(dataset_config.dim_neural_textures))  # [D]
        else:
            self.learnable_background_color = None

        # Expression Codes Cross-attention
        if model_config.temporal_config.use_expression_condition:

            if self._dataset_config.get_data_config().use_nphm and not self._dataset_config.force_flame_expression_condition:
                d_expression_codes = 200  # NPHM has 200 expression codes

                if self._dataset_config.include_eye_condition:
                    d_expression_codes += 6
            else:
                d_expression_codes = 100  # FLAME has 100 expression codes

            self.expression_codes_mlp = nn.Linear(
                d_expression_codes,
                self.unet.cross_attention_dim * model_config.temporal_config.n_expression_tokens)

        # Cast vae, unet and text_encoder to weight_dtype
        self.vae.to(dtype=self._weight_dtype)

        learnable_parameters = self.unet.get_learnable_parameters()
        learnable_parameters_float_copy = {n: p.clone() for n, p in learnable_parameters.items()}
        self.unet.to(dtype=self._weight_dtype)
        # Ensure that newly introduced layers have float32 weights.
        # This means, we have to undo parts of what self.unet.to(...) does
        for key, param in learnable_parameters_float_copy.items():
            param_dict = {k: p for k, p in self.unet.named_parameters()}
            param_dict[key].data = param.data

        self.text_encoder.to(dtype=self._weight_dtype)

        # Ignore pretrained models in the module's state dict
        # That way, their weights won't be persistent in the checkpoint which saves storage
        # Note, that if no_pretrained is set, all unet parameters will be included in get_learnable_parameters() and thus not be ignored here
        self.text_encoder._register_state_dict_hook(self._ignore_in_state_dict)
        self.vae._register_state_dict_hook(self._ignore_in_state_dict)
        self.unet._register_state_dict_hook(self._ignore_in_state_dict)

        self.text_encoder.register_load_state_dict_post_hook(self._ignore_in_load_state_dict)
        self.vae.register_load_state_dict_post_hook(self._ignore_in_load_state_dict)
        self.unet.register_load_state_dict_post_hook(self._ignore_in_load_state_dict)

    def _ignore_in_state_dict(self, module: nn.Module, state_dict: Dict[str, nn.Module], prefix: str, local_metadata):
        # Remove everything from the state dict with prefix of the module that is to be ignored
        learnable_unet_parameters = self.unet.get_learnable_parameters()
        for key in list(state_dict.keys()):
            # Do not remove parameters for temporal layers from state dict.
            # key = "unet.down_blocks.0.temporal_convolutions.0.weight"
            # learnable_unet_parameters = "down_blocks.0.temporal_convolutions.0.weight"
            if key.startswith(prefix) and not key[len(prefix):] in learnable_unet_parameters:
                del state_dict[key]

        return state_dict

    def _ignore_in_load_state_dict(self, module: nn.Module, incompatible_keys: Tuple[List[str], List[str]]) -> None:
        # Remove all items from "missing keys"
        # This module isn't initialized from a state dict
        incompatible_keys[0].clear()

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------

    @staticmethod
    def build_scheduler(model_config: DiffusionAvatarsModelConfig) -> DDPMScheduler:
        config = DDPMScheduler.load_config(model_config.diffusion_model_name, subfolder="scheduler")
        scheduler_cls = getattr(diffusers.schedulers, config['_class_name'])
        extra_kwargs = {
            'rescale_betas_zero_snr': model_config.rescale_betas_zero_snr,
        }
        if model_config.use_trailing_timestep_spacing:
            extra_kwargs['timestep_spacing'] = 'trailing'
        if model_config.n_train_steps is not None:
            extra_kwargs['num_train_timesteps'] = model_config.n_train_steps
        if model_config.predict_x0:
            extra_kwargs['prediction_type'] = 'sample'

        if model_config.n_train_steps == 1:
            extra_kwargs['rescale_betas_zero_snr'] = False
            extra_kwargs['trained_betas'] = torch.tensor([1])

        if model_config.use_original_scheduler:
            scheduler = scheduler_cls.from_pretrained(model_config.diffusion_model_name, subfolder="scheduler", **extra_kwargs)
        else:
            scheduler = DDPMScheduler.from_pretrained(model_config.diffusion_model_name, subfolder="scheduler", **extra_kwargs)
        return scheduler

    @staticmethod
    def build_tokenizer(model_config: DiffusionAvatarsModelConfig) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.diffusion_model_name, subfolder="tokenizer", revision=model_config.revision, use_fast=False,
        )
        return tokenizer

    @staticmethod
    def build_text_encoder(model_config: DiffusionAvatarsModelConfig) -> CLIPTextModel:
        text_encoder_cls = get_text_encoder_cls(model_config.diffusion_model_name, model_config.revision)
        text_encoder = text_encoder_cls.from_pretrained(
            model_config.diffusion_model_name, subfolder="text_encoder", revision=model_config.revision
        )
        return text_encoder

    @staticmethod
    def build_vae(model_config: DiffusionAvatarsModelConfig) -> AutoencoderKL:
        vae = AutoencoderKL.from_pretrained(model_config.diffusion_model_name, subfolder="vae",
                                            revision=model_config.revision)
        return vae

    @staticmethod
    def build_unet_and_controlnet(model_config: DiffusionAvatarsModelConfig) \
            -> Tuple[VideoUNet2DConditionModel, ControlNetModel]:
        if model_config.no_pretrained:
            config = VideoUNet2DConditionModel.load_config(model_config.diffusion_model_name, subfolder="unet")
            d_embedded_condition = config["block_out_channels"][0]
            config['in_channels'] = 4 + d_embedded_condition
            unet = VideoUNet2DConditionModel.from_config(config, temporal_config=model_config.temporal_config, no_pretrained=model_config.no_pretrained)

            # Note, we slightly abuse the controlnet variable here, because in the no_pretrained case, it is only the ControlNetConditioningEmbedding
            controlnet = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=d_embedded_condition,
                conditioning_channels=model_config.n_cond_channels)
        else:
            unet = VideoUNet2DConditionModel.from_pretrained(
                model_config.diffusion_model_name, subfolder="unet", revision=model_config.revision,
                temporal_config=model_config.temporal_config,
            )

            # controlnet = ControlNetModel.from_unet(unet)
            controlnet = VideoControlNetModel.from_unet(unet, temporal_config=model_config.temporal_config)

            # Adapt number of input channels for ControlNet
            block_out_channels = controlnet.config["block_out_channels"]
            conditioning_embedding_out_channels = controlnet.config["conditioning_embedding_out_channels"]
            conditioning_channels = model_config.n_cond_channels
            controlnet.controlnet_cond_embedding = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=block_out_channels[0],
                conditioning_channels=conditioning_channels,
                block_out_channels=conditioning_embedding_out_channels,
            )

            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()

        return unet, controlnet

    def change_inference_scheduler(self, scheduler: SchedulerType):
        scheduler_config = self.inference_pipeline.scheduler.config

        extra_kwargs = {
            'rescale_betas_zero_snr': self._model_config.rescale_betas_zero_snr
        }
        if self._model_config.use_trailing_timestep_spacing:
            extra_kwargs['timestep_spacing'] = 'trailing'
        if self._model_config.predict_x0:
            extra_kwargs['prediction_type'] = 'sample'
        if self._model_config.n_train_steps is not None:
            extra_kwargs['num_train_timesteps'] = self._model_config.n_train_steps
        if self._model_config.rescale_betas_zero_snr and scheduler == 'ddpm':
            betas = (
                    torch.linspace(scheduler_config.beta_start ** 0.5,
                                   scheduler_config.beta_end ** 0.5,
                                   scheduler_config.num_train_timesteps,
                                   dtype=torch.float32) ** 2
            )
            betas = rescale_zero_terminal_snr(betas)
            extra_kwargs['trained_betas'] = betas
        if self._model_config.n_train_steps == 1:
            extra_kwargs['rescale_betas_zero_snr'] = False
            extra_kwargs['trained_betas'] = torch.tensor([1])

        # Switch scheduler
        if scheduler == 'ddim':
            self.inference_pipeline.scheduler = DDIMScheduler.from_config(scheduler_config, **extra_kwargs)
        elif scheduler == 'ddpm':
            self.inference_pipeline.scheduler = DDPMScheduler.from_config(scheduler_config, **extra_kwargs)
        elif scheduler == 'linear-ddpm':
            scheduler_config['beta_schedule'] = 'linear'
            self.inference_pipeline.scheduler = DDPMScheduler.from_config(scheduler_config, **extra_kwargs)
        elif scheduler == 'no-variance-ddpm':
            self.inference_pipeline.scheduler = NoVarianceDDPMScheduler.from_config(scheduler_config, **extra_kwargs)
        elif scheduler == 'unipc':
            self.inference_pipeline.scheduler = UniPCMultistepScheduler.from_config(scheduler_config, **extra_kwargs)

    def setup_accelerator(self, accelerator_project_config: ProjectConfiguration) -> Accelerator:
        accelerator = Accelerator(
            gradient_accumulation_steps=self._train_config.gradient_accumulation_steps,
            mixed_precision=self._train_config.mixed_precision,
            log_with='wandb',
            project_config=accelerator_project_config,
        )

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            " Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training, copy of the weights should still be float32."
        )
        if self._model_config.no_pretrained:
            if accelerator.unwrap_model(self.controlnet).conv_in.weight.dtype != torch.float32:
                raise ValueError(
                    f"Controlnet loaded as datatype {accelerator.unwrap_model(self.controlnet).conv_in.weight.dtype}. {low_precision_error_string}"
                )
        else:
            if accelerator.unwrap_model(self.controlnet).dtype != torch.float32:
                raise ValueError(
                    f"Controlnet loaded as datatype {accelerator.unwrap_model(self.controlnet).dtype}. {low_precision_error_string}"
                )

        if self._dataset_config.use_neural_textures:
            if self._dataset_config.use_texture_fields:
                if accelerator.unwrap_model(self.neural_texture_fields[0]).params.dtype != torch.float32:
                    raise ValueError(
                        f"Neural Texture Lookup loaded as datatype {accelerator.unwrap_model(self.neural_texture_fields[0]).params.dtype}. {low_precision_error_string}"
                    )
            elif self._dataset_config.use_texture_triplanes:
                if accelerator.unwrap_model(self.neural_texture_lookup_uv).weight.dtype != torch.float32:
                    raise ValueError(
                        f"Neural Texture Lookup loaded as datatype {accelerator.unwrap_model(self.neural_texture_lookup_uv).dtype}. {low_precision_error_string}"
                    )
            elif self._dataset_config.use_texture_hierarchy:
                if accelerator.unwrap_model(self.neural_texture_hierarchy)[0].weight.dtype != torch.float32:
                    raise ValueError(
                        f"Neural Texture Lookup loaded as datatype {accelerator.unwrap_model(self.neural_texture_hierarchy)[0].dtype}. {low_precision_error_string}"
                    )
            else:
                if accelerator.unwrap_model(self.neural_texture_lookup).weight.dtype != torch.float32:
                    raise ValueError(
                        f"Neural Texture Lookup loaded as datatype {accelerator.unwrap_model(self.neural_texture_lookup).dtype}. {low_precision_error_string}"
                    )

        if self._dataset_config.use_ambient_textures:
            if accelerator.unwrap_model(self.ambient_texture_lookup).weight.dtype != torch.float32:
                raise ValueError(
                    f"Neural Texture Lookup loaded as datatype {accelerator.unwrap_model(self.ambient_texture_lookup).dtype}. {low_precision_error_string}"
                )

        for p in self.unet.get_learnable_parameters().values():
            if p.dtype != torch.float32:
                raise ValueError(
                    f"Temporal U-Net layers loaded as datatype {p.dtype}. {low_precision_error_string}"
                )

        self.accelerator = accelerator

        return accelerator

    def get_latent_shape(self) -> torch.Size:
        condition_height = self._dataset_config.resolution
        condition_width = self._dataset_config.resolution
        # num_channels_latents = self.unet.config.in_channels
        num_channels_latents = 4
        latent_shape = (num_channels_latents,
                        condition_height // self.inference_pipeline.vae_scale_factor,
                        condition_width // self.inference_pipeline.vae_scale_factor)
        return latent_shape

    def get_noise_texture_shape(self) -> torch.Size:
        condition_height = self._dataset_config.resolution
        condition_width = self._dataset_config.resolution
        # num_channels_latents = self.unet.config.in_channels
        num_channels_latents = 4
        latent_shape = (num_channels_latents,
                        self._model_config.remap_noise_scale * condition_height // self.inference_pipeline.vae_scale_factor,
                        self._model_config.remap_noise_scale * condition_width // self.inference_pipeline.vae_scale_factor)
        return latent_shape

    def prepare_batch(self,
                      batch: RenderingDatasetBatch,
                      latent_noise: Optional[torch.Tensor] = None,
                      neural_textures: Optional[torch.Tensor] = None,
                      neural_texture_triplanes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                      neural_texture_fields: Optional[List['tcnn.Encoding']] = None,
                      neural_texture_hierarchy: Optional[List[torch.Tensor]] = None,
                      ambient_textures: Optional[torch.Tensor] = None,
                      n_previous_frames: Optional[int] = None) -> RenderingDatasetBatch:
        B = len(batch)
        batch = batch.to(self.device)
        if self._dataset_config.use_neural_textures:
            if self._dataset_config.use_texture_fields:
                if neural_texture_fields is None:
                    neural_texture_fields = [self.neural_texture_fields[i_participant]
                                             for i_participant in batch.i_participant]
            elif self._dataset_config.use_texture_triplanes:
                if neural_texture_triplanes is None:
                    neural_textures_uv = self.neural_texture_lookup_uv(batch.i_participant)
                    neural_textures_uw = self.neural_texture_lookup_uw(batch.i_participant)
                    neural_textures_vw = self.neural_texture_lookup_vw(batch.i_participant)
                    neural_textures_uv = neural_textures_uv.reshape(
                        (B, self._dataset_config.res_neural_textures, self._dataset_config.res_neural_textures,
                         self._dataset_config.dim_neural_textures))
                    neural_textures_uw = neural_textures_uw.reshape(
                        (B, self._dataset_config.res_neural_textures, self._dataset_config.res_neural_textures,
                         self._dataset_config.dim_neural_textures))
                    neural_textures_vw = neural_textures_vw.reshape(
                        (B, self._dataset_config.res_neural_textures, self._dataset_config.res_neural_textures,
                         self._dataset_config.dim_neural_textures))
                    neural_texture_triplanes = (neural_textures_uv, neural_textures_uw, neural_textures_vw)
            elif self._dataset_config.use_texture_hierarchy:
                if neural_texture_hierarchy is None:
                    neural_texture_hierarchy = [neural_texture_lookup(batch.i_participant) for neural_texture_lookup in self.neural_texture_hierarchy]
                    neural_texture_hierarchy = [neural_texture_hierarchy[i].reshape(
                        (B,
                         self._dataset_config.res_neural_textures // 2 ** i,
                         self._dataset_config.res_neural_textures // 2 ** i,
                         self._dataset_config.dim_neural_textures))
                        for i in range(len(neural_texture_hierarchy))]
            else:
                if neural_textures is None:
                    # Use i_participant field of batch to take neural textures from train subjects
                    neural_textures = self.neural_texture_lookup(batch.i_participant)
                    neural_textures = neural_textures.reshape(
                        (B, self._dataset_config.res_neural_textures, self._dataset_config.res_neural_textures,
                         self._dataset_config.dim_neural_textures))

        if self._dataset_config.use_ambient_textures:
            if ambient_textures is None:
                # Use i_participant field of batch to take neural textures from train subjects
                ambient_textures = self.ambient_texture_lookup(batch.i_participant)
                ambient_textures = ambient_textures.reshape(
                    (B, self._dataset_config.res_neural_textures, self._dataset_config.res_neural_textures,
                     self._dataset_config.dim_neural_textures))

        background_texture = None
        if self._dataset_config.use_background_texture:
            background_texture = self.learnable_background_color

        batch = self._dataset_config.prepare_batch(batch,
                                                   self.device,
                                                   latent_noise=latent_noise,
                                                   neural_textures=neural_textures,
                                                   neural_texture_triplanes=neural_texture_triplanes,
                                                   neural_texture_fields=neural_texture_fields,
                                                   neural_texture_hierarchy=neural_texture_hierarchy,
                                                   ambient_textures=ambient_textures,
                                                   background_texture=background_texture,
                                                   n_previous_frames=n_previous_frames)

        return batch

    def configure_optimizers(self) -> Any:
        accelerator_num_processes = 1  # TODO: How is accelerator.num_processes defined

        if self._optimizer_config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizers_and_schedulers = []

        control_net_optimizer = optimizer_class(
            list(self.controlnet.parameters()),
            lr=self._optimizer_config.learning_rate,
            betas=(self._optimizer_config.adam_beta1, self._optimizer_config.adam_beta2),
            weight_decay=self._optimizer_config.adam_weight_decay,
            eps=self._optimizer_config.adam_epsilon,
        )

        control_net_lr_scheduler = get_scheduler(
            self._optimizer_config.lr_scheduler,
            optimizer=control_net_optimizer,
            num_warmup_steps=self._optimizer_config.lr_warmup_steps * accelerator_num_processes,
            num_training_steps=self._train_config.get_max_train_steps(),
            num_cycles=self._optimizer_config.lr_num_cycles,
            power=self._optimizer_config.lr_power,
        )

        optimizers_and_schedulers.append({"optimizer": control_net_optimizer, "lr_scheduler": control_net_lr_scheduler})

        learnable_unet_parameters = self.unet.get_learnable_parameters().values()

        if learnable_unet_parameters:
            learnable_unet_parameters = list(learnable_unet_parameters)

            if self._model_config.temporal_config.use_expression_condition:
                learnable_unet_parameters = learnable_unet_parameters + list(self.expression_codes_mlp.parameters())

            unet_optimizer = optimizer_class(
                learnable_unet_parameters,
                lr=self._optimizer_config.learning_rate,
                betas=(self._optimizer_config.adam_beta1, self._optimizer_config.adam_beta2),
                weight_decay=self._optimizer_config.adam_weight_decay,
                eps=self._optimizer_config.adam_epsilon,
            )

            unet_scheduler = get_scheduler(
                self._optimizer_config.lr_scheduler,
                optimizer=unet_optimizer,
                num_warmup_steps=self._optimizer_config.lr_warmup_steps * accelerator_num_processes,
                num_training_steps=self._train_config.get_max_train_steps(),
                num_cycles=self._optimizer_config.lr_num_cycles,
                power=self._optimizer_config.lr_power,
            )

            optimizers_and_schedulers.append(
                {"optimizer": unet_optimizer, "lr_scheduler": unet_scheduler})

        # Neural Textures optimizer
        params_neural_textures = []
        if self._dataset_config.use_neural_textures:
            if self._dataset_config.use_texture_fields:
                params_neural_textures = []
                for neural_texture_field in self.neural_texture_fields:
                    params_neural_textures.extend(neural_texture_field.parameters())
            elif self._dataset_config.use_texture_triplanes:
                params_neural_textures = list(self.neural_texture_lookup_uv.parameters()) + list(self.neural_texture_lookup_uw.parameters()) + list(
                    self.neural_texture_lookup_vw.parameters())
            elif self._dataset_config.use_texture_hierarchy:
                params_neural_textures = list(self.neural_texture_hierarchy.parameters())
            else:
                params_neural_textures = list(self.neural_texture_lookup.parameters())

        if self._dataset_config.use_ambient_textures:
            params_neural_textures += list(self.ambient_texture_lookup.parameters())

        if self._dataset_config.use_background_texture:
            params_neural_textures += [self.learnable_background_color]

        if self._dataset_config.use_neural_textures or self._dataset_config.use_background_texture or self._dataset_config.use_ambient_textures:
            if self._optimizer_config.use_adam_for_neural_textures:
                optimizer_neural_textures = Adam(params_neural_textures,
                                                 lr=self._optimizer_config.learning_rate_neural_textures)
            else:
                optimizer_neural_textures = optimizer_class(params_neural_textures,
                                                            lr=self._optimizer_config.learning_rate_neural_textures)
            optimizers_and_schedulers.append({"optimizer": optimizer_neural_textures})

        # Accelerator prepare()
        if self.accelerator is not None:
            # Prepare ControlNet.
            self.controlnet = self.accelerator.prepare(self.controlnet)

            # Prepare Neural Textures
            if self._dataset_config.use_neural_textures:
                if self._dataset_config.use_texture_fields:
                    self.neural_texture_fields = self.accelerator.prepare(self.neural_texture_fields)
                elif self._dataset_config.use_texture_triplanes:
                    self.neural_texture_lookup_uv = self.accelerator.prepare(self.neural_texture_lookup_uv)
                    self.neural_texture_lookup_uw = self.accelerator.prepare(self.neural_texture_lookup_uw)
                    self.neural_texture_lookup_vw = self.accelerator.prepare(self.neural_texture_lookup_vw)
                elif self._dataset_config.use_texture_hierarchy:
                    self.neural_texture_hierarchy = self.accelerator.prepare(self.neural_texture_hierarchy)
                else:
                    self.neural_texture_lookup = self.accelerator.prepare(self.neural_texture_lookup)

            if self._dataset_config.use_ambient_textures:
                self.ambient_texture_lookup = self.accelerator.prepare(self.ambient_texture_lookup)

            # Prepare optimizers and schedulers
            for optimizer_and_scheduler in optimizers_and_schedulers:
                optimizer_and_scheduler["optimizer"] = self.accelerator.prepare(optimizer_and_scheduler["optimizer"])
                if "lr_scheduler" in optimizer_and_scheduler:
                    optimizer_and_scheduler["lr_scheduler"] = self.accelerator.prepare(
                        optimizer_and_scheduler["lr_scheduler"])

        return optimizers_and_schedulers

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def predict_noise(self,
                      noisy_latents: torch.Tensor,
                      timesteps: torch.Tensor,
                      controlnet_conditions: torch.Tensor,
                      expression_codes: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = noisy_latents.shape[0]

        # Get the text embedding for conditioning
        token_ids = self.tokenizer([self._train_config.global_prompt for _ in range(B)],
                                   max_length=self.tokenizer.model_max_length,
                                   padding="max_length", truncation=True,
                                   return_tensors="pt").input_ids.clone().detach().to(self.device)
        encoder_hidden_states = self.text_encoder(token_ids)[0]

        # Expression codes encoding
        expression_codes_encoded = None
        if self._model_config.temporal_config.use_expression_condition:
            expression_codes_encoded = self.expression_codes_mlp(expression_codes)
            B = expression_codes_encoded.shape[0]
            expression_codes_encoded = expression_codes_encoded.reshape(
                (B, self._model_config.temporal_config.n_expression_tokens, -1))

        if self._model_config.no_pretrained:
            encoded_conditions = self.controlnet(controlnet_conditions)

            unet_inputs = torch.concat([noisy_latents, encoded_conditions], dim=1)
            model_pred = self.unet(
                unet_inputs,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs={"expression_codes": expression_codes_encoded}
            ).sample
        else:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_conditions,
                return_dict=False,
            )

            # Predict the noise residual
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=self._weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=self._weight_dtype),
                cross_attention_kwargs={"expression_codes": expression_codes_encoded},
            ).sample

        return model_pred

    def get_loss_dict(self, batch: RenderingDatasetBatch, skip_prepare_batch: bool = False) -> Dict[str, torch.Tensor]:
        if not skip_prepare_batch:
            self.prepare_batch(batch)

        # Autoregressive generation of previous outputs
        self.prepare_autoregressive_inference(batch)

        # Convert images to latent space
        target_images = self._dataset_config.concat_target_images(batch).to(dtype=self._weight_dtype)
        B = len(target_images)
        encoded_target_images = self.vae.encode(target_images)
        if self._train_config.use_vae_mean:
            latents = encoded_target_images.latent_dist.mean
        else:
            latents = encoded_target_images.latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample a random timestep for each image
        n_train_timesteps = self.scheduler.config.num_train_timesteps
        if self._train_config.use_full_noise:
            n_train_timesteps += 1

        # Sample noise that we'll add to the latents  # TODO: should we fix noise for autoregressive inference during training? Can it be the same as GT noise?
        if self._train_config.share_temporal_noise and self._dataset_config.temporal_batch_size > 0:
            b = latents.shape[0] // self._dataset_config.temporal_batch_size
            if self._dataset_config.remap_noise:
                noise = torch.randn((b, *self.get_noise_texture_shape()),
                                    dtype=latents.dtype, device=latents.device)
            else:
                noise = torch.randn((b, *latents.shape[1:]),
                                    dtype=latents.dtype, device=latents.device)
            noise = noise.repeat_interleave(self._dataset_config.temporal_batch_size, dim=0)
            timesteps = torch.randint(0, n_train_timesteps, (b,), device=latents.device)
            timesteps = timesteps.repeat_interleave(self._dataset_config.temporal_batch_size, dim=0)
        else:
            if self._dataset_config.remap_noise:
                noise = torch.randn((B, *self.get_noise_texture_shape()),
                                    dtype=latents.dtype, device=latents.device)
            elif self._model_config.disable_noise:
                noise = torch.zeros_like(latents)
            else:
                noise = torch.randn_like(latents)
            timesteps = torch.randint(0, n_train_timesteps, (B,), device=latents.device)

        if self._dataset_config.remap_noise:
            noise = self._remap_noise(batch, noise)

        timesteps = timesteps.long()

        if self._train_config.use_full_noise:
            idx_full_noise = timesteps == n_train_timesteps - 1
            timesteps[idx_full_noise] = 0

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        if self._train_config.use_full_noise:
            timesteps[idx_full_noise] = n_train_timesteps - 1
            noisy_latents[idx_full_noise] = noise[idx_full_noise]

        conditions = self._dataset_config.concat_conditions(batch)
        controlnet_image = conditions.to(dtype=self._weight_dtype)

        model_pred = self.predict_noise(noisy_latents, timesteps, controlnet_image,
                                        expression_codes=batch.expression_codes)

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        elif self.scheduler.config.prediction_type == "sample":
            target = latents
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        loss_dict = dict()

        # Laplacian smoothing
        if self._optimizer_config.lambda_laplacian_reg > 0 and self._dataset_config.temporal_batch_size > 2:
            laplacian_kernel = torch.tensor([1, -2, 1], dtype=model_pred.dtype, device=model_pred.device)
            laplacian_kernel = laplacian_kernel[None, None]  # [1, 1, 3]
            T = self._dataset_config.temporal_batch_size
            B_full, C, H, W = model_pred.shape
            B_t = B_full // T
            model_pred_flattened = model_pred.view(B_t, T, C, H, W).permute(0, 2, 3, 4, 1).reshape(B_t * C * H * W, 1,
                                                                                                   T)
            model_pred_laplacian = F.conv1d(model_pred_flattened, laplacian_kernel)

            # This convolution "counts" for each position in model_pred_laplacian how many of the
            # 3 values (left, center, right) where invalid
            # This is necessary, as some temporal batches will not be complete (when they are at the beginning of
            # a sequence). Hence, we can only compute the laplacian on a shorter sequence in these cases
            # Since this would require some sort of loop, we instead compute the laplacian on the full BxT tensor
            # And then mask out the entries that correspond to invalid timesteps
            # Example:
            # 1 - invalid, 0 - valid
            #   1|1 0 0 0|0  conv1d  2 1 0 0   bool  x x - -
            #   0|0 0 0 0|0    ->    0 0 0 0    ->   - - - -
            invalid_temporal_mask = (target_images == 0).all(dim=-1).all(dim=-1).all(dim=-1)  # [B]
            invalid_temporal_mask = invalid_temporal_mask.reshape(B_t, T).unsqueeze(1).float()  # [B, 1, T]
            invalid_kernel = torch.tensor([1, 1, 1], dtype=torch.float32, device=model_pred.device)[
                None, None]  # [1, 1, 3]
            laplacian_invalid_mask = F.conv1d(invalid_temporal_mask, invalid_kernel)  # [B, 1, T-2]
            laplacian_invalid_mask = laplacian_invalid_mask.squeeze(1) > 0  # [B, T-2]
            T_l = model_pred_laplacian.shape[-1]
            model_pred_laplacian = model_pred_laplacian.reshape(B_t, C, H, W, T_l).permute(0, 4, 1, 2,
                                                                                           3)  # [B_t, T, C, H, W]
            model_pred_laplacian = model_pred_laplacian[~laplacian_invalid_mask]

            loss_dict["laplacian_reg"] = (model_pred_laplacian ** 2).mean()

        if self._dataset_config.temporal_batch_size > 0:
            # Not all batches will have a full temporal stream, because batches that were sampled early in the sequence
            # might not have previous frames available.
            # In this case, the target will be all 0s (CxHxW)
            # We detect this here and exclude those from the supervision
            invalid_temporal_mask = (target_images == 0).all(dim=-1).all(dim=-1).all(dim=-1)
            model_pred = model_pred[~invalid_temporal_mask]
            target = target[~invalid_temporal_mask]

        # Mouth loss
        if self._optimizer_config.lambda_mouth_loss > 0 and self._dataset_config.include_mouth_mask:
            mouth_masks = batch.mouth_masks[:, ::8, ::8]  # [B, 64, 64]
            mouth_masks = mouth_masks.unsqueeze(1)  # [B, 1, 64, 64]
            mouth_masks = mouth_masks.expand(B, model_pred.shape[1], mouth_masks.shape[-2], mouth_masks.shape[-1])  # [B, 4, 64, 64]
            mse_mouth_loss = ((model_pred[mouth_masks] - target[mouth_masks]) ** 2).sum() / model_pred.numel()
            loss_dict[f"mse_{self.scheduler.config.prediction_type}_mouth_loss"] = mse_mouth_loss

        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss_dict[f"mse_{self.scheduler.config.prediction_type}_loss"] = mse_loss

        if self._optimizer_config.lambda_neural_texture_rgb_loss > 0:
            neural_textured_rgb_rendering = batch.neural_textured_rendering[:, -3:]
            neural_texture_rgb_loss = F.mse_loss(neural_textured_rgb_rendering, batch.target_images)
            loss_dict["neural_texture_rgb_loss"] = neural_texture_rgb_loss

        return loss_dict

    def combine_losses(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Combine losses into a single scalar by using lambda weightings
        loss = loss_dict[f"mse_{self.scheduler.config.prediction_type}_loss"]

        if self._optimizer_config.lambda_neural_texture_rgb_loss > 0:
            loss = loss + self._optimizer_config.lambda_neural_texture_rgb_loss * loss_dict["neural_texture_rgb_loss"]

        if self._optimizer_config.lambda_laplacian_reg > 0 and "laplacian_reg" in loss_dict:
            loss = loss + self._optimizer_config.lambda_laplacian_reg * loss_dict["laplacian_reg"]

        if self._optimizer_config.lambda_mouth_loss > 0 and f"mse_{self.scheduler.config.prediction_type}_mouth_loss" in loss_dict:
            loss = loss + self._optimizer_config.lambda_mouth_loss * loss_dict[f"mse_{self.scheduler.config.prediction_type}_mouth_loss"]

        return loss

    def training_step(self, batch: RenderingDatasetBatch) -> torch.Tensor:
        loss_dict = self.get_loss_dict(batch)
        loss = self.combine_losses(loss_dict)

        return loss

    def _remap_noise(self, batch: RenderingDatasetInferenceBatch, latent_noise: torch.Tensor) -> torch.Tensor:
        latent_shape = self.get_latent_shape()
        uvs = batch.conditions[RenderingName.UV]
        uvs = torch.nn.functional.interpolate(uvs, size=(latent_shape[1], latent_shape[2]), mode="nearest")
        masks = torch.nn.functional.interpolate(batch.conditions[RenderingName.MASK].to(torch.uint8), size=(latent_shape[1], latent_shape[2]),
                                                mode="nearest").bool()
        if self._dataset_config.use_canonical_coordinates and self._dataset_config.get_data_config().use_nphm:
            uvs = torch.from_numpy(NPHMProvider.canonical_coordinates_to_uv(uvs.permute(0, 2, 3, 1).cpu().numpy())).cuda()
        else:
            uvs = uvs.permute(0, 2, 3, 1).contiguous()

        uvs = (uvs + 1) / 2  # uvs should be in [0, 1]
        noise = latent_noise.permute(0, 2, 3, 1).contiguous()
        noise_textured_rendering = dr.texture(noise, uvs)

        latent_noise = torch.nn.functional.interpolate(latent_noise, size=(latent_shape[1], latent_shape[2]), mode="nearest")
        noise_textured_rendering[~masks[:, 0]] = latent_noise.permute(0, 2, 3, 1)[~masks[:, 0]]
        noise = noise_textured_rendering.permute(0, 3, 1, 2)

        return noise

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    def prepare_autoregressive_inference(self, batch: RenderingDatasetInferenceBatch):
        # Autoregressive generation of previous outputs
        if self._dataset_config.use_autoregressive and not self._dataset_config.use_teacher_forcing:
            # Assume that older timesteps should have the same latent noise as newer ones, if not explicitly
            # stated differently
            for previous_timestep in range(self._dataset_config.n_previous_frames):
                current_batch = batch.get_previous_batch(previous_timestep)
                previous_batch = batch.get_previous_batch(previous_timestep + 1)
                if previous_batch is not None \
                        and previous_batch.latent_noise is None \
                        and current_batch.latent_noise is not None:
                    previous_batch.latent_noise = current_batch.latent_noise[current_batch.previous_sample_ids]

            # From earliest previous timestep t -> latest previous timestep 1:
            #  1. Select part of batch for which we have conditions
            #  2. Call inference()
            #  3. Add generated images for timestep t to previous outputs of batch for timestep t - 1
            #  4. repeat

            # range() creates [n_previous_frames - 1, ..., 0]
            for previous_timestep in range(self._dataset_config.n_previous_frames - 1, -1, -1):
                current_batch = batch.get_previous_batch(previous_timestep)
                previous_batch = current_batch.previous_batch

                # Can also happen that none of the samples in the previous batch are valid.
                # In this case, previous batch will be None, and we do not have to do anything
                # previous_outputs is already initialized with all 0s
                if previous_batch is not None:
                    previous_inference_outputs = self.inference(previous_batch)
                    previous_outputs = torch.stack(
                        [Img.from_numpy(inference_output.prediction).to_normalized_torch().img
                         for inference_output in previous_inference_outputs])  # [B_p, 3, H, W]
                    previous_outputs = previous_outputs.to(device=self.device, dtype=self.dtype)
                    # TODO: In case n_previous_frames > 1: We also need to add the actual previous_output of previous_batch and shift it by 1 to the right
                    current_batch.previous_outputs[current_batch.previous_sample_ids, 0] = previous_outputs

    @torch.no_grad()
    def inference(self,
                  batch: RenderingDatasetInferenceBatch,
                  n_inference_steps: int = 20,
                  cfg_weight: float = 1) -> List[DiffusionAvatarsInferenceOutput]:
        # Assumes prepare_batch() has already been called, if neural textures are used
        #   (or neural textures were added manually to batch)
        # Assumes prepare_autoregressive_inference() has been called, if autoregressive is used
        #   (or previous outputs were added manually to batch)

        n_inference_steps = min(self._model_config.n_train_steps, n_inference_steps) if self._model_config.n_train_steps is not None else n_inference_steps  # Cannot do more inference steps than train steps

        # Only relevant for DDPM scheduler: Fix generator in each inference call to ensure variance noise (which is used in each step()) is the same for each
        # inference call
        if self._train_config.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(self._train_config.seed)

        # Prompts
        validation_conditions = self._dataset_config.concat_conditions(batch)
        B = len(validation_conditions)
        if batch.prompts is None:
            validation_prompts = [self._train_config.global_prompt for _ in range(B)]
        elif isinstance(batch.prompts, str):
            validation_prompts = [batch.prompts for _ in range(B)]
        else:
            validation_prompts = batch.prompts

        if self._dataset_config.temporal_batch_size > 0 and batch.latent_noise is not None:
            latent_noise = batch.latent_noise.repeat_interleave(self._dataset_config.temporal_batch_size, dim=0)
        elif self._model_config.disable_noise:
            latent_noise = torch.zeros_like(batch.latent_noise)
        else:
            latent_noise = batch.latent_noise

        if self._dataset_config.remap_noise:
            latent_noise = self._remap_noise(batch, latent_noise)

        cross_attention_kwargs = None
        if self._model_config.temporal_config.use_expression_condition:
            expression_codes = batch.expression_codes
            expression_codes_encoded = self.expression_codes_mlp(expression_codes)
            B = expression_codes_encoded.shape[0]
            expression_codes_encoded = expression_codes_encoded.reshape(
                (B, self._model_config.temporal_config.n_expression_tokens, -1))

            if cfg_weight > 1:
                # Duplicate expression codes. One of them is paired with prompt, the other is paired with unconditional forward pass
                expression_codes_encoded = torch.cat([expression_codes_encoded, expression_codes_encoded])

            cross_attention_kwargs = {"expression_codes": expression_codes_encoded}

        with torch.autocast("cuda"):
            if self._model_config.no_pretrained:
                encoded_conditions = self.controlnet(validation_conditions)

                images = self.inference_pipeline(
                    validation_prompts,
                    encoded_conditions=encoded_conditions,
                    num_inference_steps=n_inference_steps,
                    generator=generator,
                    latents=latent_noise,
                    guidance_scale=cfg_weight,
                    cross_attention_kwargs=cross_attention_kwargs,
                    output_type="latent" if self._model_config.use_consistency_decoder else 'pil',
                ).images
            else:
                images = self.inference_pipeline(
                    validation_prompts,
                    validation_conditions,
                    num_inference_steps=n_inference_steps,
                    generator=generator,
                    latents=latent_noise,
                    guidance_scale=cfg_weight,
                    cross_attention_kwargs=cross_attention_kwargs,
                    output_type="latent" if self._model_config.use_consistency_decoder else 'pil',
                ).images

            if self._model_config.use_consistency_decoder:
                images = images / self.vae.config.scaling_factor
                consistent_latent = self.consistency_decoder(images, schedule=[1.0])

                images = consistent_latent.permute(0, 2, 3, 1).cpu().numpy()
                images = (images + 1.0) * 127.5
                images = images.clip(0, 255).astype(np.uint8)

        outputs = []
        for i, image in enumerate(images):
            # for i, (validation_prompt, validation_condition) in enumerate(zip(validation_prompts, validation_conditions)):

            # with torch.autocast("cuda"):
            #     image = self.inference_pipeline(
            #         validation_prompt,
            #         validation_condition[None, ...],
            #         num_inference_steps=20,
            #         generator=generator,
            #         latents=None,
            #     ).images[0]

            image = np.asarray(image)  # PIL -> Numpy

            # Collect prediction and inference inputs into a single DiffVPInferenceOutput
            # This facilitates analysing the output of the network
            conditions_numpy = Img.from_normalized_torch(validation_conditions[i]).to_numpy().img
            conditions_numpy = self._dataset_config.split_renderings(conditions_numpy)
            conditions = dict(zip(self._dataset_config.rendering_names, conditions_numpy))

            if isinstance(batch, RenderingDatasetBatch):
                target_images = self._dataset_config.concat_target_images(batch)
                target_img = Img.from_normalized_torch(target_images[i]).to_numpy().img
            else:
                target_img = None

            if self._dataset_config.use_neural_textures:
                conditions_list = self._dataset_config.split_renderings(validation_conditions.permute(0, 2, 3, 1))
                idx_neural_textured_rendering = -2 if self._dataset_config.use_ambient_textures else -1
                neural_textured_rendering = conditions_list[idx_neural_textured_rendering][i]
                neural_textured_rendering = (neural_textured_rendering / 3).clip(min=-1, max=1)
                neural_textured_rendering = Img.from_normalized_torch(
                    neural_textured_rendering.permute(2, 0, 1)).to_numpy().img
            else:
                neural_textured_rendering = None

            if self._dataset_config.use_ambient_textures:
                conditions_list = self._dataset_config.split_renderings(validation_conditions.permute(0, 2, 3, 1))
                idx_neural_textured_rendering = -1
                ambient_textured_rendering = conditions_list[idx_neural_textured_rendering][i]
                ambient_textured_rendering = (ambient_textured_rendering / 3).clip(min=-1, max=1)
                ambient_textured_rendering = Img.from_normalized_torch(
                    ambient_textured_rendering.permute(2, 0, 1)).to_numpy().img
            else:
                ambient_textured_rendering = None

            if batch.previous_outputs is not None:
                previous_outputs = batch.previous_outputs[i]  # [N, 3, H, W]
                previous_outputs = [Img.from_normalized_torch(previous_output).to_numpy().img
                                    for previous_output in previous_outputs]
            else:
                previous_outputs = None

            single_latent_noise = Img.from_normalized_torch((latent_noise[i] / 3).clip(min=-1, max=1)).to_numpy().img

            output = DiffusionAvatarsInferenceOutput(image,
                                                     conditions,
                                                     target_image=target_img,
                                                     latent_noise=single_latent_noise,
                                                     neural_textured_rendering=neural_textured_rendering,
                                                     ambient_textured_rendering=ambient_textured_rendering,
                                                     previous_outputs=previous_outputs,
                                                     prompt=validation_prompts[i])
            outputs.append(output)

        return outputs


def get_text_encoder_cls(pretrained_model_name_or_path: str, revision: str) -> Type[PreTrainedModel]:
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
