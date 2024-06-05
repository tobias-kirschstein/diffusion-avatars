from typing import Tuple, Union, Optional, Dict, Any

import torch
from diffusers import ControlNetModel, UNet2DConditionModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.controlnet import ControlNetConditioningEmbedding, zero_module, ControlNetOutput
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, TextImageProjection, TextTimeEmbedding, \
    TextImageTimeEmbedding
from diffusers.models.unet_2d_blocks import get_down_block as get_down_block_2d, UNetMidBlock2DCrossAttn
from diffusers.utils import logging
from torch import nn

from diffusion_avatars.model.temporal.pseudo_3d_cross_attention_down_block import CrossAttnDownBlockPseudo3D
from diffusion_avatars.model.temporal.temporal_config import VideoControlNetConfig

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VideoControlNetModel(ControlNetModel):
    @register_to_config
    def __init__(
            self,
            in_channels: int = 4,
            conditioning_channels: int = 3,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
            ),
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1280,
            transformer_layers_per_block: Union[int, Tuple[int]] = 1,
            encoder_hid_dim: Optional[int] = None,
            encoder_hid_dim_type: Optional[str] = None,
            attention_head_dim: Union[int, Tuple[int]] = 8,
            num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            addition_embed_type: Optional[str] = None,
            addition_time_embed_dim: Optional[int] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",
            projection_class_embeddings_input_dim: Optional[int] = None,
            controlnet_conditioning_channel_order: str = "rgb",
            conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
            global_pool_conditions: bool = False,
            addition_embed_type_num_heads=64,

            temporal_config: VideoControlNetConfig = VideoControlNetConfig(),
    ):
        super().__init__()  # TODO: Only call super().super().__init__() ?

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kadinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )

        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kadinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # TODO: Enable expression attention?
            if temporal_config.use_temporal_controlnet_encoder \
                    and temporal_config.temporal_batch_size is not None and temporal_config.temporal_batch_size >= 1 \
                    and down_block_type == 'CrossAttnDownBlock2D':
                # Replace 'CrossAttnDownBlock2D' with 'CrossAttnDownBlockPseudo3D'
                if temporal_config.fix_pseudo_3d_block:
                    down_block = CrossAttnDownBlockPseudo3D(
                        num_layers=layers_per_block,
                        transformer_layers_per_block=transformer_layers_per_block[i],
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        add_downsample=not is_final_block,
                        resnet_eps=norm_eps,
                        resnet_act_fn=act_fn,
                        resnet_groups=norm_num_groups,
                        cross_attention_dim=cross_attention_dim,
                        downsample_padding=downsample_padding,
                        num_attention_heads=num_attention_heads[i],
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention[i],
                        upcast_attention=upcast_attention,
                        resnet_time_scale_shift=resnet_time_scale_shift,

                        temporal_config=temporal_config,
                    )
                else:
                    # Warning: This configuration is most likely wrong because attn_num_head_channels was passed
                    down_block = CrossAttnDownBlockPseudo3D(
                        input_channel,
                        output_channel,
                        time_embed_dim,
                        num_layers=layers_per_block,
                        resnet_eps=norm_eps,
                        resnet_time_scale_shift=resnet_time_scale_shift,
                        resnet_act_fn=act_fn,
                        resnet_groups=norm_num_groups,
                        attn_num_head_channels=attention_head_dim[i],
                        cross_attention_dim=cross_attention_dim,
                        downsample_padding=downsample_padding,
                        add_downsample=not is_final_block,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention[i],
                        upcast_attention=upcast_attention,

                        temporal_config=temporal_config,
                    )
            else:
                down_block = get_down_block_2d(
                    down_block_type,
                    num_layers=layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=num_attention_heads[i],
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                    downsample_padding=downsample_padding,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

    @classmethod
    def from_unet(
            cls,
            unet: UNet2DConditionModel,
            temporal_config: VideoControlNetConfig = VideoControlNetConfig(),
            controlnet_conditioning_channel_order: str = "rgb",
            conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
            load_weights_from_unet: bool = True,
    ):
        r"""
        Overwrite loading weights to ignore that temporal_convolutions are not in the pre-trained UNet

        Instantiate Controlnet class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,

            temporal_config=temporal_config,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            # Pre-trained U-Net does not have temporal convolutions
            # Hence, we just copy the weights for the new temporal modules here to avoid mismatched state_dict errors
            down_blocks_state_dict = unet.down_blocks.state_dict()

            # It might be that we already added some learnable layers to the temporal U-Net
            # Exclude them here, because ControlNet will have its own learnable layers inserted
            unet_learnable_parameters = unet.get_learnable_parameters()
            down_blocks_state_dict = {k: p for k, p in down_blocks_state_dict.items() if
                                      f"down_blocks.{k}" not in unet_learnable_parameters}
            temporal_convolutions_state_dict = {key: value
                                                for key, value in controlnet.down_blocks.state_dict().items()
                                                if 'temporal_convolutions' in key}
            temporal_attentions_state_dict = {key: value
                                              for key, value in controlnet.down_blocks.state_dict().items()
                                              if 'temporal_attentions' in key}
            down_blocks_state_dict.update(temporal_convolutions_state_dict)
            down_blocks_state_dict.update(temporal_attentions_state_dict)
            controlnet.down_blocks.load_state_dict(down_blocks_state_dict)

            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        """
        Copied from diffusers/models/controlnet.py
        Adjusted to forward denoising step to temporal attention layers

        The [`ControlNetModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.FloatTensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnet.ControlNetOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                if isinstance(downsample_block, CrossAttnDownBlockPseudo3D):
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        denoising_step=timestep.item() if len(timestep.shape) == 0 else timesteps[0].item(),
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0

            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )
