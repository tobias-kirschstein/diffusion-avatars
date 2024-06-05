from typing import Optional, Dict, Any

import torch
from diffusers.models import DualTransformer2DModel
from diffusers.models.resnet import ResnetBlock2D, Downsample2D
from torch import nn

from diffusion_avatars.model.expression.expression_attention_transformer_2d import ExpressionAttentionTransformer2DModel
from diffusion_avatars.model.temporal.temporal_attention import TemporalAttention
from diffusion_avatars.model.temporal.temporal_config import VideoControlNetConfig
from diffusion_avatars.model.temporal.temporal_convolution import TemporalConvolution


class CrossAttnDownBlockPseudo3D(nn.Module):
    """
    Adapted from CrossAttnDownBlock2D to include an additional 1D temporal convolution after each convolution
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads=1,
            cross_attention_dim=1280,
            output_scale_factor=1.0,
            downsample_padding=1,
            add_downsample=True,
            dual_cross_attention=False,
            use_linear_projection=False,
            only_cross_attention=False,
            upcast_attention=False,
            attention_type="default",

            temporal_config: VideoControlNetConfig = VideoControlNetConfig(),
    ):
        super().__init__()
        resnets = []
        attentions = []
        temporal_convolutions = []
        temporal_attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    ExpressionAttentionTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        use_expression_attention=temporal_config.use_expression_condition
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )

            # Temporal convolutions
            if temporal_config.use_temporal_convolution and temporal_config.temporal_batch_size > 0:
                temporal_convolution = TemporalConvolution(in_channels=out_channels,
                                                           out_channels=out_channels,
                                                           kernel_size=temporal_config.temporal_kernel_size,
                                                           stride=1,
                                                           padding=temporal_config.temporal_kernel_size // 2,
                                                           padding_mode=temporal_config.temporal_padding_mode,
                                                           temporal_config=temporal_config)
            else:
                temporal_convolution = None

            temporal_convolutions.append(
                temporal_convolution
            )

            # Temporal Attention
            if temporal_config.use_temporal_attention and temporal_config.temporal_batch_size > 0:
                # Self-attention: (b h w) t c
                # Queries: t vectors of size c
                # Keys/values: t vectors of size c
                temporal_attention = TemporalAttention(
                    query_dim=out_channels,
                    heads=num_attention_heads,
                    dim_head=int(out_channels // num_attention_heads * temporal_config.attention_dim_factor),
                    norm_num_groups=resnet_groups,
                    upcast_attention=upcast_attention,
                    temporal_config=temporal_config)
            else:
                temporal_attention = None

            temporal_attentions.append(temporal_attention)

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.temporal_convolutions = nn.ModuleList(temporal_convolutions)
        self.temporal_attentions = nn.ModuleList(temporal_attentions)
        self._n_timesteps = temporal_config.temporal_batch_size
        self._temporal_config = temporal_config

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            additional_residuals=None,
            denoising_step: Optional[int] = None,
            expression_codes: Optional[torch.Tensor] = None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        for resnet, temporal_convolution, attn, temporal_attention in zip(self.resnets, self.temporal_convolutions,
                                                                          self.attentions, self.temporal_attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                # spatial convolution
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)  # [(B*T), C, H, W]

                # temporal convolution
                if self._temporal_config.use_temporal_convolution and self._temporal_config.temporal_batch_size > 0:
                    hidden_states = temporal_convolution(hidden_states)

                # Expression attention
                if self._temporal_config.use_expression_condition:
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                        expression_codes=expression_codes,
                    )[0]
                else:
                    # spatial attention
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                # temporal attention
                if self._temporal_config.use_temporal_attention and self._temporal_config.temporal_batch_size > 0:
                    hidden_states = temporal_attention(hidden_states, denoising_step=denoising_step)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states
