from dataclasses import dataclass
from typing import Literal, Optional

from elias.config import Config, implicit

PositionalEncodingType = Literal['rotary']


@dataclass
class VideoControlNetConfig(Config):
    temporal_batch_size: int = implicit()  # If > 1, then special temporal convolutions (and attention) blocks will be added. A batch will consist of a set of sequential frames [B * T]

    fix_pseudo_3d_block: bool = False  # Legacy flag, older models used a broken configuration of pretrained ControlNet encoder blocks
    use_temporal_convolution: bool = True
    temporal_kernel_size: int = 3
    temporal_padding_mode: str = 'zeros'
    downscale_factor_convolution: int = 1
    downscale_threshold_convolution: int = 32
    downscale_mode: str = 'nearest'  # nearest, bilinear, bicubic, area, nearest-exact

    use_temporal_attention: bool = False
    attention_dim_factor: float = 1  # Optionally, use a smaller inner attention dimension to save space
    downscale_factor_attention: int = 1
    downscale_threshold_attention: int = 32

    use_temporal_unet_encoder: bool = False
    use_temporal_unet_decoder: bool = False
    use_temporal_controlnet_encoder: bool = True

    positional_encoding: Optional[PositionalEncodingType] = None
    enable_extended_temporal_attention: bool = False  # During inference, collect all previously computed hidden_states and provide them to temporal attention. NB: Enabling this assumes frames will are generated for inference in order. They are currently never cleared!

    # Expression condition
    use_expression_condition: bool = implicit()
    n_expression_tokens: int = 4  # Sequences length for cross attention to attend to. 4 is the default used in IP-Adapter