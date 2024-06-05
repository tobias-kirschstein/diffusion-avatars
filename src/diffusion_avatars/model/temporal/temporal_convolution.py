from typing import Union

import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn.common_types import _size_1_t
import torch.nn.functional as F

from diffusion_avatars.model.temporal.temporal_config import VideoControlNetConfig


class TemporalConvolution(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None,
                 temporal_config: VideoControlNetConfig = VideoControlNetConfig()) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         device, dtype)

        self._temporal_config = temporal_config

        # Hack: Add a new field "init" to all parameters of newly introduced layers
        #  the init field contains the actual tensor values with which the parameters should be initialized
        #  this is necessary as in accelerate, models are initialized with "with_init('meta')"
        #  which causes all model parameters to be created without actual values (hence the 'meta')
        #  This is ok for pre-trained layers, as the layers will be initialized with the weights from the
        #  checkpoint later
        #  However, for new layers, we actually want to create proper initial weights
        #  This is done now later, after the pretrained weights have been loaded. The initializations for
        #  the new temporal layers is just added to the loaded state dict imitating that these layers were
        #  already part of the pretrained model
        self.weight.init = torch.zeros_like(self.weight.data, device='cpu')
        self.bias.init = torch.zeros_like(self.bias.data, device='cpu')
        if temporal_config.downscale_factor_convolution > 1:
            # When downscaling is used, we have to use skip connections, otherwise we would loose part of
            # the input dimensionality
            nn.init.zeros_(self.weight.data)
            nn.init.zeros_(self.weight.init)
        else:
            # Without downscaling, we always feed everything through the 1D temporal convolution
            nn.init.dirac_(self.weight.data)
            nn.init.dirac_(self.weight.init)

        nn.init.zeros_(self.bias.data)
        nn.init.zeros_(self.bias.init)

    def forward(self, hidden_states: Tensor) -> Tensor:
        h_orig = hidden_states.shape[-2]
        w_orig = hidden_states.shape[-1]
        t = self._temporal_config.temporal_batch_size

        # Apparently, the batch dimension can be too large for the CUDA kernel.
        # Hence, we need to downscale large activation maps
        scale_factor = self._temporal_config.downscale_factor_convolution
        use_downscaling = scale_factor > 1 and h_orig > self._temporal_config.downscale_threshold_convolution
        if use_downscaling:
            hidden_states_convolution = F.interpolate(hidden_states,
                                                      scale_factor=1 / scale_factor,
                                                      mode=self._temporal_config.downscale_mode)
            h = h_orig // scale_factor
            w = w_orig // scale_factor
        else:
            hidden_states_convolution = hidden_states
            h = h_orig
            w = w_orig

        hidden_states_convolution = rearrange(hidden_states_convolution, '(b t) c h w -> (b h w) c t', t=t)
        hidden_states_convolution = super(TemporalConvolution, self).forward(hidden_states_convolution)
        hidden_states_convolution = rearrange(hidden_states_convolution, '(b h w) c t -> (b t) c h w', h=h, w=w)

        if use_downscaling:
            hidden_states = hidden_states + F.interpolate(hidden_states_convolution,
                                                          scale_factor=scale_factor,
                                                          mode=self._temporal_config.downscale_mode)
        else:
            if scale_factor > 1:
                # Training with skip connection
                hidden_states = hidden_states + hidden_states_convolution
            else:
                hidden_states = hidden_states_convolution

        return hidden_states
