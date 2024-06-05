import math
from collections import defaultdict
from typing import Optional, Dict, Tuple, List, Callable, Mapping, Any

import torch
import xformers
import xformers.ops
from diffusers.models.attention_processor import Attention
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from xformers.components.positional_embedding import RotaryEmbedding

from diffusion_avatars.model.temporal.temporal_attention import RotaryXFormersAttnProcessor
from diffusion_avatars.model.temporal.temporal_config import VideoControlNetConfig


class ExpressionAttention(Attention):

    def __init__(self,
                 query_dim: int,
                 cross_attention_dim: Optional[int] = None,
                 heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.0, bias=False, upcast_attention: bool = False, upcast_softmax: bool = False,
                 cross_attention_norm: Optional[str] = None, cross_attention_norm_num_groups: int = 32,
                 added_kv_proj_dim: Optional[int] = None, norm_num_groups: Optional[int] = None,
                 spatial_norm_dim: Optional[int] = None, out_bias: bool = True, scale_qk: bool = True,
                 only_cross_attention: bool = False, eps: float = 1e-5, rescale_output_factor: float = 1.0,
                 residual_connection: bool = False, _from_deprecated_attn_block=False,
                 processor: Optional["AttnProcessor"] = None,
                 temporal_config: VideoControlNetConfig = VideoControlNetConfig()):
        if temporal_config.positional_encoding == 'rotary':
            processor = RotaryXFormersAttnProcessor(head_dim=dim_head)

        super().__init__(query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention,
                         upcast_softmax, cross_attention_norm, cross_attention_norm_num_groups, added_kv_proj_dim,
                         norm_num_groups, spatial_norm_dim, out_bias, scale_qk, only_cross_attention, eps,
                         rescale_output_factor, residual_connection, _from_deprecated_attn_block, processor)

        self._temporal_config = temporal_config
        self._dim_head = dim_head

        self._expression_attention = Attention(
            query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention,
            upcast_softmax, cross_attention_norm, cross_attention_norm_num_groups, added_kv_proj_dim,
            norm_num_groups, spatial_norm_dim, out_bias, scale_qk, only_cross_attention, eps,
            rescale_output_factor, residual_connection, _from_deprecated_attn_block, processor)

        # Temporal attention is applied with skip connection. Hence, it suffices to set it all 0s in beginning
        self._expression_attention.to_q.weight.init = torch.zeros_like(self.to_q.weight, device='cpu') # self.to_q.weight
        self._expression_attention.to_k.weight.init = torch.zeros_like(self.to_k.weight, device='cpu')  # self.to_k.weight
        self._expression_attention.to_v.weight.init = torch.zeros_like(self.to_v.weight, device='cpu')  # self.to_v.weight
        if self.group_norm is not None:
            self._expression_attention.group_norm.weight.init = self.group_norm.weight
            self._expression_attention.group_norm.bias.init = self.group_norm.bias

        for to_out_param in self._expression_attention.to_out.parameters():
            nn.init.zeros_(to_out_param)
            to_out_param.init = torch.zeros_like(to_out_param.data, device='cpu')
            nn.init.zeros_(to_out_param.init)

        self._expression_attention_initialized = False

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        result = super().load_state_dict(state_dict, strict)
        # TODO: Test that loaded weights from checkpoint are not overridden!
        # Assume that we are loading expression attention weights from a proper checkpoint
        self._expression_attention_initialized = True

        return result

    def set_use_memory_efficient_attention_xformers(
            self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        if self._temporal_config.positional_encoding == 'rotary':
            # We already fixed using the RotaryXFormersAttnProcessor. Ensure that it cannot be changed
            pass
        else:
            super(ExpressionAttention, self).set_use_memory_efficient_attention_xformers(
                use_memory_efficient_attention_xformers, attention_op=attention_op)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        attention_output = super().forward(hidden_states, encoder_hidden_states, attention_mask,
                                           **cross_attention_kwargs)

        if not self._expression_attention_initialized:
            # Initialize with pretrained attention weights
            self._expression_attention.to_q.weight[:] = self.to_q.weight
            self._expression_attention.to_k.weight[:] = self.to_k.weight
            self._expression_attention.to_v.weight[:] = self.to_v.weight
            self._expression_attention_initialized = True

        expression_codes = cross_attention_kwargs["expression_codes"]
        expression_attention_output = self._expression_attention(hidden_states, expression_codes, attention_mask,
                                                                 **cross_attention_kwargs)

        return attention_output + expression_attention_output


