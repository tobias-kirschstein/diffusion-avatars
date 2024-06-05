import math
from collections import defaultdict
from typing import Optional, Dict, Tuple, List, Callable

import torch
import xformers
import xformers.ops
from diffusers.models.attention_processor import Attention
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from xformers.components.positional_embedding import RotaryEmbedding

from diffusion_avatars.model.temporal.temporal_config import VideoControlNetConfig


class RotaryXFormersAttnProcessor(nn.Module):
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None, head_dim: Optional[int] = None):
        super(RotaryXFormersAttnProcessor, self).__init__()
        self.attention_op = attention_op
        self.rotary_embedding = RotaryEmbedding(head_dim)  # TODO: Which dimension to choose?

        self.rotary_embedding._register_state_dict_hook(self._ignore_in_state_dict)
        self.rotary_embedding.register_load_state_dict_post_hook(self._ignore_in_load_state_dict)

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        B = query.shape[0]
        T_q = query.shape[1]
        T_k = key.shape[1]
        dtype = query.dtype

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        query = query.view(B, attn.heads, T_q, -1)  # [B, H, T, F]
        key = key.view(B, attn.heads, T_k, -1)

        # Rotary Positional Encoding
        query, key = self.rotary_embedding(q=query, k=key)

        query = query.view(B * attn.heads, T_q, -1).to(dtype)  # Important to cast to float16 again
        key = key.view(B * attn.heads, T_k, -1).to(dtype)

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _ignore_in_state_dict(self, module: nn.Module, state_dict: Dict[str, nn.Module], prefix: str, local_metadata):
        # Remove everything from the state dict with prefix of the module that is to be ignored
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                del state_dict[key]

        return state_dict

    def _ignore_in_load_state_dict(self, module: nn.Module, incompatible_keys: Tuple[List[str], List[str]]) -> None:
        # Remove all items from "missing keys"
        # This module isn't initialized from a state dict
        incompatible_keys[0].clear()


class TemporalAttention(Attention):

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

        # Temporal attention is applied with skip connection. Hence, it suffices to set it all 0s in beginning
        self.to_q.weight.init = torch.zeros_like(self.to_q.weight.data, device='cpu')
        self.to_k.weight.init = torch.zeros_like(self.to_k.weight.data, device='cpu')
        self.to_v.weight.init = torch.zeros_like(self.to_v.weight.data, device='cpu')
        self.group_norm.weight.init = torch.zeros_like(self.group_norm.weight.data, device='cpu')
        self.group_norm.bias.init = torch.zeros_like(self.group_norm.bias.data, device='cpu')

        init.kaiming_uniform_(self.to_q.weight.init, a=math.sqrt(5))
        init.kaiming_uniform_(self.to_k.weight.init, a=math.sqrt(5))
        init.kaiming_uniform_(self.to_v.weight.init, a=math.sqrt(5))
        init.ones_(self.group_norm.weight.init)
        init.zeros_(self.group_norm.bias.init)

        for to_out_param in self.to_out.parameters():
            nn.init.zeros_(to_out_param)
            to_out_param.init = torch.zeros_like(to_out_param.data, device='cpu')
            nn.init.zeros_(to_out_param.init)

        if temporal_config.enable_extended_temporal_attention:
            self._hidden_states_history = defaultdict(list)

    def set_use_memory_efficient_attention_xformers(
            self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        if self._temporal_config.positional_encoding == 'rotary':
            # We already fixed using the RotaryXFormersAttnProcessor. Ensure that it cannot be changed
            pass
        else:
            super(TemporalAttention, self).set_use_memory_efficient_attention_xformers(
                use_memory_efficient_attention_xformers, attention_op=attention_op)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states=None, attention_mask=None,
                denoising_step: int = None,
                **cross_attention_kwargs):
        h_orig = hidden_states.shape[-2]
        w_orig = hidden_states.shape[-1]
        t = self._temporal_config.temporal_batch_size

        # Apparently, the batch dimension can be too large for the CUDA kernel.
        # Hence, we need to downscale large activation maps
        scale_factor = self._temporal_config.downscale_factor_attention
        use_downscaling = scale_factor > 1 and h_orig > self._temporal_config.downscale_threshold_attention
        if use_downscaling:
            hidden_states_attention = F.interpolate(hidden_states,
                                                    scale_factor=1 / scale_factor,
                                                    mode=self._temporal_config.downscale_mode)

            h = h_orig // scale_factor
            w = w_orig // scale_factor
        else:
            hidden_states_attention = hidden_states

            h = h_orig
            w = w_orig

        hidden_states_attention = rearrange(hidden_states_attention, '(b t) c h w -> (b h w) t c', t=t)
        if self._temporal_config.enable_extended_temporal_attention:
            assert denoising_step is not None
            encoder_hidden_states_attention = torch.concatenate(
                [prev_hidden_states.to(hidden_states_attention) for prev_hidden_states in self._hidden_states_history[denoising_step]] + [hidden_states_attention],
                dim=1)
            if self.group_norm is not None:
                encoder_hidden_states_attention = self.group_norm(encoder_hidden_states_attention.transpose(1, 2)).transpose(1, 2)
            self._hidden_states_history[denoising_step].append(hidden_states_attention.detach().cpu())
        else:
            encoder_hidden_states_attention = None

        hidden_states_attention = super().forward(hidden_states_attention,
                                                  encoder_hidden_states=encoder_hidden_states_attention)

        hidden_states_attention = rearrange(hidden_states_attention, '(b h w) t c -> (b t) c h w', h=h, w=w)

        if use_downscaling:
            hidden_states = hidden_states + F.interpolate(hidden_states_attention,
                                                          scale_factor=scale_factor,
                                                          mode=self._temporal_config.downscale_mode)
        else:
            hidden_states = hidden_states + hidden_states_attention

        return hidden_states
