from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn

from .attention import (AudioTemporalBasicTransformerBlock, 
                        TemporalBasicTransformerBlock)


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        name=None,
        use_audio_module=False,
        depth=0,
        unet_block_name=None,
        stack_enable_blocks_name = None,
        stack_enable_blocks_depth = None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels
        self.name=name

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )

        if use_audio_module:
            self.transformer_blocks = nn.ModuleList(
                [
                    AudioTemporalBasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                        depth=depth,
                        unet_block_name=unet_block_name,
                        stack_enable_blocks_name=stack_enable_blocks_name,
                        stack_enable_blocks_depth=stack_enable_blocks_depth,
                    )
                    for d in range(num_layers)
                ]
            )


        else:
            # Define transformers blocks
            self.transformer_blocks = nn.ModuleList(
                [
                    TemporalBasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                        name=f"{self.name}_{d}_TransformerBlock" if self.name else None,
                    )
                    for d in range(num_layers)
                ]
            )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0
            )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        full_mask=None,
        face_mask=None,
        body_mask=None,
        motion_scale=None,
        mode=None,
        timestep=None,
        return_dict: bool = True,
        self_attention_additional_feats=None,
    ):
        # Input
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        if self.use_audio_module:
            encoder_hidden_states = rearrange(
                encoder_hidden_states,
                "bs f margin dim -> (bs f) margin dim",
            )
        else:
            if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
                encoder_hidden_states = repeat(
                    encoder_hidden_states, "b n c -> (b f) n c", f=video_length
                )

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * weight, inner_dim
            )
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * weight, inner_dim
            )
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        # for i, block in enumerate(self.transformer_blocks):
            
        #     if self.training and self.gradient_checkpointing:

        #         def create_custom_forward(module, return_dict=None):
        #             def custom_forward(*inputs):
        #                 if return_dict is not None:
        #                     return module(*inputs, return_dict=return_dict)
        #                 else:
        #                     return module(*inputs)

        #             return custom_forward

        #         # if hasattr(self.block, 'bank') and len(self.block.bank) > 0:
        #         #     hidden_states
        #         hidden_states = torch.utils.checkpoint.checkpoint(
        #             create_custom_forward(block),
        #             hidden_states,
        #             encoder_hidden_states=encoder_hidden_states,
        #             timestep=timestep,
        #             attention_mask=None,
        #             video_length=video_length,
        #             self_attention_additional_feats=self_attention_additional_feats,
        #             mode=mode,
        #         )
        #     else:

        #         hidden_states = block(
        #             hidden_states,
        #             encoder_hidden_states=encoder_hidden_states,
        #             timestep=timestep,
        #             self_attention_additional_feats=self_attention_additional_feats,
        #             mode=mode,
        #             video_length=video_length,
        #         )
        
        for _, block in enumerate(self.transformer_blocks):
            if isinstance(block, TemporalBasicTransformerBlock):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    video_length=video_length,
                )
            else:
                hidden_states = block(
                    hidden_states,  # shape [2, 4096, 320]
                    encoder_hidden_states=encoder_hidden_states,  # shape [2, 20, 640]
                    attention_mask=attention_mask,
                    full_mask=full_mask,
                    face_mask=face_mask,
                    body_mask=body_mask,
                    timestep=timestep,
                    video_length=video_length,
                    motion_scale=motion_scale,
                )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)
