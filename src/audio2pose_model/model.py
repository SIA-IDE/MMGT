from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from audio2pose_model.rotary_embedding_torch import RotaryEmbedding
from audio2pose_model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
from einops import reduce  # 确保已安装 einops

def batch_mask(normalized_keypoints, device):

    batch_size, seq_len = normalized_keypoints.shape[0], normalized_keypoints.shape[1]  
    keypoints = normalized_keypoints.reshape(batch_size, seq_len, 134, 3)  # 头部的关键点范围是（24:92）
    
    # 创建全为1的mask，shape为 (batch_size, seq_len, 134, 3)
    mask = np.ones(keypoints.shape, dtype=np.uint8)
    
    # 关键点索引
    lip_indices = range(72, 92)      # 嘴唇
    face_indices = range(24, 92)     
    # chin_indices = range(28, 37)     # 下巴
    # left_eye_indices = range(60, 66) # 左眼
    # right_eye_indices = range(66, 72)# 右眼  
    
    # 对每个样本的特定区域进行mask
    # mask[:, :, chin_indices, :] = 0
    # mask[:, :, left_eye_indices, :] = 0
    # mask[:, :, right_eye_indices, :] = 0
    mask[:, :, face_indices, :] = 0

    mask = torch.tensor(mask, dtype=torch.uint8).to(device)    
    # 应用mask
    normalized_keypoints_face = keypoints * (1 - mask)
    normalized_keypoints_body = keypoints * mask 
    # 重整为原始的形状 (batch_size, seq_len, 402)
    normalized_keypoints_face = normalized_keypoints_face.reshape(batch_size, seq_len, -1)
    normalized_keypoints_body = normalized_keypoints_body.reshape(batch_size, seq_len, -1)
    return normalized_keypoints_face, normalized_keypoints_body


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift



class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class FiLMTransformerDecoderLayer_split(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        # 面部的自注意力和交叉注意力
        self.face_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.face_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # 身体的自注意力和交叉注意力
        self.body_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.body_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # 合并后的 Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # 前馈层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm_face_1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_face_2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_face_3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.norm_body_1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_body_2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_body_3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # 合并后的 Self-Attention 层的归一化
        self.norm_final = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        # FiLM 调制
        self.film_face_1 = DenseFiLM(d_model)
        self.film_face_2 = DenseFiLM(d_model)
        self.film_face_3 = DenseFiLM(d_model)

        self.film_body_1 = DenseFiLM(d_model)
        self.film_body_2 = DenseFiLM(d_model)
        self.film_body_3 = DenseFiLM(d_model)

        # 合并后的 FiLM 调制
        self.film_final = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        x_face,     # 面部输入
        x_body,     # 身体输入
        cond_tokens,  # 条件嵌入
        t,          # 时间步嵌入
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Face block processing
        face_output = self._process_block(
            x_face, self.face_self_attn, self.face_cross_attn,
            self.norm_face_1, self.norm_face_2, self.norm_face_3,
            self.film_face_1, self.film_face_2, self.film_face_3,
            cond_tokens, t, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask
        )

        # Body block processing
        body_output = self._process_block(
            x_body, self.body_self_attn, self.body_cross_attn,
            self.norm_body_1, self.norm_body_2, self.norm_body_3,
            self.film_body_1, self.film_body_2, self.film_body_3,
            cond_tokens, t, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask
        )

        # 合并 face 和 body 的输出，逐元素相加
        
        merged_output = face_output + body_output

        # feedforward -> FiLM -> residual
        if self.norm_first:
            merged_output_2 = self._ff_block(self.norm_final(merged_output))
        merged_output = merged_output + featurewise_affine(merged_output_2, self.film_final(t))

        return merged_output

    def _process_block(
        self, x, self_attn, cross_attn, norm1, norm2, norm3,
        film1, film2, film3, cond_tokens, t,
        tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask
    ):
        if self.norm_first:
            # self-attention -> FiLM -> residual
            x_1 = self._sa_block(norm1(x), self_attn, tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, film1(t))

            # cross-attention -> FiLM -> residual
            x_2 = self._mha_block(norm2(x), cross_attn, cond_tokens, memory_mask, memory_key_padding_mask)
            x = x + featurewise_affine(x_2, film2(t))
        else:
            x = norm1(x + featurewise_affine(self._sa_block(x, self_attn, tgt_mask, tgt_key_padding_mask), film1(t)))
            x = norm2(x + featurewise_affine(self._mha_block(x, cross_attn, cond_tokens, memory_mask, memory_key_padding_mask), film2(t)))
        return x

    # 自注意力块
    def _sa_block(self, x, attn, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)


    # 特征融合方式的修改LayerNorm (LN) + MLP (即 _ff_block) + 残差连接（Residual Connection, RS）
    def _mha_block_new(self, x, attn, mem, attn_mask, key_padding_mask):

        mem_resized = F.interpolate(mem.permute(0, 2, 1), size=x.shape[1], mode='linear', align_corners=False)
        mem_resized = mem_resized.permute(0, 2, 1)
        x = x + mem_resized
        # 1. 进行 LayerNorm 处理
        x = self.norm_final(x)
        
        # 2. 通过 MLP (即 _ff_block)
        x_mlp = self._ff_block(x)
        
        # 3. 残差连接，将原始输入 x 与 MLP 输出 x_mlp 相加
        x = x + x_mlp
        
        return x


    # 交叉注意力块
    def _mha_block(self, x, attn, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # 前馈块
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)




class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, y, cond, t):
        for layer in self.stack:
            x = layer(x, y, cond, t)
        return x


class GestureDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int, # 402
        seq_len: int = 100,  # 4 seconds, 25 fps
        latent_dim: int = 256, # 804 就变成512了
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats * 2, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        # split face and gesture
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer_split(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)
        
        self.epsilon = 0.00001

    def guided_forward(self, x, cond_frame, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_frame, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_frame, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight
    
    def vae_encoder(self, x):
        x = torch.cat([x, x], dim=-1)
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)
        return x

    def forward(
        self, x: Tensor, cond_frame: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device
        # mask face and mask body
        face_x, body_x = batch_mask(x, device)
        face_cond_frame = cond_frame.unsqueeze(1)
        face_cond_frame, body_cond_frame = batch_mask(face_cond_frame, device)
        
        x_face = torch.cat([face_x, face_cond_frame.repeat(1, x.shape[1], 1)], dim=-1)
        x_body = torch.cat([body_x, body_cond_frame.repeat(1, x.shape[1], 1)], dim=-1)
        # encoder
        x_face = self.input_projection(x_face)
        # # add the positional embeddings of the input sequence to provide temporal information
        x_face = self.abs_pos_encoding(x_face)

        x_body = self.input_projection(x_body)
        # add the positional embeddings of the input sequence to provide temporal information
        x_body = self.abs_pos_encoding(x_body)
        ######################################################################################

        # create audio conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_tokens = self.cond_projection(cond_embed)
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # create the diffusion timestep embedding, add the extra audio projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        output = self.seqTransDecoder(x_face, x_body, cond_tokens, t)

        # 只是一个线性层
        output = self.final_layer(output)

        return output
