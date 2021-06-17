# coding=utf-8
# Copyright 2020 The Nested-Transformer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific Nested-Transformer governing permissions and
# limitations under the License.
# ==============================================================================
"""Nested Transformer."""
import functools
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from libml import attn_utils
from libml import self_attention
import numpy as np

default_kernel_init = attn_utils.trunc_normal(stddev=0.02)
default_bias_init = jax.nn.initializers.zeros


class NestNet(nn.Module):
  """Nested Transformer Net."""
  num_classes: int
  config: ml_collections.ConfigDict
  train: bool = False
  dtype: int = jnp.float32
  activation_fn: Any = nn.gelu

  @nn.compact
  def __call__(self, inputs):
    config = self.config
    num_layers_per_block = config.num_layers_per_block
    num_blocks = len(num_layers_per_block)
    # Here we just assume image/patch size are squared.
    assert inputs.shape[1] == inputs.shape[2]
    assert inputs.shape[1] % config.init_patch_embed_size == 0
    input_size_after_patch = inputs.shape[1] // config.init_patch_embed_size
    assert input_size_after_patch % config.patch_size == 0
    down_sample_ratio = input_size_after_patch // config.patch_size
    # There are 4 child nodes for each node.
    assert num_blocks == int(np.log(down_sample_ratio) / np.log(2) + 1)

    # If `scale_hidden_dims` is provided, at every block, it increases hidden
    # dimension and num_heads by `scale_hidden_dims`. Set `scale_hidden_dims=2`
    # overall is a common design, so we do not gives the flexibility to control
    # layer-wise `scale_hidden_dims` to simplify the architecture.
    scale_hidden_dims = config.get("scale_hidden_dims", None)

    norm_fn = attn_utils.get_norm_layer(
        self.train, self.dtype, norm_type=config.norm_type)
    conv_fn = functools.partial(
        nn.Conv, dtype=self.dtype, kernel_init=default_kernel_init)
    dense_fn = functools.partial(
        nn.Dense, dtype=self.dtype, kernel_init=default_kernel_init)
    encoder_dict = dict(
        num_heads=config.num_heads,
        norm_fn=norm_fn,
        mlp_ratio=config.mlp_ratio,
        attn_type=config.attn_type,
        dense_fn=dense_fn,
        activation_fn=self.activation_fn,
        qkv_bias=config.qkv_bias,
        attn_drop=config.attn_drop,
        proj_drop=config.proj_drop,
        train=self.train,
        dtype=self.dtype)
    x = self_attention.PatchEmbedding(
        conv_fn=conv_fn,
        patch_size=(config.init_patch_embed_size, config.init_patch_embed_size),
        embedding_dim=config.embedding_dim)(
            inputs)
    x = attn_utils.block_images(x, (config.patch_size, config.patch_size))
    block_idx = 0
    total_block_num = np.sum(num_layers_per_block)
    path_drop = np.linspace(0, config.stochastic_depth_drop, total_block_num)
    for i in range(num_blocks):
      x = self_attention.PositionEmbedding()(x)
      if scale_hidden_dims and i != 0:
        # Overwrite the original num_heads value in encoder_dict so num_heads
        # multipled by scale_hidden_dims continueously.
        encoder_dict.update(
            {"num_heads": encoder_dict["num_heads"] * scale_hidden_dims})
      for _ in range(num_layers_per_block[i]):
        x = self_attention.EncoderNDBlock(
            **encoder_dict, path_drop=path_drop[block_idx])(
                x)
        block_idx = block_idx + 1
      if i < num_blocks - 1:
        grid_size = int(math.sqrt(x.shape[1]))
        if scale_hidden_dims:
          output_dim = x.shape[-1] * scale_hidden_dims
        else:
          output_dim = None

        x = self_attention.ConvPool(
            grid_size=(grid_size, grid_size),
            patch_size=(config.patch_size, config.patch_size),
            conv_fn=conv_fn,
            dtype=self.dtype,
            output_dim=output_dim)(
                x)
    assert x.shape[1] == 1
    assert x.shape[2] == config.patch_size**2

    x = norm_fn()(x)
    x_pool = jnp.mean(x, axis=(1, 2))
    out = dense_fn(self.num_classes)(x_pool)
    return out


MODELS = {}


def register(f):
  MODELS[f.__name__] = f
  return f


def default_config():
  """Shared configs for models."""
  nest = ml_collections.ConfigDict()
  nest.norm_type = "LN"
  nest.attn_type = "local_multi_head"
  nest.mlp_ratio = 4
  nest.qkv_bias = True
  nest.attn_drop = 0.0
  nest.proj_drop = 0.0
  nest.stochastic_depth_drop = 0.1
  return nest


@register
def nest_tiny_s16_32(config):
  """NesT tiny version with sequence length 16 for 32x32 inputs."""
  nest = default_config()
  # Encode one pixel as a word vector.
  nest.init_patch_embed_size = 1
  # Default max sequencee length is 4x4=16, so it has 4 layers.
  nest.patch_size = 4
  nest.num_layers_per_block = [3, 3, 3, 3]
  nest.embedding_dim = 192
  nest.num_heads = 3
  nest.attn_type = "local_multi_query"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(NestNet, config=nest)


@register
def nest_small_s16_32(config):
  """NesT small version with sequence length 16 for 32x32 inputs."""
  nest = default_config()
  nest.init_patch_embed_size = 1
  nest.patch_size = 4
  nest.num_layers_per_block = [3, 3, 3, 3]
  nest.embedding_dim = 384
  nest.num_heads = 6
  nest.attn_type = "local_multi_query"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(NestNet, config=nest)


@register
def nest_base_s16_32(config):
  """NesT base version with sequence length 16 for 32x32 inputs."""
  nest = default_config()
  nest.init_patch_embed_size = 1
  nest.patch_size = 4
  nest.num_layers_per_block = [3, 3, 3, 3]
  nest.embedding_dim = 768
  nest.num_heads = 12
  nest.attn_type = "local_multi_query"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(NestNet, config=nest)


@register
def nest_tiny_s196_224(config):
  """NesT tiny version with sequence length 49 for 224x224 inputs."""
  nest = default_config()
  # Encode 4x4 pixel as a word vector.
  nest.init_patch_embed_size = 4
  # Default max sequencee length is 14x14=196, so it has 3 layers:
  # Spatial image size: [56, 28, 14]
  nest.patch_size = 14
  nest.num_layers_per_block = [2, 2, 8]
  nest.embedding_dim = 96
  nest.num_heads = 3
  nest.scale_hidden_dims = 2
  nest.stochastic_depth_drop = 0.2
  nest.attn_type = "local_multi_head"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(NestNet, config=nest)


@register
def nest_small_s196_224(config):
  """NesT small version with sequence length 196 for 224x224 inputs."""
  nest = default_config()
  nest.init_patch_embed_size = 4
  nest.patch_size = 14
  nest.num_layers_per_block = [2, 2, 20]
  nest.embedding_dim = 96
  nest.num_heads = 3
  nest.scale_hidden_dims = 2
  nest.stochastic_depth_drop = 0.3
  nest.attn_type = "local_multi_head"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(NestNet, config=nest)


@register
def nest_base_s196_224(config):
  """NesT base version with sequence length 196 for 224x224 inputs."""
  nest = default_config()
  nest.init_patch_embed_size = 4
  nest.patch_size = 14
  nest.num_layers_per_block = [2, 2, 20]
  nest.embedding_dim = 128
  nest.num_heads = 4
  nest.scale_hidden_dims = 2
  nest.stochastic_depth_drop = 0.5
  nest.attn_type = "local_multi_head"

  if config.get("nest"):
    nest.update(config.nest)
  return functools.partial(NestNet, config=nest)


def create_model(name, config):
  """Creates model partial function."""
  if name not in MODELS:
    raise ValueError(f"Model {name} does not exist.")
  return MODELS[name](config)
