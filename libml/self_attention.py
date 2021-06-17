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
"""Self Attention."""

from collections.abc import Iterable  # pylint: disable=g-importing-member
from typing import Any, Optional, Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from libml import attn_utils

default_kernel_init = attn_utils.trunc_normal(stddev=0.02)
default_bias_init = jax.nn.initializers.zeros


class MultiHeadAttention(nn.Module):
  """Multi Head Attention."""
  num_heads: int
  hidden_dims: Optional[int] = None
  qkv_bias: bool = False
  attn_drop: float = 0.1
  train: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):  # x is blocked with shape (b,..., N, C)
    if self.hidden_dims is None:
      hidden_dims = x.shape[-1]
    else:
      hidden_dims = self.hidden_dims
    assert hidden_dims % self.num_heads == 0
    head_dim = hidden_dims // self.num_heads
    query = nn.DenseGeneral(
        features=(self.num_heads, head_dim),
        use_bias=self.qkv_bias,
        dtype=self.dtype,
        kernel_init=default_kernel_init)(
            x)
    kv = nn.DenseGeneral(
        features=(self.num_heads, 2 * head_dim),
        use_bias=self.qkv_bias,
        dtype=self.dtype,
        kernel_init=default_kernel_init)(
            x)
    perm = tuple(range(query.ndim - 3)) + (query.ndim - 2, query.ndim - 3,
                                           query.ndim - 1)
    query = jnp.transpose(query, perm)
    kv = jnp.transpose(kv, perm)
    key, value = jnp.split(kv, 2, axis=-1)
    query /= jnp.sqrt(query.shape[-1]).astype(self.dtype)
    logits = jnp.einsum("b...hld,b...hmd->b...hlm", query, key)
    attention_weights = nn.softmax(logits, axis=-1)
    attention_weights = nn.Dropout(
        self.attn_drop, deterministic=not self.train)(
            attention_weights)
    output = jnp.einsum("b...hlm,b...hmd->b...hld", attention_weights, value)
    kernel = self.param("proj_kernel", default_kernel_init,
                        (output.shape[-3], output.shape[-1], self.hidden_dims))
    kernel = jnp.asarray(kernel, self.dtype)
    proj = jnp.einsum("b...hld,hdf->b...lf", output, kernel)
    bias = self.param("bias", default_bias_init, (proj.shape[-1],))
    bias = jnp.asarray(bias, self.dtype)
    proj = proj + bias
    return proj


class MultiQueryAttention(nn.Module):
  """Multi Query Attention.

  Follow the implementation in
  Fast Transformer Decoding: One Write-Head is All You Need
  https://arxiv.org/abs/1911.02150
  """
  num_heads: int
  hidden_dims: Optional[int]
  qkv_bias: bool = False
  attn_drop: float = 0.1
  train: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    if self.hidden_dims is None:
      hidden_dims = x.shape[-1]
    else:
      hidden_dims = self.hidden_dims
    assert hidden_dims % self.num_heads == 0
    head_dim = hidden_dims // self.num_heads
    query = nn.DenseGeneral(
        features=(self.num_heads, head_dim),
        use_bias=self.qkv_bias,
        dtype=self.dtype,
        kernel_init=default_kernel_init)(
            x)
    kv = nn.Dense(
        features=2 * head_dim,
        use_bias=self.qkv_bias,
        dtype=self.dtype,
        kernel_init=default_kernel_init)(
            x)
    perm = tuple(range(query.ndim - 3)) + (query.ndim - 2, query.ndim - 3,
                                           query.ndim - 1)
    query = jnp.transpose(query, perm)
    key, value = jnp.split(kv, 2, axis=-1)
    query /= jnp.sqrt(query.shape[-1]).astype(self.dtype)
    logits = jnp.einsum("b...hld,b...md->b...hlm", query, key)
    attention_weights = nn.softmax(logits, axis=-1)
    attention_weights = nn.Dropout(
        self.attn_drop, deterministic=not self.train)(
            attention_weights)
    output = jnp.einsum("b...hlm,b...md->b...hld", attention_weights, value)
    kernel = self.param("proj_kernel", default_kernel_init,
                        (output.shape[-3], output.shape[-1], hidden_dims))
    kernel = jnp.asarray(kernel, self.dtype)
    proj = jnp.einsum("b...hld,hdf->b...lf", output, kernel)
    bias = self.param("bias", default_bias_init, (proj.shape[-1],))
    bias = jnp.asarray(bias, self.dtype)
    proj = proj + bias
    return proj


class MlpBlock(nn.Module):
  """MLP blocks.

  Same as original Flax.
  """
  norm_fn: Any
  mlp_dim: Optional[int] = None
  out_dim: Optional[int] = None
  dense_fn: Callable = nn.Dense  # pylint: disable=g-bare-generic
  activation_fn: Any = nn.relu
  proj_drop: float = 0.1
  train: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    actual_out_dim = (x.shape[-1] if self.out_dim is None else self.out_dim)
    mlp_dim = x.shape[-1] if self.mlp_dim is None else self.mlp_dim
    x = self.dense_fn(
        mlp_dim, dtype=self.dtype, kernel_init=default_kernel_init)(
            x)
    x = self.activation_fn(x)
    x = nn.Dropout(self.proj_drop, deterministic=not self.train)(x)
    x = self.dense_fn(
        actual_out_dim, dtype=self.dtype, kernel_init=default_kernel_init)(
            x)
    x = nn.Dropout(self.proj_drop, deterministic=not self.train)(x)
    return x


class EncoderNDBlock(nn.Module):
  """Encoder ND Block."""
  num_heads: int
  norm_fn: Any
  hidden_dims: Optional[int] = None
  mlp_ratio: int = 4
  attn_type: str = "local_multi_head"
  dense_fn: Callable = nn.Dense  # pylint: disable=g-bare-generic
  activation_fn: Any = nn.relu
  qkv_bias: bool = False
  path_drop: float = 0.0
  attn_drop: float = 0.0
  proj_drop: float = 0.0
  train: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    if self.hidden_dims is not None:
      hidden_dims = self.hidden_dims
    else:
      hidden_dims = x.shape[-1]
    identity = x
    attn_kwargs = dict(
        hidden_dims=hidden_dims,
        num_heads=self.num_heads,
        qkv_bias=self.qkv_bias,
        attn_drop=self.attn_drop,
        train=self.train,
        dtype=self.dtype)
    mlp_kwargs = dict(
        norm_fn=self.norm_fn,
        mlp_dim=hidden_dims * self.mlp_ratio,
        dense_fn=self.dense_fn,
        activation_fn=self.activation_fn,
        proj_drop=self.proj_drop,
        train=self.train,
        dtype=self.dtype,
    )

    x = self.norm_fn()(x)
    if self.attn_type == "local_multi_head":
      x = MultiHeadAttention(**attn_kwargs)(x)
    elif self.attn_type == "local_multi_query":
      # Use one head for query.
      x = MultiQueryAttention(**attn_kwargs)(x)
    else:
      raise NotImplementedError
    x = nn.Dropout(self.proj_drop, deterministic=not self.train)(x)
    x = attn_utils.DropPath(self.path_drop, deterministic=not self.train)(x)
    x = x + identity

    # MLP block
    y = self.norm_fn()(x)
    y = MlpBlock(**mlp_kwargs)(y)
    y = attn_utils.DropPath(self.path_drop, deterministic=not self.train)(y)
    return x + y


class PositionEmbedding(nn.Module):
  """Position Embedding Layer."""
  embed_axis: Optional[Any] = None

  @nn.compact
  def __call__(self, inputs):
    axis = self.embed_axis
    if axis is None:
      axis = list(range(1, inputs.ndim))
    if not isinstance(axis, Iterable):
      axis = (axis,)
    param_shape = []
    inputs_shape = inputs.shape
    for i in range(inputs.ndim):
      if i in axis:
        param_shape.append(inputs_shape[i])
      else:
        param_shape.append(1)
    pos_embedding = self.param("pos_embedding", default_kernel_init,
                               param_shape)
    return inputs + pos_embedding


class PatchEmbedding(nn.Module):
  """Patch Embedding Layer."""
  patch_size: Tuple[int, int]
  embedding_dim: int
  conv_fn: Any = nn.Conv

  @nn.compact
  def __call__(self, inputs):
    assert inputs.shape[1] % self.patch_size[0] == 0
    assert inputs.shape[2] % self.patch_size[1] == 0
    out = self.conv_fn(
        self.embedding_dim,
        kernel_size=self.patch_size,
        strides=self.patch_size)(
            inputs)
    return out


class ConvPool(nn.Module):
  """Downsampling layer with conv_pool."""
  patch_size: Tuple[int, int]
  grid_size: Tuple[int, int]
  output_dim: Optional[int] = None
  conv_fn: Any = nn.Conv
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = attn_utils.unblock_images(
        x, grid_size=self.grid_size, patch_size=self.patch_size)
    if self.output_dim is None:
      output_dim = x.shape[-1]
    else:
      output_dim = self.output_dim
    x = self.conv_fn(output_dim, kernel_size=(3, 3))(x)
    x = nn.LayerNorm(dtype=self.dtype, epsilon=1e-6)(x)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
    x = attn_utils.block_images(x, patch_size=self.patch_size)
    return x
