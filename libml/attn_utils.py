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
"""Utility functions for attention."""
import functools

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def trunc_normal(stddev=1e-2, lower=-2, upper=2, dtype=jnp.float32):

  def init(key, shape, dtype=dtype):
    return jax.random.truncated_normal(key, lower, upper, shape, dtype) * stddev

  return init


def block_images(x, patch_size):
  """Image to patches."""

  batch, height, width, depth = x.shape
  assert height % patch_size[0] == 0
  assert width % patch_size[1] == 0
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = jnp.reshape(
      x, (batch, grid_height, patch_size[0], grid_width, patch_size[1], depth))
  x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
  x = jnp.reshape(x, (batch, grid_height * grid_width, -1, depth))
  return x


def unblock_images(x, grid_size, patch_size):
  """patches to images."""

  batch, grid_length, patch_length, depth = x.shape
  assert np.prod(grid_size) == grid_length
  assert np.prod(patch_size) == patch_length
  new_shape = (batch, grid_size[0], grid_size[1], patch_size[0], patch_size[1],
               depth)
  height = grid_size[0] * patch_size[0]
  width = grid_size[1] * patch_size[1]
  x = jnp.reshape(x, new_shape)
  x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
  x = jnp.reshape(x, (batch, height, width, depth))
  return x


def get_norm_layer(train, dtype, norm_type="LN"):
  """Normalization layer."""
  if norm_type == "BN":
    norm_fn = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name=None,
        axis_index_groups=None,
        dtype=jnp.float32)
  elif norm_type == "LN":
    norm_fn = functools.partial(nn.LayerNorm, epsilon=1e-6, dtype=dtype)
  elif norm_type == "GN":
    norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
  else:
    raise NotImplementedError
  return norm_fn


class DropPath(nn.Module):
  """Create a stochastic depth layer.

  Follows dropout implementation from
  flax/linen/stochastic.py

    Attributes:
      rate: the drop probability.  (_not_ the keep rate!)
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
  """
  rate: float
  deterministic: Optional[bool] = None

  @nn.compact
  def __call__(self, inputs):
    if self.rate == 0.:
      return inputs
    keep_prob = 1. - self.rate
    if self.deterministic:
      return inputs
    else:
      # just use the same set of naming with dropout
      rng = self.make_rng("dropout")
      mask_shape = [inputs.shape[0]] + [1 for _ in inputs.shape[1:]]
      mask = jax.random.bernoulli(rng, p=keep_prob, shape=mask_shape)
      mask = jnp.tile(mask, [1] + list(inputs.shape[1:]))
      return jax.lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
