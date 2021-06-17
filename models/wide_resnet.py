# coding=utf-8
# Copyright 2020 The nestvit Authors.
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
# See the License for the specific nestvit governing permissions and
# limitations under the License.
# ==============================================================================
# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wide Resnet Model.

Reference:

Wide Residual Networks, Sergey Zagoruyko, Nikos Komodakis
https://arxiv.org/abs/1605.07146

Initially forked from
github.com/google/flax/blob/master/examples/cifar10/models/wideresnet.py

This implementation mimics the one from
github.com/tensorflow/models/blob/master/research/autoaugment/wrn.py
that is widely used as a benchmark.

It uses identity + zero padding skip connections, with kaiming normal
initialization for convolutional kernels (mode = fan_out, gain=2.0).
The final dense layer uses a uniform distribution U[-scale, scale] where
scale = 1 / sqrt(num_classes) as per the autoaugment implementation.

Using the default initialization instead gives error rates approximately 0.5%
greater on cifar100, most likely because the parameters used in the literature
were finetuned for this particular initialization.

Finally, the autoaugment implementation adds more residual connections between
the groups (instead of just between the blocks as per the original paper and
most implementations). It is possible to safely remove those connections without
degrading the performance, which we do by default to match the original
wideresnet paper. Setting `use_additional_skip_connections` to True will add
them back and then reproduces exactly the model used in autoaugment.
"""
import functools
from typing import Tuple

from absl import flags
import flax.linen as nn
import jax
from jax import numpy as jnp

FLAGS = flags.FLAGS

_BATCHNORM_MOMENTUM = 0.9
_BATCHNORM_EPSILON = 1e-5
_USE_ADDITIONAL_SKIP_CONNECTIONS = True  # Match the AutoAugment settings


def activation(x, train, apply_relu=True, name=''):
  """Applies BatchNorm and then (optionally) ReLU.

  Args:
    x: Tensor on which the activation should be applied.
    train: If False, will use the moving average for batch norm statistics.
      Else, will use statistics computed on the batch.
    apply_relu: Whether or not ReLU should be applied after batch normalization.
    name: How to name the BatchNorm layer.

  Returns:
    The input tensor where BatchNorm and (optionally) ReLU where applied.
  """
  batch_norm = functools.partial(
      nn.BatchNorm,
      use_running_average=not train,
      momentum=_BATCHNORM_MOMENTUM,
      epsilon=_BATCHNORM_EPSILON)
  x = batch_norm(name=name)(x)
  if apply_relu:
    x = jax.nn.relu(x)
  return x


# Kaiming initialization with fan out mode. Should be used to initialize
# convolutional kernels.
conv_kernel_init_fn = jax.nn.initializers.variance_scaling(
    2.0, 'fan_out', 'normal')


def dense_layer_init_fn(key, shape, dtype=jnp.float32):
  """Initializer for the final dense layer.

  Args:
    key: PRNG key to use to sample the weights.
    shape: Shape of the tensor to initialize.
    dtype: Data type of the tensor to initialize.

  Returns:
    The initialized tensor.
  """
  num_units_out = shape[1]
  unif_init_range = 1.0 / (num_units_out)**(0.5)
  return jax.random.uniform(key, shape, dtype, -1) * unif_init_range


def _output_add(block_x, orig_x):
  """Add two tensors, padding them with zeros or pooling them if necessary.

  Args:
    block_x: Output of a resnet block.
    orig_x: Residual branch to add to the output of the resnet block.

  Returns:
    The sum of blocks_x and orig_x. If necessary, orig_x will be average pooled
      or zero padded so that its shape matches orig_x.
  """
  stride = orig_x.shape[-2] // block_x.shape[-2]
  strides = (stride, stride)
  if block_x.shape[-1] != orig_x.shape[-1]:
    orig_x = nn.avg_pool(orig_x, strides, strides)
    channels_to_add = block_x.shape[-1] - orig_x.shape[-1]
    orig_x = jnp.pad(orig_x, [(0, 0), (0, 0), (0, 0), (0, channels_to_add)])
  return block_x + orig_x


class WideResnetBlock(nn.Module):
  """Defines a single WideResnetBlock.

  Attributes:
    channels: How many channels to use in the convolutional layers.
    strides: Strides for the pooling.
    activate_before_residual: True if the batch norm and relu should be applied
      before the residual branches out (should be True only for the first block
      of the model).
  """

  channels: int
  strides: Tuple[int, int] = (1, 1)
  activate_before_residual: bool = False

  @nn.compact
  def __call__(self, x, train=True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, features]
        where dim is the resolution (width and height if the input is an image).
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the resnet block.
    """
    if self.activate_before_residual:
      x = activation(x, train, name='init_bn')
      orig_x = x
    else:
      orig_x = x

    block_x = x
    if not self.activate_before_residual:
      block_x = activation(block_x, train, name='init_bn')

    block_x = nn.Conv(
        self.channels, (3, 3),
        self.strides,
        padding='SAME',
        use_bias=False,
        kernel_init=conv_kernel_init_fn,
        name='conv1')(
            block_x)
    block_x = activation(block_x, train=train, name='bn_2')
    block_x = nn.Conv(
        self.channels, (3, 3),
        padding='SAME',
        use_bias=False,
        kernel_init=conv_kernel_init_fn,
        name='conv2')(
            block_x)

    return _output_add(block_x, orig_x)


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup.

  Attributes:
    blocks_per_group: How many resnet blocks to add to each group (should be 4
      blocks for a WRN28, and 6 for a WRN40).
    channels: How many channels to use in the convolutional layers.
    strides: Strides for the pooling.
    activate_before_residual: True if the batch norm and relu should be applied
      before the residual branches out (should be True only for the first group
      of the model).
  """

  blocks_per_group: int
  channels: int
  strides: Tuple[int, int] = (1, 1)
  activate_before_residual: bool = False
  use_additional_skip_connections: bool = _USE_ADDITIONAL_SKIP_CONNECTIONS

  @nn.compact
  def __call__(self, x, train=True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, features]
        where dim is the resolution (width and height if the input is an image).
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the resnet block.
    """
    orig_x = x
    for i in range(self.blocks_per_group):
      x = WideResnetBlock(
          self.channels,
          self.strides if i == 0 else (1, 1),
          activate_before_residual=self.activate_before_residual and
          not i)(x, train)
    if self.use_additional_skip_connections:
      x = _output_add(x, orig_x)
    return x


class WideResnet(nn.Module):
  """Defines the WideResnet Model."""

  blocks_per_group: int
  channel_multiplier: int
  num_classes: int
  use_additional_skip_connections: bool = _USE_ADDITIONAL_SKIP_CONNECTIONS
  train: bool = False

  @nn.compact
  def __call__(self, x):
    """Implements a WideResnet module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, 3] where
        dim is the resolution of the image.

    Returns:
      The output of the WideResnet, a tensor of shape [batch_size, num_classes].
    """
    first_x = x
    x = nn.Conv(
        16, (3, 3),
        padding='SAME',
        name='init_conv',
        kernel_init=conv_kernel_init_fn,
        use_bias=False)(
            x)
    x = WideResnetGroup(
        self.blocks_per_group,
        16 * self.channel_multiplier,
        activate_before_residual=True)(x, self.train)
    x = WideResnetGroup(self.blocks_per_group, 32 * self.channel_multiplier,
                        (2, 2))(x, self.train)
    x = WideResnetGroup(self.blocks_per_group, 64 * self.channel_multiplier,
                        (2, 2))(x, self.train)
    if self.use_additional_skip_connections:
      x = _output_add(x, first_x)
    x = activation(x, train=self.train, name='pre-pool-bn')
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, kernel_init=dense_layer_init_fn)(x)
    return x


WRN28_2 = functools.partial(
    WideResnet, blocks_per_group=4, channel_multiplier=2)

WRN28_10 = functools.partial(
    WideResnet, blocks_per_group=4, channel_multiplier=10)


def create_model(model_name, config):
  """Creates model partial function."""
  del config
  if model_name == 'wrn28-10':
    model_cls = WRN28_10
  elif model_name == 'wrn28-2':
    model_cls = WRN28_2
  return model_cls
