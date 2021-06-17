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
"""Utility functions."""

import functools
import os
import time
from typing import Any, Dict, Sequence

from absl import logging
from clu import checkpoint
from clu import platform
import flax
import jax
import jax.numpy as jnp
import numpy as np
import scipy

POS_EMBED = "PositionEmbedding"  # Match the class name of PositionEmbedding
HEAD = "Dense"




def compute_flops(model_cls: Any,
                  variables: Dict[str, Any],
                  input_shape: Sequence[int],
                  fuse_multiply_add: bool = True) -> str:
  """Performs static analysis of the graph to compute theoretical FLOPs."""
  if input_shape[0] != 1:
    raise ValueError("FLOP test requires batch size dim is 1.")
  model = model_cls(train=False)

  def apply_fn(x):
    return model.apply(variables, x, mutable=False)

  model_input = jnp.ones(input_shape, dtype=jnp.float32)
  # jax.xla_computation must accept a function that takes input argument only.
  m = jax.xla_computation(apply_fn)(model_input).as_hlo_module()
  client = jax.lib.xla_bridge.get_backend()
  analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)  # pylint: disable=protected-access
  flops = analysis["flops"]
  if fuse_multiply_add:
    flops = flops / 2
  gflops = flops / (10**9)
  logging.info("Module: GFLOPs %0.3f for input shape %s", gflops, input_shape)
  message = "GFLOPS: %0.3f" % gflops
  return message


def log_throughput(model_cls: Any,
                   variables: Dict[str, Any],
                   input_shape: Sequence[int],
                   iterations: int = 500) -> str:
  """Log throughput of models."""
  model = model_cls(train=False)

  inputs = jnp.ones(input_shape, jnp.float32)
  batch_size = inputs.shape[0]
  logging.info("Start to compute throughput for input %s...", input_shape)

  apply_fn = jax.jit(functools.partial(model.apply, mutable=False))
  # Let it compile first with zombie runs.
  for _ in range(10):
    y = apply_fn(variables, inputs)

  start = time.time()
  for _ in range(iterations):
    y = apply_fn(variables, inputs)
  y.block_until_ready()
  total_time = time.time() - start

  logging.info("Cuda time cost per iteration %.3f", total_time / iterations)
  message = "Throughput: %.3f image/s" % (iterations * batch_size / total_time)
  logging.info(message)
  return message


def cosine_decay(lr: float, step: float, total_steps: int):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def linear_decay(lr: float, step: float, total_steps: int):
  ratio = jnp.maximum(0., step / total_steps)
  return lr * (1 - ratio)


def get_learning_rate(step: int,
                      *,
                      base_learning_rate: float,
                      steps_per_epoch: int,
                      num_epochs: int,
                      schedule: str = "cosine",
                      warmup_epochs: int = 5,
                      min_learning_rate: float = 0.):
  """Cosine learning rate schedule."""
  logging.info(
      "get_learning_rate(step=%s, base_learning_rate=%s, steps_per_epoch=%s, num_epochs=%s",
      step, base_learning_rate, steps_per_epoch, num_epochs)
  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch should be a positive integer but was "
                     f"{steps_per_epoch}.")
  if warmup_epochs >= num_epochs:
    raise ValueError(f"warmup_epochs should be smaller than num_epochs. "
                     f"Currently warmup_epochs is {warmup_epochs}, "
                     f"and num_epochs is {num_epochs}.")
  epoch = step / steps_per_epoch
  if schedule == "cosine":
    lr = cosine_decay(base_learning_rate, epoch - warmup_epochs,
                      num_epochs - warmup_epochs)
  elif schedule == "linear":
    lr = linear_decay(base_learning_rate, epoch - warmup_epochs,
                      num_epochs - warmup_epochs)
  elif schedule == "constant":
    lr = jnp.array(base_learning_rate)
  warmup = jnp.minimum(1., epoch / warmup_epochs)
  return jnp.where(warmup < 1, lr * warmup,
                   jnp.maximum(lr * warmup, min_learning_rate))


def _reshape_position_embeddings(pa: jnp.ndarray, ratio: float) -> jnp.ndarray:
  """Resizes position embeddings with scipy zoom like ViT."""
  b, n, s, d = pa.shape
  h = w = int(np.sqrt(s))
  # Two dimension spline interpolation.
  pa = jnp.reshape(pa, (b, n, h, w, d))
  newh = neww = int(jnp.ceil(h * ratio))
  pa_new_numpy = scipy.ndimage.zoom(
      np.array(pa), (1, 1, newh / h, neww / w, 1), order=1)
  pa_new = jax.numpy.asarray(pa_new_numpy)
  pa_new = jnp.reshape(pa_new, (b, n, newh * neww, d))
  return pa_new


def load_and_custom_init_checkpoint(init_state: Any,
                                    checkpoint_path: str,
                                    *,
                                    resize_posembed: float = 1.0,
                                    reinit_head: str = None) -> Any:
  """Load checkpoint for finetuing task, e.g. 384 ImageNet classification."""

  def _find_var_names(s):
    return [i for i in s.keys()]

  logging.info("Load finetune checkpoint from %s", checkpoint_path)
  # 1) Copy model params init_param_dict.
  state = checkpoint.load_state_dict(os.path.split(checkpoint_path)[0])
  init_param_dict = state["optimizer"]["target"]
  state_params = flax.core.freeze(init_param_dict)

  if resize_posembed != 1:
    # resize_posembed represents the image size ratio (new size / orignal
    # size in the checkpoint).
    # 2) Resize POS_EMBED variables and update to init_param_dict
    for pkey in init_param_dict.keys():
      # POS_EMBED is assumed to exist in the first level of init_param_dict.
      if POS_EMBED in pkey:
        # Find variable name for POS_EMBED.
        var_names = _find_var_names(init_param_dict[pkey])
        assert len(var_names) == 1
        var_name = var_names[0]
        pa = state_params[pkey][var_name]
        pa_new = _reshape_position_embeddings(pa, resize_posembed)
        init_param_dict[pkey][var_name] = pa_new
        pa_expected_shape = init_state.optimizer.target[pkey][var_name].shape
        assert jnp.array_equal(pa_expected_shape, pa_new.shape)
        logging.info("Reshape %s.%s from %s to %s", pkey, var_name, pa.shape,
                     pa_new.shape)
  if reinit_head:
    count = 1
    # 3) Re-init classification head parameters.
    for pkey in init_param_dict.keys():
      # kernel/bias are assumed to exist in the first level of init_param_dict.
      if HEAD in pkey:
        var_names = _find_var_names(init_param_dict[pkey])
        for var_name in var_names:
          count += 1
          pa = state_params[pkey][var_name]
          if reinit_head == "zero_all":
            pa_new = jnp.zeros_like(init_state.optimizer.target[pkey][var_name])
          else:
            raise NotImplementedError(
                f"reinit_head mode {reinit_head} not found.")
          init_param_dict[pkey][var_name] = pa_new
          logging.info("Zero init %s.%s (%s)", pkey, var_name, pa_new.shape)
    assert count, "Does not found head parameters"

  # 4): Copy model params to init_state.
  optimizer = init_state.optimizer.replace(
      target=flax.core.freeze(init_param_dict))
  init_state = init_state.replace(optimizer=optimizer)
  return init_state
