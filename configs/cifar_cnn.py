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
"""A config for training convnet on CIFAR."""

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # Note: This is currently not used but it's nice to document the itended
  # binary and allows sharing a XM launcher script for multiple binaries.
  config.dataset = "cifar10"
  config.model_name = "resnet18_cifar"
  config.learning_rate = 0.2
  config.grad_clip_max_norm = 0
  config.learning_rate_schedule = "cosine"
  config.optim = "sgd"
  config.warmup_epochs = 5
  config.sgd_momentum = 0.9
  config.weight_decay = 0.0005
  config.num_epochs = 200
  config.num_train_steps = -1
  config.num_eval_steps = 100
  config.per_device_batch_size = 128
  config.eval_pad_last_batch = True

  config.log_loss_every_steps = 500
  config.eval_every_steps = 200
  config.checkpoint_every_steps = 5000
  config.shuffle_buffer_size = 10000

  config.seed = 42

  return config
