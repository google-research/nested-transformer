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
"""A config for training NesT on ImageNet."""

import ml_collections


def get_config():
  """Configs.

  Configurations for NesT-B, Nest-S, Nest-T respectively.
    [{
        "model_name": "nest_base_s196_224",
        "per_device_batch_size": 16.
    },
    {
        "model_name": "nest_small_s196_224",
        "per_device_batch_size": 64.
    },
    {
        "model_name": "nest_tiny_s196_224",
        "per_device_batch_size": 64.
    }]
  """
  config = ml_collections.ConfigDict()
  config.model_name = "nest_base_s196_224"
  config.per_device_batch_size = 16

  config.dataset = "imagenet"
  config.learning_rate = 2.5e-4
  config.optim = "adamw"
  config.optim_wd_ignore = ["pos_embedding"]
  config.grad_clip_max_norm = 0
  config.learning_rate_schedule = "cosine"
  config.warmup_epochs = 20
  config.weight_decay = 0.05
  config.num_epochs = 300
  config.num_train_steps = -1
  config.num_eval_steps = -1

  config.eval_pad_last_batch = True
  config.log_loss_every_steps = 500
  config.eval_every_steps = -1
  config.eval_per_epochs = 10
  config.checkpoint_every_steps = 5000
  config.shuffle_buffer_size = 1000

  config.seed = 42
  config.trial = 0  # Dummy for repeated runs.

  # Add randaugment.
  config.augment = ml_collections.ConfigDict()
  config.augment.type = "randaugment"  # Set to `default` to disable
  # All parameters start with `config.augment.randaugment_`.
  config.augment.randaugment_num_layers = 2
  config.augment.randaugment_cutout = False
  config.augment.randaugment_magnitude = 9
  config.augment.randaugment_magstd = 0.5
  config.augment.randaugment_prob_to_apply = 0.5
  config.augment.size = 224
  # Add random erasing.
  config.randerasing = ml_collections.ConfigDict()
  config.randerasing.erase_prob = 0.25  # Set to 0 to disable
  # Add mix style augmentation.
  config.mix = ml_collections.ConfigDict()
  config.mix.mixup_alpha = 0.8
  config.mix.prob_to_apply = 1.0  # Set to 0 to disable
  config.mix.smoothing = 0.1

  # Add color jitter.
  # It uses default impl='simclrv2'
  config.colorjitter = ml_collections.ConfigDict()
  config.colorjitter.type = "colorjitter"  # Set to `default` to disable
  config.colorjitter.colorjitter_strength = 0.3
  config.colorjitter.size = 224

  config.eval_only = False
  config.init_checkpoint = ""

  return config
