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
"""Deterministic input pipeline for ImageNet."""

import functools
from typing import Callable, Dict, Tuple, Union

from absl import logging
from clu import deterministic_data
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from libml import preprocess

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]
RANDOM_ERASING = preprocess.RANDOM_ERASING
AUGMENT = preprocess.AUGMENT
MIX = preprocess.MIX
COLORJITTER = preprocess.COLORJITTER


def preprocess_with_per_batch_rng(ds: tf.data.Dataset,
                                  preprocess_fn: Callable[[Features], Features],
                                  *, rng: jnp.ndarray) -> tf.data.Dataset:
  """Maps batched `ds` using the preprocess_fn and a deterministic RNG per batch.

  This preprocess_fn usually contains data preprcess needs a batch of data, like
  Mixup.

  Args:
    ds: Dataset containing Python dictionary with the features. The 'rng'
      feature should not exist.
    preprocess_fn: Preprocessing function that takes a Python dictionary of
      tensors and returns a Python dictionary of tensors. The function should be
      convertible into a TF graph.
    rng: Base RNG to use. Per example RNGs will be derived from this by folding
      in the example index.

  Returns:
    The dataset mapped by the `preprocess_fn`.
  """
  rng = list(jax.random.split(rng, 1)).pop()

  def _fn(example_index: int, features: Features) -> Features:
    example_index = tf.cast(example_index, tf.int32)
    features["rng"] = tf.random.experimental.stateless_fold_in(
        tf.cast(rng, tf.int64), example_index)
    processed = preprocess_fn(features)
    if isinstance(processed, dict) and "rng" in processed:
      del processed["rng"]
    return processed

  return ds.enumerate().map(
      _fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_dataset_fns(
    config: ml_collections.ConfigDict
) -> Tuple[tfds.core.DatasetBuilder, tfds.core.ReadInstruction, Callable[
    [Features], Features], Callable[[Features], Features], str, Union[Callable[
        [Features], Features], None]]:
  """Gets dataset specific functions."""
  # Use config.augment.type to control custom aug vs default aug, it makes sweep
  # parameter setting easier.
  use_custom_process = (
      config.get(AUGMENT) or config.get(RANDOM_ERASING) or
      config.get(COLORJITTER))
  use_batch_process = config.get(MIX)

  label_key = "label"
  image_key = "image"
  if config.dataset.startswith("imagenet"):
    dataset_builder = tfds.builder("imagenet2012")
    train_split = deterministic_data.get_read_instruction_for_host(
        "train", dataset_builder.info.splits["train"].num_examples)
    test_split_name = "validation"

    # If there is resource error during preparation, checkout
    # https://github.com/tensorflow/datasets/issues/1441#issuecomment-581660890
    dataset_builder.download_and_prepare()

    # Default image size is 224, one can use a different one by setting
    # config.input_size. Note that some augmentation also requires specifying
    # input_size through respective config.
    input_size = config.get("input_size", 224)
    # Create augmentaton fn.
    if use_custom_process:
      # When using custom augmentation, we use mean/std normalization.
      logging.info("Configure augmentation type %s", config.augment.type)
      mean = tf.constant(
          preprocess.IMAGENET_DEFAULT_MEAN, dtype=tf.float32, shape=[1, 1, 3])
      std = tf.constant(
          preprocess.IMAGENET_DEFAULT_STD, dtype=tf.float32, shape=[1, 1, 3])
      basic_preprocess_fn = functools.partial(
          preprocess.train_preprocess, input_size=input_size)
      preprocess_fn = preprocess.get_augment_preprocess(
          config.get(AUGMENT),
          colorjitter_params=config.get(COLORJITTER),
          randerasing_params=config.get(RANDOM_ERASING),
          mean=mean,
          std=std,
          basic_process=basic_preprocess_fn)
      eval_preprocess_fn = functools.partial(
          preprocess.eval_preprocess, mean=mean, std=std, input_size=input_size)
    else:
      # Standard imagenet preprocess with 0-1 normalization
      preprocess_fn = functools.partial(
          preprocess.train_preprocess, input_size=input_size)
      eval_preprocess_fn = functools.partial(
          preprocess.eval_preprocess, input_size=input_size)

  elif config.dataset.startswith("cifar"):
    assert config.dataset in ("cifar10", "cifar100")
    dataset_builder = tfds.builder(config.dataset)
    dataset_builder.download_and_prepare()

    train_split = deterministic_data.get_read_instruction_for_host(
        "train", dataset_builder.info.splits["train"].num_examples)
    # Create augmentaton fn.
    test_split_name = "test"
    # When using custom augmentation, we use mean/std normalization.
    if config.dataset == "cifar10":
      mean, std = preprocess.CIFAR10_MEAN, preprocess.CIFAR10_STD
    else:
      mean, std = preprocess.CIFAR100_MEAN, preprocess.CIFAR100_STD
    mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 3])
    std = tf.constant(std, dtype=tf.float32, shape=[1, 1, 3])
    input_size = config.get("input_size", 32)

    if input_size != 32:
      # Finetune cifar from imagenet pretrained models.
      logging.info("Use %s input size for cifar.", input_size)
      factor = 256 / 224
      new_input_size = int(input_size * factor)

      # Resize small images by a factor of ratio.
      def train_preprocess(features):
        image = tf.io.decode_jpeg(features["image"])
        image = tf.image.resize(image, (new_input_size, new_input_size))
        features["image"] = tf.io.encode_jpeg(tf.cast(image, tf.uint8))
        return preprocess.train_preprocess(features, input_size)

      def eval_preprocess(features, mean, std):
        features["image"] = tf.cast(
            tf.image.resize(features["image"],
                            (new_input_size, new_input_size)), tf.uint8)
        return preprocess.eval_preprocess(features, mean, std, input_size)
    else:
      train_preprocess = preprocess.train_cifar_preprocess
      eval_preprocess = preprocess.cifar_eval_preprocess

    if use_custom_process:
      logging.info("Configure augmentation type %s", config.augment.type)
      # The augmentor util uses augment.size to size specific augmentation.
      config.augment.size = input_size
      preprocess_fn = preprocess.get_augment_preprocess(
          config.get(AUGMENT),
          colorjitter_params=config.get(COLORJITTER),
          randerasing_params=config.get(RANDOM_ERASING),
          mean=mean,
          std=std,
          basic_process=train_preprocess)
      eval_preprocess_fn = functools.partial(
          eval_preprocess, mean=mean, std=std)
    else:
      # When not using use_custom_process, we use 0-1 normalization.
      preprocess_fn = train_preprocess
      eval_preprocess_fn = functools.partial(
          eval_preprocess, mean=None, std=None)
  else:
    raise ValueError(f"Dataset {config.dataset} does not exist.")

  if use_batch_process:
    logging.info("Configure mix augmentation type %s", config.mix)
    # When config.mix batch augmentation is enabled.
    batch_preprocess_fn = preprocess.create_mix_augment(
        num_classes=dataset_builder.info.features[label_key].num_classes,
        **config.mix.to_dict())
  else:
    batch_preprocess_fn = None


  return (dataset_builder, train_split, preprocess_fn, eval_preprocess_fn,
          test_split_name, batch_preprocess_fn)


def create_datasets(
    config: ml_collections.ConfigDict,
    data_rng) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset, tf.data.Dataset]:
  """Create datasets for training and evaluation.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.

  Returns:
    A tuple with the dataset info, the training dataset and the evaluation
    dataset.
  """
  (dataset_builder, train_split, preprocess_fn, eval_preprocess_fn,
   test_split_name, batch_preprocess_fn) = get_dataset_fns(config)
  data_rng1, data_rng2 = jax.random.split(data_rng, 2)
  skip_batching = batch_preprocess_fn is not None
  batch_dims = [jax.local_device_count(), config.per_device_batch_size]
  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      rng=data_rng1,
      preprocess_fn=preprocess_fn,
      cache=False,
      decoders={"image": tfds.decode.SkipDecoding()},
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=batch_dims if not skip_batching else None,
      num_epochs=config.num_epochs,
      shuffle=True,
  )

  if batch_preprocess_fn:
    # Perform batch augmentation on each device and them batch devices.
    train_ds = train_ds.batch(batch_dims[-1], drop_remainder=True)
    train_ds = preprocess_with_per_batch_rng(
        train_ds, batch_preprocess_fn, rng=data_rng2)
    for batch_size in reversed(batch_dims[:-1]):
      train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(4)

  options = tf.data.Options()
  options.experimental_external_state_policy = (
      tf.data.experimental.ExternalStatePolicy.WARN)
  train_ds = train_ds.with_options(options)

  num_validation_examples = (
      dataset_builder.info.splits[test_split_name].num_examples)
  eval_split = deterministic_data.get_read_instruction_for_host(
      test_split_name, num_validation_examples, drop_remainder=False)

  eval_num_batches = None
  if config.eval_pad_last_batch:
    eval_batch_size = jax.local_device_count() * config.per_device_batch_size
    eval_num_batches = int(
        np.ceil(num_validation_examples / eval_batch_size /
                jax.process_count()))
  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      preprocess_fn=eval_preprocess_fn,
      # Only cache dataset in distributed setup to avoid consuming a lot of
      # memory in Colab and unit tests.
      cache=jax.process_count() > 1,
      batch_dims=[jax.local_device_count(), config.per_device_batch_size],
      num_epochs=1,
      shuffle=False,
      pad_up_to_batches=eval_num_batches,
  )
  return dataset_builder.info, train_ds, eval_ds
