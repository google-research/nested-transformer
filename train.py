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
"""Methods for training ResNet-50 on ImageNet using JAX."""

import functools
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from libml import input_pipeline
from libml import losses
from libml import utils
from models import nest_net
from models import resnet_v1
from models import wide_resnet


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer
  model_state: Any


def create_train_state(config: ml_collections.ConfigDict, rng: np.ndarray,
                       input_shape: Sequence[int],
                       num_classes: int) -> Tuple[Any, TrainState]:
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shape: Shape of the inputs fed into the model.
    num_classes: Number of classes in the output layer.

  Returns:
    The initialized TrainState with the optimizer.
  """
  # Create model function.
  if config.model_name.startswith("resnet"):
    model_cls = resnet_v1.create_model(config.model_name, config)
  elif config.model_name.startswith("nest"):
    model_cls = nest_net.create_model(config.model_name, config)
  else:
    raise ValueError(f"Model {config.model_name} not supported.")
  model = functools.partial(model_cls, num_classes=num_classes)
  variables = model(train=False).init(rng, jnp.ones(input_shape))
  model_state = dict(variables)
  params = model_state.pop("params")
  parameter_overview.log_parameter_overview(params)
  if config.get("log_model_profile"):  # Be True or [1, 2]
    message_1 = utils.log_throughput(model, variables, input_shape)
    message_2 = utils.compute_flops(model, variables,
                                    [1] + list(input_shape[1:]))
    count = parameter_overview.count_parameters(params)
    message_3 = "Params: {:,}".format(count)
    message = ", ".join([message_1, message_2, message_3])
    logging.info("Profile results %s", message)
    if (isinstance(config.log_model_profile, (int,)) and
        config.log_model_profile >= 2):
      sys.exit(0)
  # Create optimizer.
  if config.optim in ("adamw", "adam"):
    if config.get("optim_wd_ignore"):
      # Allow zero weight decay for certain parameters listed in optim_wd_ignore
      igns = config.optim_wd_ignore
      p = flax.optim.ModelParamTraversal(
          lambda path, _: not any([i in path for i in igns]))
      p_nowd = flax.optim.ModelParamTraversal(
          lambda path, _: any([i in path for i in igns]))
      p_opt = flax.optim.Adam(weight_decay=config.weight_decay)
      p_nowd_opt = flax.optim.Adam(weight_decay=0)
      optimizer = flax.optim.MultiOptimizer((p, p_opt),
                                            (p_nowd, p_nowd_opt)).create(params)
    else:
      optimizer = flax.optim.Adam(
          weight_decay=config.weight_decay).create(params)
  elif config.optim == "sgd":
    optimizer = flax.optim.Momentum(
        beta=config.sgd_momentum, nesterov=True).create(params)
  else:
    raise NotImplementedError(f"{config.optim} does not exist.")

  return model, TrainState(step=0, optimizer=optimizer, model_state=model_state)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):

  accuracy: metrics.Accuracy
  eval_loss: metrics.Average.from_output("loss")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):

  train_accuracy: metrics.Accuracy
  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")
  l2_grads: metrics.Average.from_output("l2_grads")


def train_step(
    model: Any,
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: np.ndarray,
    learning_rate_fn: Callable[[int], float],
    weight_decay: float,
    grad_clip_max_norm: Optional[float] = None
) -> Tuple[TrainState, metrics.Collection]:
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    rng: Random seed.
    learning_rate_fn: Function that computes the learning rate given the step
      number.
    weight_decay: Weighs L2 regularization term.
    grad_clip_max_norm: Gradient norm max value. Default to be None.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.info("train_step(batch=%s)", batch)

  step = state.step + 1
  lr = learning_rate_fn(step)
  # Convert one-hot labels to single values if appliable.
  u_labels = (
      jnp.argmax(batch["label"], 1)
      if len(batch["label"].shape) > 1 else batch["label"])

  def loss_fn(params):
    variables = {"params": params}
    variables.update(state.model_state)
    logits, new_model_state = model(train=True).apply(
        variables,
        batch["image"],
        mutable=["batch_stats"],
        rngs={"dropout": rng})
    loss = jnp.mean(
        losses.softmax_cross_entropy_loss(logits=logits, labels=batch["label"]))
    if weight_decay > 0:
      weight_penalty_params = jax.tree_leaves(variables["params"])
      weight_l2 = sum(
          [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
      weight_penalty = weight_decay * 0.5 * weight_l2
      loss = loss + weight_penalty
    new_model_state = dict(new_model_state)
    return loss, (new_model_state, logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_model_state, logits)), grad = grad_fn(state.optimizer.target)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")

  # Compute l2 grad always for training debugging.
  grads, _ = jax.tree_flatten(grad)
  l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
  if grad_clip_max_norm:
    g_factor = jnp.minimum(1.0, grad_clip_max_norm / (l2_g + 1e-6))
    grad = jax.tree_map(lambda p: g_factor * p, grad)

  new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
  new_state = state.replace(  # pytype: disable=attribute-error
      step=step,
      optimizer=new_optimizer,
      model_state=new_model_state)

  metrics_update = TrainMetrics.gather_from_model_output(
      loss=loss,
      logits=logits,
      labels=u_labels,
      learning_rate=lr,
      l2_grads=l2_g)
  return new_state, metrics_update


@functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=0)
def eval_step(model: Any, state: TrainState,
              batch: Dict[str, jnp.ndarray]) -> metrics.Collection:
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: Replicate model state.
    batch: Inputs that should be evaluated.

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.info("eval_step(batch=%s)", batch)
  variables = {
      "params": state.optimizer.target,
  }
  variables.update(state.model_state)
  logits = model(train=False).apply(variables, batch["image"], mutable=False)
  loss = jnp.mean(
      losses.cross_entropy_loss(logits=logits, labels=batch["label"]))
  return EvalMetrics.gather_from_model_output(
      logits=logits,
      labels=batch["label"],
      loss=loss,
      mask=batch.get("mask"),
  )


class StepTraceContextHelper:
  """Helper class to use jax.profiler.StepTraceAnnotation."""

  def __init__(self, name: str, init_step_num: int):
    self.name = name
    self.step_num = init_step_num

  def __enter__(self):
    self.context = jax.profiler.StepTraceAnnotation(
        self.name, step_num=self.step_num)
    self.step_num += 1
    self.context.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.context.__exit__(exc_type, exc_value, tb)
    self.context = None

  def next_step(self):
    self.context.__exit__(None, None, None)
    self.__enter__()


def evaluate(model: nn.Module,
             state: TrainState,
             eval_ds: tf.data.Dataset,
             num_eval_steps: int = -1) -> Union[None, EvalMetrics]:
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = None
  with StepTraceContextHelper("eval", 0) as trace_context:
    for step, batch in enumerate(eval_ds):  # pytype: disable=wrong-arg-types
      batch = jax.tree_map(np.asarray, batch)
      metrics_update = flax_utils.unreplicate(eval_step(model, state, batch))
      eval_metrics = (
          metrics_update
          if eval_metrics is None else eval_metrics.merge(metrics_update))
      if num_eval_steps > 0 and step + 1 == num_eval_steps:
        break
      trace_context.next_step()
  return eval_metrics


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  tf.io.gfile.makedirs(workdir)
  rng = jax.random.PRNGKey(config.seed)

  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  ds_info, train_ds, eval_ds = input_pipeline.create_datasets(config, data_rng)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types

  # Learning rate schedule.
  global_batch_size = config.per_device_batch_size * jax.device_count()
  num_train_steps = config.num_train_steps
  if num_train_steps == -1:
    num_train_steps = train_ds.cardinality().numpy()
    assert num_train_steps > 0

  steps_per_epoch = num_train_steps // config.num_epochs
  if config.eval_every_steps == -1 or config.get("eval_per_epochs"):
    # Show plots in the epoch view (x-axis).
    eval_every_steps = steps_per_epoch * config.get("eval_per_epochs", 1)
    summary_step_div = steps_per_epoch
  else:
    eval_every_steps = config.eval_every_steps
    summary_step_div = 1
  logging.info(
      "global_batch_size=%d, num_train_steps=%d, steps_per_epoch=%d, eval_every_steps=%d",
      global_batch_size, num_train_steps, steps_per_epoch, eval_every_steps)
  # We treat the learning rate in the config as the learning rate for batch size
  # 256 but scale it according to our batch size.
  base_learning_rate = config.learning_rate * global_batch_size / 256.0
  learning_rate_fn = functools.partial(
      utils.get_learning_rate,
      base_learning_rate=base_learning_rate,
      steps_per_epoch=steps_per_epoch,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs,
      schedule=config.learning_rate_schedule,
      min_learning_rate=config.get("min_learning_rate", 0.) *
      global_batch_size / 256.0)

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, state = create_train_state(
      config,
      model_rng,
      input_shape=train_ds.element_spec["image"].shape[1:],
      num_classes=ds_info.features["label"].num_classes)

  if config.get("init_checkpoint"):
    state = utils.load_and_custom_init_checkpoint(
        state,
        config.init_checkpoint,
        resize_posembed=config.get("resize_posembed", 1),
        reinit_head=config.get("reinit_head", None))
  # Conduct evaluation only.
  if config.get("eval_only"):
    assert config.get("init_checkpoint")
    state = flax_utils.replicate(state)
    eval_writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0)
    eval_metrics = evaluate(model, state, eval_ds, config.num_eval_steps)
    eval_results = eval_metrics.compute()
    eval_writer.write_scalars(0, eval_results)
    logging.info("Eval results %s", eval_results)
    sys.exit(0)
  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=2)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  disable_l2_wd = config.optim == "adamw"
  # Distribute training.
  state = flax_utils.replicate(state)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          weight_decay=0 if disable_l2_wd else config.weight_decay,
          grad_clip_max_norm=config.get("grad_clip_max_norm")),
      axis_name="batch")

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if initial_step == 1:
    writer.write_hparams(dict(config))

  logging.info("Starting training loop at step %d.", initial_step)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  train_metrics = None
  rng, drop_out_rng = jax.random.split(rng, 2)
  drop_out_rng = jax.random.fold_in(drop_out_rng, jax.process_index())

  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      is_last_step = step == num_train_steps

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = jax.tree_map(np.asarray, next(train_iter))
        drop_out_rng_step = jax.random.fold_in(drop_out_rng, step)
        drop_out_rng_step_all = jax.random.split(drop_out_rng_step,
                                                 jax.local_device_count())
        state, metrics_update = p_train_step(
            state=state, batch=batch, rng=drop_out_rng_step_all)
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      if step % config.log_loss_every_steps == 0 or is_last_step:
        writer.write_scalars(step // summary_step_div, train_metrics.compute())
        train_metrics = None

      if step % eval_every_steps == 0 or is_last_step:
        with report_progress.timed("eval"):
          eval_metrics = evaluate(model, state, eval_ds, config.num_eval_steps)
        writer.write_scalars(step // summary_step_div, eval_metrics.compute())

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        with report_progress.timed("checkpoint"):
          ckpt.save(flax_utils.unreplicate(state))

  logging.info("Finishing training at step %d", num_train_steps)
