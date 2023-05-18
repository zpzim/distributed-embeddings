# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmarks of synthetic models"""

import os
from time import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.python.ops.ragged import ragged_tensor

tf.keras.utils.set_random_seed(12345)

import horovod.tensorflow.keras as hvd

from config_v3 import synthetic_models_v3
from config_v3 import generate_custom_config

from synthetic_models import SyntheticModelTFDE, SyntheticModelNative, InputGenerator

from distributed_embeddings.python.layers import dist_model_parallel as dmp

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'
np.set_printoptions(precision=3, suppress=True)


# pylint: disable=line-too-long
# yapf: disable
flags.DEFINE_integer("batch_size", 4, help="Global batch size")
flags.DEFINE_integer("num_data_batches", 1, help="Number of batches of synthetic data to generate")
flags.DEFINE_float("alpha", 1.05, help="Exponent to generate power law distributed data")
flags.DEFINE_integer("num_steps", 100, help="Number of steps to benchmark")
flags.DEFINE_bool("dp_input", False, help="Use data parallel input")
flags.DEFINE_string("model", "tiny", help="Choose model size to run benchmark")
flags.DEFINE_enum("optimizer", "sgd", ["sgd", "adagrad", "adam"], help="Optimizer")
flags.DEFINE_integer("column_slice_threshold", None, help="Upper bound of elements count in each column slice")
flags.DEFINE_bool("use_model_fit", False, help="Use Keras model.fit")
flags.DEFINE_string("embedding_device", "/GPU:0", help="device to place embedding. inputs are placed on same device")
flags.DEFINE_enum("embedding_api", "tfde", ["native", "tfde"], help="embedding to use.")
flags.DEFINE_bool("amp", False, help="Use mixed precision")
flags.DEFINE_float("mean_dynamic_hotness_ratio", 1.0, help="For enabling dynamic hot input. Ratio of nnz to set the average hotness of data generated for a particular feature. Set between [0,1]") 
flags.DEFINE_string("nnzfile", None, help="See config_v3.py for information on the format of this file.")
flags.DEFINE_string("modelconfig", None, help="See config_v3.py for information on the format of this file.")
flags.DEFINE_string('input_file_fmt_string', None, help='Specifies the format string for the input file path. e.g. data/train_{0}.pkl, If specified uses data from the input file rather than randomly generating.')
flags.DEFINE_list('custom_mlp_sizes', [256, 128], help='Size of mlp layers for custom model')
flags.DEFINE_integer('custom_numeric_features', 1, help='Number of numeric features for a custom model')
flags.DEFINE_integer('custom_interact_stride', None, help='Interact stride for custom model.')
flags.DEFINE_integer('custom_cross_count', None, help='Number of DCN crosses for custom model.')
flags.DEFINE_string('custom_cross_activation', None, help='Preactivation for DCN crosses for custom model. e.g. relu or tanh.')
flags.DEFINE_float('custom_cross_projection', 1.0, help='Size of the DCN projection dimension as a fraction of the input size.')
flags.DEFINE_bool('custom_cross_use_bias', False, help='Whether to use bias for the DCN crosses in custom model.')
flags.DEFINE_string('custom_combiner', 'sum', help='Combiner for custom model.')
flags.DEFINE_string('lookups_file', None, help='File with lookup frequency info.')
flags.DEFINE_bool('dump_lookups', False, help='Whether to dump the lookup frequency per table after the run.')
# yapf: enable
# pylint: enable=line-too-long

FLAGS = flags.FLAGS


def main(_):
  hvd.init()
  hvd_rank = hvd.rank()
  hvd_size = hvd.size()
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  if FLAGS.amp:
    mixed_precision.set_global_policy('mixed_float16')

  if FLAGS.batch_size % hvd_size != 0:
    raise ValueError(F"Batch size ({FLAGS.batch_size}) is not divisible by world size ({hvd_size})")


  if FLAGS.modelconfig and FLAGS.nnzfile:
    custom_cross_params = None
    if FLAGS.custom_cross_count is not None:
      num_crosses = FLAGS.custom_cross_count
      proj = FLAGS.custom_cross_projection
      bias = FLAGS.custom_cross_use_bias
      activation = FLAGS.custom_cross_activation
      custom_cross_params = [num_crosses, proj, bias, activation]
    mlp_sizes = None
    if FLAGS.custom_mlp_sizes:
      mlp_sizes = []
      for elem in FLAGS.custom_mlp_sizes:
        mlp_sizes.append(int(elem))
    model_config, table_to_features_map = generate_custom_config(FLAGS.modelconfig, FLAGS.nnzfile, mlp_sizes, FLAGS.custom_numeric_features, FLAGS.custom_interact_stride, FLAGS.custom_combiner, custom_cross_params, FLAGS.lookups_file)
  else:
    model_config = synthetic_models_v3[FLAGS.model]
    feat_id = 0
    table_id = 0
    table_to_features_map = []
    for conf in model_config.embedding_configs:
      for table in range(conf.num_tables):
        feats = []
        for nnz in conf.nnz:
          feats.append(feat_id)
          feat_id += 1
        table_to_features_map.append(feats)
        table_id += 1
  if hvd_rank == 0:
    print(f'Tables to features: {table_to_features_map}')
  if hvd_rank == 0:
    for embedding_config in model_config.embedding_configs:
      print(embedding_config)
  if FLAGS.embedding_api == "tfde":
    if FLAGS.embedding_device != "/GPU:0":
      raise ValueError(
          F"distributed-embeddings api is not supported on device {FLAGS.embedding_device}.")
    model = SyntheticModelTFDE(model_config,
                               column_slice_threshold=FLAGS.column_slice_threshold,
                               dp_input=FLAGS.dp_input)
  elif FLAGS.embedding_api == "native":
    if FLAGS.dp_input is False or FLAGS.column_slice_threshold is not None:
      raise ValueError(
          "Model parallel inputs and column slicing are not supported with native embedding api.")
    model = SyntheticModelNative(model_config, embedding_device=FLAGS.embedding_device)
  else:
    raise ValueError(F"Unknown embedding api {FLAGS.embedding_api}.")

  mp_input_ids = None if FLAGS.dp_input else model.embeddings.strategy.input_ids_list[hvd_rank]
  if FLAGS.input_file_fmt_string:
    if not FLAGS.dp_input:
      raise ValueError("Model parallel inputs are not supported with a custom input file.")
    input_filename = FLAGS.input_file_fmt_string.format(hvd_rank)
  else:
    input_filename = None
  if hvd_rank == 0:
    print(F"Construct input gen")
  input_gen = InputGenerator(model_config,
                             FLAGS.batch_size,
                             alpha=FLAGS.alpha,
                             mp_input_ids=mp_input_ids,
                             num_batches=FLAGS.num_data_batches,
                             input_filename=input_filename,
                             embedding_device=FLAGS.embedding_device,
                             mean_hotness_ratio=FLAGS.mean_dynamic_hotness_ratio)
  if hvd_rank == 0:
    print(F"Constructed input gen")

  if FLAGS.optimizer == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0)
  if FLAGS.optimizer == "adagrad":
    optimizer = tf.keras.optimizers.Adagrad()
  if FLAGS.optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam()

  bce = keras.losses.BinaryCrossentropy(from_logits=True)
  if FLAGS.use_model_fit:
    optimizer = dmp.DistributedOptimizer(optimizer)
    model.compile(optimizer=optimizer, loss=bce)
    callbacks = [dmp.BroadcastGlobalVariablesCallback(0)]
    epochs = FLAGS.num_steps // FLAGS.num_data_batches
    model.fit(input_gen,
              epochs=epochs,
              batch_size=FLAGS.batch_size,
              callbacks=callbacks,
              verbose=1 if hvd_rank == 0 else 0)
    return

  if hvd_rank == 0:
    print(F"Gen input")
  # Not using model.fit() api. benchmark custom loop below
  # Run one step to init
  (numerical_features, cat_features), labels = input_gen[-1]
  if hvd_rank == 0:
    print(F"First model step")
  model((numerical_features, cat_features))
  dmp.broadcast_variables(model.variables, root_rank=0)

  @tf.function
  def train_step(numerical_features, categorical_features, labels):
    with tf.GradientTape() as tape:
      predictions = model((numerical_features, categorical_features))
      loss = bce(labels, predictions)
    tape = dmp.DistributedGradientTape(tape)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
  if hvd_rank == 0:
    print(F"Warm up.")
  model((numerical_features, cat_features))
  # Run 5 steps to compile and warm up
  (numerical_features, cat_features), labels = input_gen[-1]
  for _ in range(5):
    loss = train_step(numerical_features, cat_features, labels)
  loss = hvd.allreduce(loss, name="mean_loss", op=hvd.Average)
  # printing initial loss here to force sync before we start timer
  print(F"Initial loss: {loss:.3f}")

  strategy = model.get_embedding_strategy()

  lookups_per_feature = None
  sharded_lookups_per_feature = None

  start = time()
  if hvd_rank == 0:
    logging.info("Starting benchmark steps...")
  # Input data consumes a lot of memory. Instead of generating num_steps batch of synthetic data,
  # We generate smaller amount of data and loop over them
  for step in range(FLAGS.num_steps):
    inputs = input_gen[step % FLAGS.num_data_batches]
    (numerical_features, cat_features), labels = inputs
    if lookups_per_feature is None:
      lookups_per_feature = np.zeros(shape=(len(cat_features),), dtype=np.int64)
    for i, inp in enumerate(cat_features):
      if isinstance(inp, ragged_tensor.RaggedTensor):
        lookups_per_feature[i] += len(inp.values)
      else:
        lookups_per_feature[i] += tf.size(inp, out_type=tf.int64)
    loss = train_step(numerical_features, cat_features, labels)
    if step % 50 == 0:
      loss = hvd.allreduce(loss, name="mean_loss", op=hvd.Average)
      if hvd_rank == 0:
        print(F"Benchmark step [{step}/{FLAGS.num_steps}]")

  loss = hvd.allreduce(loss, name="mean_loss", op=hvd.Average)
  if hvd_rank == 0:
    # printing GPU tensor forces a sync. loss was allreduced, printing on one GPU is enough
    # for computing time so we don't print noisy messages from all ranks
    print(F"loss: {loss:.3f}")
    stop = time()
    print(F"Iteration time: {(stop - start) * 1000 / FLAGS.num_steps:.3f} ms")

  lookups_per_feature = hvd.allreduce(lookups_per_feature, name="get_num_lookups_per_feature", op=hvd.Sum)

  total_samples = FLAGS.num_steps * FLAGS.num_data_batches * FLAGS.batch_size

  if hvd_rank == 0:
    total_input_lookups = tf.math.reduce_sum(lookups_per_feature)
    lookups_each_device = np.array([sum([lookups_per_feature[feat] for feat in rank]) for rank in strategy.input_ids_list])
    total_lookups = tf.math.reduce_sum(lookups_each_device)
    print(f"Avg lookups per sample (before sharding): across all devices {total_input_lookups / total_samples}")
    print(f"Avg lookups per sample (inc sharded lookups): across all devices {total_lookups / total_samples}")
    lookup_ratio_per_device = tf.cast(lookups_each_device, tf.float64) / tf.cast(total_lookups, tf.float64)
    print(f'Avg lookups per sample on each device (inc sharded lookups): {lookups_each_device / total_samples}')
    print(f'Share of total lookups on each device (inc sharded lookups): {lookup_ratio_per_device}')
    print(F"Avg lookups per feature per sample (before sharding): {lookups_per_feature / total_samples}")
    if table_to_features_map is not None:
      num_tables = len(model_config.embedding_configs)
      lookups_per_table = []
      for feats in table_to_features_map:
        lookups_for_table = 0
        for feat in feats:
          if feat < len(lookups_per_feature):
            lookups_for_table += lookups_per_feature[feat]
        lookups_per_table.append(lookups_for_table)
      #lookups_per_table = [sum(lookups_per_feature[table_to_features_map[i]]) for i in range(num_tables)]
      lookups_per_table = np.array(lookups_per_table)
      lookups_per_table_per_sample = lookups_per_table / total_samples
      print(F"Avg lookups per table per sample (before sharding): {lookups_per_table_per_sample}")
      if FLAGS.dump_lookups:
        with open('lookup_frequency.pkl', 'wb') as f:
          pickle.dump(lookups_per_table_per_sample.tolist(), f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
