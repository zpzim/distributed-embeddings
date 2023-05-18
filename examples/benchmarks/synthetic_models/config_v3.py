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
"""synthetic model configs"""

import pickle

from collections import namedtuple

# nnz is a list of integer(s). If not shared total number of embedding tables will
# be num_tables * len(nnz)
EmbeddingConfig = namedtuple("EmbeddingConfig",
                             ["num_tables", "nnz", "num_rows", "width", "shared", "lookup_frequency"])

# The the last MLP layer project to 1 should be omitted in mlp_sizes
# interact_stride is stride of 1d pooling which is used emulate memory limited interaction/FM
ModelConfig = namedtuple(
    "ModelConfig",
    ["name", "embedding_configs", "mlp_sizes", "num_numerical_features", "interact_stride", "combiner", "cross_params"])

def generate_custom_config(modelconfig, nnzfile, mlp_sizes, num_numerical_features, interact_stride, combiner, cross_params, lookup_file=None):

  # Load input shapes from pickle
  # Input shapes contains several lists which contain the embedding table information
  # 'table_info' contains the vocab size and embedding dimension for each table
  # 'table_to_input_mapping' maps each table_id to the list of input_ids that share it.
  with open(modelconfig,'rb') as f:
      input_shapes = pickle.load(f)

  # Maps each input id to its corresponding table and nnz info
  # Mapping looks like:
  # input_id -> (table_id, max_nnz, min_nnz, mean_nnz)
  with open(nnzfile, 'rb') as f:
      max_nnz_dict = pickle.load(f)

  lookup_freqs = None
  if lookup_file is not None:
    with open(lookup_file, 'rb') as f:
      lookup_freqs = pickle.load(f)

  embedding_data = input_shapes['table_info']
  table_to_input_mapping = input_shapes['tag_table_info']

  embedding_configs = []
  slots_per_table = []
  for i, embedding_info in enumerate(embedding_data):
    nnz_per_slot_for_table = []
    embedding_vocab = embedding_info[0]
    embedding_dim = embedding_info[1]
    slots = []
    for j, input_id in enumerate(table_to_input_mapping[i]):
      nnz_per_slot_for_table.append(max_nnz_dict[input_id][1])
      slots.append(max_nnz_dict[input_id][0])
    slots_per_table.append(slots)
    if lookup_freqs is not None:
      lookup_freq = lookup_freqs[i]
    else:
      lookup_freq = None
    embedding_configs.append(EmbeddingConfig(1,  nnz_per_slot_for_table, embedding_vocab, embedding_dim, True, lookup_freq)) 
  custom_config = ModelConfig(name="Custom", embedding_configs=embedding_configs, mlp_sizes=mlp_sizes, num_numerical_features=num_numerical_features, interact_stride=interact_stride, combiner=combiner, cross_params=cross_params)
  return custom_config, slots_per_table

model_tiny = ModelConfig(name="Tiny V3",
                         embedding_configs=[
                             EmbeddingConfig(1, [1, 10], 10000, 8, True, None),
                             EmbeddingConfig(1, [1, 10], 1000000, 16, True, None),
                             EmbeddingConfig(1, [1, 10], 25000000, 16, True, None),
                             EmbeddingConfig(1, [1], 25000000, 16, False, None),
                             EmbeddingConfig(16, [1], 10, 8, False, None),
                             EmbeddingConfig(10, [1], 1000, 8, False, None),
                             EmbeddingConfig(4, [1], 10000, 8, False, None),
                             EmbeddingConfig(2, [1], 100000, 16, False, None),
                             EmbeddingConfig(19, [1], 1000000, 16, False, None),
                         ],
                         mlp_sizes=[256, 128],
                         num_numerical_features=10,
                         interact_stride=None,
                         combiner='sum',
                         cross_params=None)

model_small = ModelConfig(name="Small V3",
                          embedding_configs=[
                              EmbeddingConfig(5, [1, 30], 10000, 16, True, None),
                              EmbeddingConfig(3, [1, 30], 4000000, 32, True, None),
                              EmbeddingConfig(1, [1, 30], 50000000, 32, True, None),
                              EmbeddingConfig(1, [1], 50000000, 32, False, None),
                              EmbeddingConfig(30, [1], 10, 16, False, None),
                              EmbeddingConfig(30, [1], 1000, 16, False, None),
                              EmbeddingConfig(5, [1], 10000, 16, False, None),
                              EmbeddingConfig(5, [1], 100000, 32, False, None),
                              EmbeddingConfig(27, [1], 4000000, 32, False, None),
                          ],
                          mlp_sizes=[512, 256, 128],
                          num_numerical_features=10,
                          interact_stride=None,
                          combiner='sum',
                          cross_params=None)

model_medium = ModelConfig(name="Medium v3",
                           embedding_configs=[
                               EmbeddingConfig(20, [1, 50], 100000, 64, True, None),
                               EmbeddingConfig(5, [1, 50], 10000000, 64, True, None),
                               EmbeddingConfig(1, [1, 50], 100000000, 128, True, None),
                               EmbeddingConfig(1, [1], 100000000, 128, False, None),
                               EmbeddingConfig(80, [1], 10, 32, False, None),
                               EmbeddingConfig(60, [1], 1000, 32, False, None),
                               EmbeddingConfig(80, [1], 100000, 64, False, None),
                               EmbeddingConfig(24, [1], 200000, 64, False, None),
                               EmbeddingConfig(40, [1], 10000000, 64, False, None),
                           ],
                           mlp_sizes=[1024, 512, 256, 128],
                           num_numerical_features=25,
                           interact_stride=7,
                           combiner='sum',
                           cross_params=None)

model_large = ModelConfig(name="Large v3",
                          embedding_configs=[
                              EmbeddingConfig(40, [1, 100], 100000, 64, True, None),
                              EmbeddingConfig(16, [1, 100], 15000000, 64, True, None),
                              EmbeddingConfig(1, [1, 100], 200000000, 128, True, None),
                              EmbeddingConfig(1, [1], 200000000, 128, False, None),
                              EmbeddingConfig(100, [1], 10, 32, False, None),
                              EmbeddingConfig(100, [1], 10000, 32, False, None),
                              EmbeddingConfig(160, [1], 100000, 64, False, None),
                              EmbeddingConfig(50, [1], 500000, 64, False, None),
                              EmbeddingConfig(144, [1], 15000000, 64, False, None),
                          ],
                          mlp_sizes=[2048, 1024, 512, 256],
                          num_numerical_features=100,
                          interact_stride=8,
                          combiner='sum',
                          cross_params=None)

model_jumbo = ModelConfig(name="Jumbo v3",
                          embedding_configs=[
                              EmbeddingConfig(50, [1, 200], 100000, 128, True, None),
                              EmbeddingConfig(24, [1, 200], 20000000, 128, True, None),
                              EmbeddingConfig(1, [1, 200], 400000000, 256, True, None),
                              EmbeddingConfig(1, [1], 400000000, 256, False, None),
                              EmbeddingConfig(100, [1], 10, 32, False, None),
                              EmbeddingConfig(200, [1], 10000, 64, False, None),
                              EmbeddingConfig(350, [1], 100000, 128, False, None),
                              EmbeddingConfig(80, [1], 1000000, 128, False, None),
                              EmbeddingConfig(216, [1], 20000000, 128, False, None),
                          ],
                          mlp_sizes=[2048, 1024, 512, 256],
                          num_numerical_features=200,
                          interact_stride=20,
                          combiner='sum',
                          cross_params=None)

model_colossal = ModelConfig(name="Colossal v3",
                             embedding_configs=[
                                 EmbeddingConfig(100, [1, 300], 100000, 128, True, None),
                                 EmbeddingConfig(50, [1, 300], 40000000, 256, True, None),
                                 EmbeddingConfig(1, [1, 300], 2000000000, 256, True, None),
                                 EmbeddingConfig(1, [1], 1000000000, 256, False, None),
                                 EmbeddingConfig(100, [1], 10, 32, False, None),
                                 EmbeddingConfig(400, [1], 10000, 128, False, None),
                                 EmbeddingConfig(100, [1], 100000, 128, False, None),
                                 EmbeddingConfig(800, [1], 1000000, 128, False, None),
                                 EmbeddingConfig(450, [1], 40000000, 256, False, None),
                             ],
                             mlp_sizes=[4096, 2048, 1024, 512, 256],
                             num_numerical_features=500,
                             interact_stride=30,
                             combiner='sum',
                             cross_params=None)

model_criteo = ModelConfig(name="Criteo-dlrm-like",
                           embedding_configs=[
                               EmbeddingConfig(26, [1], 100000, 128, False, None),
                           ],
                           mlp_sizes=[512, 256, 128],
                           num_numerical_features=13,
                           interact_stride=None,
                           combiner='sum',
                           cross_params=None)

synthetic_models_v3 = {
    "criteo": model_criteo,
    "tiny": model_tiny,
    "small": model_small,
    "medium": model_medium,
    "large": model_large,
    "jumbo": model_jumbo,
    "colossal": model_colossal
}
