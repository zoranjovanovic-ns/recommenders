# Copyright 2023 The TensorFlow Recommenders Authors.
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

"""Keras interface for GPU Embeddings in TF2."""

from typing import Iterable, Optional, Union, Any, Dict

import tensorflow.compat.v2 as tf

from tensorflow_recommenders.layers.embedding import tpu_embedding_layer

EMBEDDING_GPU = tf.tpu.experimental.HardwareFeature.EmbeddingFeature.GPU

class GPUEmbedding(tpu_embedding_layer.TPUEmbedding):
  """A Keras layer for accelerating embedding lookups for large tables with GPU.
  Based on tpu_embedding_layer, but using MirroredStrategy
  Note that instead of sharding the table across devices, the table will be
  replicated across them.
  """

  def __init__(
      self,
      feature_config: Union[tf.tpu.experimental.embedding.FeatureConfig,
                            Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[Union[tf.tpu.experimental.embedding.SGD,
                                tf.tpu.experimental.embedding.Adagrad,
                                tf.tpu.experimental.embedding.Adam,
                                tf.tpu.experimental.embedding.FTRL]],
      pipeline_execution_with_tensor_core: bool = False,
      batch_size: Optional[int] = None,
      embedding_feature: Optional[
          tf.tpu.experimental.HardwareFeature.EmbeddingFeature] = None):
    """A Keras layer for accelerated embedding lookups on TPU.

    Args:
      feature_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`, a Keras optimizer or a string name
        of an optimizer (see `tf.keras.optimizers.get`). Or, if not created
        under a TPU strategy, None, which will avoid creation of the optimizer
        slot variable do reduce memory consumption during export.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old with potential correctness drawbacks). Set to True
        for improved performance.
      batch_size: Batch size of the input feature. Deprecated, support backward
        compatibility.
      embedding_feature: EmbeddingFeature enum, inidicating which version of TPU
        hardware the layer should run on.
    """
    super(tpu_embedding_layer.TPUEmbedding, self).__init__()
    self._feature_config, self._table_config_map = (
        tpu_embedding_layer._clone_and_prepare_features(feature_config))
    self._optimizer = tpu_embedding_layer._normalize_and_prepare_optimizer(optimizer)

    self._strategy = tf.distribute.get_strategy()
    self._using_tpu = False

    self._embedding_feature = embedding_feature

    # Create GPU embedding APIs according to the embedding feature
    # setting.
    self._tpu_embedding = self._create_gpu_embedding_mid_level_api(
        self._embedding_feature,
        pipeline_execution_with_tensor_core)

    self.batch_size = batch_size

    self._tpu_call_id = 0

  def _create_gpu_embedding_mid_level_api(
      self, embedding_feature: Optional[
          tf.tpu.experimental.HardwareFeature.EmbeddingFeature],
      pipeline_execution_with_tensor_core: bool
  ) -> Union[tf.tpu.experimental.embedding.GPUEmbeddingV0,
             tf.tpu.experimental.embedding.TPUEmbeddingForServing]:
    """Creates GPU Embedding API instance based on settings.

    Args:
      embedding_feature: EmbeddingFeature enum, indicating which version of TPU
        TPU hardware the layer is running on.
      pipeline_execution_with_tensor_core: Whether the GPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old with potential correctness drawbacks). Only used
        when the embedding feature is set to be v1.

    Returns:
      Instance of the TPU/GPUEmbedding API.

    Raises:
      ValueError: If the embedding_feature if not one of the EmbeddingFeature
        Enum.
    """

    if embedding_feature is None:
        return tf.tpu.experimental.embedding.TPUEmbeddingForServing(
            self._feature_config, self._optimizer)
    elif embedding_feature is EMBEDDING_GPU:
        return tf.tpu.experimental.embedding.GPUEmbeddingV0(
            self._feature_config, self._optimizer)
    else:
      raise ValueError("Unknown embedding feature {}".format(embedding_feature))
