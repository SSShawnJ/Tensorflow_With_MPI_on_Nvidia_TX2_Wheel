# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.maximum(input_, leak * input_, name=name)


def reverse_gradient(x):
  return -x + tf.stop_gradient(2 * x)


@registry.register_model
class BasicFcRelu(t2t_model.T2TModel):
  """Basic fully-connected + ReLU model."""

  def body(self, features):
    hparams = self.hparams
    x = features["inputs"]
    shape = common_layers.shape_list(x)
    x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
    for i in range(hparams.num_hidden_layers):
      l = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)

      with tf.variable_scope("layer_%d" % i, reuse=True):
        weights = tf.get_variable('kernel')
        bias = tf.get_variable('bias')

      # 2 devices
      # weights1, weights2 = tf.split(weights, num_or_size_splits=2, axis=1)
      # bias1, bias2 = tf.split(bias, num_or_size_splits=2, axis=0)

      # with tf.device('job:worker/task:0'):
      #   x1 = tf.nn.bias_add(tf.matmul(x, weights1),bias1);
      #   x1 = tf.nn.dropout(x1, keep_prob=1.0 - hparams.dropout)
      #   x1 = tf.nn.relu(x1)
      # with tf.device('job:worker/task:1'):
      #   x2 = tf.nn.bias_add(tf.matmul(x, weights2),bias2);
      #   x2 = tf.nn.dropout(x2, keep_prob=1.0 - hparams.dropout)
      #   x2 = tf.nn.relu(x2)

      # x = tf.concat([x1,x2], axis = 1);


      # 4 devices
      weights1, weights2, weights3, weights4 = tf.split(weights, num_or_size_splits=4, axis=1)
      bias1, bias2, bias3, bias4 = tf.split(bias, num_or_size_splits=4, axis=0)

      with tf.device('job:worker/task:0'):
        x1 = tf.nn.bias_add(tf.matmul(x, weights1),bias1);
        x1 = tf.nn.dropout(x1, keep_prob=1.0 - hparams.dropout)
        x1 = tf.nn.relu(x1)
      with tf.device('job:worker/task:1'):
        x2 = tf.nn.bias_add(tf.matmul(x, weights2),bias2);
        x2 = tf.nn.dropout(x2, keep_prob=1.0 - hparams.dropout)
        x2 = tf.nn.relu(x2)
      with tf.device('job:worker/task:2'):
        x3 = tf.nn.bias_add(tf.matmul(x, weights3),bias3);
        x3 = tf.nn.dropout(x3, keep_prob=1.0 - hparams.dropout)
        x3 = tf.nn.relu(x3)
      with tf.device('job:worker/task:3'):
        x4 = tf.nn.bias_add(tf.matmul(x, weights4),bias4);
        x4 = tf.nn.dropout(x4, keep_prob=1.0 - hparams.dropout)
        x4 = tf.nn.relu(x4)

      x = tf.concat([x1,x2,x3,x4], axis = 1);

    return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # 4D For T2T.


@registry.register_model
class BasicAutoencoder(t2t_model.T2TModel):
  """A basic autoencoder, try with image_mnist_rev or image_cifar10_rev."""

  def __init__(self, *args, **kwargs):
    super(BasicAutoencoder, self).__init__(*args, **kwargs)
    self._cur_bottleneck_tensor = None
    self.is1d = None

  def bottleneck(self, x):
    with tf.variable_scope("bottleneck"):
      hparams = self.hparams
      x = tf.layers.dense(x, hparams.bottleneck_bits, name="bottleneck")
      if hparams.mode == tf.estimator.ModeKeys.TRAIN:
        noise = 2.0 * tf.random_uniform(common_layers.shape_list(x)) - 1.0
        return tf.tanh(x) + noise * hparams.bottleneck_noise, 0.0
      return tf.tanh(x), 0.0

  def discriminator(self, x, is_training):
    """Discriminator architecture based on InfoGAN.

    Args:
      x: input images, shape [bs, h, w, channels]
      is_training: boolean, are we in train or eval model.

    Returns:
      out_logit: the output logits (before sigmoid).
    """
    hparams = self.hparams
    with tf.variable_scope(
        "gate",
        initializer=tf.random_normal_initializer(stddev=0.02)):
      batch_size, height, width = common_layers.shape_list(x)[:3]
      # Mapping x from [bs, h, w, c] to [bs, 1]
      net = tf.layers.conv2d(x, 64, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv1")
      # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv2")
      # [bs, h/4, w/4, 128]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn2")
      net = lrelu(net)
      size = height * width
      net = tf.reshape(net, [batch_size, size * 8])  # [bs, h * w * 8]
      net = tf.layers.dense(net, 1024, name="d_fc3")  # [bs, 1024]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn3")
      net = lrelu(net)
      return net

  def unbottleneck(self, x, res_size, reuse=None):
    with tf.variable_scope("unbottleneck", reuse=reuse):
      x = tf.layers.dense(x, res_size, name="dense")
      return x

  def make_even_size(self, x):
    if not self.is1d:
      return common_layers.make_even_size(x)
    shape1 = x.get_shape().as_list()[1]
    if shape1 is not None and shape1 % 2 == 0:
      return x
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2, axis=1)
    return x

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      # Down-convolutions.
      for i in range(hparams.num_hidden_layers):
        x = self.make_even_size(x)
        x = tf.layers.conv2d(
            x, hparams.hidden_size * 2**(i + 1), kernel, strides=strides,
            padding="SAME", activation=common_layers.belu, name="conv_%d" % i)
        x = common_layers.layer_norm(x)
      return x

  def decoder(self, x):
    with tf.variable_scope("decoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      # Up-convolutions.
      for i in range(hparams.num_hidden_layers):
        j = hparams.num_hidden_layers - i - 1
        x = tf.layers.conv2d_transpose(
            x, hparams.hidden_size * 2**j, kernel, strides=strides,
            padding="SAME", activation=common_layers.belu, name="deconv_%d" % j)
        x = common_layers.layer_norm(x)
      return x

  def body(self, features):
    hparams = self.hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      x = features["targets"]
      shape = common_layers.shape_list(x)
      is1d = shape[2] == 1
      self.is1d = is1d
      # Run encoder.
      x = self.encoder(x)
      # Bottleneck (mix during early training, not too important but stable).
      b, b_loss = self.bottleneck(x)
      self._cur_bottleneck_tensor = b
      b = self.unbottleneck(b, common_layers.shape_list(x)[-1])
      b = common_layers.mix(b, x, hparams.bottleneck_warmup_steps, is_training)
      if hparams.add_gan_loss:
        # Add a purely sampled batch on which we'll compute the GAN loss.
        g = self.unbottleneck(self.sample(), common_layers.shape_list(x)[-1],
                              reuse=True)
        b = tf.concat([g, b], axis=0)
      # With probability bottleneck_max_prob use the bottleneck, otherwise x.
      if hparams.bottleneck_max_prob < -1.0:
        x = tf.where(tf.less(tf.random_uniform([]),
                             hparams.bottleneck_max_prob), b, x)
      else:
        x = b
    else:
      if self._cur_bottleneck_tensor is None:
        b = self.sample()
      else:
        b = self._cur_bottleneck_tensor
      res_size = self.hparams.hidden_size * 2**self.hparams.num_hidden_layers
      res_size = min(res_size, hparams.max_hidden_size)
      x = self.unbottleneck(b, res_size)
    # Run decoder.
    x = self.decoder(x)
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      return x, {"bottleneck_loss": 0.0}
    # Cut to the right size and mix before returning.
    res = x[:, :shape[1], :shape[2], :]
    # Add GAN loss if requested.
    gan_loss = 0.0
    if hparams.add_gan_loss:
      # Split back if we added a purely sampled batch.
      res_gan, res = tf.split(res, 2, axis=0)
      num_channels = self.hparams.problem.num_channels
      res_rgb = common_layers.convert_real_to_rgb(tf.nn.sigmoid(
          tf.layers.dense(res_gan, num_channels, name="gan_rgb")))
      tf.summary.image("gan", tf.cast(res_rgb, tf.uint8), max_outputs=1)
      orig_rgb = tf.to_float(features["targets_raw"])
      def discriminate(x):
        return self.discriminator(x, is_training=is_training)
      gan_loss = common_layers.sliced_gan_loss(
          orig_rgb, reverse_gradient(res_rgb),
          discriminate, self.hparams.num_sliced_vecs)
      gan_loss *= common_layers.inverse_lin_decay(
          hparams.bottleneck_warmup_steps)
    # Mix the final result and return.
    res = common_layers.mix(res, features["targets"],
                            hparams.bottleneck_warmup_steps // 2, is_training)
    return res, {"bottleneck_loss": b_loss, "gan_loss": - gan_loss}

  def sample(self, features=None, shape=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
            hp.bottleneck_bits]
    # Sample in [-1, 1] as the bottleneck is under tanh.
    return 2.0 * tf.random_uniform(size) - 1.0

  def encode(self, x):
    """Auto-encode x and return the bottleneck."""
    features = {"targets": x}
    self(features)  # pylint: disable=not-callable
    res = tf.maximum(0.0, self._cur_bottleneck_tensor)  # Be 0/1 and not -1/1.
    self._cur_bottleneck_tensor = None
    return res

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    """Produce predictions from the model by sampling."""
    del args, kwargs
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Sample and decode.
    # TODO(lukaszkaiser): is this a universal enough way to get channels?
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    if "targets" not in features:
      features["targets"] = tf.zeros(
          [self.hparams.batch_size, 1, 1, num_channels],
          dtype=tf.int32)
    logits, _ = self(features)  # pylint: disable=not-callable
    samples = tf.argmax(logits, axis=-1)

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return samples.
    return samples

  def decode(self, bottleneck):
    """Auto-decode from the bottleneck and return the result."""
    # Get the shape from bottleneck and num channels.
    shape = common_layers.shape_list(bottleneck)
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    dummy_targets = tf.zeros(shape[:-1] + [num_channels])
    # Set the bottleneck to decode.
    if len(shape) > 4:
      bottleneck = tf.squeeze(bottleneck, axis=[1])
    bottleneck = 2 * bottleneck - 1  # Be -1/1 instead of 0/1.
    self._cur_bottleneck_tensor = bottleneck
    # Run decoding.
    res = self.infer({"targets": dummy_targets})
    self._cur_bottleneck_tensor = None
    return res

  def _get_kernel_and_strides(self):
    hparams = self.hparams
    kernel = (hparams.kernel_height, hparams.kernel_width)
    kernel = (hparams.kernel_height, 1) if self.is1d else kernel
    strides = (2, 1) if self.is1d else (2, 2)
    return (kernel, strides)


@registry.register_hparams
def basic_fc_small():
  """Small fully connected model."""
  hparams = common_hparams.basic_params1()
  hparams.learning_rate = 0.1
  hparams.batch_size = 128
  hparams.hidden_size = 256
  hparams.num_hidden_layers = 2
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.dropout = 0.0
  return hparams
  
@registry.register_hparams
def basic_fc_2():
  """Small fully connected model."""
  hparams = basic_fc_small()
  hparams.hidden_size = 2048
  hparams.num_hidden_layers = 2
  hparams.dropout = 0.1
  
  hparams.learning_rate_decay_scheme = "cosine"
  hparams.learning_rate_cosine_cycle_steps = 20000
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 100  # That's basically unused.
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-4
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  
  return hparams
  
@registry.register_hparams
def basic_fc_4():
  """Small fully connected model."""
  hparams = basic_fc_small()
  hparams.hidden_size = 2048
  hparams.num_hidden_layers = 4
  hparams.dropout = 0.1
  
  hparams.learning_rate_decay_scheme = "cosine"
  hparams.learning_rate_cosine_cycle_steps = 20000
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 100  # That's basically unused.
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-4
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  return hparams

@registry.register_hparams
def basic_fc_8():
  """Small fully connected model."""
  hparams = basic_fc_small()
  hparams.hidden_size = 2048
  hparams.num_hidden_layers = 8
  hparams.dropout = 0.1
  
  hparams.learning_rate_decay_scheme = "cosine"
  hparams.learning_rate_cosine_cycle_steps = 20000
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 100  # That's basically unused.
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-4
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  
  return hparams
  
@registry.register_hparams
def basic_fc_16():
  """Small fully connected model."""
  hparams = basic_fc_small()
  hparams.hidden_size = 2048
  hparams.num_hidden_layers = 16
  hparams.dropout = 0.1
  
  hparams.learning_rate_decay_scheme = "cosine"
  hparams.learning_rate_cosine_cycle_steps = 20000
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 100  # That's basically unused.
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-4
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  
  return hparams


@registry.register_hparams
def basic_autoencoder():
  """Basic autoencoder model."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "Adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 64
  hparams.num_hidden_layers = 5
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.dropout = 0.1
  hparams.add_hparam("max_hidden_size", 1024)
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("bottleneck_noise", 0.1)
  hparams.add_hparam("bottleneck_warmup_steps", 3000)
  hparams.add_hparam("bottleneck_max_prob", 1.0)
  hparams.add_hparam("sample_height", 32)
  hparams.add_hparam("sample_width", 32)
  hparams.add_hparam("discriminator_batchnorm", True)
  hparams.add_hparam("num_sliced_vecs", 4096)
  hparams.add_hparam("add_gan_loss", False)
  return hparams
