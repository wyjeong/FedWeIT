
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

class DecomposedDense(tf.keras.layers.Dense):
  """ Custom dense layer that decomposes parameters into shared and specific parameters.
  
  Base code is referenced from official tensorflow code (https://github.com/tensorflow/tensorflow/)
  
  Created by:
      Wonyong Jeong (wyjeong@kaist.ac.kr)
  """
  def __init__(self,
               units,
               activation=None,
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               lambda_l1=None,
               lambda_mask=None,
               shared=None,
               adaptive=None,
               from_kb=None,
               atten=None,
               mask=None,
               bias=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(DecomposedDense, self).__init__(
               units=units,
               activation=activation,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               activity_regularizer=activity_regularizer,
               kernel_constraint=kernel_constraint,
               bias_constraint=bias_constraint,
               **kwargs)

    self.sw   = shared
    self.aw   = adaptive
    self.mask = mask
    self.bias = bias
    self.aw_kb = from_kb
    self.atten = atten
    self.lambda_l1   = lambda_l1
    self.lambda_mask = lambda_mask

  def l1_pruning(self, weights, hyp):
    hard_threshold = tf.cast(tf.greater(tf.abs(weights), hyp), tf.float32)
    return tf.multiply(weights, hard_threshold)
  
  def call(self, inputs):
    #####################################################################
    aw = self.aw if tf.keras.backend.learning_phase() else self.l1_pruning(self.aw, self.lambda_l1)
    mask = self.mask if tf.keras.backend.learning_phase() else self.l1_pruning(self.mask, self.lambda_mask)
    atten = self.atten
    aw_kbs = self.aw_kb
    ############################### Decomposed Kernel #################################
    self.my_theta = self.sw * mask + aw + tf.keras.backend.sum(aw_kbs * atten, axis=-1) 
    ###################################################################################
    rank = len(inputs.shape)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.my_theta, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = math_ops.cast(inputs, self._compute_dtype)
      if K.is_sparse(inputs):
        outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.my_theta)
      else:
        outputs = gen_math_ops.mat_mul(inputs, self.my_theta)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs


class DecomposedConv(tf.keras.layers.Conv2D):
  """ Custom conv layer that decomposes parameters into shared and specific parameters.
  
  Base code is referenced from official tensorflow code (https://github.com/tensorflow/tensorflow/)

  Created by:
      Wonyong Jeong (wyjeong@kaist.ac.kr)
  """
  def __init__(self, 
               filters,
               kernel_size,
               rank=2,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               lambda_l1=None,
               lambda_mask=None,
               shared=None,
               adaptive=None,
               from_kb=None,
               atten=None,
               mask=None,
               bias=None,
               **kwargs):
    
    super(DecomposedConv, self).__init__(
               filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               data_format=data_format,
               dilation_rate=dilation_rate,
               activation=activation,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               activity_regularizer=activity_regularizer,
               kernel_constraint=kernel_constraint,
               bias_constraint=bias_constraint,
               trainable=trainable,
               name=name, **kwargs)
    
    self.sw   = shared
    self.aw   = adaptive
    self.mask = mask
    self.bias = bias
    self.aw_kb = from_kb
    self.atten = atten
    self.lambda_l1   = lambda_l1
    self.lambda_mask = lambda_mask

  def l1_pruning(self, weights, hyp):
    hard_threshold = tf.cast(tf.greater(tf.abs(weights), hyp), tf.float32)
    return tf.multiply(weights, hard_threshold)
  
  def call(self, inputs):
    ###################################################################################
    aw = self.aw if tf.keras.backend.learning_phase() else self.l1_pruning(self.aw, self.lambda_l1)
    mask = self.mask if tf.keras.backend.learning_phase() else self.l1_pruning(self.mask, self.lambda_mask)
    atten = self.atten
    aw_kbs = self.aw_kb
    ############################### Decomposed Kernel #################################
    self.my_theta = self.sw * mask + aw + tf.keras.backend.sum(aw_kbs * atten, axis=-1)
    ###################################################################################

    # if self._recreate_conv_op(inputs):
    self._convolution_op = nn_ops.Convolution(
        inputs.get_shape(),
        filter_shape=self.my_theta.shape,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self._padding_op,
        data_format=self._conv_op_data_format)

    # Apply causal padding to inputs for Conv1D.
    if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
   
    outputs = self._convolution_op(inputs, self.my_theta)

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        else:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

