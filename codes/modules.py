# From Tensorflow fully convolutional densenet:
# https://github.com/HasnainRaz/FC-DenseNet-TensorFlow
import tensorflow as tf


def style_mod(x, dlatent, num):
    with tf.variable_scope('StyleMod_{}'.format(num)):
        style = tf.layers.dense(dlatent, units=x.shape[3]*2, activation=None)
        style = tf.reshape(style, [-1, 2, 1, 1, x.shape[3]])
        return x * (style[:, 0] + 1) + style[:, 1]


def batch_norm(x, training, name):
    """
    Wrapper for batch normalization in tensorflow, updates moving batch statistics
    if training, uses trained parameters if inferring.
    Args:
        x: Tensor, the input to normalize.
        training: Boolean tensor, indicates if training or not.
        name: String, name of the op in the graph.

    Returns:
        x: Batch normalized input.
    """
    with tf.variable_scope(name):
        x = tf.cond(training, lambda: tf.contrib.layers.batch_norm(x, is_training=True, scope=name+'_batch_norm'),
                    lambda: tf.contrib.layers.batch_norm(x, is_training=False, scope=name+'_batch_norm', reuse=True))
    return x


def conv_layer( x, training, filters, name):
    """
    Forms the atomic layer of the tiramisu, does three operation in sequence:
    batch normalization -> Relu -> 2D Convolution.
    Args:
        x: Tensor, input feature map.
        training: Bool Tensor, indicating whether training or not.
        filters: Integer, indicating the number of filters in the output feat. map.
        name: String, naming the op in the graph.

    Returns:
        x: Tensor, Result of applying batch norm -> Relu -> Convolution.
    """
    with tf.name_scope(name):
        x = batch_norm(x, training, name=name+'_bn')
        x = tf.nn.relu(x, name=name+'_relu')
        x = tf.layers.conv2d(x,
                                filters=filters,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='SAME',
                                dilation_rate=[1, 1],
                                activation=None,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                name=name+'_conv3x3')
        x = tf.layers.dropout(x, rate=0.2, training=training, name=name+'_dropout')

    return x


def dense_block(x, training, block_nb, layers_per_block, growth_k, name):
    """
    Forms the dense block of the Tiramisu to calculate features at a specified growth rate.
    Each conv layer in the dense block calculate growth_k feature maps, which are sequentially
    concatenated to build a larger final output.
    Args:
        x: Tensor, input to the Dense Block.
        training: Bool Tensor, indicating whether training or testing.
        block_nb: Int, identifying the block in the graph.
        name: String, identifying the layers in the graph.

    Returns:
        x: Tesnor, the output of the dense block.
    """
    dense_out = []
    with tf.name_scope(name):
        for i in range(layers_per_block[block_nb]):
            conv = conv_layer(x, training, growth_k, name=name+'_layer_'+str(i))
            x = tf.concat([conv, x], axis=3)
            dense_out.append(conv)

        x = tf.concat(dense_out, axis=3)

    return x


def transition_down(x, training, filters, name):
    """
    Down-samples the input feature map by half using maxpooling.
    Args:
        x: Tensor, input to downsample.
        training: Bool tensor, indicating whether training or inferring.
        filters: Integer, indicating the number of output filters.
        name: String, identifying the ops in the graph.

    Returns:
        x: Tensor, result of downsampling.
    """
    with tf.name_scope(name):
        x = batch_norm(x, training, name=name+'_bn')
        x = tf.nn.relu(x, name=name+'relu')
        x = tf.layers.conv2d(x,
                                filters=filters,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                padding='SAME',
                                dilation_rate=[1, 1],
                                activation=None,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                name=name+'_conv1x1')
        x = tf.layers.dropout(x, rate=0.2, training=training, name=name+'_dropout')
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name+'_maxpool2x2')

    return x


def transition_up(x, filters, name):
    """
    Up-samples the input feature maps using transpose convolutions.
    Args:
        x: Tensor, input feature map to upsample.
        filters: Integer, number of filters in the output.
        name: String, identifying the op in the graph.

    Returns:
        x: Tensor, result of up-sampling.
    """
    with tf.name_scope(name):
        x = tf.layers.conv2d_transpose(x,
                                        filters=filters,
                                        kernel_size=[3, 3],
                                        strides=[2, 2],
                                        padding='SAME',
                                        activation=None,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        name=name+'_trans_conv3x3')

    return x
