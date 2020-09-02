import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from codes.base import BaseModel
from codes import modules


class MNISTModel_digit(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.define_iterator()
        self.build_model()
        if self.config['prior'] in ['ours', 'GMM']:
            self.define_GM_prior()
        if self.config['prior'] in ['ours', 'hierarchical']:
            self.define_inner_VAE_prior()
        elif self.config['prior'] == 'vampPrior':
            self.define_vampPrior()
        self.define_loss()
        self.training_variables()
        self.compute_gradients()
        self.init_saver()

    def define_iterator(self):
        self.original_signal = tf.placeholder(tf.float32,
                                              [None,
                                               self.config['dim_input_x'],
                                               self.config['dim_input_y'],
                                               1])
        self.seed = tf.placeholder(tf.int64, shape=())
        self.dataset = tf.data.Dataset.from_tensor_slices(self.original_signal)
        self.dataset = self.dataset.shuffle(buffer_size=60000, seed=self.seed)
        self.dataset = self.dataset.repeat(8000)
        self.dataset = self.dataset.batch(
          self.config['batch_size'], drop_remainder=True)
        self.iterator = self.dataset.make_initializable_iterator()

        self.input_image = self.iterator.get_next()

        self.code_input = tf.placeholder(tf.float32,
                                         [None, self.config['code_size']])
        self.is_code_input = tf.placeholder(tf.bool)

    def encoder_mapping(self, init, input):
        kernel_size = self.config['kernel_size']
        padded_input = tf.pad(input,
                            [[0, 0], [2, 2], [2, 2], [0, 0]],
                            "SYMMETRIC")
        conv_1 = tf.layers.conv2d(inputs=padded_input,
                                filters=self.config['num_hidden_units'] // 16,
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same',
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=init)
        conv_2 = tf.layers.conv2d(inputs=conv_1,
                                filters=self.config['num_hidden_units'] // 4,
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same',
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=init)
        conv_3 = tf.layers.conv2d(inputs=conv_2,
                                filters=self.config['num_hidden_units'],
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same',
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=init)
        encoded_signal = tf.layers.flatten(conv_3)
        encoded_signal = tf.layers.dense(encoded_signal,
                                       units=self.config['num_hidden_units'] // 4,
                                       kernel_initializer=init,
                                       activation=tf.nn.leaky_relu)
        return encoded_signal

    def build_model(self):
        # network architecture
        # define the encoder
        init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('encoder'):
            encoded_signal = self.encoder_mapping(init, input=self.original_signal)
            self.code_mean = tf.layers.dense(encoded_signal,
                                             units=self.config['code_size'],
                                             activation=None,
                                             kernel_initializer=init,
                                             name='code_mean')
            self.code_std_dev = tf.layers.dense(encoded_signal,
                                                units=self.config['code_size'],
                                                activation=tf.nn.relu,
                                                kernel_initializer=init,
                                                name='code_std_dev')
            self.code_std_dev = self.code_std_dev + self.config['latent_variance_precision']

        mvn = tfp.distributions.MultivariateNormalDiag(
            loc=self.code_mean,
            scale_diag=self.code_std_dev)
        encoder_code_sample = mvn.sample()
        print("finish encoder:\n{}\n".format(encoder_code_sample))

        self.code_sample = encoder_code_sample
        print("code_sample:\n{}\n".format(self.code_sample))

        with tf.variable_scope('decoder'):
            encoded = tf.cond(self.is_code_input, lambda: self.code_input, lambda: self.code_sample)

            encoded = tf.layers.dense(encoded, units=4 * 4 * self.config['num_hidden_units'], activation=tf.nn.leaky_relu)
            encoded = tf.reshape(encoded, [-1, 1, 1, 4 * 4 * self.config['num_hidden_units']])
            print(encoded)
            # decode the code to generate original sequence
            decoded_1 = tf.nn.depth_to_space(input=encoded,
                                             block_size=4)
            decoded_2 = tf.layers.conv2d(inputs=decoded_1,
                                         filters=self.config['num_hidden_units'],
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=init)
            decoded_2 = tf.nn.depth_to_space(input=decoded_2,
                                             block_size=2)
            decoded_3 = tf.layers.conv2d(inputs=decoded_2,
                                         filters=self.config['num_hidden_units'] / 4,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=init)
            decoded_3 = tf.nn.depth_to_space(input=decoded_3,
                                             block_size=2)
            decoded_4 = tf.layers.conv2d(inputs=decoded_3,
                                         filters=self.config['num_hidden_units'] / 16,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=init)
            decoded_ = tf.nn.depth_to_space(input=decoded_4,
                                            block_size=2)
            self.decoded = tf.layers.conv2d(inputs=decoded_,
                                            filters=1,
                                            kernel_size=5,
                                            strides=1,
                                            padding='valid',
                                            activation=tf.nn.relu,
                                            kernel_initializer=init)
        print("finish decoder:\n{}\n".format(self.decoded))

        # define sigma as a trainable variable
        with tf.variable_scope('sigma'):
            self.sigma = tf.Variable(self.config['sigma'], dtype=tf.float32, trainable=True)
            self.sigma = tf.square(self.sigma)
            self.sigma = tf.sqrt(self.sigma)
            recons_error = tf.abs(self.decoded - self.original_signal)
            self.mean_pixel_error = tf.reduce_mean(recons_error)
            if self.config['TRAIN_sigma'] == 1:
                self.sigma = tf.maximum(self.sigma, self.mean_pixel_error)
        print("sigma:\n{}\n".format(self.sigma))


class MNISTModel_fashion(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.define_iterator()
        self.build_model()
        if self.config['prior'] in ['ours', 'GMM']:
            self.define_GM_prior()
        if self.config['prior'] in ['ours', 'hierarchical']:
            self.define_inner_VAE_prior()
        elif self.config['prior'] == 'vampPrior':
            self.define_vampPrior()
        self.define_loss()
        self.training_variables()
        self.compute_gradients()
        self.init_saver()

    def define_iterator(self):
        self.original_signal = tf.placeholder(tf.float32,
                                              [None,
                                               self.config['dim_input_x'],
                                               self.config['dim_input_y'],
                                               1])
        self.seed = tf.placeholder(tf.int64, shape=())
        self.dataset = tf.data.Dataset.from_tensor_slices(self.original_signal)
        self.dataset = self.dataset.shuffle(buffer_size=60000, seed=self.seed)
        self.dataset = self.dataset.repeat(8000)
        self.dataset = self.dataset.batch(
            self.config['batch_size'], drop_remainder=True)
        self.iterator = self.dataset.make_initializable_iterator()

        self.input_image = self.iterator.get_next()

        self.code_input = tf.placeholder(tf.float32,
                                         [None, self.config['code_size']])
        self.is_code_input = tf.placeholder(tf.bool)

    def encoder_mapping(self, init, input):
        padded_input = tf.pad(input,
                              [[0, 0], [2, 2], [2, 2], [0, 0]],
                              "SYMMETRIC")
        conv_1 = tf.layers.conv2d(inputs=padded_input,
                                  filters=self.config['num_hidden_units'] // 4,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=init)
        conv_2 = tf.layers.conv2d(inputs=conv_1,
                                  filters=self.config['num_hidden_units'] // 4,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=init)
        conv_3 = tf.layers.conv2d(inputs=conv_2,
                                  filters=self.config['num_hidden_units'] // 2,
                                  kernel_size=3,
                                  strides=2,
                                  padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=init)
        conv_4 = tf.layers.conv2d(inputs=conv_3,
                                  filters=self.config['num_hidden_units'] // 2,
                                  kernel_size=3,
                                  padding='valid',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=init)
        encoded_signal = tf.layers.flatten(conv_4)
        encoded_signal = tf.layers.dense(encoded_signal,
                                         units=self.config['num_hidden_units'],
                                         kernel_initializer=init,
                                         activation=tf.nn.leaky_relu)
        return encoded_signal

    def build_model(self):
        # network architecture
        # define the encoder
        init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('encoder'):
            encoded_signal = self.encoder_mapping(init, input=self.original_signal)
            self.code_mean = tf.layers.dense(encoded_signal,
                                             units=self.config['code_size'],
                                             activation=None,
                                             kernel_initializer=init,
                                             name='code_mean')
            self.code_std_dev = tf.layers.dense(encoded_signal,
                                                units=self.config['code_size'],
                                                activation=tf.nn.relu,
                                                kernel_initializer=init,
                                                name='code_std_dev')
            self.code_std_dev = self.code_std_dev + self.config['latent_variance_precision']

        mvn = tfp.distributions.MultivariateNormalDiag(
            loc=self.code_mean,
            scale_diag=self.code_std_dev)
        encoder_code_sample = mvn.sample()
        print("finish encoder:\n{}\n".format(encoder_code_sample))

        self.code_sample = encoder_code_sample
        print("code_sample:\n{}\n".format(self.code_sample))

        with tf.variable_scope('decoder'):
            encoded = tf.cond(self.is_code_input, lambda: self.code_input, lambda: self.code_sample)

            encoded = tf.layers.dense(encoded, units=self.config['num_hidden_units'], activation=tf.nn.leaky_relu)
            encoded = tf.reshape(encoded, [-1, 1, 1, self.config['num_hidden_units']])
            print(encoded)
            # decode the code to generate original sequence
            decoded_1 = tf.nn.depth_to_space(input=encoded,
                                             block_size=2)
            decoded_1 = tf.layers.conv2d(inputs=decoded_1,
                                         filters=self.config['num_hidden_units'],
                                         kernel_size=1,
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=init)
            decoded_2 = tf.nn.depth_to_space(input=decoded_1,
                                             block_size=2)
            decoded_2 = tf.layers.conv2d(inputs=decoded_2,
                                         filters=self.config['num_hidden_units'],
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=init)
            decoded_3 = tf.nn.depth_to_space(input=decoded_2,
                                             block_size=2)
            decoded_3 = tf.layers.conv2d(inputs=decoded_3,
                                         filters=self.config['num_hidden_units'],
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=init)
            decoded_4 = tf.nn.depth_to_space(input=decoded_3,
                                             block_size=2)
            decoded_4 = tf.layers.conv2d(inputs=decoded_4,
                                         filters=self.config['num_hidden_units'],
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=init)
            decoded_ = tf.nn.depth_to_space(input=decoded_4,
                                            block_size=2)
            self.decoded = tf.layers.conv2d(inputs=decoded_,
                                            filters=1,
                                            kernel_size=5,
                                            strides=1,
                                            padding='valid',
                                            activation=tf.nn.relu,
                                            kernel_initializer=init)
        print("finish decoder:\n{}\n".format(self.decoded))

        # define sigma as a trainable variable
        with tf.variable_scope('sigma'):
            self.sigma = tf.Variable(self.config['sigma'], dtype=tf.float32, trainable=True)
            self.sigma = tf.square(self.sigma)
            self.sigma = tf.sqrt(self.sigma)
            recons_error = tf.abs(self.decoded - self.original_signal)
            self.mean_pixel_error = tf.reduce_mean(recons_error)
            if self.config['TRAIN_sigma'] == 1:
                self.sigma = tf.maximum(self.sigma, self.mean_pixel_error)
        print("sigma:\n{}\n".format(self.sigma))


class CelebAModel_densenet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.define_iterator()
        self.build_model()
        if self.config['prior'] in ['ours', 'GMM']:
            self.define_GM_prior()
        if self.config['prior'] in ['ours', 'hierarchical']:
            self.define_inner_VAE_prior()
        elif self.config['prior'] == 'vampPrior':
            self.define_vampPrior()
        self.define_loss()
        self.training_variables()
        self.compute_gradients()
        self.init_saver()

    def define_iterator(self):
        self.data_file = tf.placeholder(tf.string)
        self.original_signal = tf.placeholder(tf.float32,
                                              [None,
                                               self.config['dim_input_x'],
                                               self.config['dim_input_y'],
                                               self.config['dim_input_channel']])

        def decode(serialized_example):
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'X': tf.FixedLenFeature([], tf.string)
                })

            image = tf.decode_raw(features['X'], tf.uint8)
            image.set_shape(
                [self.config['dim_input_x'] * self.config['dim_input_y'] * self.config['dim_input_channel']])
            image = tf.reshape(image, [self.config['dim_input_x'], self.config['dim_input_y'],
                                       self.config['dim_input_channel']])

            return image

        def normalize(image):
            image = tf.cast(image, tf.float32) * (1. / 255)
            return image

        self.dataset = tf.data.TFRecordDataset(self.data_file)
        self.dataset = self.dataset.map(decode)
        self.dataset0 = self.dataset.map(normalize)
        self.dataset1 = self.dataset0.batch(self.config['batch_size'])
        self.iterator_test = self.dataset1.make_initializable_iterator()

        self.test_image = self.iterator_test.get_next()

        self.dataset2 = self.dataset0.shuffle(1000 + 3 * self.config['batch_size'])
        self.dataset2 = self.dataset2.repeat(8000)
        self.dataset2 = self.dataset2.batch(self.config['batch_size'])
        self.iterator = self.dataset2.make_initializable_iterator()

        self.input_image = self.iterator.get_next()
        self.is_code_input = tf.placeholder(tf.bool)
        self.code_input = tf.placeholder(tf.float32,
                                         [None,
                                          self.config['code_size']])

    def encoder_mapping(self, init, input):
        kernel_size = self.config['kernel_size']
        padded_input = tf.pad(input,
                              [[0, 0], [0, 0], [0, 0], [0, 0]],
                              "SYMMETRIC")
        # conv_1: 64*64*h/4
        conv_1 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=padded_input,
                                                                                 filters=self.config[
                                                                                             'num_hidden_units'] / 4,
                                                                                 kernel_size=kernel_size,
                                                                                 strides=2,
                                                                                 padding='same',
                                                                                 activation=None,
                                                                                 kernel_initializer=init),
                                                                training=self.is_training))
        print("conv_1: {}".format(conv_1))
        # conv_2: 32*32*h/14
        conv_2 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_1,
                                                                                 filters=self.config[
                                                                                             'num_hidden_units'] / 4,
                                                                                 kernel_size=kernel_size,
                                                                                 strides=2,
                                                                                 padding='same',
                                                                                 activation=None,
                                                                                 kernel_initializer=init),
                                                                training=self.is_training))
        print("conv_2: {}".format(conv_2))
        # conv_3: 16*16*h/2
        conv_3 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_2,
                                                                                 filters=self.config[
                                                                                             'num_hidden_units'] / 2,
                                                                                 kernel_size=kernel_size,
                                                                                 strides=2,
                                                                                 padding='same',
                                                                                 activation=None,
                                                                                 kernel_initializer=init),
                                                                training=self.is_training))
        print("conv_3: {}".format(conv_3))
        # conv_4: 8*8*h/2
        conv_4 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_3,
                                                                                 filters=self.config[
                                                                                             'num_hidden_units'] / 2,
                                                                                 kernel_size=kernel_size,
                                                                                 strides=2,
                                                                                 padding='same',
                                                                                 activation=None,
                                                                                 kernel_initializer=init),
                                                                training=self.is_training))
        print("conv_4: {}".format(conv_4))
        # conv_5: 4*4*h
        conv_5 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_4,
                                                                                 filters=self.config[
                                                                                     'num_hidden_units'],
                                                                                 kernel_size=kernel_size,
                                                                                 strides=2,
                                                                                 padding='same',
                                                                                 activation=None,
                                                                                 kernel_initializer=init),
                                                                training=self.is_training))
        print("conv_5: {}".format(conv_5))
        # conv_5: 1*1*h
        conv_6 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs=conv_5,
                                                                                 filters=self.config[
                                                                                     'num_hidden_units'],
                                                                                 kernel_size=kernel_size,
                                                                                 padding='valid',
                                                                                 activation=None,
                                                                                 kernel_initializer=init),
                                                                training=self.is_training))
        x = conv_6
        print("conv_6: {}".format(x))
        encoded_signal = tf.layers.flatten(x)
        return encoded_signal

    def build_model(self):
        init = tf.contrib.layers.xavier_initializer()
        # network architecture
        # define the encoder
        # if self.config['TRAIN_VAE'] == 1:
        self.is_training = tf.constant(True, dtype=tf.bool)
        # else:
        #   self.is_training = tf.constant(False, dtype=tf.bool)
        kernel_size = self.config['kernel_size']
        with tf.variable_scope('encoder'):
            # after padding 128*128*3
            encoded_signal = self.encoder_mapping(init, input=self.original_signal)
            self.code_mean = tf.layers.dense(encoded_signal,
                                             units=self.config['code_size'],
                                             kernel_initializer=init,
                                             activation=None,
                                             name='code_mean')
            self.code_std_dev = tf.layers.dense(encoded_signal,
                                                units=self.config['code_size'],
                                                kernel_initializer=init,
                                                activation=tf.nn.relu,
                                                name='code_std_dev')
            self.code_std_dev = self.code_std_dev + self.config['latent_variance_precision']

        mvn = tfp.distributions.MultivariateNormalDiag(
            loc=self.code_mean,
            scale_diag=self.code_std_dev)
        encoder_code_sample = mvn.sample()
        print("finish encoder:\n{}\n".format(encoder_code_sample))

        self.code_sample = encoder_code_sample
        print("code_sample:\n{}\n".format(self.code_sample))

        with tf.variable_scope('decoder'):
            encoded = tf.cond(self.is_code_input, lambda: self.code_input, lambda: self.code_sample)
            encoded = tf.layers.dense(encoded,
                                      units=self.config['num_hidden_units'],
                                      activation=tf.nn.leaky_relu)
            print(encoded)
            dlatent_ = encoded
            for _ in range(8):
                dlatent_ = tf.layers.dense(dlatent_,
                                           units=self.config['num_hidden_units'],
                                           activation=tf.nn.leaky_relu)
                dlatent = dlatent_

            # decode the code to generate original sequence
            # decoded_1: 2*2*h
            decoded_1 = tf.layers.conv2d(tf.reshape(encoded, [-1, 1, 1, int(self.config['num_hidden_units'])]),
                                         filters=self.config['num_hidden_units'],
                                         kernel_size=1,
                                         padding='SAME',
                                         activation=None)
            decoded_1 = tf.image.resize_images(decoded_1, [2, 2])
            print("decoded_1: {}".format(decoded_1))
            # decoded_2: 2*2*h
            decoded_2 = tf.contrib.layers.instance_norm(tf.layers.conv2d(decoded_1,
                                                                         filters=self.config['num_hidden_units'],
                                                                         kernel_size=3,
                                                                         padding='SAME',
                                                                         activation=None), scale=False, center=False,
                                                        trainable=False)
            decoded_2 = tf.nn.leaky_relu(modules.style_mod(decoded_2, dlatent, 0))
            print("decoded_2: {}".format(decoded_2))
            # decoded_3: 16*16*h
            decoded_3 = tf.contrib.layers.instance_norm(tf.layers.conv2d(decoded_2,
                                                                         filters=int(self.config['num_hidden_units']),
                                                                         kernel_size=3,
                                                                         padding='SAME',
                                                                         activation=None), scale=False, center=False,
                                                        trainable=False)
            decoded_3 = tf.nn.leaky_relu(modules.style_mod(decoded_3, dlatent, 1))
            decoded_3 = tf.image.resize_images(decoded_3, [8, 8])
            decoded_3 = tf.layers.conv2d(decoded_3,
                                         filters=int(self.config['num_hidden_units']),
                                         kernel_size=3,
                                         padding='SAME',
                                         activation=tf.nn.leaky_relu)
            decoded_3 = tf.image.resize_images(decoded_3, [16, 16])
            print("decoded_3: {}".format(decoded_3))
            # decoded_4: 64*64*h/2
            decoded_4 = tf.contrib.layers.instance_norm(tf.layers.conv2d(decoded_3,
                                                                         filters=int(
                                                                             self.config['num_hidden_units'] / 2),
                                                                         kernel_size=3,
                                                                         padding='SAME',
                                                                         activation=None), scale=False, center=False,
                                                        trainable=False)
            decoded_4 = tf.nn.leaky_relu(modules.style_mod(decoded_4, dlatent, 2))
            decoded_4 = tf.image.resize_images(decoded_4, [32, 32])
            decoded_4 = tf.layers.conv2d(decoded_4,
                                         filters=int(self.config['num_hidden_units'] / 2),
                                         kernel_size=3,
                                         padding='SAME',
                                         activation=tf.nn.leaky_relu)
            decoded_4 = tf.image.resize_images(decoded_4, [64, 64])
            print("decoded_4: {}".format(decoded_4))
            # decoded_5: 128*128*h/4
            decoded_5 = tf.contrib.layers.instance_norm(tf.layers.conv2d(decoded_4,
                                                                         filters=int(
                                                                             self.config['num_hidden_units'] / 4),
                                                                         kernel_size=3,
                                                                         padding='SAME',
                                                                         activation=None), scale=False, center=False,
                                                        trainable=False)
            decoded_5 = tf.nn.leaky_relu(modules.style_mod(decoded_5, dlatent, 3))
            decoded_5 = tf.image.resize_images(decoded_5, [128, 128])
            decoded_5 = tf.layers.conv2d(decoded_5,
                                         filters=int(self.config['num_hidden_units'] / 4),
                                         kernel_size=3,
                                         padding='SAME',
                                         activation=tf.nn.leaky_relu)
            decoded_5 = tf.image.resize_images(decoded_5, [128, 128])
            print("decoded_5: {}".format(decoded_5))
            # decoded_6: 128*128*3
            decoded_6 = tf.layers.conv2d(decoded_5,
                                         filters=3,
                                         kernel_size=1,
                                         padding='SAME',
                                         activation=None)
            print("decoded_6: {}".format(decoded_6))
            self.decoded = tf.cond(self.is_training, lambda: decoded_6, lambda: tf.clip_by_value(decoded_6, 0, 1))
        print("finish decoder:\n{}\n".format(self.decoded))

        # define sigma as a trainable variable
        with tf.variable_scope('sigma'):
            self.sigma = tf.Variable(self.config['sigma'], dtype=tf.float32, trainable=True)
            self.sigma = tf.square(self.sigma)
            self.sigma = tf.sqrt(self.sigma)
            recons_error = tf.abs(self.decoded - self.original_signal)
            self.mean_pixel_error = tf.reduce_mean(recons_error)
            self.sigma = tf.maximum(self.sigma, self.mean_pixel_error)
        print("sigma:\n{}\n".format(self.sigma))
