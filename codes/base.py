import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import random
import time
import os.path
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, savefig, cla, figure
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.stats import norm, multivariate_normal

from codes.utils import count_trainable_variables


class BaseDataGenerator:
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

    # separate training and val sets
    def separate_train_and_val_set(self):
        self.n_train = int(np.floor((self.config.n_samples * 0.9)))
        self.n_val = self.config.n_samples - self.n_train
        idx_train = random.sample(range(self.config.n_samples), self.n_train)
        idx_val = list(set(idx_train) ^ set(range(self.config.n_samples)))
        return idx_train, idx_val


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.two_pi = tf.constant(2 * np.pi)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver_path_ae = os.path.join(self.config['checkpoint_dir'], 'vae-model')
        self.saver_ae = tf.train.Saver(
            max_to_keep=self.config['max_to_keep'],
            var_list=self.VAE_outer_vars + self.sigma_vars)

        if self.config['prior'] in ["ours", "hierarchical", "vampPrior"]:
            self.saver_path_prior = os.path.join(self.config['checkpoint_dir'], 'prior-model')
            self.saver_prior = tf.train.Saver(
                max_to_keep=self.config['max_to_keep'],
                var_list=self.prior_vars)

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, model):
        print("Saving model...")
        if model == "VAE":
            self.saver_ae.save(sess, self.saver_path_ae)
            print("Outer VAE model saved.")
        elif model == "prior":
            self.saver_prior.save(sess, self.saver_path_prior)
            print("Prior model saved.")
        elif model == "joint":
            if self.config['TRAIN_VAE'] == 1:
                self.saver_ae.save(sess, self.saver_path_ae)
                print("Outer VAE model saved.")
            if self.config['TRAIN_prior'] == 1:
                self.saver_prior.save(sess, self.saver_path_prior)
                print("Prior model saved.")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess, model):
        print("\ncheckpoint_dir to be loaded:\n{}\n".format(
            self.config['checkpoint_dir']))

        if model == "VAE":
            if os.path.isfile(self.saver_path_ae + '.meta'):
                # print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                self.saver_ae.restore(sess, self.saver_path_ae)
                print("Outer VAE model loaded.")
            else:
                print("No outer VAE model found. No VAE model loaded.")
        elif model == "prior":
            if os.path.isfile(self.saver_path_prior + '.meta'):
                # print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                self.saver_prior.restore(sess, self.saver_path_prior)
                print("Prior model loaded.")
            else:
                print("No prior model found. No prior model loaded.")

    # Define GM prior: only needed for our ladder prior and GMM prior
    def define_GM_prior(self):
        # numpy prior GM
        n_mixtures = self.config['n_mixtures']
        if self.config['prior'] == 'ours':
            n_dims = self.config['representation_size']
            self.GM_prior_training = BayesianGaussianMixture(n_components=n_mixtures,
                                                             covariance_type='full',
                                                             max_iter=1000,
                                                             n_init=1,
                                                             weight_concentration_prior_type='dirichlet_distribution',
                                                             weight_concentration_prior=0.1,
                                                             warm_start=True)
        elif self.config['prior'] == 'GMM':
            n_dims = self.config['code_size']
            self.GM_prior_training = GaussianMixture(n_components=n_mixtures,
                                                     covariance_type='full',
                                                     max_iter=1000,
                                                     n_init=1,
                                                     warm_start=True)

        # create a corresponding placeholder for the Gaussian mixture (hyper) prior in tensorflow
        with tf.name_scope("GMM"):
            self.prior_mean = tf.placeholder(tf.float32, [n_mixtures, n_dims])
            self.prior_cov = tf.placeholder(tf.float32, [n_mixtures, n_dims, n_dims])
            self.prior_weight = tf.placeholder(tf.float32, [n_mixtures])
            mixtures = []
            for i in range(n_mixtures):
                self.mean = tf.squeeze(tf.slice(self.prior_mean, begin=[i, 0], size=[1, n_dims]))
                self.cov = tf.squeeze(tf.slice(self.prior_cov, begin=[i, 0, 0], size=[1, n_dims, n_dims]))

                single_mixture = tfd.MultivariateNormalFullCovariance(
                    loc=self.mean,
                    covariance_matrix=self.cov)
                mixtures.append(single_mixture)
            self.prior_GM_tf = tfd.Mixture(
                cat=tfd.Categorical(probs=self.prior_weight),
                components=mixtures)

    # Define a hyper prior: only needed for our ladder prior and hierarchical prior
    def define_inner_VAE_prior(self):
        init = tf.contrib.layers.xavier_initializer()
        n_layers = self.config['n_layers_inner_VAE']
        self.representation_input = tf.placeholder(tf.float32, [None, self.config['representation_size']])
        self.is_representation_input = tf.placeholder(tf.bool)
        self.is_outer_VAE_input = tf.placeholder(tf.bool)
        self.customised_inner_VAE_input = tf.placeholder(tf.float32, [None, self.config['code_size']])
        if self.config['inner_activation'] == 'tanh':
            activation = tf.nn.tanh
        elif self.config['inner_activation'] == 'relu':
            activation = tf.nn.relu
        elif self.config['inner_activation'] == 'leaky_relu':
            activation = tf.nn.leaky_relu

        with tf.variable_scope('prior'):
            encoder_input = tf.cond(self.is_outer_VAE_input,
                                    lambda: self.code_sample,
                                    lambda: self.customised_inner_VAE_input)
            inner_activation_ = tf.layers.dense(encoder_input,
                                                units=self.config['num_hidden_units_inner_VAE'],
                                                kernel_initializer=init,
                                                activation=activation)
            for i in range(n_layers - 1):
                inner_activation_ = tf.layers.dense(inner_activation_,
                                                    units=self.config['num_hidden_units_inner_VAE'],
                                                    kernel_initializer=init,
                                                    activation=activation)
            self.representation_mean = tf.layers.dense(inner_activation_,
                                                       units=self.config['representation_size'],
                                                       kernel_initializer=init,
                                                       activation=None)
            self.representation_std_dev = tf.layers.dense(inner_activation_,
                                                          units=self.config['representation_size'],
                                                          kernel_initializer=init,
                                                          activation=tf.nn.relu)
            self.representation_std_dev = self.representation_std_dev + self.config['latent_variance_precision']

            mvn = tfp.distributions.MultivariateNormalDiag(
                loc=self.representation_mean,
                scale_diag=self.representation_std_dev)
            self.representation_sample = mvn.sample()
            print("finish creating inner VAE encoder:\n{}\n".format(self.representation_sample))
            print("\n")

            decoder_input = tf.cond(self.is_representation_input,
                                    lambda: self.representation_input,
                                    lambda: self.representation_sample)
            inner_decoder_activation_ = tf.layers.dense(decoder_input,
                                                        units=self.config['num_hidden_units_inner_VAE'],
                                                        kernel_initializer=init,
                                                        activation=activation)
            for i in range(n_layers - 1):
                inner_decoder_activation_ = tf.layers.dense(inner_decoder_activation_,
                                                            units=self.config['num_hidden_units_inner_VAE'],
                                                            kernel_initializer=init,
                                                            activation=activation)
            self.decoded_code = tf.layers.dense(inner_decoder_activation_,
                                                units=self.config['code_size'],
                                                kernel_initializer=init,
                                                activation=None)
            if self.config['TRAIN_decoded_z_std'] == 1:
                inner_decoder_activation_std_ = tf.layers.dense(decoder_input,
                                                                units=self.config['num_hidden_units_inner_VAE'],
                                                                kernel_initializer=init,
                                                                activation=activation)
                for i in range(n_layers - 1):
                    inner_decoder_activation_std_ = tf.layers.dense(inner_decoder_activation_std_,
                                                                    units=self.config['num_hidden_units_inner_VAE'],
                                                                    kernel_initializer=init,
                                                                    activation=activation)
                self.decoded_code_std = tf.layers.dense(inner_decoder_activation_std_,
                                                        units=self.config['code_size'],
                                                        kernel_initializer=init,
                                                        activation=None)
        print("finish inner VAE decoder:\n{}\n".format(self.decoded_code))

        # define sigma as a trainable variable
        with tf.variable_scope('inner_sigma'):
            self.inner_sigma = tf.Variable(self.config['inner_sigma'], dtype=tf.float32, trainable=True)
            self.inner_sigma = tf.square(self.inner_sigma)
            self.inner_sigma = tf.sqrt(self.inner_sigma)
            recons_error_code = tf.abs(self.decoded_code - self.code_sample)
            self.mean_code_error = tf.reduce_mean(recons_error_code)
            if self.config['TRAIN_inner_sigma'] == 1:
                self.inner_sigma = tf.minimum(tf.maximum(self.inner_sigma, self.config['inner_sigma_lb']),
                                              self.config['inner_sigma_ub'])
        print("inner sigma:\n{}\n".format(self.inner_sigma))

    # Define VampPrior: only needed for VampPrior
    def define_vampPrior(self):
        n_mixtures = self.config['n_mixtures']
        n_dims = self.config['code_size']
        dx = self.config['dim_input_x']
        dy = self.config['dim_input_y']
        dc = self.config['dim_input_channel']
        kernel_size = self.config['kernel_size']
        with tf.variable_scope('prior'):
            self.psedeu_input = tf.Variable(tf.random.normal([n_mixtures, dx, dy, dc]),
                                            dtype=tf.float32, trainable=True)

        init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('encoder', reuse=True):
            encoded_signal = self.encoder_mapping(init, input=self.psedeu_input)
            self.code_mean_prior = tf.layers.dense(encoded_signal,
                                                   units=self.config['code_size'],
                                                   activation=None,
                                                   name='code_mean')
            code_std_dev_prior = tf.layers.dense(encoded_signal,
                                                 units=self.config['code_size'],
                                                 activation=tf.nn.relu,
                                                 name='code_std_dev')
            self.code_std_dev_prior = code_std_dev_prior + self.config['latent_variance_precision']

        # tf prior GM - placeholder
        with tf.name_scope("prior"):
            prior_weight = tf.constant(1 / n_mixtures, shape=[n_mixtures], dtype=tf.float32)
            mixtures = []
            for i in range(n_mixtures):
                mean = tf.squeeze(tf.slice(self.code_mean_prior, begin=[i, 0], size=[1, n_dims]))
                scale = tf.squeeze(tf.slice(self.code_std_dev_prior, begin=[i, 0], size=[1, n_dims]))

                single_mixture = tfd.MultivariateNormalDiag(
                    loc=mean,
                    scale_diag=scale)
                mixtures.append(single_mixture)
            self.psedeu_prior = tfd.Mixture(
                cat=tfd.Categorical(probs=prior_weight),
                components=mixtures)

    # Define the ELBO loss for different prior models
    def define_loss(self):
        self.use_standard_gaussian_prior = tf.placeholder(tf.bool)
        self.use_mask = tf.placeholder(tf.bool)
        with tf.name_scope("loss"):
            # computer L1 norm of standard deviation of the sample-wise posterior
            self.std_dev_code = tf.reduce_mean(self.code_std_dev, axis=0)
            if self.config['prior'] in ["ours", "hierarchical"]:
                self.std_dev_representation = tf.reduce_mean(self.representation_std_dev, axis=0)

            # define ELBO loss for VAE model
            # part I: entropy of q(z|x)
            # term 2 in Eq (2)
            entropy_z = - 0.5 * self.config['code_size'] * tf.log(self.two_pi) - \
                        0.5 * self.config['code_size'] - \
                        0.5 * tf.reduce_sum(2. * tf.log(self.code_std_dev), 1)
            self.entropy_z = tf.squeeze(tf.reduce_mean(entropy_z))

            # part II: cross-entropy between q(z|x) and prior p(z)
            # crossEntropy_prior wrt standard gaussian distribution
            # term 3 in Eq (2)
            crossEntropy_prior_sg = - 0.5 * self.config['code_size'] * tf.log(self.two_pi) - \
                                    0.5 * (tf.reduce_sum(tf.square(self.code_mean), 1) + \
                                           tf.reduce_sum(tf.square(self.code_std_dev), 1))
            self.crossEntropy_prior_sg = tf.squeeze(tf.reduce_mean(crossEntropy_prior_sg))
            if self.config['prior'] == 'standard_gaussian':
                self.crossEntropy_prior = self.crossEntropy_prior_sg

            # our LaDDer prior - use another VAE network to model the prior distribution p(z)
            elif self.config['prior'] == "ours":
                recons_error = tf.square(self.code_sample - self.decoded_code)
                # expanded_std_dev = tf.tile(tf.expand_dims(self.std_dev_code, 0), [self.config['batch_size'], 1])
                masked_recons_error = tf.where(self.code_std_dev > 1., x=0. * recons_error, y=1. * recons_error)
                code_recons_error = tf.cond(self.use_mask,
                                            lambda: masked_recons_error,
                                            lambda: recons_error)
                code_reconstruction_likelihood = tf.reduce_sum(
                    code_recons_error / (2 * tf.square(self.inner_sigma)), 1)
                code_reconstruction_likelihood = tf.reduce_mean(code_reconstruction_likelihood)
                self.code_reconstruction_likelihood = - tf.squeeze(code_reconstruction_likelihood)
                code_l1_reconstruction_error = tf.reduce_sum(tf.sqrt(code_recons_error), 1)
                self.code_l1_reconstruction_error = tf.reduce_mean(code_l1_reconstruction_error)

                self.representation_regularisor = - self.config['code_size'] * tf.log(self.inner_sigma) - \
                                                  0.5 * self.config['code_size'] * tf.log(self.two_pi)

                entropy_t = - 0.5 * self.config['representation_size'] * tf.log(self.two_pi) - \
                            0.5 * self.config['representation_size'] - \
                            0.5 * tf.reduce_sum(2. * tf.log(self.representation_std_dev), 1)
                self.entropy_t = tf.reduce_mean(entropy_t)

                # draw samples from a batch of posteriors (estimated from encoder)
                q_t_batch = tfd.MultivariateNormalDiag(loc=self.representation_mean,
                                                       scale_diag=self.representation_std_dev)
                L = self.config['n_MC_samples']
                samples = q_t_batch.sample(L)
                # evaluate the difference log prob of the qt and pt
                self.crossEntropy_representation = tf.reduce_mean(self.prior_GM_tf.log_prob(samples))
                elbo_prior = self.code_reconstruction_likelihood + self.representation_regularisor - \
                             self.entropy_t + self.crossEntropy_representation

                self.elbo_prior = tf.squeeze(elbo_prior)
                self.crossEntropy_prior = tf.cond(self.use_standard_gaussian_prior,
                                                  lambda: self.crossEntropy_prior_sg,
                                                  lambda: self.elbo_prior)

            # GMM prior
            elif self.config['prior'] == "GMM":
                # draw samples from a batch of z posteriors (estimated from encoder)
                q_z_batch = tfd.MultivariateNormalDiag(loc=self.code_mean, scale_diag=self.code_std_dev)
                L = self.config['n_MC_samples']
                samples = q_z_batch.sample(L)
                # evaluate the difference log prob of the qz and pz
                self.crossEntropy_prior = tf.reduce_mean(self.prior_GM_tf.log_prob(samples))

            # hierarchical VAE prior
            elif self.config['prior'] == "hierarchical":
                # inner_sigma = self.config['inner_sigma']
                recons_error = tf.square(self.code_sample - self.decoded_code)
                code_reconstruction_likelihood = tf.reduce_sum(
                    recons_error / (2 * tf.square(self.inner_sigma)), 1)
                code_reconstruction_likelihood = tf.reduce_mean(code_reconstruction_likelihood)
                self.code_reconstruction_likelihood = - tf.squeeze(code_reconstruction_likelihood)
                code_l1_reconstruction_error = tf.reduce_sum(tf.sqrt(recons_error), 1)
                self.code_l1_reconstruction_error = tf.reduce_mean(code_l1_reconstruction_error)

                self.representation_regularisor = - self.config['code_size'] * tf.log(self.inner_sigma) - \
                                                  0.5 * self.config['code_size'] * tf.log(self.two_pi)

                entropy_t = - 0.5 * 2 * tf.log(self.two_pi) - 0.5 * 2 - \
                            0.5 * tf.reduce_sum(2. * tf.log(self.representation_std_dev), 1)
                self.entropy_t = tf.reduce_mean(entropy_t)

                crossEntropy_representation = - 0.5 * self.config['representation_size'] * tf.log(self.two_pi) - \
                                              0.5 * (tf.reduce_sum(tf.square(self.representation_mean), 1) + \
                                                     tf.reduce_sum(tf.square(self.representation_std_dev), 1))
                self.crossEntropy_representation = tf.squeeze(tf.reduce_mean(crossEntropy_representation))

                elbo_prior = self.code_reconstruction_likelihood + self.representation_regularisor - \
                             self.entropy_t + self.crossEntropy_representation
                self.elbo_prior = tf.squeeze(elbo_prior)
                self.crossEntropy_prior = tf.cond(self.use_standard_gaussian_prior,
                                                  lambda: self.crossEntropy_prior_sg,
                                                  lambda: self.elbo_prior)

            # VampPrior
            elif self.config['prior'] == "vampPrior":
                # draw samples from a batch of z posteriors (estimated from encoder)
                q_z_batch = tfd.MultivariateNormalDiag(loc=self.code_mean, scale_diag=self.code_std_dev)
                L = self.config['n_MC_samples']
                samples = q_z_batch.sample(L)
                # evaluate the difference log prob of the qz and pz
                self.crossEntropy_prior = tf.cond(self.use_standard_gaussian_prior,
                                                  lambda: self.crossEntropy_prior_sg,
                                                  lambda: tf.reduce_mean(self.psedeu_prior.log_prob(samples)))

            # reconstruction error
            # sum of squares
            l2_reconstruction_error = tf.reduce_sum(
                tf.square(self.original_signal - self.decoded), [1, 2, 3])
            self.l2_reconstruction_error = tf.reduce_mean(l2_reconstruction_error)

            # L1 norm
            l1_reconstruction_error = tf.reduce_sum(
                tf.sqrt(tf.square(self.original_signal - self.decoded)), [1, 2, 3])
            self.l1_reconstruction_error = tf.reduce_mean(l1_reconstruction_error)

            # reconstruction likelihood
            # See Supplementary Material Part C for more details
            pixel_error = self.original_signal - self.decoded
            reconstruction_likelihood = tf.reduce_sum(
                tf.abs(pixel_error), [1, 2, 3])
            reconstruction_likelihood = tf.reduce_mean(reconstruction_likelihood)
            self.reconstruction_likelihood = - reconstruction_likelihood / self.sigma
            print("reconstruction likelihood:\n{}\n".format(self.reconstruction_likelihood))

            # sigma regularisor
            # See Supplementary Material Part C for more details
            self.input_dim = self.config['dim_input_x'] * self.config['dim_input_y'] * self.config['dim_input_channel']
            self.sigma_regularisor = - self.input_dim * tf.log(2 * self.sigma)
            self.sigma_regularisor = tf.squeeze(self.sigma_regularisor)

            # Assemble all the parts together to make ELBO loss
            elbo = self.reconstruction_likelihood + self.sigma_regularisor - \
                   self.entropy_z + self.crossEntropy_prior
            self.elbo = tf.squeeze(elbo)
            self.negative_elbo = - self.elbo

            self.loss_ae = self.negative_elbo
            print("Loss for the outer auto-encoder model is:\n{}\n".format(self.loss_ae))

            if self.config['prior'] in ["ours", "hierarchical", "vampPrior"]:
                if self.config['prior'] == "vampPrior":
                    self.loss_prior = self.negative_elbo
                    # self.loss_prior = - self.crossEntropy_prior
                else:
                    self.loss_prior = - self.elbo_prior
                print("Loss for prior model is:\n{}\n".format(self.loss_prior))

    def training_variables(self):
        # Define training variables for VAE network
        encoder_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
        decoder_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
        self.VAE_outer_vars = encoder_vars + decoder_vars
        self.sigma_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "sigma")
        if self.config['prior'] in ["ours", "hierarchical", "vampPrior"]:
            self.prior_vars_ae = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "prior")
            if self.config['prior'] in ["ours", "hierarchical"]:
                self.prior_vars_sigma = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner_sigma")
                self.prior_vars = self.prior_vars_ae + self.prior_vars_sigma
            else:
                self.prior_vars = self.prior_vars_ae
            print("prior model variable:\n{}\n".format(self.prior_vars))

        if self.config['prior'] in ['standard_gaussian', 'GMM']:
            self.all_vars = self.VAE_outer_vars + self.sigma_vars
        elif self.config['prior'] in ['ours', 'hierarchical', 'vampPrior']:
            self.all_vars = self.VAE_outer_vars + self.sigma_vars + self.prior_vars

        # compute the number of trainable parameters in the VAE network:
        self.num_encoder = count_trainable_variables('encoder')
        self.num_decoder = count_trainable_variables('decoder')
        self.num_sigma = count_trainable_variables('sigma')
        if self.config['prior'] in ["ours", "hierarchical", "vampPrior"]:
            self.num_prior_ae = count_trainable_variables('prior')
            if self.config['prior'] in ["ours", "hierarchical"]:
                self.num_prior_sigma = count_trainable_variables('inner_sigma')
            else:
                self.num_prior_sigma = 0
        else:
            self.num_prior_sigma = 0
            self.num_prior_ae = 0
        self.num_para_list = [self.num_encoder, self.num_decoder, self.num_sigma, self.num_prior_ae,
                              self.num_prior_sigma]

        print(
            "Total number of trainable parameters in VAE network is:\n{}k\n".format(
                np.around(sum(self.num_para_list)/1000, 2)))

    def compute_gradients(self):
        self.lr_ae = tf.placeholder(tf.float32, [])
        opt_ae = tf.train.AdamOptimizer(learning_rate=self.lr_ae,
                                        beta1=0.9,
                                        beta2=0.95)
        self.gvs_ae = opt_ae.compute_gradients(self.loss_ae, var_list=self.VAE_outer_vars)
        print('gvs autoencoder: {}'.format(self.gvs_ae))
        capped_gvs_ae = [(self.ClipIfNotNone(grad), var) for grad, var in self.gvs_ae]
        if self.config['TRAIN_sigma'] == 1:
            self.lr_sigma = tf.placeholder(tf.float32, [])
            opt_sigma = tf.train.AdamOptimizer(learning_rate=self.lr_sigma,
                                               beta1=0.9,
                                               beta2=0.95)
            self.gvs_sigma = opt_sigma.compute_gradients(self.loss_ae, var_list=self.sigma_vars)
            print('gvs sigma: {}'.format(self.gvs_sigma))
            capped_gvs_sigma = [(self.ClipIfNotNone(grad), var) for grad, var in self.gvs_sigma]

        if self.config['prior'] in ["ours", "hierarchical", "vampPrior"]:
            self.lr_prior = tf.placeholder(tf.float32, [])
            opt_prior = tf.train.AdamOptimizer(learning_rate=self.lr_prior,
                                               beta1=0.9,
                                               beta2=0.95)
            self.gvs_prior = opt_prior.compute_gradients(self.loss_prior, var_list=self.prior_vars_ae)
            print('gvs prior: {}'.format(self.gvs_prior))
            capped_gvs_prior = [(self.ClipIfNotNone(grad), var) for grad, var in self.gvs_prior]

            if self.config['prior'] in ["ours", "hierarchical"] and self.config['TRAIN_inner_sigma'] == 1:
                self.lr_inner_sigma = tf.placeholder(tf.float32, [])
                opt_inner_sigma = tf.train.AdamOptimizer(learning_rate=self.lr_inner_sigma,
                                                         beta1=0.9,
                                                         beta2=0.95)
                self.gvs_inner_sigma = opt_inner_sigma.compute_gradients(self.loss_prior,
                                                                         var_list=self.prior_vars_sigma)
                print('gvs inner sigma: {}'.format(self.gvs_inner_sigma))
                capped_gvs_inner_sigma = [(self.ClipIfNotNone(grad), var) for grad, var in self.gvs_inner_sigma]
            if self.config['prior'] in ["ours", "hierarchical"] and self.config['TRAIN_decoded_z_std'] == 1:
                opt_z_std = tf.train.AdamOptimizer(learning_rate=self.lr_prior,
                                                   beta1=0.9,
                                                   beta2=0.95)
                self.gvs_z_std = opt_z_std.compute_gradients(self.decoded_z_std_loss, var_list=self.prior_vars_ae)
                print('gvs z std: {}'.format(self.gvs_z_std))
                capped_gvs_z_std = [(self.ClipIfNotNone(grad), var) for grad, var in self.gvs_z_std]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step_ae = opt_ae.apply_gradients(capped_gvs_ae)
            if self.config['TRAIN_sigma'] == 1:
                self.train_step_sigma = opt_sigma.apply_gradients(capped_gvs_sigma)
            if self.config['prior'] in ["ours", "hierarchical", "vampPrior"]:
                if self.config['TRAIN_prior'] == 1:
                    self.train_step_prior = opt_prior.apply_gradients(capped_gvs_prior)
                if self.config['prior'] in ["ours", "hierarchical"] and self.config['TRAIN_inner_sigma'] == 1:
                    self.train_step_inner_sigma = opt_inner_sigma.apply_gradients(capped_gvs_inner_sigma)
                if self.config['prior'] in ["ours", "hierarchical"] and self.config['TRAIN_decoded_z_std'] == 1:
                    self.train_step_z_std = opt_z_std.apply_gradients(capped_gvs_z_std)
        print("Finish the gradient computation for the VAE model.\n")

    def ClipIfNotNone(self, grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)


class BaseTrain:
    def __init__(self, sess, model, data, config):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
        self.sess.run(self.init)
        self.cur_epoch = 0

        # keep a record of the training result
        self.train_loss = []
        self.train_loss_prior = []
        self.val_loss = []
        self.val_loss_prior = []
        self.train_loss_ave_epoch = []
        self.val_loss_ave_epoch = []
        self.elbo_train = []
        self.elbo_val = []
        self.recons_error_train = []
        self.recons_error_val = []
        self.entropy_z_train = []
        self.entropy_z_val = []
        self.crossEntropy_prior_train = []
        self.crossEntropy_prior_val = []
        self.vampPrior_crossEntropy_prior_val = []
        self.vampPrior_crossEntropy_prior_train = []
        self.sigma_reguarisor_train = []
        self.sigma_reguarisor_val = []
        self.code_elbo_train = []
        self.code_elbo_val = []
        self.entropy_t_train = []
        self.entropy_t_val = []
        self.crossEntropy_t_train = []
        self.crossEntropy_t_val = []
        self.code_recons_error_train = []
        self.code_recons_error_val = []
        self.code_recons_likelihood_train = []
        self.code_inner_sigma_train = []
        self.iter_epochs_list = []
        self.test_batch_code_mean = []
        self.test_batch_code_std_dev = []
        self.test_sigma = []
        self.sigma_train = []
        self.classifier_accuracy = []
        self.n_train_iter = []
        self.n_val_iter = []
        self.gmm_mean = []
        self.gmm_cov = []
        self.gmm_weight = []

    def compute_execution_time(self, cur_epoch, total_epoch):
        # cur_epoch is zero start
        # compute current execution time
        self.current_time = time.time()
        elapsed_time = (self.current_time - self.start_time) / 60
        print("Already trained for {} min.".format(elapsed_time))

        est_remaining_time = (self.current_time - self.start_time) / (cur_epoch + 1) * total_epoch
        est_remaining_time = est_remaining_time / 60 - elapsed_time
        print("Remaining {} min.\n".format(est_remaining_time))

    def train_step_ae(self, cur_lr, batch_data):
        feed_dict = self.compute_feeddict(batch_data=batch_data, model_to_train="VAE")
        feed_dict[self.model.lr_ae] = cur_lr

        train_loss, train_elbo, train_recons_loss, train_entropy_z, train_crossEntropy_prior, train_sigma_regularisor, _ = self.sess.run(
            [self.model.loss_ae,
             self.model.elbo,
             self.model.l1_reconstruction_error,
             self.model.entropy_z,
             self.model.crossEntropy_prior,
             self.model.sigma_regularisor,
             self.model.train_step_ae], feed_dict=feed_dict)
        self.recons_error_train.append(np.squeeze(train_recons_loss))
        self.entropy_z_train.append(train_entropy_z)
        self.crossEntropy_prior_train.append(train_crossEntropy_prior)
        self.sigma_reguarisor_train.append(train_sigma_regularisor)
        self.elbo_train.append(train_elbo)

        if self.config['TRAIN_sigma'] == 1:
            feed_dict[self.model.lr_sigma] = self.config['learning_rate_sigma'] * (0.99 ** (self.cur_epoch - 1))
            sigma, _ = self.sess.run(
                [self.model.sigma,
                 self.model.train_step_sigma], feed_dict=feed_dict)
            self.sigma_train.append(sigma)

        return train_loss

    def train_step_prior(self, batch_data):
        feed_dict = self.compute_feeddict(batch_data=batch_data, model_to_train="prior")
        feed_dict[self.model.lr_prior] = self.config['learning_rate_prior'] * (1.01 ** (self.cur_epoch - 1))

        if self.config['prior'] in ['ours', 'hierarchical']:
            elbo_prior, code_recons_loss, code_recons_likelihood, entropy_t, crossEntropy_t, inner_sigma, _ = self.sess.run(
                [self.model.elbo_prior,
                 self.model.code_l1_reconstruction_error,
                 self.model.code_reconstruction_likelihood,
                 self.model.entropy_t,
                 self.model.crossEntropy_representation,
                 self.model.inner_sigma,
                 self.model.train_step_prior], feed_dict=feed_dict)
            self.code_recons_error_train.append(code_recons_loss)
            self.code_recons_likelihood_train.append(code_recons_likelihood)
            self.entropy_t_train.append(entropy_t)
            self.crossEntropy_t_train.append(crossEntropy_t)
            self.code_elbo_train.append(elbo_prior)
            self.code_inner_sigma_train.append(inner_sigma)
        else:
            prior_crossEntropy, prior_loss, _ = self.sess.run([self.model.crossEntropy_prior,
                                                               self.model.loss_prior,
                                                               self.model.train_step_prior], feed_dict=feed_dict)
            self.train_loss_prior.append(prior_loss)
            self.vampPrior_crossEntropy_prior_train.append(prior_crossEntropy)

        if self.config['prior'] in ["ours", "hierarchical"] and self.config['TRAIN_inner_sigma'] == 1:
            feed_dict[self.model.lr_inner_sigma] = self.config['learning_rate_inner_sigma'] * (
                        1.01 ** (self.cur_epoch - 1))
            _ = self.sess.run(self.model.train_step_inner_sigma, feed_dict=feed_dict)
        if self.config['prior'] in ["ours", "hierarchical"] and self.config['TRAIN_decoded_z_std'] == 1:
            _ = self.sess.run(self.model.train_step_z_std, feed_dict=feed_dict)

    def val_step(self, model_to_train, batch_data):
        feed_dict = self.compute_feeddict(batch_data=batch_data, model_to_train=model_to_train)

        if model_to_train == "VAE":
            elbo_val, recons_loss_val, entropy_z_val, crossEntropy_prior_val, val_loss = self.sess.run(
                [self.model.elbo,
                 self.model.l1_reconstruction_error,
                 self.model.entropy_z,
                 self.model.crossEntropy_prior,
                 self.model.loss_ae],
                feed_dict=feed_dict)
            self.val_loss.append(val_loss)
            self.recons_error_val.append(recons_loss_val)
            self.entropy_z_val.append(entropy_z_val)
            self.elbo_val.append(elbo_val)
            self.crossEntropy_prior_val.append(crossEntropy_prior_val)
        else:
            if self.config['prior'] in ['ours', 'hierarchical']:
                elbo_val, recons_loss_val, entropy_z_val, crossEntropy_prior_val, val_loss = self.sess.run(
                    [self.model.elbo_prior,
                     self.model.code_l1_reconstruction_error,
                     self.model.entropy_t,
                     self.model.crossEntropy_representation,
                     self.model.loss_prior],
                    feed_dict=feed_dict)
                self.val_loss_prior.append(val_loss)
                self.code_recons_error_val.append(recons_loss_val)
                self.entropy_t_val.append(entropy_z_val)
                self.code_elbo_val.append(elbo_val)
                self.crossEntropy_t_val.append(crossEntropy_prior_val)
            else:
                crossEntropy_prior_val, val_loss = self.sess.run([self.model.crossEntropy_prior,
                                                                  self.model.loss_prior], feed_dict=feed_dict)
                self.val_loss_prior.append(val_loss)
                self.vampPrior_crossEntropy_prior_val.append(crossEntropy_prior_val)

        return val_loss

    def fit_GMM_VI(self, iterator, mode="fast", space="z"):
        if mode == "fast":
            n_batch = 2000 // self.config['batch_size'] + 1
            if space == "t":
                for i in range(n_batch):
                    batch_signal = self.sess.run(iterator)
                    feed_dict = {self.model.original_signal: batch_signal,
                                 self.model.is_code_input: False,
                                 self.model.code_input: np.zeros((1, self.config['code_size'])),
                                 self.model.is_outer_VAE_input: True,
                                 self.model.customised_inner_VAE_input: np.zeros((1, self.config['code_size'])),
                                 self.model.is_representation_input: False,
                                 self.model.representation_input: np.zeros((1, self.config['representation_size']))}
                    t_sample = self.sess.run(self.model.representation_sample, feed_dict=feed_dict)
                    if i == 0:
                        samples = t_sample
                    else:
                        samples = np.concatenate((samples, t_sample), axis=0)
            elif space == "z":
                for i in range(n_batch):
                    batch_signal = self.sess.run(iterator)
                    feed_dict = {self.model.original_signal: batch_signal,
                                 self.model.is_code_input: False,
                                 self.model.code_input: np.zeros((1, self.config['code_size']))}
                    z_sample = self.sess.run(self.model.code_sample, feed_dict=feed_dict)
                    if i == 0:
                        samples = z_sample
                    else:
                        samples = np.concatenate((samples, z_sample), axis=0)
            self.model.GM_prior_training.fit(samples)
            idx_valid_mixture = np.squeeze(np.argwhere(self.model.GM_prior_training.weights_ >= 1e-2)).tolist()
            if type(idx_valid_mixture) is int:
                print("There are 1 active mixtures.")
                print("The current GM prior estimate has following weights:\n{}".format(
                    self.model.GM_prior_training.weights_[idx_valid_mixture]))
            elif len(idx_valid_mixture) == 0:
                print("There are 0 active mixtures.")
            else:
                print("There are {} active mixtures.".format(len(idx_valid_mixture)))
                print("The current GM prior estimate has following weights:\n{}".format(
                    self.model.GM_prior_training.weights_[idx_valid_mixture]))
        else:
            n_batch = 20000 // self.config['batch_size'] + 1
            if space == "t":
                for i in range(n_batch):
                    batch_signal = self.sess.run(iterator)
                    feed_dict = {self.model.original_signal: batch_signal,
                                 self.model.is_code_input: False,
                                 self.model.code_input: np.zeros((1, self.config['code_size'])),
                                 self.model.is_outer_VAE_input: True,
                                 self.model.customised_inner_VAE_input: np.zeros((1, self.config['code_size'])),
                                 self.model.is_representation_input: False,
                                 self.model.representation_input: np.zeros((1, self.config['representation_size']))}
                    t_sample = self.sess.run(self.model.representation_sample, feed_dict=feed_dict)
                    if i == 0:
                        samples = t_sample
                    else:
                        samples = np.concatenate((samples, t_sample), axis=0)
                gemma = 0.1
                self.GM_prior_final = BayesianGaussianMixture(n_components=self.config['n_mixtures'],
                                                              covariance_type='full',
                                                              max_iter=2000,
                                                              n_init=self.config['GM_fit_restart'],
                                                              weight_concentration_prior_type='dirichlet_process',
                                                              weight_concentration_prior=gemma,
                                                              warm_start=False,
                                                              verbose=2,
                                                              verbose_interval=100)
            elif space == "z":
                for i in range(n_batch):
                    batch_signal = self.sess.run(iterator)
                    feed_dict = {self.model.original_signal: batch_signal,
                                 self.model.is_code_input: False,
                                 self.model.code_input: np.zeros((1, self.config['code_size']))}
                    z_sample = self.sess.run(self.model.code_sample, feed_dict=feed_dict)
                    if i == 0:
                        samples = z_sample
                    else:
                        samples = np.concatenate((samples, z_sample), axis=0)
                self.GM_prior_final = GaussianMixture(n_components=self.config['n_mixtures'],
                                                      covariance_type='full',
                                                      max_iter=2000,
                                                      n_init=1,
                                                      warm_start=False,
                                                      verbose=2,
                                                      verbose_interval=100)
            self.GM_prior_final.fit(samples)
            idx_valid_mixture = np.squeeze(np.argwhere(self.GM_prior_final.weights_ >= 1e-2)).tolist()
            filename = "{}GM_prior_info.npz".format(self.config['result_dir'])
            adjusted_w = self.GM_prior_final.weights_[idx_valid_mixture]
            adjusted_w = adjusted_w / sum(adjusted_w)
            np.savez(filename, w_active=adjusted_w,
                     m_active=self.GM_prior_final.means_[idx_valid_mixture],
                     K_active=self.GM_prior_final.covariances_[idx_valid_mixture],
                     w_full=self.GM_prior_final.weights_,
                     m_full=self.GM_prior_final.means_,
                     K_full=self.GM_prior_final.covariances_)
            if type(idx_valid_mixture) is int:
                print("There are 1 active mixtures.")
                print("The current GM prior estimate has following weights:\n{}".format(
                    self.GM_prior_final.weights_[idx_valid_mixture]))
            elif len(idx_valid_mixture) == 0:
                print("There are 0 active mixtures.")
            else:
                print("There are {} active mixtures.".format(len(idx_valid_mixture)))
                print("The current GM prior estimate has following weights:\n{}".format(
                    self.GM_prior_final.weights_[idx_valid_mixture]))
            print("Final fitted prior saved.")
        return samples

    def save_variables_VAE(self):
        # save some variables for later inspection
        file_name = "{}{}-result.npz".format(self.config['result_dir'], self.config['exp_name'])
        np.savez(file_name,
                 iter_list_val=self.iter_epochs_list,
                 n_train_iter=self.n_train_iter,
                 n_val_iter=self.n_val_iter,
                 train_loss=self.train_loss,
                 elbo_train=self.elbo_train,
                 val_loss=self.val_loss,
                 elbo_val=self.elbo_val,
                 train_loss_prior=self.train_loss_prior,
                 val_loss_prior=self.val_loss_prior,
                 code_elbo_train=self.code_elbo_train,
                 code_elbo_val=self.code_elbo_val,
                 recons_loss_train=self.recons_error_train,
                 recons_loss_val=self.recons_error_val,
                 recons_loss_prior_train=self.code_recons_error_train,
                 recons_loss_prior_val=self.code_recons_error_val,
                 entropy_z_train=self.entropy_z_train,
                 entropy_z_val=self.entropy_z_val,
                 entropy_t_train=self.entropy_t_train,
                 entropy_t_val=self.entropy_t_val,
                 crossentropy_z_train=self.crossEntropy_prior_train,
                 crossentropy_z_val=self.crossEntropy_prior_val,
                 crossentropy_t_train=self.crossEntropy_t_train,
                 crossentropy_t_val=self.crossEntropy_t_val,
                 vampPrior_crossEntropy_z_train_prior=self.vampPrior_crossEntropy_prior_train,
                 vampPrior_crossEntropy_z_val_prior=self.vampPrior_crossEntropy_prior_val,
                 sigma_regularisor_train=self.sigma_reguarisor_train,
                 sigma_regularisor_val=self.sigma_reguarisor_val,
                 num_para_VAE=self.model.num_para_list,
                 sigma=self.test_sigma)

    def draw_ellipse(self, position, covariance, weight, ax=None, color='r'):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        nsig = 2
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, color=color, fill=False, lw=weight * 10))


class BaseTrain_joint(BaseTrain):
    def __init__(self, sess, model, data, config):
        super().__init__(sess, model, data, config)

    def train(self):
        self.start_time = time.time()
        for cur_epoch in range(0, self.config['num_epochs'], 1):
            self.train_epoch()

            # save the model after finishing one epoch
            if self.config['prior'] in ["ours", "hierarchical", "vampPrior"]:
                self.model.save(self.sess, model="joint")
            elif self.config['prior'] in ["standard_gaussian", "GMM"]:
                self.model.save(self.sess, model="VAE")

            # compute current execution time
            self.compute_execution_time(self.cur_epoch - 1, self.config['num_epochs'])

    def compute_feeddict(self, batch_data, model_to_train=None):
        if self.config['prior'] == 'standard_gaussian':
            feed_dict = {self.model.original_signal: batch_data,
                         self.model.is_code_input: False,
                         self.model.code_input: np.zeros((1, self.config['code_size']))}
        elif self.config['prior'] == 'ours':
            if self.cur_epoch <= self.config['sg_pretraining']:
                feed_dict = {self.model.original_signal: batch_data,
                             self.model.prior_mean: np.zeros((self.config['n_mixtures'],
                                                              self.config['representation_size'])),
                             self.model.prior_cov: np.tile(np.expand_dims(np.diag(
                                 np.ones(self.config['representation_size'])), 0),
                                 (self.config['n_mixtures'], 1, 1)),
                             self.model.prior_weight: 1 / self.config['n_mixtures'] * np.ones(
                                 self.config['n_mixtures']),
                             self.model.use_standard_gaussian_prior: True,
                             self.model.is_code_input: False,
                             self.model.code_input: np.zeros((1, self.config['code_size'])),
                             self.model.is_outer_VAE_input: True,
                             self.model.customised_inner_VAE_input: np.zeros((1, self.config['code_size'])),
                             self.model.is_representation_input: False,
                             self.model.representation_input: np.zeros((1, self.config['representation_size']))}
            else:
                feed_dict = {self.model.original_signal: batch_data,
                             self.model.prior_mean: self.model.GM_prior_training.means_,
                             self.model.prior_cov: self.model.GM_prior_training.covariances_,
                             self.model.prior_weight: self.model.GM_prior_training.weights_,
                             self.model.use_standard_gaussian_prior: False,
                             self.model.is_code_input: False,
                             self.model.code_input: np.zeros((1, self.config['code_size'])),
                             self.model.is_outer_VAE_input: True,
                             self.model.customised_inner_VAE_input: np.zeros((1, self.config['code_size'])),
                             self.model.is_representation_input: False,
                             self.model.representation_input: np.zeros((1, self.config['representation_size']))}
            if self.cur_epoch >= self.config['use_mask_start']:
                feed_dict[self.model.use_mask] = True
            else:
                feed_dict[self.model.use_mask] = False

        elif self.config['prior'] == "hierarchical":
            feed_dict = {self.model.original_signal: batch_data,
                         self.model.is_code_input: False,
                         self.model.code_input: np.zeros((1, self.config['code_size'])),
                         self.model.is_outer_VAE_input: True,
                         self.model.customised_inner_VAE_input: np.zeros((1, self.config['code_size'])),
                         self.model.is_representation_input: False,
                         self.model.representation_input: np.zeros((1, self.config['representation_size'])),
                         self.model.use_standard_gaussian_prior: False}
            if self.cur_epoch <= self.config['sg_pretraining']:
                feed_dict[self.model.use_standard_gaussian_prior] = True
        elif self.config['prior'] == 'GMM':
            if self.cur_epoch == 1:
                feed_dict = {self.model.original_signal: batch_data,
                             self.model.prior_mean: np.zeros((self.config['n_mixtures'],
                                                              self.config['code_size'])),
                             self.model.prior_cov: np.tile(np.expand_dims(np.diag(
                                 np.ones(self.config['code_size'])), 0),
                                 (self.config['n_mixtures'], 1, 1)),
                             self.model.prior_weight: 1 / self.config['n_mixtures'] * np.ones(
                                 self.config['n_mixtures']),
                             self.model.is_code_input: False,
                             self.model.code_input: np.zeros((1, self.config['code_size']))}
            else:
                feed_dict = {self.model.original_signal: batch_data,
                             self.model.prior_mean: self.model.GM_prior_training.means_,
                             self.model.prior_cov: self.model.GM_prior_training.covariances_ + np.tile(
                                 np.expand_dims(np.diag(
                                     0.01 * np.ones(self.config['code_size'])), 0),
                                 (self.config['n_mixtures'], 1, 1)),
                             self.model.prior_weight: self.model.GM_prior_training.weights_,
                             self.model.is_code_input: False,
                             self.model.code_input: np.zeros((1, self.config['code_size']))}
        elif self.config['prior'] == 'vampPrior':
            feed_dict = {self.model.original_signal: batch_data,
                         self.model.is_code_input: False,
                         self.model.code_input: np.zeros((1, self.config['code_size']))}
            if self.cur_epoch <= self.config['sg_pretraining']:
                feed_dict[self.model.use_standard_gaussian_prior] = True
            else:
                feed_dict[self.model.use_standard_gaussian_prior] = False
        return feed_dict

    def test_step(self, batch_data, print_result=False):
        feed_dict = self.compute_feeddict(batch_data=batch_data)

        output_test, test_recons_loss, test_pixel_recons_error, test_entropy_z, test_crossEntropy_z, test_elbo, test_sigma_regularisor = self.sess.run(
            [self.model.decoded,
             self.model.l1_reconstruction_error,
             self.model.mean_pixel_error,
             self.model.entropy_z,
             self.model.crossEntropy_prior,
             self.model.elbo,
             self.model.sigma_regularisor],
            feed_dict=feed_dict)
        self.output_test = np.squeeze(output_test)

        if print_result:
            print(
                "test loss: elbo: {:.4f}, recons_loss_l1: {:.4f}, entropy z: {:.4f}, cross entropy z: {:.4f}, sigma_regularisor: {:.4f}".format(
                    test_elbo,
                    test_recons_loss,
                    test_entropy_z,
                    test_crossEntropy_z,
                    test_sigma_regularisor))

        sigma = self.sess.run(self.model.sigma, feed_dict=feed_dict)
        sigma_mean = np.mean(sigma)
        self.test_sigma.append(sigma_mean)
        print("current sigma: mean: {:.7f}; pixel mean error: {:.7f}".format(
            sigma_mean, test_pixel_recons_error))

        if self.config['prior'] in ["ours", "hierarchical"]:
            z_std, t_std, inner_sigma, code_error = self.sess.run([self.model.std_dev_code,
                                                                   self.model.std_dev_representation,
                                                                   self.model.inner_sigma,
                                                                   self.model.mean_code_error], feed_dict=feed_dict)
            if print_result:
                print("current z std: {}".format(z_std))
                print("current t std: {}".format(t_std))
                print("current inner VAE sigma: {}".format(inner_sigma))
                print("current code prediction error per channel: {}".format(code_error))
        else:
            z_std = self.sess.run(self.model.std_dev_code, feed_dict=feed_dict)
            if print_result:
                print("current z std: {}".format(z_std))

    def fit_GM(self, iterator):
        # fit a GM for p(t) or p(z)
        if self.config['prior'] == "ours":
            samples = self.fit_GMM_VI(iterator=iterator, mode="fast", space="t")
            if self.config['representation_size'] == 2:
                self.plot_prior_distribution(samples, mode="crude-GM", style='circle')
                self.plot_prior_distribution(samples, mode="crude-GM", style='density')
            if self.cur_epoch % self.config['accurate_fit'] == 0 or self.cur_epoch == self.config['num_epochs']:
                samples = self.fit_GMM_VI(iterator=iterator, mode="accurate", space="t")
                if self.config['representation_size'] == 2:
                    self.plot_prior_distribution(samples, mode="accurate-GM", style='circle')
                    self.plot_prior_distribution(samples, mode="accurate-GM", style='density')
        elif self.config['prior'] == "GMM":
            if self.cur_epoch < self.config['num_epochs']:
                samples = self.fit_GMM_VI(iterator=iterator, mode="fast", space="z")
                if self.config['code_size'] == 2:
                    self.plot_prior_distribution(samples, mode="crude-GM", style='circle')
                    self.plot_prior_distribution(samples, mode="crude-GM", style='density')
            else:
                samples = self.fit_GMM_VI(iterator=iterator, mode="accurate", space="z")
                if self.config['code_size'] == 2:
                    self.plot_prior_distribution(samples, mode="accurate-GM", style='circle')
                    self.plot_prior_distribution(samples, mode="accurate-GM", style='density')

    def plot_prior_distribution(
        self, samples, 
        mode="crude-GM", style='circle', show_img=False, axis_scale=10):
        if mode == "crude-GM":
            w = self.model.GM_prior_training.weights_
            m = self.model.GM_prior_training.means_
            K = self.model.GM_prior_training.covariances_
        else:
            w = self.GM_prior_final.weights_
            m = self.GM_prior_final.means_
            K = self.GM_prior_final.covariances_
        idx_valid_mixture = np.squeeze(np.argwhere(w >= 1e-2)).tolist()
        fig, axs = plt.subplots(1, 1, figsize=(6, 6), edgecolor='k')
        if style == 'circle':
            axs.scatter(samples[:, 0], samples[:, 1], s=1, c='b')
            for i in idx_valid_mixture:
                self.draw_ellipse(m[i], K[i], weight=w[i])
            axs.set_xlim([-axis_scale, axis_scale])
            axs.set_ylim([-axis_scale, axis_scale])
            axs.set(aspect='equal')
            axs.set_title("Fitting a GMM to a batch of encodings")
            if show_img:
                plt.show()
            savefig(self.config['result_dir'] + 'prior_estimate_circle_{}_{}.pdf'.format(self.cur_epoch, mode))
        elif style == 'density':
            # grid point
            x, y = np.mgrid[-axis_scale:axis_scale:.05, -axis_scale:axis_scale:.05]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            ticks = np.arange(0, axis_scale*20*2, 40)
            labels = tuple(np.arange(-axis_scale, axis_scale, 2))
            for i in idx_valid_mixture:
                rv = multivariate_normal(m[i], K[i])
                if i == idx_valid_mixture[0]:
                    gm_pdf = rv.pdf(pos)
                else:
                    gm_pdf = gm_pdf + rv.pdf(pos)
            gm_pdf = gm_pdf / len(idx_valid_mixture) + 1e-8

            im = axs.imshow(np.log(gm_pdf), cmap='viridis', vmin=-12, vmax=0)
            axs.set_title("Estimate GM prior (log)")
            axs.set_xticks(ticks)
            axs.set_xticklabels(labels)
            axs.set_yticks(ticks)
            axs.set_yticklabels(labels)
            fig.colorbar(im)
            if show_img:
                plt.show()
            savefig(self.config['result_dir'] + "prior_estimate_density_{}_{}.pdf".format(self.cur_epoch, mode))
        fig.clf()
        plt.close()

    def generate_samples_from_prior_by_method(self, mode="crude-GM", n_sample=10, method=None):
        if method is None:
            method = self.config['prior']

        if method == 'standard_gaussian':
            rv = multivariate_normal(np.zeros(self.config['code_size']), np.diag(np.ones(self.config['code_size'])))
            # Generate a batch size of samples from the prior samples
            samples_code_prior = rv.rvs(n_sample ** 2)
            filename = self.config['result_dir'] + 'generated_samples_prior_{}.pdf'.format(self.cur_epoch)
        elif method == 'GMM':
            if mode == "crude-GM":
                samples = self.model.GM_prior_training.sample(n_sample ** 2)
            else:
                samples = self.GM_prior_final.sample(n_sample ** 2)
            filename = self.config['result_dir'] + 'generated_samples_prior_{}_{}.pdf'.format(self.cur_epoch, mode)
            samples_code_prior = samples[0]
        elif method == 'ours':
            if mode == "crude-GM":
                samples = self.model.GM_prior_training.sample(n_sample ** 2)
            else:
                samples = self.GM_prior_final.sample(n_sample ** 2)
            sample_t = samples[0]
            filename = self.config['result_dir'] + 'generated_samples_prior_{}_{}.pdf'.format(self.cur_epoch, mode)
            samples_code_prior = self.sess.run(self.model.decoded_code,
                                               feed_dict={self.model.original_signal: np.zeros((1,
                                                                                                self.config[
                                                                                                    'dim_input_x'],
                                                                                                self.config[
                                                                                                    'dim_input_y'],
                                                                                                self.config[
                                                                                                    'dim_input_channel'])),
                                                          self.model.is_outer_VAE_input: True,
                                                          self.model.customised_inner_VAE_input: np.zeros(
                                                              (1, self.config['code_size'])),
                                                          self.model.representation_input: sample_t,
                                                          self.model.is_representation_input: True})
        elif method == 'hierarchical':
            rv = multivariate_normal(np.zeros(self.config['representation_size']),
                                     np.diag(np.ones(self.config['representation_size'])))
            sample_t = rv.rvs(n_sample ** 2)
            filename = self.config['result_dir'] + 'generated_samples_prior_{}.pdf'.format(self.cur_epoch)
            samples_code_prior = self.sess.run(self.model.decoded_code,
                                               feed_dict={self.model.original_signal: np.zeros((1,
                                                                                                self.config[
                                                                                                    'dim_input_x'],
                                                                                                self.config[
                                                                                                    'dim_input_y'],
                                                                                                self.config[
                                                                                                    'dim_input_channel'])),
                                                          self.model.is_outer_VAE_input: True,
                                                          self.model.customised_inner_VAE_input: np.zeros(
                                                              (1, self.config['code_size'])),
                                                          self.model.representation_input: sample_t,
                                                          self.model.is_representation_input: True})
        elif method == 'vampPrior':
            samples_code_prior = self.sess.run(self.model.psedeu_prior.sample(n_sample ** 2))
            filename = self.config['result_dir'] + 'generated_samples_prior_{}.pdf'.format(self.cur_epoch)
        return samples_code_prior, filename

    def plot_generated_samples_from_prior(self, samples_code_prior, filename, n_sample=10):
        sampled_images = self.sess.run(self.model.decoded,
                                       feed_dict={self.model.original_signal: np.zeros((1,
                                                                                        self.config['dim_input_x'],
                                                                                        self.config['dim_input_y'],
                                                                                        self.config[
                                                                                            'dim_input_channel'])),
                                                  self.model.code_input: samples_code_prior,
                                                  self.model.is_code_input: True})
        sampled_images = np.squeeze(sampled_images)
        fig, axs = plt.subplots(n_sample, n_sample, figsize=(12, 12), edgecolor='k')
        fig.subplots_adjust(hspace=.0, wspace=.0)
        axs = axs.ravel()
        for i in range(n_sample ** 2):
            axs[i].imshow(sampled_images[i])
            axs[i].grid(False)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        plt.suptitle('Prior method: {}'.format(self.config['prior']))
        savefig(filename)
        fig.clf()
        plt.close()

    def generate_samples_from_prior(self):
        if self.config['prior'] == 'ours':
            if self.cur_epoch <= self.config['sg_pretraining']:
                samples, filename = self.generate_samples_from_prior_by_method(method='standard_gaussian')
                self.plot_generated_samples_from_prior(samples, filename)
            else:
                if self.cur_epoch % self.config['accurate_fit'] == 0 or self.cur_epoch == self.config['num_epochs']:
                    samples, filename = self.generate_samples_from_prior_by_method(mode="accurate-GM")
                    self.plot_generated_samples_from_prior(samples, filename)
                else:
                    samples, filename = self.generate_samples_from_prior_by_method(mode="crude-GM")
                    self.plot_generated_samples_from_prior(samples, filename)
        elif self.config['prior'] == 'GMM':
            if self.cur_epoch < self.config['num_epochs']:
                samples, filename = self.generate_samples_from_prior_by_method(mode="crude-GM")
                self.plot_generated_samples_from_prior(samples, filename)
            else:
                samples, filename = self.generate_samples_from_prior_by_method(mode="accurate-GM")
                self.plot_generated_samples_from_prior(samples, filename)
        else:
            samples, filename = self.generate_samples_from_prior_by_method()
            self.plot_generated_samples_from_prior(samples, filename)

    def plot_train_and_val_loss(self, model_to_train):
        # plot the training and validation loss over epochs
        if model_to_train == "VAE":
            fig, axs = plt.subplots(1, 1, figsize=(8, 6), edgecolor='k')
            axs.plot(self.train_loss, 'b-')
            axs.plot(self.iter_epochs_list, self.val_loss_ave_epoch, 'r-')
            axs.legend(('training loss (total)', 'validation loss'))
            axs.set_title('Negative ELBO over iterations (val @ epochs)')
            axs.set_ylabel('total loss')
            axs.set_xlabel('iterations')
            axs.set_xlim([0, len(self.train_loss)])
            axs.grid(True)
            savefig(self.config['result_dir'] + '/loss-elbo.pdf')
            plt.close()

            # plot individual components of validation loss over epochs
            fig, axs = plt.subplots(1, 4, figsize=(14, 2), edgecolor='k')
            fig.subplots_adjust(hspace=.4, wspace=.4)
            axs = axs.ravel()
            axs[0].plot(self.recons_error_val, 'b-')
            axs[0].set_xlim([0, len(self.recons_error_val)])
            axs[0].set_title("Reconstruction error")
            axs[0].grid(True)
            axs[1].plot(self.entropy_z_val, 'b-')
            axs[1].set_xlim([0, len(self.entropy_z_val)])
            axs[1].set_title("Entropy q(z|x)")
            axs[1].grid(True)
            axs[2].plot(self.crossEntropy_prior_val, 'b-')
            axs[2].set_xlim([0, len(self.crossEntropy_prior_val)])
            axs[2].set_title("Cross entropy q(z|x) || p(z)")
            axs[2].grid(True)
            axs[3].plot(self.elbo_val, 'b-')
            axs[3].set_xlim([0, len(self.elbo_val)])
            axs[3].set_title("ELBO")
            axs[3].grid(True)
            # plt.suptitle("Outer VAE val losses")
            savefig(self.config['result_dir'] + '/loss-outer-VAE-val.pdf')
            plt.close()

            # plot sigma2 over epochs
            if self.config['TRAIN_sigma'] == 1:
                figure(num=1, figsize=(8, 6))
                plot(self.test_sigma, 'b-')
                plt.title('scale parameter over training')
                plt.ylabel('sigma')
                plt.xlabel('epoch (zero index)')
                plt.ylim([0, self.config['sigma']])
                plt.xlim([0, len(self.test_sigma)])
                plt.grid(True)
                savefig(self.config['result_dir'] + '/sigma.pdf')
                plt.close()

        # plot inner VAE training loss
        if model_to_train == "prior":
            if self.config['prior'] in ["ours", "hierarchical"]:
                fig, axs = plt.subplots(2, 4, figsize=(16, 5), edgecolor='k')
                fig.subplots_adjust(hspace=.4, wspace=.4)
                axs = axs.ravel()
                axs[0].plot(self.code_recons_error_train, 'b-')
                axs[0].set_xlim([0, len(self.code_recons_error_train)])
                axs[0].set_title("Reconstruction error")
                axs[0].grid(True)
                axs[1].plot(self.entropy_t_train, 'b-')
                axs[1].set_xlim([0, len(self.entropy_t_train)])
                axs[1].set_title("Entropy q(t|z)")
                axs[1].grid(True)
                axs[2].plot(self.crossEntropy_t_train, 'b-')
                axs[2].set_xlim([0, len(self.crossEntropy_t_train)])
                axs[2].set_title("Cross entropy q(t|z) || p(t)")
                axs[2].grid(True)
                axs[3].plot(self.code_elbo_train, 'b-')
                axs[3].set_xlim([0, len(self.code_elbo_train)])
                axs[3].set_title("ELBO")
                axs[3].grid(True)
                axs[4].plot(self.code_recons_likelihood_train, 'b-')
                axs[4].set_xlim([0, len(self.code_recons_likelihood_train)])
                axs[4].set_title("Reconstruction likelihood")
                axs[4].grid(True)
                axs[5].plot(self.code_inner_sigma_train, 'b-')
                axs[5].set_xlim([0, len(self.code_inner_sigma_train)])
                axs[5].set_title("Inner VAE sigma")
                axs[5].grid(True)
                axs[6].axis('off')
                axs[7].axis('off')
                plt.suptitle("Inner VAE losses")
                savefig(self.config['result_dir'] + '/loss-inner-VAE.pdf')
                plt.close()
            else:
                fig, axs = plt.subplots(1, 2, figsize=(8, 2), edgecolor='k')
                fig.subplots_adjust(hspace=.4, wspace=.4)
                axs = axs.ravel()
                axs[0].plot(self.train_loss_prior, 'b-')
                axs[0].set_xlim([0, len(self.train_loss_prior)])
                axs[0].set_title("Prior loss")
                axs[0].grid(True)
                axs[1].plot(self.vampPrior_crossEntropy_prior_train, 'b-')
                axs[1].set_xlim([0, len(self.vampPrior_crossEntropy_prior_train)])
                axs[1].set_title("Cross entropy q(z|x) || p(z)")
                axs[1].grid(True)
                plt.suptitle("VampPrior prior loss")
                savefig(self.config['result_dir'] + '/vampPrior-prior-loss.pdf')
                plt.close()