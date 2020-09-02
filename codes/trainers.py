# import matplotlib
# matplotlib.use('Agg')

from codes.base import BaseTrain, BaseTrain_joint
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, savefig, figure
import time
from tqdm import tqdm


class MNISTTrainer_joint_training(BaseTrain_joint):
    def __init__(self, sess, model, data, config):
        super().__init__(sess, model, data, config)
        self.test_batch = self.data.test_set['image']
        self.n_train_iter = self.data.n_train // self.config['batch_size']
        self.n_val_iter = self.data.n_val // self.config['batch_size']
        self.idx_check_point = np.arange(0, self.n_train_iter - 1, self.n_train_iter // self.config['num_iter_to_plot'])

        # plot the ground truth test set
        self.plot_ground_truth_test_set()

    def train_epoch(self):
        self.cur_epoch = self.cur_epoch + 1
        print("{}/{}:".format(self.cur_epoch, self.config['num_epochs']))
        self.sess.run(self.model.iterator.initializer,
                      feed_dict={self.model.original_signal: self.data.train_set['image'],
                                 self.model.seed: self.cur_epoch})
        train_loss_cur_epoch = 0.0
        self.cur_lr = self.config['learning_rate_ae'] * (0.99 ** (self.cur_epoch - 1))

        # train outer VAE model parameters
        for i in range(self.n_train_iter):
            training_batch = self.sess.run(self.model.input_image)
            if self.config['TRAIN_VAE'] == 1:
                loss = self.train_step_ae(cur_lr=self.cur_lr, batch_data=training_batch)
                self.train_loss.append(np.squeeze(loss))
                train_loss_cur_epoch = train_loss_cur_epoch + loss
            if self.cur_epoch > self.config['sg_pretraining']-1 and self.config['prior'] in ['ours', 'hierarchical', 'vampPrior'] and self.config['TRAIN_prior'] == 1:
                self.train_step_prior(batch_data=training_batch)

        if self.config['TRAIN_VAE'] == 1:
            self.train_loss_ave_epoch.append(train_loss_cur_epoch / self.n_train_iter)
            self.iter_epochs_list.append(len(self.train_loss) - 1)

        # fit a GM in representation or code space
        if self.cur_epoch > self.config['sg_pretraining']-1 and self.config['prior'] in ["ours", "GMM"]:
            self.fit_GM(iterator=self.model.input_image)

        # generate samples from prior
        self.generate_samples_from_prior()

        # print a few losses to inspect in the test_step
        self.test_step(batch_data=self.test_batch, print_result=True)

        # validation
        self.sess.run(self.model.iterator.initializer,
                      feed_dict={self.model.original_signal: self.data.val_set['image'],
                                 self.model.seed: self.cur_epoch})
        val_loss_cur_epoch = 0.0
        for i in range(self.n_val_iter):
            val_batch = self.sess.run(self.model.input_image)
            val_loss = self.val_step(batch_data=val_batch, model_to_train="VAE")
            if self.cur_epoch > self.config['sg_pretraining']-1 and self.config['prior'] in ['ours', 'hierarchical', 'vampPrior']:
                _ = self.val_step(batch_data=val_batch, model_to_train="prior")
            val_loss_cur_epoch = val_loss_cur_epoch + val_loss
        self.val_loss_ave_epoch.append(val_loss_cur_epoch / self.n_val_iter)
        if self.config['TRAIN_VAE'] == 1:
            print("Average overall negative ELBO loss:\ntrain: {:.4f}, val: {:.4f}".format(
                self.train_loss_ave_epoch[self.cur_epoch - 1],
                self.val_loss_ave_epoch[self.cur_epoch - 1]))

        # reconstruction plot
        self.plot_reconstructed_data(self.output_test, title=False, narrow_space=True)

        # Use the baseTrain function to save all the training results to keep record
        self.save_variables_VAE()

        # plot the training and validation loss over iterations/epochs
        if self.config['TRAIN_VAE'] == 1:
            self.plot_train_and_val_loss(model_to_train="VAE")
        if self.cur_epoch > self.config['sg_pretraining'] and self.config['prior'] in ['ours', 'hierarchical', 'vampPrior'] and self.config['TRAIN_prior'] == 1:
            self.plot_train_and_val_loss(model_to_train="prior")

    def plot_reconstructed_data(self, images, save_name=None, title=True, narrow_space=False):
        images = np.squeeze(images)
        n_images = images.shape[0]
        for j in range(n_images // 64):
            # plot the reconstructed image for a shape
            fig, axs = plt.subplots(8, 8, figsize=(12, 14), edgecolor='k')
            if narrow_space:
                fig.subplots_adjust(hspace=0., wspace=0.)
            else:
                fig.subplots_adjust(hspace=.4, wspace=.4)
            axs = axs.ravel()
            for i in range(8 * 8):
                axs[i].imshow(images[i + 64 * j], vmin=0., vmax=1.)
                axs[i].grid(False)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                if title:
                    axs[i].set_title("{}".format(
                        self.data.test_set['attrib'][i + 64 * j]))
                # plt.tight_layout()
            if save_name:
                savefig(self.config['result_dir'] + save_name + '_{}_{}.pdf'.format(self.cur_epoch, j))
            else:
                savefig(self.config['result_dir'] + 'test_reconstructed_{}_{}.pdf'.format(self.cur_epoch, j))
            fig.clf()
            plt.close()

    def plot_ground_truth_test_set(self):
        image = np.squeeze(self.data.test_set['image'])
        for j in range(self.config['batch_size'] // 64):
            # plot the ground truth image
            fig, axs = plt.subplots(8, 8, figsize=(12, 14), edgecolor='k')
            fig.subplots_adjust(hspace=0., wspace=0.)
            axs = axs.ravel()
            for i in range(8 * 8):
                axs[i].imshow(image[i + 64 * j], vmin=0., vmax=1.)
                axs[i].grid(False)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            savefig(self.config['result_dir'] +
                    '/test_original_{}.pdf'.format(j))
            fig.clf()
            plt.close()


class CelebATrainer_joint_training(BaseTrain_joint):
    def __init__(self, sess, model, data, config):
        super().__init__(sess, model, data, config)
        # Generate the test batch
        self.sess.run(self.model.iterator_test.initializer,
                      feed_dict={self.model.data_file: self.config['data_path'] + 'celebA_test.tfrecords'})
        self.test_batch = self.sess.run(self.model.test_image)
        self.n_train_iter = self.data.n_train // self.config['batch_size']
        self.n_val_iter = self.data.n_val // self.config['batch_size']
        self.idx_check_point = np.arange(0, self.n_train_iter-1, self.n_train_iter//self.config['num_iter_to_plot'])

    def train_epoch(self):
        self.cur_epoch = self.cur_epoch + 1
        print('Training epoch: {}/{}'.format(self.cur_epoch, self.config['num_epochs']))
        self.sess.run(self.model.iterator.initializer,
                      feed_dict={self.model.data_file: self.config['data_path']+'celebA_train.tfrecords'})
        train_loss_cur_epoch = 0.0
        self.compute_cur_lr()
        for i in tqdm(range(self.n_train_iter)):
            training_batch = self.sess.run(self.model.input_image)
            if self.config['TRAIN_VAE'] == 1:
                loss = self.train_step_ae(cur_lr=self.cur_lr, batch_data=training_batch)
                self.train_loss.append(np.squeeze(loss))
                train_loss_cur_epoch = train_loss_cur_epoch + loss
            if self.cur_epoch > self.config['sg_pretraining']-1 and self.config['prior'] in ['ours', 'hierarchical', 'vampPrior'] and self.config['TRAIN_prior'] == 1:
                self.train_step_prior(batch_data=training_batch)
            if self.config['num_iter_to_plot'] > 1 and np.any(self.idx_check_point == i):
                self.test_step(batch_data=self.test_batch, print_result=False)
                self.plot_reconstructed_image(i, self.output_test, self.test_batch)

        if self.config['TRAIN_VAE'] == 1:
            self.train_loss_ave_epoch.append(train_loss_cur_epoch/self.n_train_iter)
            self.iter_epochs_list.append(len(self.train_loss)-1)

        # fit a GM in representation or code space
        if self.cur_epoch > self.config['sg_pretraining']-1 and self.config['prior'] in ["ours", "GMM"]:
            self.fit_GM(iterator=self.model.input_image)

        # generate samples from prior
        self.generate_samples_from_prior()

        # print a few losses to inspect in the test_step
        self.test_step(batch_data=self.test_batch, print_result=True)

        # validation
        self.sess.run(self.model.iterator.initializer,
                      feed_dict={self.model.data_file: self.config['data_path']+'celebA_val.tfrecords'})
        val_loss_cur_epoch = 0.0
        for i in range(self.n_val_iter):
            val_batch = self.sess.run(self.model.input_image)
            if self.config['TRAIN_VAE'] == 1:
                val_loss = self.val_step(batch_data=val_batch, model_to_train="VAE")
                val_loss_cur_epoch = val_loss_cur_epoch + val_loss
            if self.cur_epoch > self.config['sg_pretraining']-1 and self.config['TRAIN_prior'] == 1 and self.config['prior'] in ['ours', 'hierarchical', 'vampPrior']:
                _ = self.val_step(batch_data=val_batch, model_to_train="prior")
        self.val_loss_ave_epoch.append(val_loss_cur_epoch/self.n_val_iter)
        if self.config['TRAIN_VAE'] == 1:
            print("Average:\ntrain: {:.4f}, val: {:.4f}".format(
                self.train_loss_ave_epoch[self.cur_epoch-1],
                self.val_loss_ave_epoch[self.cur_epoch-1]))

        # Use the baseTrain function to save all the training results to keep record
        self.save_variables_VAE()

        # plot the training and validation loss over iterations/epochs
        if self.config['TRAIN_VAE'] == 1:
            self.plot_train_and_val_loss(model_to_train="VAE")
        if self.config['TRAIN_prior'] == 1 and self.config['prior'] in ['ours', 'hierarchical', 'vampPrior']:
            self.plot_train_and_val_loss(model_to_train="prior")

    def compute_cur_lr(self):
        if self.cur_epoch <= 25:
            cur_lr = self.config['learning_rate_ae'] * (0.99 ** (self.cur_epoch - 1))
        elif self.cur_epoch <= 50:
            cur_lr = self.config['learning_rate_ae'] / 2 * (0.99 ** (self.cur_epoch - 25))
        elif self.cur_epoch <= 75:
            cur_lr = self.config['learning_rate_ae'] / 5 * (0.99 ** (self.cur_epoch - 50))
        else:
            cur_lr = self.config['learning_rate_ae'] / 10 * (0.99 ** (self.cur_epoch - 75))
        self.cur_lr = cur_lr

    def plot_reconstructed_image(self, idx_iter, images, gt_images, save_name=None):
        if self.config['batch_size'] > 64:
            n_images = 64
        else:
            n_images = self.config['batch_size']
        for j in range(self.config['batch_size'] // n_images):
            if n_images // 8 > 4:
                fig, axs = plt.subplots(8, 8, figsize=(16, 18), edgecolor='k')
                fig.subplots_adjust(hspace=0.0, wspace=0.0)
                axs = axs.ravel()
                for i in range(8):
                    for k in range(8):
                        if i % 2 == 0:
                            axs[k+i*8].imshow(gt_images[k+i*8//2+n_images*j])
                        else:
                            axs[k+i*8].imshow(images[k+(i-1)*8//2+n_images*j])
                        axs[k+i*8].grid(False)
                        axs[k+i*8].set_xticks([])
                        axs[k+i*8].set_yticks([])
            else:
                fig, axs = plt.subplots((n_images // 8)*2, 8, figsize=(16, 18), edgecolor='k')
                fig.subplots_adjust(hspace=0.0, wspace=0.0)
                axs = axs.ravel()
                for i in range((n_images // 8)*2):
                    for k in range(8):
                        if i % 2 == 0:
                            axs[k+i*8].imshow(gt_images[k+i*8//2+n_images*j])
                        else:
                            axs[k+i*8].imshow(images[k+(i-1)*8//2+n_images*j])
                        axs[k+i*8].grid(False)
                        axs[k+i*8].set_xticks([])
                        axs[k+i*8].set_yticks([])
            if save_name:
                savefig(self.config['result_dir'] + save_name +'_{}_{}_{}.pdf'.format(self.cur_epoch, idx_iter, j))
            else:
                savefig(self.config['result_dir'] + 'test_reconstructed_{}_{}_{}.pdf'.format(self.cur_epoch, idx_iter, j))
            fig.clf()
            plt.close()

