import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, savefig
import matplotlib._color_data as mcd

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def plot_images_and_its_reconstruction(x, x_decoded, config,
                                       x_from_t=None, save_plot=False, idx=0):
    save_dir = config['result_dir']
    matplotlib.rcParams.update({'font.size': 12})
    if config['prior'] in ['ours', 'hierarchical']:
        fig, axs = plt.subplots(1, 3, figsize=(6, 2), edgecolor='k')
    else:
        fig, axs = plt.subplots(1, 2, figsize=(4, 2), edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.4)
    axs = axs.ravel()
    axs[0].imshow(np.squeeze(x))
    axs[0].set_title('original')
    axs[1].imshow(np.squeeze(x_decoded))
    axs[1].set_title('decoded from z')
    for i in range(2):
        axs[i].grid(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    if config['prior'] in ['ours', 'hierarchical']:
        axs[2].imshow(np.squeeze(x_from_t))
        axs[2].set_title('decoded from t')
        axs[2].grid(False)
        axs[2].set_xticks([])
        axs[2].set_yticks([])
    if save_plot:
        savefig(save_dir + 'original_image_{}.pdf'.format(idx))
    show()


def get_embeddings_from_val_set(idx, config, exp_name, sess, data, model, trainer,
                                save_plot=False):
    if exp_name == 'mnist_digit':
        x = data.val_set['image']
    else:
        x = trainer.test_batch
    if config['prior'] in ['ours', 'hierarchical']:
        feed_dict = {model.original_signal: x,
                     model.is_code_input: False,
                     model.code_input: np.zeros((1, config['code_size'])),
                     model.is_outer_VAE_input: True,
                     model.customised_inner_VAE_input: np.zeros((1, config['code_size'])),
                     model.is_representation_input: False,
                     model.representation_input: np.zeros((1, config['representation_size']))}
        embedding, x_decoded = sess.run([model.representation_mean, model.decoded], feed_dict=feed_dict)

        feed_dict[model.is_representation_input] = True
        feed_dict[model.representation_input] = embedding
        z_decoded = sess.run(model.decoded_code, feed_dict=feed_dict)
        x_decoded = np.clip(x_decoded, 0., 1.)

        feed_dict = {model.original_signal: x,
                     model.is_code_input: True,
                     model.code_input: z_decoded}
        x_from_t = sess.run(model.decoded, feed_dict=feed_dict)
        x_from_t = np.clip(x_from_t, 0., 1.)
        plot_images_and_its_reconstruction(x[idx], x_decoded[idx], config, x_from_t[idx], save_plot=save_plot, idx=idx)
    else:
        feed_dict = {model.original_signal: x,
                     model.is_code_input: False,
                     model.code_input: np.zeros((1, config['code_size']))}
        embedding, x_decoded = sess.run([model.code_mean, model.decoded], feed_dict=feed_dict)
        x_decoded = np.clip(x_decoded, 0., 1.)
        plot_images_and_its_reconstruction(x[idx], x_decoded[idx], config, save_plot=save_plot, idx=idx)
    return np.squeeze(embedding[idx])


# a function to generate a batch of embeddings from prior
def define_prior_distribution(config, sess, model, gmm_info=None):
    if config['prior'] == 'standard_gaussian':
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(config['code_size'], dtype=np.float32),
                                           scale_diag=np.ones(config['code_size'], dtype=np.float32))
        return prior
    elif config['prior'] in ['GMM', 'ours']:
        n_mixtures = len(gmm_info['w'])
        mixtures = []
        for i in range(n_mixtures):
            single_mixture = tfd.MultivariateNormalFullCovariance(
                loc=gmm_info['m'][i],
                covariance_matrix=gmm_info['K'][i])
            mixtures.append(single_mixture)
        prior_GM = tfd.Mixture(cat=tfd.Categorical(probs=gmm_info['w']),
                               components=mixtures)
        return prior_GM
    elif config['prior'] == 'hierarchical':
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(config['representation_size'], dtype=np.float32),
                                           scale_diag=np.ones(config['representation_size'], dtype=np.float32))
        return prior
    elif config['prior'] == 'vampPrior':
        feed_dict = {model.original_signal: sess.run(model.psedeu_input),
                     model.code_input: np.zeros((1, config['code_size'])),
                     model.is_code_input: False}
        prior_mean, prior_std = sess.run([model.code_mean, model.code_std_dev], feed_dict=feed_dict)

        n_mixtures = config['n_mixtures']
        prior_weight = tf.constant(1 / n_mixtures, shape=[n_mixtures], dtype=tf.float32)
        mixtures = []
        for i in range(n_mixtures):
            single_mixture = tfd.MultivariateNormalDiag(
                loc=prior_mean[i],
                scale_diag=prior_std[i])
            mixtures.append(single_mixture)
        prior_GM = tfd.Mixture(cat=tfd.Categorical(prior_weight),
                               components=mixtures)
        return prior_GM


def generate_prior_embeddings(prior, sess, n_embeddings):
    samples_prior = sess.run(prior.sample(n_embeddings))
    return samples_prior


def plot_interpolation_losses(loss, path_length_record, step_var_record, neg_ll_record, n_iter,
                              idx_start, idx_end, n_step, config):
    fig, axs = plt.subplots(1, 4, figsize=(15, 2.5), edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.4)
    axs = axs.ravel()
    axs[0].plot(loss)
    axs[0].set_title('Overall loss')
    axs[0].grid(True)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[1].plot(path_length_record, label='SLP', lw=2)
    axs[1].axhline(y=path_length_record[0], xmin=0, xmax=n_iter, color='r', ls='--', lw=2, label='SP')
    axs[1].set_title('Path length')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlabel('Iteration')
    axs[2].plot(step_var_record, lw=2)
    axs[2].axhline(y=step_var_record[0], xmin=0, xmax=n_iter, color='r', lw=2, ls='--')
    axs[2].set_title('Step variance')
    axs[2].grid(True)
    axs[2].set_xlabel('Iteration')
    axs[3].plot(neg_ll_record, lw=2)
    axs[3].axhline(y=neg_ll_record[0], xmin=0, xmax=n_iter, color='r', lw=2, ls='--')
    axs[3].set_title('Negative LL')
    axs[3].grid(True)
    axs[3].set_xlabel('Iteration')
    for i in range(4):
        axs[i].set_xlim(0, n_iter)

    if config['prior'] in ['ours', 'hierarchical']:
        dim = config['representation_size']
    else:
        dim = config['code_size']
    savefig(config['result_dir'] + 'loss_image{}-{}_{}_zdim_{}_nstep_{}.pdf'.format(
        idx_start, idx_end,
        config['prior'], dim, n_step))
    show()


# plot the interpolated images along the path
def plot_interpolated_images(
    interpolated_embeddings, config, exp_name, sess, data, model, trainer,
    n_step, idx_start, idx_end, font_size=16, save_plot=False, name_input=''):
    save_dir = config['result_dir']
    if "mnist" in exp_name:
        x = np.expand_dims(data.val_set['image'][0], axis=0)
    elif exp_name == "celeba":
        x = np.expand_dims(trainer.test_batch[0], axis=0)
    if config['prior'] in ['ours', 'hierarchical']:
        feed_dict = {model.original_signal: x,
                     model.is_code_input: False,
                     model.code_input: np.zeros((1, config['code_size'])),
                     model.is_outer_VAE_input: True,
                     model.customised_inner_VAE_input: np.zeros((1, config['code_size'])),
                     model.is_representation_input: True,
                     model.representation_input: interpolated_embeddings}
        z_decoded = sess.run(model.decoded_code, feed_dict=feed_dict)
    else:
        z_decoded = interpolated_embeddings
    feed_dict = {model.original_signal: x,
                 model.is_code_input: True,
                 model.code_input: z_decoded}
    interpolated_images = sess.run(model.decoded, feed_dict=feed_dict)
    interpolated_images = np.clip(interpolated_images, 0., 1.)

    matplotlib.rcParams.update({'font.size': font_size})
    fig, axs = plt.subplots(1, n_step + 2, figsize=(2 * n_step, 2), edgecolor='k')
    fig.subplots_adjust(hspace=.0, wspace=.0)
    axs = axs.ravel()

    axs[0].set_title('Start')
    axs[n_step + 1].set_title('Target')

    for i in range(n_step + 2):
        axs[i].imshow(np.squeeze(interpolated_images[i]))
        axs[i].grid(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if i >= 1 and i <= n_step:
            axs[i].set_title("Step {}".format(i))

    if save_plot:
        if config['prior'] in ['ours', 'hierarchical']:
            dim = config['representation_size']
        else:
            dim = config['code_size']
        savefig(save_dir + 'interpolated_image{}-{}_{}_zdim_{}_nstep_{}_{}.pdf'.format(
            idx_start, idx_end,
            config['prior'], dim, n_step, name_input))
    show()


# plot the path
def plot_optimised_path(cur_pts, config, GM, sess, model, trainer,
                        embedding_start, embedding_end, idx_start, idx_end, n_step,
                        prior=None, plot_prior='circle', w=2.,
                        save_plot=False, font_size=18, grid_size=8., name_input='', c='b'):
    save_dir = config['result_dir']

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), edgecolor='k')
    matplotlib.rcParams.update({'font.size': font_size})
    if plot_prior == 'circle':
        if config['prior'] in ['ours', 'GMM']:
            n_mixture = len(GM['w'])
            for i in range(n_mixture):
                trainer.draw_ellipse(GM['m'][i], GM['K'][i], GM['w'][i]*w,
                                     ax=axs, color='k')
        elif config['prior'] == 'vampPrior':
            feed_dict = {model.original_signal: sess.run(model.psedeu_input),
                         model.is_code_input: False,
                         model.code_input: np.zeros((1, config['code_size']))}
            GM_prior_mean, GM_prior_K_diag = sess.run([model.code_mean, model.code_std_dev], feed_dict=feed_dict)
            n_mixture = config['n_mixtures']
            for i in range(n_mixture):
                trainer.draw_ellipse(GM_prior_mean[i],
                                     np.diag(GM_prior_K_diag[i]),
                                     1/n_mixture*w,
                                     ax=axs, color='k')
        elif config['prior'] in ['standard_gaussian', 'hierarchical']:
            trainer.draw_ellipse(np.zeros(config['code_size']), np.diag(np.ones(config['code_size'])), 1.,
                                     ax=axs, color='k')
    elif plot_prior == 'density':
        # grid point
        x, y = np.mgrid[-grid_size:grid_size:.05, -grid_size:grid_size:.05]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        ticks = np.arange(0, grid_size//0.05*2, 4//0.05+1)
        labels = list(np.arange(-grid_size,grid_size,4))

        # prior
#         if config['prior'] == 'standard_gaussian':
        prior_pdf = sess.run(prior.prob(pos)) + 1e-8
#         elif config['prior'] in ['GMM', 'ours']:
#             prior_pdf = sess.run(prior.prob(pos)) + 1e-8
#         elif config['prior'] == 'vampPrior':
#             prior_pdf = sess.run(prior.prob(pos)) + 1e-8
        im = axs.imshow(np.log(prior_pdf), cmap='viridis', vmin=-14, vmax=0)
#         axs.set_title("aggregate posteriors (log)")
        axs.set_xticks(ticks)
        axs.set_xticklabels(labels)
        axs.set_yticks(ticks)
        axs.set_yticklabels(labels)
        fig.colorbar(im)

    pts_start = np.concatenate([np.expand_dims(embedding_start, 0), cur_pts], axis=0)
    pts_start = (pts_start + grid_size) // 0.05
    pts_end = np.concatenate([cur_pts, np.expand_dims(embedding_end, 0)], axis=0)
    pts_end = (pts_end + grid_size) // 0.05
    for i in range(n_step+1):
        axs.plot([pts_start[i, 1], pts_end[i, 1]], [pts_start[i, 0], pts_end[i, 0]], '-', color=c, lw=4, zorder=1)
    axs.plot(pts_start[1:, 1], pts_start[1:, 0], '.', color=c, ms=15, zorder=50, label='Interpolation')
    axs.scatter(pts_start[0,1], pts_start[0,0], c=mcd.CSS4_COLORS['beige'], s=80, label='Start', zorder=120)
    axs.scatter(pts_end[-1,1], pts_end[-1,0], c=mcd.CSS4_COLORS['orangered'], s=80, label='Target', zorder=120)
    axs.legend()
    plt.title('interpolation method: {}'.format(name_input))
    if save_plot:
        if config['prior'] in ['ours', 'hierarchical']:
            dim = config['representation_size']
        else:
            dim = config['code_size']
        savefig(save_dir + 'interpolated_path{}-{}_{}_zdim_{}_nstep_{}_{}.pdf'.format(
            idx_start, idx_end,
            config['prior'], dim,
            n_step, name_input))
    show()