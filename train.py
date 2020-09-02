import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append("..")

import tensorflow as tf
from codes.data_loader import DataGenerator
from codes.models import MNISTModel_digit, MNISTModel_fashion, CelebAModel_densenet
from codes.trainers import MNISTTrainer_joint_training, CelebATrainer_joint_training
from codes.utils import process_config, create_dirs, get_args, save_config


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir']])

    # save the config in a txt file
    save_config(config)

    # create tensorflow session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # create your data loader
    data = DataGenerator(config, sess)

    # create VAE models
    if config['exp_name'] == 'mnist_digit':
        model = MNISTModel_digit(config)
    elif config['exp_name'] == 'mnist_fashion':
        model = MNISTModel_fashion(config)
    elif config['exp_name'] == 'celeba':
        model = CelebAModel_densenet(config)
    print("Created a VAE model.")
    print("The current dataset is {}, num hidden units: {}.\n".format(
        config['exp_name'],
        config['num_hidden_units']))

    # here you train your model
    if config['TRAIN_VAE'] or config['TRAIN_sigma'] or config['TRAIN_prior']:
        if config['exp_name'] == 'mnist_digit' or config['exp_name'] == 'mnist_fashion':
            trainer_VAE = MNISTTrainer_joint_training(sess, model, data, config)
        elif config['exp_name'] == 'celeba':
            trainer_VAE = CelebATrainer_joint_training(sess, model, data, config)

        # load model if exists: after the init in creating trainer
        model.load(sess, model="VAE")

        if config['prior'] in ["ours", "hierarchical", "vampPrior"]:
            model.load(sess, model="prior")

        # start training
        if config['num_epochs'] > 0:
            trainer_VAE.train()


if __name__ == '__main__':
    main()