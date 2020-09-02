""" util function.py """

import os
import argparse
import tensorflow as tf
import numpy as np
import json
from datetime import datetime


def get_config_from_json(json_file):
    """
        Get the config from a json file
        param: json_file;
        return: config(dictionary).
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict


def save_config(config):
    """
        Save the current config dict in a txt file for future reference.
        param: python dict;
        return: None.
    """
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")
    filename = config['checkpoint_dir'] + 'training_config_{}.txt'.format(timestampStr)
    config_to_save = json.dumps(config)
    f = open(filename, "w")
    f.write(config_to_save)
    f.close()
    print('The current config is saved at {}'.format(filename))


def process_config(json_file):
    """
        Process the config json file and create various directories to save experiment results.
        param: json_file;
        return: config(dictionary).
    """
    config = get_config_from_json(json_file)
    print("The current config is:\n{}\n".format(config))

    save_name = 'prior-{}-{}-{}-{}-{}-{}-mixture-{}'.format(
        config['prior'],
        config['num_hidden_units'],
        config['code_size'],
        config['representation_size'],
        config['inner_activation'],
        config['n_layers_inner_VAE'],
        config['n_mixtures'])
    print("Experiment results will be saved at:\n{}\n".format(save_name))

    # if config['exp_name'] == "celebA":
    #     config['data_path'] = '/home/shuyu/Documents/dataset/celeba/'

    if config['load_dir'] == "default":
        save_dir = "./experiments/{}/batch-{}".format(
            config['exp_name'],
            config['batch_size'])
        config['summary_dir'] = os.path.join(save_dir, save_name, "summary/")
        config['result_dir'] = os.path.join(save_dir, save_name, "result/")
        config['checkpoint_dir'] = os.path.join(save_dir, save_name, "checkpoint/")
    else:
        save_dir = config['load_dir']
        config['summary_dir'] = "./figures/{}/summary/".format(config['exp_name'])
        config['result_dir'] = "./figures/{}/result/".format(config['exp_name'])
        config['checkpoint_dir'] = os.path.join(save_dir, config['exp_name'])
    print("Models will be saved / loaded at:\n{}".format(config['checkpoint_dir']))
    print("Results will be saved at:\n{}\n".format(config['result_dir']))

    return config


def create_dirs(dirs):
    """
        dirs - a list of directories to create if these directories are not found
        param: dirs;
        return: exit_code: 0 - success;  -1 - failed.
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def count_trainable_variables(scope_name):
    """
        compute the total number of trainable parameters in a network defined in the scope_name.
        param: scope name of a network;
        return: an integer indicating the total number of trainable parameters.
    """
    total_parameters = 0
    for variable in tf.trainable_variables(scope_name):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(
        'The total number of trainable parameters in the {} model is: {}k.'.format(
            scope_name, np.around(total_parameters/1000, 2)))
    return total_parameters


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def preprocess_input_original(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def preprocess_input_generated(x):
    x = np.clip(x, 0., 1.)
    x -= 0.5
    x *= 2.
    return x


def compute_FID_score(data_file1, data_file2, FID_network, pooling_option, second_set='generated'):
    dataset1 = np.load(data_file1)
    dataset1 = dataset1['sampled_images']
    dataset1 = dataset1.astype(np.float32)
    dataset1 = preprocess_input_original(dataset1)

    dataset2 = np.load(data_file2)
    dataset2 = dataset2['sampled_images']
    dataset2 = dataset2.astype(np.float32)
    if second_set == 'generated':
        dataset2 = preprocess_input_generated(dataset2)
    else:
        dataset2 = preprocess_input_original(dataset2)

    sess = tf.compat.v1.InteractiveSession()
    dataset1 = tf.image.resize_images(dataset1, (64, 64))
    dataset2 = tf.image.resize_images(dataset2, (64, 64))
    dataset1 = dataset1.eval()
    dataset2 = dataset2.eval()
    sess.close()

    if FID_network == "inception":
        inception_v3_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(140, 140, 3),
            pooling=pooling_option)
        inception_v3_model.trainable = False
        inception_v3_model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                                   loss='mse',  # mean squared error
                                   metrics=['mae'])

        sess = tf.compat.v1.InteractiveSession()
        resized_input1 = tf.image.resize_images(dataset1, (140, 140))
        resized_input2 = tf.image.resize_images(dataset2, (140, 140))
        resized_input1 = resized_input1.eval()
        resized_input2 = resized_input2.eval()
        sess.close()

        act1 = inception_v3_model.predict(resized_input1, batch_size=100)
        act2 = inception_v3_model.predict(resized_input2, batch_size=100)

    elif FID_network == "VGG":
        vgg16_model = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(128, 128, 3),
            pooling=pooling_option)
        vgg16_model.trainable = False
        vgg16_model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                            loss='mse',  # mean squared error
                            metrics=['mae'])
        act1 = vgg16_model.predict(dataset1, batch_size=100)
        act2 = vgg16_model.predict(dataset2, batch_size=100)

    sess = tf.Session()
    FID_score = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(
        tf.constant(act1),
        tf.constant(act2))
    print("FID score between {} and {} is:\n{}".format(data_file1, data_file2, sess.run(FID_score)))