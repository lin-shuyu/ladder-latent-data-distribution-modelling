import numpy as np
import tensorflow as tf

from codes.base import BaseDataGenerator


class DataGenerator(BaseDataGenerator):
    def __init__(self, config, sess):
        super(DataGenerator, self).__init__(config, sess)
        # load data here: generate 3 state variables: train_set, val_set and test_set
        if self.config['exp_name'] == 'mnist_digit':
            self.load_MNIST_dataset('digit')
        elif self.config['exp_name'] == 'mnist_fashion':
            self.load_MNIST_dataset('fashion')
        elif self.config['exp_name'] == 'celeba':
            self.n_train = 180000
            self.n_val = 20000

    def load_MNIST_dataset(self, choice):
        if choice == 'digit':
            mnist = tf.keras.datasets.mnist
        else:
            mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        self.n_train = x_train.shape[0]
        self.n_val = x_test.shape[0]
        self.train_set = dict(
            attrib=y_train,
            image=np.expand_dims(x_train, -1))
        self.val_set = dict(
            attrib=y_test,
            image=np.expand_dims(x_test, -1))

        # prepare the test set - a batch of randomly chosen digits from validation set;
        # each digit has either 6/7 instances (even across different classes)
        if self.config['batch_size'] == 64:
            number_digits = (7, 7, 7, 7, 6, 6, 6, 6, 6, 6)
        elif self.config['batch_size'] == 128:
            number_digits = (13, 13, 13, 13, 13, 13, 13, 13, 12, 12)
        elif self.config['batch_size'] == 256:
            number_digits = (26, 26, 26, 26, 26, 26, 25, 25, 25, 25)
        elif self.config['batch_size'] == 512:
            number_digits = (51, 51, 51, 51, 51, 51, 51, 51, 52, 52)
        x_selected_set = np.zeros((self.config['batch_size'], 28, 28))
        y_selected_set = np.zeros((self.config['batch_size'],), dtype='uint8')
        count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        idx_val = 0
        while sum(count) < self.config['batch_size']:
            if count[y_test[idx_val]] < number_digits[y_test[idx_val]]:
                x_selected_set[sum(number_digits[:y_test[idx_val]]) + count[y_test[idx_val]]] = x_test[idx_val]
                y_selected_set[sum(number_digits[:y_test[idx_val]]) + count[y_test[idx_val]]] = y_test[idx_val]
                count[y_test[idx_val]] = count[y_test[idx_val]] + 1
            idx_val = idx_val + 1

        self.test_set = dict(
            attrib=y_selected_set,
            image=np.expand_dims(x_selected_set, -1))
        if choice == 'fashion':
            self.class_name = (
            'top', 'trousers', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

    def load_celebA_dataset(self):
        # load pre-created tf.records for celebA dataset
        pass