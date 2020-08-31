import numpy as np
import tensorflow as tf

from base import BaseDataGenerator


class DataGenerator(BaseDataGenerator):
    def __init__(self, config, sess):
        super(DataGenerator, self).__init__(config, sess)
        # load data here: generate 3 state variables: train_set, val_set and test_set
        if self.config['exp_name'] == 'MNIST_digit':
            if self.config['one_image'] == 1:
                self.load_MNIST_dataset_single_image()
            else:
                self.load_MNIST_dataset('digit')
        elif self.config['exp_name'] == 'MNIST_fashion':
            self.load_MNIST_dataset('fashion')
        elif self.config['exp_name'] == 'celebA':
            self.n_train = 180000
            self.n_val = 20000
        elif self.config['exp_name'] == 'interiorNet':
            # create a training batch iterator
            training_set_path = self.config['data_path'] + 'training/'
            print(training_set_path)
            self.training_batch = self.load_interiorNet_dataset(load_dir=training_set_path)
            # create a validation batch iterator
            val_set_path = self.config['data_path'] + 'val/'
            self.val_batch = self.load_interiorNet_dataset(load_dir=val_set_path)
            # create a batch as test set
            self.test_batch = self.sess.run(self.val_batch)
            print("The size of the test batch for interiorNet is {}".format(self.test_batch.shape))
            self.n_train = self.config['n_seq_train'] * 1000
            self.n_val = self.config['n_seq_val'] * 1000

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
        # load the data
        # load the attribute list
        file_name = self.config.data_path+'/list_attr_celeba.txt'
        results = []
        with open(file_name) as inputfile:
            for line in inputfile:
                results.append(line.strip().split(' '))

        n_image = int(results[0][0])
        print(n_image)
        celebA_attrib = dict(attrib_full_list=results[1],
                             attrib_full_label=np.zeros((n_image, 40)),
                             attrib_selected_list=[],
                             attrib_selected_label=np.zeros((n_image, 10)))

        # remove empty thing
        thing = ''
        for i in range(n_image):
            while thing in results[i+2]:
                results[i+2].remove(thing)
            celebA_attrib['attrib_full_label'][i] = np.asarray(
                np.array(results[i+2][1:], dtype=int))

        #  select wanted attributes
        idx_selected_attrib = (4, 13, 15, 20, 21, 22, 24, 31, 35, 39)
        for i in idx_selected_attrib:
            celebA_attrib['attrib_selected_list'].append(results[1][i])
        celebA_attrib['attrib_selected_label'] = celebA_attrib['attrib_full_label'][:,
                                                                                    idx_selected_attrib]
        # load the preprocessed image file
        image_test = np.reshape(np.load(self.config.data_path+'/celebA_image_rescaled_128_test.npy').astype(np.uint8),[1000,128,128,3])

        # keep 1000 images as test set for visualisation and comparison
        idx_test = np.arange(0, 1000, 1, dtype=np.int)
        # n_repeat = int(np.ceil(2*self.config['batch_size']/100))
        self.test_set = dict(
            attrib_list=celebA_attrib['attrib_selected_list'],
            attrib=celebA_attrib['attrib_selected_label'][idx_test],
            image=image_test.copy().astype(np.float32)/255.0)

        # divide into train, val and test sets
        idx_train, idx_val = self.separate_train_and_val_set()
        # n_train and n_val are defined in base.py
        self.n_train = len(idx_train)
        self.n_val = len(idx_val)
        self.n_test = 1000
        self.n_test_set = self.n_test