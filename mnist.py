# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy as np
import scipy.io as sio
import scipy

import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import os.path

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def read_MNIST_dataset(size=1000, one_hot=False, dtype=dtypes.float64, reshape=True):

    TRAIN_IMAGES = os.path.dirname(__file__) + '/datasets/train-images-idx3-ubyte'
    TRAIN_LABELS = os.path.dirname(__file__) + '/datasets/train-labels-idx1-ubyte'
    # TEST_IMAGES = os.path.dirname(__file__) + '/../datasets/MNIST/t10k-images-idx3-ubyte'
    # TEST_LABELS = os.path.dirname(__file__) + '/../datasets/MNIST/t10k-labels-idx1-ubyte'

    with open(TRAIN_IMAGES, 'rb') as f:
        train_images = extract_images(f)

    with open(TRAIN_LABELS, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    # with open(TEST_IMAGES, 'rb') as f:
    #     test_images = extract_images(f)
    #
    # with open(TEST_LABELS, 'rb') as f:
    #     test_labels = extract_labels(f, one_hot=one_hot)
        
    # print('mnist train labels:')
    # print(train_labels)
    #
    # N1 = sizes[0]
    # N2 = sizes[0]+sizes[1]
    # N3 = sizes[0]+sizes[1]+sizes[2]
    #
    # print('N1, N2, N3')
    # print(N1)
    # print(N2)
    # print(N3)
    
    train_images_tmp = train_images[:size]
    train_labels_tmp = train_labels[:size]
    # validation_images = train_images[N1:N2]
    # validation_labels = train_labels[N1:N2]
    # test_images = test_images[:sizes[2]]
    # test_labels = test_labels[:sizes[2]]

    return train_images_tmp, train_labels_tmp


def read_USPS_dataset(sizes=[1000, 100, 100], one_hot=False, dtype=dtypes.float32, Lresize=None):
    IMAGES = 'datasets/USPS/usps.mat'
    usps_dataset = sio.loadmat(IMAGES)

    usps_images = usps_dataset['xtest']
    usps_labels = usps_dataset['ytest']
    usps_labels = usps_labels - 1

    if Lresize:
        print('resizing usps images (L=%d)' % Lresize)
        usps_images_resized = np.zeros((usps_images.shape[0], Lresize * Lresize))
        usps_images_rs = np.reshape(usps_images, (usps_images.shape[0], 16, 16))
        for i in xrange(usps_images.shape[0]):
            image_resized = scipy.misc.imresize(usps_images_rs[i, :, :], size=(Lresize, Lresize), interp='bilinear',
                                                mode=None)
            usps_images_resized[i, :] = np.reshape(image_resized, (Lresize * Lresize))
        usps_images = usps_images_resized

    print('usps image shape:')
    print(usps_images.shape)

    # print(np.reshape(usps_images[100,:], (Lresize, Lresize)))
    # plt.imshow(np.reshape(usps_images[100,:], (Lresize, Lresize)))
    # plt.show()

    print('usps labels:')
    print(usps_labels)

    N1 = sizes[0]
    N2 = sizes[0] + sizes[1]
    N3 = sizes[0] + sizes[1] + sizes[2]

    train_images = usps_images[:N1]
    train_labels = usps_labels[:N1]
    validation_images = usps_images[N1:N2]
    validation_labels = usps_labels[N1:N2]
    test_images = usps_images[N2:N3]
    test_labels = usps_labels[N2:N3]

    reshape = False

    train = GrayscaleImageDataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = GrayscaleImageDataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
    test = GrayscaleImageDataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    return train, validation, test


class GrayscaleImageDataSet(object):

    def __init__(self, images, labels, L=28, one_hot=False, dtype=dtypes.float64, reshape=True, whitening=False):

        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """

        self.L = L
        self.d = 2

        x = np.arange(0, self.L, 1).astype(dtype=np.float64)
        y = np.arange(0, self.L, 1).astype(dtype=np.float64)
        xv, yv = np.meshgrid(x, y)
        self.vectors = np.vstack((xv.ravel(), yv.ravel()))

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32, dtypes.float64):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32 or float64' % dtype)

        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self.num_examples = images.shape[0]

        print('self.num_examples = %d' % self.num_examples)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            # if dtype == dtypes.float64:
            # Convert from [0, 255] -> [0.0, 1.0].
            # images = images.astype(np.float64)
            # images = np.multiply(images, 1.0 / 255.0)

        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def indices_for_label(self, label):
        return np.where(self.labels == label)[0]

    def features_for_label(self, label):
        I = self.indices_for_label(label)
        return self.images[I, :]

    def next_classif_batch(self, batch_size):

        batch_start = self.index_in_epoch

        if batch_start + batch_size > self.num_examples:
            batch_start = 0

        if batch_start == 0:
            # print('shuffle')
            perm0 = np.arange(self.num_examples)
            np.random.shuffle(perm0)
            self.images = self.images[perm0]
            self.labels = self.labels[perm0]

        images_batch = self.images[batch_start:(batch_start + batch_size), :]
        labels_batch = self.labels[batch_start:(batch_start + batch_size)]

        self.index_in_epoch = batch_start + batch_size

        return images_batch, labels_batch


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 np array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 np array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with f as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 np array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 np array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  
  with f as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


