import numpy as np
import gzip

from os import makedirs, stat
from os.path import join, exists
from urllib.request import urlretrieve

class MnistDataLoader():

  @staticmethod
  def download(filename, config):
    """ get the data if not present in local disk"""
    if not exists(config['work_dir']):
        makedirs(config['work_dir'])
    filepath = join(config['work_dir'], filename)
    if not exists(filepath):
        filepath, _ = urlretrieve(config['source_url'] + filename, filepath)
        size = stat(filepath).st_size
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

  @staticmethod
  def extract_data(filename, num_images, config):
      """ Extract the images into a 4D tensor [image_index, y, x, channels]
          Rescale values from [0, 255] down to [-0.5, 0.5]
      """
      print('Extracting', filename)
      with gzip.open(filename) as bytestream:
          # header is 16 bytes
          bytestream.read(16)
          buf = bytestream.read(config['image_size'] * config['image_size'] * num_images * config['num_channels'])
          data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
          data = (data - (config['pixel_depth'] / 2.0)) / config['pixel_depth']
          data = data.reshape(num_images, config['image_size'], config['image_size'], config['num_channels'])
          return data

  @staticmethod
  def extract_labels(filename, num_images):
      """Extract the labels into a vector of int64 label IDs"""
      print('Extracting', filename)
      with gzip.open(filename) as bytestream:
          bytestream.read(8)
          buf = bytestream.read(1 * num_images)
          labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
      return labels

  @staticmethod
  def get_data(config):
    # Get the data.
    train_data_filename = MnistDataLoader.download('train-images-idx3-ubyte.gz', config)
    train_labels_filename = MnistDataLoader.download('train-labels-idx1-ubyte.gz', config)
    test_data_filename = MnistDataLoader.download('t10k-images-idx3-ubyte.gz', config)
    test_labels_filename = MnistDataLoader.download('t10k-labels-idx1-ubyte.gz', config)

    # Extract it into NumPy arrays.
    train_data = MnistDataLoader.extract_data(train_data_filename, 60000, config)
    train_labels = MnistDataLoader.extract_labels(train_labels_filename, 60000)
    test_data = MnistDataLoader.extract_data(test_data_filename, 10000, config)
    test_labels = MnistDataLoader.extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    # Choosing the last slice for the validation is not 
    # correct for all the problems
    validation_data = train_data[:config['validation_size'], ...]
    validation_labels = train_labels[:config['validation_size']]
    train_data = train_data[config['validation_size']:, ...]
    train_labels = train_labels[config['validation_size']:]

    data = {
      'train_x': train_data,
      'train_y': train_labels,
      'val_x': validation_data,
      'val_y': validation_labels,
      'test_x': test_data,
      'test_y': test_labels,
    }

    return data
