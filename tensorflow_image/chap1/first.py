import tensorflow as tf
import numpy as np

np.random.seed(0)
data, labels = tf.contrib.learn.datasets.load_dataset("iris")
num_elements = len(labels)

shuffled_indices = np.arange(len(labels))
np.random.shuffle(shuffled_indices)
shuffled_data = data[shuffled_indices]
shuffled_label = labels[shuffled_indices]

one_hot_labels = np.zeros([num_elements, 3], dtype=int)
one_hot_labels[np.arange(num_elements), shuffled_label] = 1

train_data = shuffled_data[0:105]
train_labels = shuffled_label[0:105]
test_data = shuffled_data[105:]
test_labels = shuffled_label[105:]

