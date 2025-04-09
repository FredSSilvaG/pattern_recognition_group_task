#Author: mmj
#DATE: 09.04.2025
import os
import pandas as pd
from PIL import Image
import numpy as np

def load_image_and_label(image_path, label):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    return image, label

def create_dataset(tsv_path, parent_path, num_samples=None):
    # load file
    df = pd.read_csv(tsv_path, sep='\t', header=None)
    df.columns = ['path', 'label']

    # If the number of samples is specified, the num_samples bar data is taken at random
    if num_samples is not None:
        df = df.sample(n=num_samples)

    # get image path and label
    image_paths = [os.path.join(parent_path, path) for path in df['path']]
    labels = df['label'].values

    # loading image
    images = []
    for path, label in zip(image_paths, labels):
        image, label = load_image_and_label(path, label)
        images.append(image)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def get_dataset(parent_path, file_path, num_samples=None):

    X, y = create_dataset(file_path, parent_path, num_samples)
    X = X.reshape(X.shape[0], -1)
    return X, y

def load_mnist_data():
    parent_path = 'MNIST-full'  # Dataset root directory
    train_tsv_path = os.path.join(parent_path, 'gt-train.tsv')
    X_train, y_train = get_dataset(parent_path, file_path=train_tsv_path, num_samples=None)
    test_tsv_path = os.path.join(parent_path, 'gt-test.tsv')
    X_test, y_test = get_dataset(parent_path, file_path=test_tsv_path, num_samples=None)
    return X_train, y_train, X_test, y_test