import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam


train_df = pd.read_csv('MNIST-full/gt-train.tsv', sep='\t', header=None, names=['path', 'class'])
test_df = pd.read_csv('MNIST-full/gt-test.tsv', sep='\t', header=None, names=['path', 'class'])


def load_images_from_df(df, target_size=(28, 28)):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = 'MNIST-full/' + row['path']
        img = load_img(img_path, color_mode='grayscale')
        img = img_to_array(img) 
        images.append(img)
        labels.append(row['class'])
    return np.array(images), np.array(labels)

X_train, y_train = load_images_from_df(train_df)
X_test, y_test = load_images_from_df(test_df)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
