'''Load data and build generators.'''

from data_preprocesser import normalize_image, random_crop_image, center_crop_image
from data_preprocesser import resize_image, horizontal_flip_image
from data_preprocesser import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import pandas as pd
import numpy as np

def train_preprocessing(x, size_target=(448, 448)):
    '''Preprocessing for train dataset image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Preprocessed image.
    '''
    return normalize_image(
        random_crop_image(
            horizontal_flip_image(
                resize_image(
                    x,
                    size_target=size_target,
                    flg_keep_aspect=True
                )
            )
        ),
        mean=[123.82988033, 127.3509729, 110.25606303]
    )

def valid_preprocessing(x, size_target=(448, 448)):
    '''Preprocessing for validation dataset image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Preprocessed image.
    '''
    return normalize_image(
        center_crop_image(
            resize_image(
                x,
                size_target=size_target,
                flg_keep_aspect=True
            )
        ),
        mean=[123.82988033, 127.3509729, 110.25606303]
    )

def build_generator(
        train_dir=None,
        valid_dir=None,
        batch_size=128
    ):
    '''Build train and validation dataset generators.

    Args:
        train_dir: train dataset directory.
        valid_dir: validation dataset directory.
        batch_size: batch size.

    Returns:
        Train generator and validation generator.
    '''
    results = []
    if train_dir:
        train_datagen = ImageDataGenerator(
            preprocessing_function=train_preprocessing
        )
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(448, 448),
            batch_size=batch_size,
            class_mode='categorical'
        )
        results += [train_generator]

    if valid_dir:
        valid_datagen = ImageDataGenerator(
            preprocessing_function=valid_preprocessing
        )
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(448, 448),
            batch_size=batch_size,
            class_mode='categorical'
        )
        results += [valid_generator]

    return results

def load_koniq(counts=100, target_size=(192, 256)):
    label_path = "/data/yscode/QA_dataset/koniq10k_scores_and_distributions/koniq10k_scores_and_distributions.csv"
    dataset_path = "/data/yscode/QA_dataset/koniq10k_1024x768/1024x768"
    labels = pd.read_csv(label_path)
    y = np.array(labels.iloc[:,1:6])
    y = y[:counts]
    y = y / np.sum(y, axis = 1, keepdims=True)

    X = []
    for fn in labels["image_name"][:counts]:
        path = dataset_path + "/" + fn
        img = Image.open(path)
        img = resize_image(img, size_target=target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        X.append(img)
        if len(X) % 1000 == 0:
            print("Loaded " + str(len(X)) + " images...")
    X = np.concatenate(X, axis=0)

    return X, y

if __name__ == "__main__":
    pass
