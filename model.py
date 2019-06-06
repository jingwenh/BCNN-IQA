import tensorflow as tf
from tensorflow.keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import data_preprocesser as preprocessor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.resnet import ResNet101, preprocess_input
from data_loader import *
from components import *
from model import *

def naive_model(shape=(192,256,3)):
    x = Input(shape=shape)
    a1 = Conv2D(50, kernel_size=7, activation='relu')(x)
    a2 = Conv2D(50, kernel_size=7, activation='relu')(a1)
    b1 = GlobalMaxPooling2D()(a2)
    b2 = GlobalMinPooling2D()(a2)
    b3 = GlobalAveragePooling2D()(a2)
    m = Concatenate()([b1, b2, b3])
    d1 = Dense(800, activation='relu')(m)
    drop = Dropout(rate = 0.5)(d1)
    d2 = Dense(800, activation='relu')(drop)
    out = Dense(5, activation='softmax')(d2)
    model = Model(inputs=x, outputs=out)
    print(model.summary())

    return model

def DeepRN():
    base_model = ResNet101(include_top=False, weights='imagenet')
    # for layer in base_model.layers:
    #     layer.trainable = False
    x = SpatialPyramidPooling([3])(base_model.output)
    # x = Dropout(0.75)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(5, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.summary()

    return model