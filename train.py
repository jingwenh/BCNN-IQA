import tensorflow as tf
import keras
from tensorflow.keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
import data_preprocesser as preprocessor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from data_loader import *
from components import *
from model import *

X, y = load_koniq(counts=10000, target_size=(192, 256))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = DeepRN()
#compile model using accuracy to measure model performance
checkpointCallback = ModelCheckpoint("weights/iqa_model_classification.h5", monitor='val_loss', verbose=0, 
                                     save_best_only=True, save_weights_only=True, 
                                     mode='auto', period=1)
evaluateCallback = EvaluateCorrelation(X_test, y_test, 5)
model.compile(optimizer='adam', loss=categorical_huber_loss, metrics=['accuracy'])

#train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[checkpointCallback, evaluateCallback])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=100)

# Train with data augmentation
# datagen = ImageDataGenerator(rotation_range=5)
# datagen.fit(X_train)
# model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
#                     steps_per_epoch=len(X_train) / 128, epochs=100)

