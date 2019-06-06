import tensorflow as tf
from keras.layers import Layer, Input, Reshape, Dense, Lambda, Activation
import keras.backend as K
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.layers import AveragePooling2D, Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D

def categorical_huber_loss(y_true, y_pred, clip_delta=1/9):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta

    squared_loss = 0.5 * K.square(error)
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

    return K.mean(tf.where(cond, squared_loss, linear_loss))

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

class MinPooling2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, 
               padding='valid', data_format=None, **kwargs):
        super(MaxPooling2D, self).__init__(pool_size, strides, padding,
                                       data_format, **kwargs)
    def pooling_function(inputs, pool_size, strides, padding, data_format):
        return -K.pool2d(-inputs, pool_size, strides, padding, data_format,
                                                         pool_mode='max')

class GlobalMinPooling2D(GlobalMaxPooling2D):
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.min(inputs, axis=[1, 2])
        else:
            return K.min(inputs, axis=[2, 3])

class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'th':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            num_rows = input_shape[1]
            num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        if self.dim_ordering == 'th':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]
                        x_crop = x[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

        elif self.dim_ordering == 'tf':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                        x_crop = x[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        if self.dim_ordering == 'th':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'tf':
            #outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs

class MinPooling2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, 
               padding='valid', data_format=None, **kwargs):
        super(MaxPooling2D, self).__init__(pool_size, strides, padding,
                                       data_format, **kwargs)
    def pooling_function(inputs, pool_size, strides, padding, data_format):
        return -K.pool2d(-inputs, pool_size, strides, padding, data_format,
                                                         pool_mode='max')
class GlobalMinPooling2D(GlobalMaxPooling2D):
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.min(inputs, axis=[1, 2])
        else:
            return K.min(inputs, axis=[2, 3])

class EvaluateCorrelation(Callback):
    def __init__(self, X_test, y_test, num_categories):
        super(Callback, self).__init__()
        self.X_test = X_test
        self.y_test = np.sum(y_test[:,] * np.arange(num_categories), axis = 1)
        self.num_categories = num_categories
    def on_epoch_end(self, batch, logs={}):
        def compute(sq, q):
            srocc = stats.spearmanr(sq, q)[0]
            krocc = stats.stats.kendalltau(sq, q)[0]
            plcc = stats.pearsonr(sq, q)[0]
            rmse = np.sqrt(((sq - q) ** 2).mean())
            mae = np.abs((sq - q)).mean()
            return srocc, krocc, plcc, rmse, mae
        y_pred = self.model.predict(self.X_test)
        y_pred = np.sum(y_pred[:,] * np.arange(self.num_categories), axis = 1)
        srocc, krocc, plcc, rmse, mae = compute(self.y_test, np.squeeze(y_pred))
        print("srocc: %f, krocc: %f, plcc: %f, rmse: %f, mae: %f" % (srocc, krocc, plcc, rmse, mae))
