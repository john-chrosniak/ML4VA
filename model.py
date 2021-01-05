import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, UpSampling2D, Reshape, Activation
from tensorflow.keras.models import Model

def create_dataset(image_path, mask_path):
  X = np.zeros((5000,128,128,3))
  Y = np.zeros((5000,128*128,2))
  for i in tqdm(range(5000)):
    try:
      img = Image.open(image_path + "{0:0=5d}".format(i) + '.jpg')
      img_arr = np.array(img)
      img_arr = tf.cast(img_arr, tf.float32) / 255.0
      X[i] = img_arr
      mask = Image.open(mask_path + "{0:0=5d}".format(i) + '.png').convert('L')
      mask_arr = tf.keras.utils.to_categorical(np.array(mask).reshape(128*128))
      mask_arr = tf.cast(mask_arr, tf.float32)
      Y[i] = mask_arr
    except:
      pass
  X_prep = X[~(X==0).all(axis=(1,2,3))]
  Y_prep = Y[~(Y==0).all(axis=(1,2))]
  print("Number of examples:", Y_prep.shape[0]+1)
  return X_prep, Y_prep

def create_segnet_model(img_height, img_width):
    kernel = 3
    pool_size = 2
    kernel_size = 64

    input_shape = Input(shape=(img_height,img_width,3))
    model = tf.keras.applications.ResNet152V2(weights="imagenet", include_top=False, input_tensor=input_shape, pooling=None)
    # Decoder Layers

    # Block 1
    o = UpSampling2D(size=(pool_size, pool_size), data_format='channels_last')(model.layers[-3].output)
    o = Conv2D(kernel_size*8, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)

    # Block 2
    o = UpSampling2D(size=(pool_size, pool_size), data_format='channels_last')(o)
    o = Conv2D(kernel_size*4, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)
    o = Conv2D(kernel_size*4, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)


    # Block 3
    o = UpSampling2D(size=(pool_size, pool_size), data_format='channels_last')(o)
    o = Conv2D(kernel_size*2, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)
    o = Conv2D(kernel_size*2, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)

    # Block 4
    o = UpSampling2D(size=(pool_size, pool_size), data_format='channels_last')(o)
    o = Conv2D(kernel_size*2, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)
    o = Conv2D(kernel_size*2, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)

    # Block 5
    o = UpSampling2D(size=(pool_size, pool_size), data_format='channels_last')(o)
    o = Conv2D(kernel_size, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)
    o = Conv2D(kernel_size, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = BatchNormalization()(o)

    # Output Block
    o = Conv2D(2, (kernel, kernel), padding='same', activation='relu', data_format='channels_last', kernel_initializer='he_normal')(o)
    o = Reshape((128*128,2))(o)
    o = Activation('softmax')(o)

    return Model(inputs=model.inputs, outputs=o)

def create_test_dataset(image_path, mask_path):
  X = np.zeros((120,128,128,3))
  Y = np.zeros((120,128*128,2))
  for i in tqdm(range(120)):
    try:
      img = Image.open(image_path + "{0:0=3d}".format(i) + '.png')
      img_arr = np.array(img)
      img_arr = tf.cast(img_arr, tf.float32) / 255.0
      X[i] = img_arr
      mask = Image.open(mask_path + "{0:0=3d}".format(i) + '.png').convert('L')
      mask_arr = np.array(mask).reshape(128*128)
      one_indices = mask_arr != 0
      zero_indices = mask_arr == 0
      mask_arr[one_indices] = 0
      mask_arr[zero_indices] = 1
      mask_arr = tf.keras.utils.to_categorical(mask_arr)
      mask_arr = tf.cast(mask_arr, tf.float32)
      Y[i] = mask_arr
    except:
      pass
  print("Number of examples:", Y.shape[0]+1)
  return X, Y