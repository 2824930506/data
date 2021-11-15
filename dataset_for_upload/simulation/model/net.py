from __future__ import print_function
import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint


def Conv_Block(input_tensor, filters, bottleneck=False, weight_decay=1e-4):
    """    封装卷积层
    :param input_tensor: 输入张量
    :param filters: 卷积核数目
    :param bottleneck: 是否使用bottleneck
    :param dropout_rate: dropout比率
    :param weight_decay: 权重衰减率
    :return:
    """
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)

    return x


def dens_block(input_tensor, nb_filter):
    x1 = Conv_Block(input_tensor, nb_filter)
    add1 = Concatenate(axis=-1)([x1, input_tensor])
    x2 = Conv_Block(add1, nb_filter)
    add2 = Concatenate(axis=-1)([x1, input_tensor, x2])
    x3 = Conv_Block(add2, nb_filter)
    return x3

# model definition
def unet(input_shape=(256, 256, 1)):

    inputs = Input(input_shape)
    # x  = Conv2D(32, 1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = Conv2D(32, 7, kernel_initializer='he_normal', padding='same', strides=1, use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    # down first
    down1 = dens_block(x, nb_filter=64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)  # 256
    # down second
    down2 = dens_block(pool1,nb_filter=64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down2)  # 128
    # down third
    down3 = dens_block(pool2,nb_filter=128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)  # 64
    # down four
    down4 = dens_block(pool3, nb_filter=256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)  # 32
    # center
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # up first
    up6 = UpSampling2D(size=(2, 2))(drop5)
    add6 = Concatenate(axis=-1)([down4, up6])
    up6 = dens_block(add6, nb_filter=256)
    # up second
    up7 = UpSampling2D(size=(2, 2))(up6)
    add7 = Concatenate(axis=-1)([down3, up7])
    up7 = dens_block(add7, nb_filter=128)
    # up third
    up8 = UpSampling2D(size=(2, 2))(up7)
    add8 = Concatenate(axis=-1)([down2, up8])
    up8 = dens_block(add8, nb_filter=64)
    # up four
    up9 =UpSampling2D(size=(2, 2))(up8)
    add9 = Concatenate(axis=-1)([down1, up9])
    up9 = dens_block(add9, nb_filter=64)
    # output
    conv10 = Conv2D(32, 7, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv10 = Conv2D(3, 1, activation='sigmoid')(conv10)
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mae'])
    # print(model.summary())
    return model

def get_data():
    path_in = ''
    path_out = ''
    t_in = []
    t_out = []
    img_in_list = os.listdir(path_in)
    img_out_list = os.listdir(path_out)
    for name in img_in_list:
        holo_count = name.split('_')[0][2:]
        holo_distance = name.split('_')[-1][2:6]
        img_in = cv2.imread(path_in + '/' + name, cv2.IMREAD_GRAYSCALE)
        img_in = np.resize(img_in, (256, 256, 1))
        for name_gt in img_out_list:
            if name_gt == 'holo' + holo_count + '_' + holo_distance + '.bmp':
                n = 'holo' + holo_count + '_' + holo_distance + '.bmp'
                img_out = cv2.imread(path_out + '/' + name_gt, cv2.IMREAD_GRAYSCALE)
                img_out = np.resize(img_out, (256, 256, 1))
                t_in.append(img_in)
                t_out.append(img_out)
    return np.array(t_in), np.array(t_out)


[date_in, data_out] = get_data()
print(date_in.shape)
model = unet()
model_checkpoint = ModelCheckpoint('rectangle.h5', monitor='loss', verbose=1, save_best_only=True)
model.fit(date_in, data_out, batch_size=2, epochs=350, verbose=2, callbacks=[model_checkpoint])

