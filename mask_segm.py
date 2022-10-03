# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 22:27:38 2022

@author: adeju
"""

from PyQt5 import uic, QtWidgets
# from PyQt5.QtGui import QPixmap, QImage
import sys
import numpy as np
from PIL import Image
import glob
import os

import tensorflow as tf
from keras import backend as K
from keras.models import Model#, load_model
from keras.layers import Input
from keras.layers.core import Dropout#, Lambda, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose # , UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization



# Set image size inputs
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

smoothness = 1.0
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

#Define dice coeficient metric
def dice_coefficient(y1, y2):
  y1 = K.flatten(y1)
  y2 = K.flatten(y2)
  return (2. * K.sum(y1 * y2) + smoothness) / (K.sum(y1) + K.sum(y2) + smoothness)

def dice_coefficient_np(y1, y2):
  y1 = y1.flatten()
  y2 = y2.flatten()
  return (2. * np.sum(y1 * y2) + smoothness) / (np.sum(y1) + np.sum(y2) + smoothness)


#Define dice coeficient loss
def dice_coefficient_loss(y1, y2):
  return -dice_coefficient(y1, y2)

def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

chanDim = -1
if K.image_data_format() == "channels_first":
  #inputShape = (depth, height, width)
  chanDim = 1

def Unet(inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))):
  #s = Lambda(lambda x: x / 255) (inputs)

  c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
  c1 = BatchNormalization(axis = chanDim)(c1)
  c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
  c1 = BatchNormalization(axis = chanDim)(c1)
  p1 = MaxPooling2D((2, 2)) (c1)
  p1 = Dropout(0.1) (p1)

  c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
  c2 = BatchNormalization(axis = chanDim)(c2)
  c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
  c2 = BatchNormalization(axis = chanDim)(c2)
  p2 = MaxPooling2D((2, 2)) (c2)
  p2 = Dropout(0.1) (p2)

  c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
  c3 = BatchNormalization(axis = chanDim)(c3)
  c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
  c3 = BatchNormalization(axis = chanDim)(c3)
  p3 = MaxPooling2D((2, 2)) (c3)
  p3 = Dropout(0.1) (p3)

  c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
  c4 = BatchNormalization(axis = chanDim)(c4)
  c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
  c4 = BatchNormalization(axis = chanDim)(c4)
  p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
  p4 = Dropout(0.2) (p4)

  c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
  c5 = BatchNormalization(axis = chanDim)(c5)
  c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
  c5 = BatchNormalization(axis = chanDim)(c5)

  u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
  c6 = BatchNormalization(axis = chanDim)(c6)
  c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
  c6 = BatchNormalization(axis = chanDim)(c6)

  u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
  c7 = BatchNormalization(axis = chanDim)(c7)
  c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
  c7 = BatchNormalization(axis = chanDim)(c7)

  u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
  c8 = BatchNormalization(axis = chanDim)(c8)
  c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
  c8 = BatchNormalization(axis = chanDim)(c8)

  u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
  c9 = BatchNormalization(axis = chanDim)(c9)
  c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
  c9 = BatchNormalization(axis = chanDim)(c9)

  outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

  model = Model(inputs=[inputs], outputs=[outputs])
  #model.compile(optimizer= Adam(), loss='binary_crossentropy', metrics=[iou_loss_core]) # 'binary_crossentropy'
  model.compile(optimizer= 'adam', loss=dice_coefficient_loss, metrics=['accuracy', dice_coefficient, iou_loss_core])

  return model



def convert_img_to_tiles(in_path, input_filename, count): # in_path, input_filename, out_path, output_filename, count

    temp_image = Image.open(os.path.join(in_path, input_filename))
    
    image_tiles = []

    IMG_WIDTH, IMG_HEIGHT = 256, 256

    image_array = np.asarray(temp_image)
    image_shape = np.shape(temp_image)

    n_col = int(image_shape[0]/IMG_WIDTH)
    n_row = int(image_shape[1]/IMG_HEIGHT)

    count = 0
    grid = []

    for i in range(0, n_col+1):
        for j in range(0, n_row+1):
            x0, y0 = i*IMG_WIDTH, j*IMG_HEIGHT 
            x1, y1 = x0+IMG_WIDTH, y0+IMG_HEIGHT
            if x1 > image_shape[0]:
                x1 = image_shape[0]
            if y1 > image_shape[1]:
                y1 = image_shape[1]
            #print(x0,x1, y0,y1)
            tile = image_array[x0:x1, y0:y1]
            grid.append([x0, y0, count])
            
            image_tiles.append(tile)

            count+=1

    return count, grid, image_tiles


def predict_tiles(image_tiles, unet):
    
    predicted_tiles = []

    for i in range(np.size(image_tiles)):
        image = Image.fromarray(image_tiles[i])
        
        
        initial_shape = np.shape(image)
        temp_image = image.resize((IMG_HEIGHT,IMG_WIDTH))
        
        processed_image = unet.predict(np.reshape(temp_image, (1, np.shape(temp_image)[0], np.shape(temp_image)[1], np.shape(temp_image)[2])))
       
        processed_image = Image.fromarray(np.uint8(np.reshape(processed_image[0], (IMG_HEIGHT,IMG_WIDTH))))
        
        processed_image = processed_image.resize((initial_shape[1], initial_shape[0]))
        
        predicted_tiles.append(np.uint8(processed_image))
        
    return predicted_tiles


def merge_tiles(output_file, predicted_tiles, image_size, grid, count):
    # output_file = 'mosaic_output.png'
    
    mosaic_output = np.zeros((image_size[0], image_size[1]))
    
    for i in range(0, count):
        # print(i)
        # temp_image = Image.open(processed_tiles_path+'tile_{}.tif'.format(int(i)))
        temp_image = predicted_tiles[i]
        shape = np.shape(temp_image)
        mosaic_output[grid[i][0]:grid[i][0]+shape[0], grid[i][1]:grid[i][1]+shape[1]] = temp_image

    Image.fromarray(np.uint8(mosaic_output*255)).save(output_file[:-4]+"_mask.png")


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('view.ui', self)
        self.input_path = ''
        self.output_path = ''
        self.unet = Unet(inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
        self.unet.load_weights("weights2.h5")
        self.init_Ui()
        self.show()

    def init_Ui(self):
        self.openInputFolder.clicked.connect(lambda: self.open_folder(0))
        self.selectOutputFolder.clicked.connect(lambda: self.open_folder(1))
        self.alignButton.clicked.connect(self.align_photos)
        self.progressBar.setValue(0)
        self.statusBar().showMessage("Idle")

    def open_folder(self, mode):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open a folder")
        if path != ('', ''):
            if mode == 0:
                self.input_path = path + '/'
                self.labelInputFolder.setText(self.input_path)
            if mode == 1:
                self.output_path = path + '/'
                self.labelOutputFolder.setText(self.output_path)

    def align_photos(self):

        os.chdir(self.input_path)

        image_list = glob.glob('*.JPG')
        
        print(image_list)

        progress_bar_max = np.size(image_list)
        progress_bar_count = 0
        self.progressBar.setValue(0)
        self.statusBar().showMessage("Processing")

        for image_address in image_list:            
            image_size = np.shape(Image.open(image_address))

            
            self.statusBar().showMessage("Converting "+image_address+" to tiles.")
            count, grid, image_tiles = convert_img_to_tiles('', image_address, 0)
            
            self.statusBar().showMessage("Predicting mask of "+image_address)    
            predicted_tiles = predict_tiles(image_tiles, self.unet)

            self.statusBar().showMessage("Saving "+image_address[:-4]+"_mask.png"+" into the output folder.")
            
            merge_tiles(self.output_path+image_address, predicted_tiles, image_size, grid, count)
            
            progress_bar_count += 1
            

            self.progressBar.setValue(int((progress_bar_count/progress_bar_max)*100))
            QtWidgets.QApplication.processEvents()
            
            
        self.statusBar().showMessage("Done!")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec()
