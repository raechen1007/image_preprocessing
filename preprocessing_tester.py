# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:06:04 2019

@author: Administrator
"""
from keras import backend as K
from ImageProcess import multi_images_plot
import ImageProcess

img_width,img_height=299,299
directory='oriTrial'


if K.image_data_format()=='channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)

'''
@preprocess input by ImageDataGenerator
'''
from keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator(
        rescale=1./255,
#        featurewise_center=True,
#        samplewise_center=True,
#        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
#        zca_whitening=True,
#        shear_range=.3,
#        zoom_range=.3,
        fill_mode='constant', 
        cval=0
        )

batch=generator.flow_from_directory(directory=directory,
                                    target_size=(img_width,img_height),
                                    batch_size=8,
                                    subset='training',
                                    seed=1)
x_batch=next(batch)[0]
multi_images_plot(x_batch)



'''
@preprocess input by inceptionV3 preprocess
inceptionV3.preprocess_input() take a 4D ndarray as input, it is same as 
imagnet_utils.preprocess_input()
'''
from keras.applications import inception_v3
x_batch=ImageProcess.BatchImagesRead('./WR/noTag/')
multi_images_plot(inception_v3.preprocess_input(x_batch))


'''
@preprocess from original inceptv3 model
'''
generator= ImageDataGenerator(
        preprocessing_function=inception_v3.preprocess_input,
    #    zca_whitening=True,
    #    zca_epsilon=.1,
    #    rotation_range=360,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='constant', 
        cval=0,
    )

batch=generator.flow_from_directory(directory=directory,
                                    target_size=(img_width,img_height),
                                    batch_size=8,
                                    subset='training',
                                    seed=1)
x_batch=next(batch)[0]
multi_images_plot(x_batch)

'''
@ImageEnhance
'''
x_batch=ImageProcess.BatchImageEnhancer('./WR/noTag/', SharpnessFactor=1.7, BrightnessFactor=1., ColorFactor=1.5)
multi_images_plot(x_batch)

ImageProcess.BatchSave(x_batch, './enhanceWR/noTag/')