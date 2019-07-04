# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:18:20 2019

@author: Administrator
"""

#-*- coding:utf-8 -*-
import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Dropout,Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras import backend as K
import numpy as np




nb_classes=4
img_width,img_height=299,299
nb_epoch=50
batch_size=80
seed=1
directory='./train_bak_pro'


if K.image_data_format()=='channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)

base_model = InceptionV3(weights='./imagenet.h5',include_top=False)

#def preprocess_input_inception(x):
#    X=np.expand_dims(x,axis=0)
#    X=preprocess_input(X)
#    return X[0]

x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Dense(1024,activation='relu',name='fc1')(x)
# x = Dense(512,activation='relu',name='fc2')(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_classes,activation='softmax',name='predictions')(x)
model = Model(inputs=base_model.input,outputs=predictions)
model.summary()



for layer in model.layers:
    if layer.name in ['predictions']:
        continue
    layer.trainable = False




train_datagen = ImageDataGenerator(
#        preprocessing_function=preprocess_input_inception,
        rescale=1./255,
        samplewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='constant', 
        cval=0)

val_datagen = ImageDataGenerator(
#        preprocessing_function=preprocess_input_inception,
        rescale=1./255,
        samplewise_std_normalization=True,
        validation_split=0.2,
        fill_mode='constant', 
        cval=0)

train_generator = train_datagen.flow_from_directory(directory=directory,
                                                    target_size=(img_width,img_height),#Inception V3规定大小
                                                    batch_size=batch_size,
                                                    subset='training',
                                                    seed=seed)
val_generator = train_datagen.flow_from_directory(directory=directory,
                                                  target_size=(img_width,img_height),
                                                  batch_size=batch_size,
                                                  subset='validation',
                                                  seed=seed
                                                  )




s_time=time.strftime("%Y%m%d%H%M%S",time.localtime())

logs_path='./logs/inception_%s' %(s_time)

try:
    os.makedirs(logs_path)
except:
    pass


tensorboard=TensorBoard(log_dir=logs_path,histogram_freq=0,write_graph=True,write_images=True)


def recall(y_true, y_pred):
    """Recall metric.
-    Only computes a batch-wise average of recall.
-    Computes the recall, a metric for multi-label classification of
-    how many relevant items are selected.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


filepath="./model_save/inception-01-{epoch:02d}.h5"
checkpointer = ModelCheckpoint(filepath,
                               monitor='loss',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='auto',
                               period=5)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8,
                              patience=3, min_lr=1e-6, verbose=1)

sgd=SGD(lr=0.01,momentum=0.9,nesterov=True)

# def setup_to_transfer_learning(model,base_model):
#     for layer in base_model.layers:
#         layer.trainable = False
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy',recall])

# setup_to_transfer_learning(model,base_model)

history_tl = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples//batch_size,
                                 epochs=nb_epoch,
                                 validation_data=val_generator,
                                 validation_steps=val_generator.samples//batch_size,
                                 class_weight='auto',
                                 callbacks=[tensorboard,checkpointer,reduce_lr]
                                 )


model.save('./model_save/proModel.h5')
