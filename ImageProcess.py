# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:05:56 2019

@author: RaeChen
"""

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from PIL import Image
import glob

def MakeChannelsFirst(ImArr):
    ImArr=np.rollaxis(ImArr, 2, 0)  
    return ImArr
    
def ZeroPaddings(image, padding_size):
    '''
    padding_size=(row_plus, column_plus)
    '''
    im_row, im_col=image.shape[1], image.shape[2]
    new=[]
    for cut in image:
        cut=np.vstack((cut, np.zeros((padding_size[0], im_row), dtype='uint8')))
        cut=np.hstack((cut, np.zeros((im_col+padding_size[0], padding_size[1]), 
                                      dtype='uint8')))
        new.append(cut)
    return array(new)

'''
############################################################################################
@BatchProcessors
    Processors that run on a batch of images, takes 4d-array as input
############################################################################################
'''

def multi_images_plot(X, img_width=299, img_height=299, col=3, row=3):
    fig=plt.figure(figsize=(8,8))
    
    for i in range(1, int(col*row)):
        img=X[i-1,:].reshape(img_width, img_height, 3)
        fig.add_subplot(row, col, i)
        plt.imshow(img)
    plt.show()

def BatchZCA(path, epsilon=.1, img_width=299 ,img_height=299):
    files=glob.glob(path+'/*jpeg')
    
    ls_im=[]
    ind=0
    for im in files:
        im=Image.open(im)
        im=im.resize((img_width, img_height), Image.ANTIALIAS)
        ls_im.append(array(im))
        ind+=1
        print(ind)
        
    X=array(ls_im)
    
    X=X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    
    X_norm=X/np.max(X)
    X_norm.mean(axis=0)
    X_norm=X_norm-X_norm.mean(axis=0)
    
    cov=np.cov(X_norm, rowvar=True)
    U, S, V=np.linalg.svd(cov) #singular value decomposition
    
    X_ZCA=U.dot(np.diag(1.0/np.sqrt(S+epsilon))).dot(U.T).dot(X_norm)
    X_ZCA_resc=(X_ZCA-X_ZCA.min())/(X_ZCA.max()-X_ZCA.min())
    
    return X_ZCA_resc

def BatchImagesRead(path, img_width=299, img_height=299):
    files=glob.glob(path+'/*.jpg')
    ls_im=[Image.open(im) for im in files]
    
    x_batch=[]
    for item in ls_im:
        x_batch.append(array(item.resize((img_width, img_height), Image.ANTIALIAS)))
    x_batch=array(x_batch)
    return x_batch
        
def BatchImageEnhancer(path, img_width=299, img_height=299, SharpnessFactor=2, 
                  ContrastFactor=2, ColorFactor=2, BrightnessFactor=2):
    files=glob.glob(path+'/*.jpg')
    ls_im=[Image.open(im) for im in files]
    
    from PIL import ImageEnhance
    for i in range(len(ls_im)):
        ls_im[i]=ls_im[i].resize((img_width, img_height), Image.ANTIALIAS)
        ls_im[i]=ImageEnhance.Sharpness(ls_im[i]).enhance(SharpnessFactor)
        ls_im[i]=ImageEnhance.Contrast(ls_im[i]).enhance(ContrastFactor)
        ls_im[i]=ImageEnhance.Color(ls_im[i]).enhance(ColorFactor)
        ls_im[i]=ImageEnhance.Brightness(ls_im[i]).enhance(BrightnessFactor)
        
    x_batch=[]
    for item in ls_im:
        x_batch.append(array(item))
    x_batch=array(x_batch)
    return x_batch  

def BatchSave(x_batch, new_path, img_width=299, img_height=299):
    for j in range(x_batch.shape[0]):
        arr=x_batch[j,:].reshape(img_width, img_height, 3)
        img=Image.fromarray((arr*255).astype('uint8'))
        img.save(new_path+str(j)+'.jpg')
        
