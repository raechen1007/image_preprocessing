# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:09:45 2019

@author: Administrator
"""

import glob
from PIL import Image
import cv2

def tif2jpg_file_converter(read_dir, write_dir):
    '''
    Converting all images in .tiff from read_dir to .jpg to write_dir
    '''
    '''
    read_dir='path'
    write_dir='path'
    tif2jpg_file_converter(read_dir, write_dir)
    '''
    path=read_dir+'/*.tif'
    files=glob.glob(path)
    
    name=0
    for im in files:
        im=Image.open(im)
        im.save(write_dir+str(name)+'.jpg')
        name+=1
        
def grayscale_file_converter(read_dir, write_dir):
    path=read_dir+'/*.jpg'
    files=glob.glob(path)
    
    name=0
    for im in files:
        im=cv2.imread(im)
        im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(write_dir+str(name)+'.jpg', im)
        name+=1
        
read_dir='/Users/Administrator/Desktop/DR数据/kaggle/label4-crops224/3'
write_dir='/Users/Administrator/Desktop/DR数据/kaggle/label4-crops224/3fine/'