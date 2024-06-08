import os
import numpy as np
import cv2

ImgSplits_path = os.getcwd() + "\Stanford40\ImageSplits"
Imgpath = os.getcwd() + "\Stanford40\JPEGImages"

train_path = ImgSplits_path + "//train.txt"
print (train_path)
# Using readlines()
train_file = open(train_path, 'r')
Lines = train_file.readlines()

train_images_paths = []
# Strips the newline character
for line in Lines:
    train_images_paths.append(line.strip())







def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img,(150,150))