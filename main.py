import os
import numpy as np
import pandas as pd

ImgSplits_path = os.getcwd() + "\Stanford40\ImageSplits"
Imgpath = os.getcwd() + "\Stanford40\JPEGImages"

train_path = ImgSplits_path + "//train.txt"
actions_path = ImgSplits_path + "//actions.txt"
#read and parse the .csv features file for A1-turbine normalized data
df = pd.read_csv(actions_path, delimiter = '\t+')
df.head()