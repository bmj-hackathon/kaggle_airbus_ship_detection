# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os



#%%
data_path = "/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/ASSETS/Dogs vs Cats"