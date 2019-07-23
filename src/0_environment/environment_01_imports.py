# import the necessary packages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import imutils
from imutils.mj_paper import PAPER

import numpy as np
import argparse
import cv2
import os

import dash

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from mpl_toolkits.mplot3d import Axes3D

import sklearn as sk

import seaborn as sns
sns.set()

import random
import pandas as pd
from pathlib import Path
import zipfile