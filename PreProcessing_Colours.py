# Import libraries
import numpy as np
import cv2
import glob
import os


def preprocessing1(path):
    f = []
    for file in glob.iglob(path+'*.JPG'):
        im = cv2.imread(file, 1)
        im = np.array(im)
        f.append(im)

    return f


#print(preprocessing1(os.getcwd() + "/page 1/"))
