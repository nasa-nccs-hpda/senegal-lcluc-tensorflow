import os
import numpy as np
from glob import glob

data_filename = glob('/explore/nobackup/projects/3sl/development/cnn_landcover/crop.srv.v4/images/*.npy')
label_filename = glob('/explore/nobackup/projects/3sl/development/cnn_landcover/crop.srv.v4/labels/*.npy')

for xf, yf in zip(data_filename, label_filename):
    x, y = np.load(xf), np.load(yf)
    xshape, yshape = x.shape, y.shape
    if xshape[0] != 256 or xshape[1] != 256 or xshape[-1] != 8 or \
            yshape[0] != 256 or yshape[1] != 256:
        print(xf, yf)
        os.remove(xf)
        os.remove(yf)

