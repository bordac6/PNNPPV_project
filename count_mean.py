import os
import sys
sys.path.insert(0, "src/tools/")

import numpy as np
import config_reader
from scipy.io import loadmat
import imageio

dataset_path = config_reader.load_path('config.json')[6:]
annot_data = loadmat(os.path.join(dataset_path, 'joint_data.mat'))
annot = annot_data['joint_uvd']
nsamples = annot.shape[1]
train_val_treshold = int(np.ceil(nsamples * 0.8))

red, green, blue = 0, 0, 0
for i in range(train_val_treshold):
    imagefile = 'rgb_1_'+ str(i+1).zfill(7) +'.jpg'
    image = imageio.imread(os.path.join(dataset_path, imagefile))
    red += np.mean(image[:,:,0])
    green += np.mean(image[:,:,1])
    blue += np.mean(image[:,:,2])

print(red)
print(green)
print(blue)
print('r: {}; g: {}; b: {}'.format(red/train_val_treshold, green/train_val_treshold, blue/train_val_treshold))

with open(os.path.join('./', 'mean_dataset_color.txt'), 'a+') as xfile:
    xfile.write('r: {}; g: {}; b: {}'.format(red/train_val_treshold, green/train_val_treshold, blue/train_val_treshold))