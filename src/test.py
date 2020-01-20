import numpy as np
import scipy.misc
import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Circle

from scipy.io import loadmat

##########################################

##Util
def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

##########################################

from shutil import copy

dst = os.path.join('D:\\nyu_reduced\\')
dataset_path = os.path.join('D:\\', 'data', 'nyu_hand', 'train')
img_idx = 72756
for i in range(img_idx):
    img_number = str(i+1).zfill(7)
    src = os.path.join(dataset_path, 'rgb_1_'+ img_number +'.png')
    copy(src, dst)

# hand_points = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 35]

# annot = loadmat(os.path.join(dataset_path, 'joint_data.mat'))

# print(annot['joint_uvd'].shape)

# joint_names = annot['joint_names']
# joint_xyz =annot['joint_xyz']
# joint_uvd = annot['joint_uvd'][0,img_idx, :, :]

# fig, ax = plt.subplots(1)
# ax.imshow(image)
# for i in hand_points:
#     x = joint_uvd[i, 0]
#     y = joint_uvd[i, 1]
#     ax.add_patch( Circle((x, y), 5, color='r') )

# plt.show()

# heat_map = np.zeros(shape=(image.shape[0], image.shape[1], len(hand_points)))
# hm = np.zeros(shape=(120, 160, len(hand_points)))

# for i in range(len(hand_points)):
#     visibility = joint_uvd[i, 2]
#     if visibility > 0:
#         heat_map[:,:,i] = draw_labelmap(heat_map[:,:,i], joint_uvd[i], 5)
#         hm[:,:,i] = cv2.resize(heat_map[:,:,i], dsize=(160, 120), interpolation=cv2.INTER_CUBIC)

# plt.imshow(hm[:,:,5])
# plt.show()