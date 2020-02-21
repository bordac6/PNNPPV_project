import os
from shutil import copy

dst = os.path.join('D:\\', 'DP')
dataset_path = os.path.join('D:\\', 'nyu_256')
img_number = str(34948).zfill(7)
src = os.path.join(dataset_path, 'rgb_1_'+ img_number +'.jpg')
copy(src, dst)
# dst = os.path.join('D:\\', 'nyu_png')
# dataset_path = os.path.join('D:\\', 'data', 'nyu_hand', 'train')
# img_idx = 72756
# for i in range(img_idx):
#     img_number = str(i+1).zfill(7)
#     src = os.path.join(dataset_path, 'rgb_1_'+ img_number +'.png')
#     copy(src, dst)