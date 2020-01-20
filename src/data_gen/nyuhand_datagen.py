import os
import numpy as np
from random import shuffle
import scipy.misc
import json
import data_process
import random
from scipy.io import loadmat
import cv2
import imageio

class NYUHandDataGen(object):

    def __init__(self, matfile, imgpath, inres, outres, is_train):
        self.matfile = matfile
        self.imgpath = imgpath
        self.inres = inres
        self.outres = outres
        self.is_train = is_train
        self.nparts = 11
        self.anno = self._load_image_annotation()

    def _load_image_annotation(self):
        # load train or val annotation
        annot_data = loadmat(os.path.join(self.imgpath, self.matfile))
        annot = annot_data['joint_uvd']
        nsamples = annot.shape[1]
        train_val_treshold = nsamples * 0.8
        hand_points = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 35]

        val_anno, train_anno = [], []
        for i in range(nsamples):
            if  i < train_val_treshold:
                train_anno.append(annot[0, i, hand_points, :])
            else:
                val_anno.append(annot[0, i, hand_points, :])

        if self.is_train:
            return train_anno
        else:
            return val_anno

    def get_dataset_size(self):
        return len(self.anno)

    def get_color_mean(self, image):
        mean = np.array([np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])], dtype=np.float)
        return mean

    def get_annotations(self):
        return self.anno

    def generator(self, batch_size, num_hgstack, sigma=1, with_meta=False, is_shuffle=False,
                  rot_flag=False, scale_flag=False, flip_flag=False):
        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''
        train_input = np.zeros(shape=(batch_size, self.inres[0], self.inres[1], 3), dtype=np.float)
        gt_heatmap = np.zeros(shape=(batch_size, self.outres[0], self.outres[1], self.nparts), dtype=np.float)
        meta_info = list()

        if not self.is_train:
            assert (is_shuffle == False), 'shuffle must be off in val model'
            assert (rot_flag == False), 'rot_flag must be off in val model'

        while True:
            if is_shuffle:
                shuffle(self.anno)

            for i, kpanno in enumerate(self.anno):

                _imageaug, _gthtmap, _meta = self.process_image(i, kpanno, sigma, rot_flag, scale_flag, flip_flag)
                _index = i % batch_size

                train_input[_index, :, :, :] = _imageaug
                gt_heatmap[_index, :, :, :] = _gthtmap
                meta_info.append(_meta)

                if i % batch_size == (batch_size - 1):
                    out_hmaps = []
                    for m in range(num_hgstack):
                        out_hmaps.append(gt_heatmap)

                    if with_meta:
                        yield train_input, out_hmaps, meta_info
                        meta_info = []
                    else:
                        yield train_input, out_hmaps

    def process_image(self, sample_index, kpanno, sigma, rot_flag, scale_flag, flip_flag):
        imagefile = 'rgb_1_'+ str(sample_index+1).zfill(7) +'.jpg'
        image = imageio.imread(os.path.join(self.imgpath, imagefile))

        norm_image = data_process.normalize(image, self.get_color_mean(image))

        # create heatmaps
        heatmaps = data_process.generate_gtmap(kpanno, sigma, self.inres, self.outres)

        # meta info
        metainfo = {'sample_index': sample_index, 'tpts': kpanno, 'name': imagefile}

        return norm_image, heatmaps, metainfo

    @classmethod
    def get_kp_keys(cls):
        keys = ['r_ankle', 'r_knee', 'r_hip',
                'l_hip', 'l_knee', 'l_ankle',
                'plevis', 'thorax', 'upper_neck', 'head_top',
                'r_wrist', 'r_elbow', 'r_shoulder',
                'l_shoulder', 'l_elbow', 'l_wrist']
        return keys

    def flip(self, image, joints, center):

        import cv2

        joints = np.copy(joints)

        matchedParts = (
            [0, 5],  # ankle
            [1, 4],  # knee
            [2, 3],  # hip
            [10, 15],  # wrist
            [11, 14],  # elbow
            [12, 13]  # shoulder
        )

        org_height, org_width, channels = image.shape

        # flip image
        flipimage = cv2.flip(image, flipCode=1)

        # flip each joints
        joints[:, 0] = org_width - joints[:, 0]

        for i, j in matchedParts:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

        # center
        flip_center = center
        flip_center[0] = org_width - center[0]

        return flipimage, joints, flip_center
