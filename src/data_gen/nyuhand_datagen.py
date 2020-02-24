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

    def __init__(self, matfile, imgpath, inres, outres, is_train, is_testtrain):
        self.matfile = matfile
        self.imgpath = imgpath
        self.inres = inres
        self.outres = outres
        self.is_train = is_train
        self.is_testtrain = is_testtrain
        self.anno, self.anno_idx = self._load_image_annotation()
        self.nparts = np.array(self.anno).shape[1]
        print('number of heatmaps: {}'.format(self.nparts))

        self.debug = False

    def _load_image_annotation(self):
        # load train or val annotation
        annot_data = loadmat(os.path.join(self.imgpath, self.matfile))
        annot = annot_data['joint_uvd']
        nsamples = annot.shape[1]
        train_val_treshold = int(np.ceil(nsamples * 0.8))
        hand_points = [18]#, 3, 6, 9, 12, 15, 18, 21, 24, 27, 35]
        annot_idx = np.arange(nsamples)

        # val_anno, train_anno = [], []
        _anno = []
        for i in range(nsamples):
            _anno.append(annot[0, i, hand_points, :])
            # if  i < train_val_treshold:
                # train_anno.append(annot[0, i, hand_points, :])
            # else:
            #     val_anno.append(annot[0, i, hand_points, :])

        if self.is_train and self.is_testtrain:
            # train_annot_idx = annot_idx[:train_val_treshold]
            # shuffle(train_annot_idx)
            # return _anno, train_annot_idx[:32]
            return _anno, np.array([34948])
    #         , 41447,  6279, 15487, 16105, 12193,
    #    39944, 16401, 50508, 16298, 52362, 55999, 38257, 44611,  2843,
    #    25869, 39627, 47312, 38578, 15636, 53584, 12798, 20677, 15582,
    #    32204, 35710, 41101, 27014, 15693])
        elif self.is_train:
            return _anno, annot_idx[:train_val_treshold]
        else:
            # return _anno, annot_idx[train_val_treshold:]
            return _anno, np.array([34948])
    #         , 41447,  6279, 15487, 16105, 12193,
    #    39944, 16401, 50508, 16298, 52362, 55999, 38257, 44611,  2843,
    #    25869, 39627, 47312, 38578, 15636, 53584, 12798, 20677, 15582,
    #    32204, 35710, 41101, 27014, 15693])

    def get_dataset_size(self):
        return len(self.anno_idx)

    def get_color_mean(self):
        mean = np.array([0.285, 0.292, 0.304])
        return mean

    def get_annotations(self):
        return self.anno[self.anno_idx]

    def generator(self, batch_size, num_hgstack, sigma=3, with_meta=False, is_shuffle=False):
        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''
        train_input = np.zeros(shape=(batch_size, self.inres[0], self.inres[1], 3), dtype=np.float)
        gt_heatmap = np.zeros(shape=(batch_size, self.outres[0], self.outres[1], self.nparts), dtype=np.float)
        meta_info = list()

        if not self.is_train:
            assert (is_shuffle == False), 'shuffle must be off in val model'

        while True:
            if is_shuffle:
                shuffle(self.anno_idx)

            for i, kpanno_idx in enumerate(self.anno_idx):
                kpanno = self.anno[kpanno_idx]
                _imageaug, _gthtmap, _meta = self.process_image(kpanno_idx, kpanno, sigma)
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

    def process_image(self, sample_index, kpanno, sigma):
        imagefile = 'rgb_1_'+ str(sample_index+1).zfill(7) +'.jpg'
        image = imageio.imread(os.path.join(self.imgpath, imagefile))
    
        norm_image = data_process.normalize(image, self.get_color_mean()) #FIXME
        # norm_image = image / 255.0

        # create heatmaps
        heatmaps, orig_size_map = data_process.generate_gtmap(kpanno, sigma, self.outres)

        if self.debug:
            orig_image = cv2.resize(image, dsize=(480, 480), interpolation=cv2.INTER_CUBIC) / 255.0
            im = np.concatenate([orig_image, np.sum(orig_size_map, axis=-1)[:,:,np.newaxis]], axis=-1)
            
            for i in range(kpanno.shape[0]):
                x = kpanno[i, 0]
                y = kpanno[i, 1]
                orig_image = cv2.circle(orig_image, (int(x), int(y)), 5, (0,0,255), 2)

            cv2.imshow('orig {} with heatmaps GENERATOR'.format(sample_index), orig_image)
            cv2.imshow('orig {} heatmap GENERATOR'.format(sample_index), np.sum(orig_size_map, axis=-1))
            cv2.imshow('gt heatmap GENERATOR', np.sum(heatmaps, axis=-1))
            
            cv2.waitKey(0) # FIXME

        # meta info
        metainfo = {'sample_index': sample_index, 'tpts': kpanno, 'name': imagefile, 'scale': 7.5}

        return norm_image, heatmaps, metainfo

    @classmethod
    def get_kp_keys(cls):
        keys = ['pinky_fingertip', 'pinky',
                'ring_fingertip', 'ring',
                'middle_fingertip', 'middle',
                'index_fingertip', 'index',
                'thumb_fingertip', 'thumb',
                'wrist'
                ]
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
