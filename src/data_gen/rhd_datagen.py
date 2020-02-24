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

class RhdDataGen(object):

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
        nsamples = 41258 #len(annot_data)
        # hand_points = [5]
        hand_points_left = np.array([0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
        hand_points_right = hand_points_left + 21
        annot_idx = np.arange(nsamples)

        # val_anno, train_anno = [], []
        _anno = []
        for i in range(nsamples):
            key_points = annot_data['frame'+str(i)]['uv_vis'][0,0]
            left = np.sum(key_points[hand_points_left, 2])
            right = np.sum(key_points[hand_points_right, 2])
            hand_points = key_points[np.array([5]), :] if left > right else key_points[np.array([5])+21, :]

            _anno.append(hand_points)

        if self.is_train and self.is_testtrain:
            # shuffle(annot_idx)
            # return _anno, annot_idx
            return _anno, np.arange(43, 43+16)
        elif self.is_train:
            return _anno, annot_idx
        else:
            # return _anno, annot_idx[train_val_treshold:]
            return _anno, np.arange(43, 43+16)
    #         return _anno, np.array([43, 50,  51, 56, 430, 4300,
    #    403, 4030, 62, 60, 61, 44, 45, 46,  47,
    #    48])

    def get_dataset_size(self):
        return len(self.anno_idx)

    def get_color_mean(self):
        # mean = np.array([0.285, 0.292, 0.304])
        mean = np.array([0.279815019304931, 0.27522995093505725, 0.26897174478554214])
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
        imagefile = str(sample_index).zfill(5) +'.png'
        image = imageio.imread(os.path.join(self.imgpath, imagefile))
    
        norm_image = data_process.normalize(image, self.get_color_mean()) #FIXME
        # norm_image = image / 255.0

        # create heatmaps
        heatmaps, orig_size_map = data_process.generate_gtmap(kpanno, sigma, self.outres)

        if self.debug:
            orig_image = cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_CUBIC) / 255.0
            
            for i in range(kpanno.shape[0]):
                x = kpanno[i, 0]
                y = kpanno[i, 1]
                print('X: {}, Y: {}'.format(x, y))
                orig_image = cv2.circle(orig_image, (int(x), int(y)), 5, (0,0,255), 2)

            cv2.imshow('orig {} with heatmaps GENERATOR'.format(sample_index), orig_image)
            cv2.imshow('orig {} heatmap GENERATOR'.format(sample_index), np.sum(orig_size_map, axis=-1))
            cv2.imshow('gt heatmap GENERATOR', np.sum(heatmaps, axis=-1))
            
            cv2.waitKey(0) # FIXME

        # meta info
        metainfo = {'sample_index': sample_index, 'tpts': kpanno, 'name': imagefile, 'scale': 4}

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
