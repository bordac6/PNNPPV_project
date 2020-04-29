import sys

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../eval/")
sys.path.insert(0, "../tools/")

from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import config_reader
from hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile
from nyuhand_datagen import NYUHandDataGen
from keras.callbacks import CSVLogger, ModelCheckpoint, LambdaCallback
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
import datetime
import scipy.misc
from data_process import normalize
import numpy as np
from eval_callback import EvalCallBack
import imageio
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
import keras

class HourglassNet(object):

    def __init__(self, num_classes, num_stacks, num_channels, inres, outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.num_channels = num_channels
        self.inres = inres
        self.outres = outres

    def build_model(self, mobile=False, show=False):
        if mobile:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, bottleneck_mobile)
        else:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, bottleneck_block)
        # show model summary and layer name
        if show:
            self.model.summary()

    def train(self, batch_size, model_path, epochs):
        # dataset_path = os.path.join('D:\\', 'nyu_croped')
        # dataset_path = '/home/tomas_bordac/nyu_croped'
        # dataset_path = '../../data/nyu_croped/'
        dataset_path = config_reader.load_path('dataset_path_nyu')
        train_dataset = NYUHandDataGen('joint_data.mat', dataset_path, inres=self.inres, outres=self.outres, is_train=True, is_testtrain=False)
        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma=3, is_shuffle=True)

        csvlogger = CSVLogger(
            os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
        modelfile = os.path.join(model_path, 'weights_{epoch:02d}_{loss:.2f}.hdf5')

        print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: [ cv2.self.model.layers[i].get_weights() for i in range(len(self.model.layers)) ])
        checkpoint = EvalCallBack(model_path, self.inres, self.outres)
        lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, cooldown=2, mode='min', min_lr=5e-6)

        xcallbacks = [csvlogger, checkpoint, lr_reducer]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=(train_dataset.get_dataset_size() // batch_size) * 4,
                                 epochs=epochs, callbacks=xcallbacks)

    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):

        self.load_model(model_json, model_weights)
        self.model.compile(optimizer=Adam(lr=5e-2), loss=self.euclidean_loss, metrics=["accuracy"])

        dataset_path = config_reader.load_path('dataset_path_nyu')
        train_dataset = NYUHandDataGen('joint_data.mat', dataset_path, inres=self.inres, outres=self.outres, is_train=True, is_testtrain=False)
        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma=3, is_shuffle=True)

        model_dir = os.path.dirname(os.path.abspath(model_json))

        csvlogger = CSVLogger(
            os.path.join(model_dir, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
        modelfile = os.path.join(model_dir, 'weights_{epoch:02d}_{loss:.2f}.hdf5')

        print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: [ cv2.self.model.layers[i].get_weights() for i in range(len(self.model.layers)) ])
        checkpoint = EvalCallBack(model_dir, self.inres, self.outres)
        lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, cooldown=2, mode='min', min_lr=5e-6)
        
        xcallbacks = [csvlogger, checkpoint, lr_reducer]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=(train_dataset.get_dataset_size() // batch_size) * 4,
                                 epochs=epochs, callbacks=xcallbacks)

    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)

    def inference_rgb(self, rgbdata, orgshape, mean=None):

        scale = (orgshape[0] * 1.0 / self.inres[0], orgshape[1] * 1.0 / self.inres[1])
        imgdata = scipy.misc.imresize(rgbdata, self.inres)

        if mean is None:
            #nyu 
            # mean = np.array([0.285, 0.292, 0.304])
            mean = np.array([0.4486, 0.4269, 0.3987])

        imgdata = normalize(imgdata, mean)

        inp = imgdata[np.newaxis, :, :, :]

        out = self.model.predict(inp)
        return out[-1], scale

    def inference_file(self, imgfile, mean=None):
        imgdata = imageio.imread(imgfile)
        ret = self.inference_rgb(imgdata, imgdata.shape, mean)
        return ret

    # used when training is resumed
    def euclidean_loss(self, x, y):
        # return K.sqrt(K.sum(K.square(x[:,:,:,0] - y[:,:,:,0])))
        return K.sqrt(K.sum(K.square(x[:,:,:,0:9:2] - y[:,:,:,0:9:2])))
        # return K.sqrt(K.sum(K.square(x - y)))

import cv2
class visualizeWeightsCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        layer_names = []
        for layer in self.model.layers:
            layer_names.append(layer.name)
        
        for layer_name, layer_act in zip(layer_names, [self.model.layers[i].get_weights() for i in range(len(layer_names))]):

            layer_activation = np.array(layer_act)
            try:
                if layer_activation.shape[0] == 0 or layer_activation.shape[1] == 0:
                    continue
            except:
                continue

            scale_percent = 1000 # percent of original size
            width = int(layer_activation.shape[1] * scale_percent / 100)
            height = int(layer_activation.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(layer_activation, dim, interpolation = cv2.INTER_AREA) 
            cv2.imshow('WEIGHTS {}'.format(layer_name), resized)
        cv2.waitKey(1000)
