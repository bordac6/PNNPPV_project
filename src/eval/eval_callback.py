import keras
import os
import datetime
import config_reader
from time import time
from nyuhand_datagen import NYUHandDataGen
from eval_heatmap import cal_heatmap_acc


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath, inres, outres):
        self.foldpath = foldpath
        self.inres = inres
        self.outres = outres

    def get_folder_path(self):
        return self.foldpath

    def run_eval(self, epoch):
        # dataset_path = os.path.join('D:\\', 'nyu_croped')
        # dataset_path = '/home/tomas_bordac/nyu_croped'
        dataset_path = config_reader.load_path()
        valdata = NYUHandDataGen('joint_data.mat', dataset_path, inres=self.inres, outres=self.outres, is_train=False, is_testtrain=False)

        total_suc, total_fail = 0, 0
        threshold = 0.5

        count = 0
        batch_size = 1
        for _img, _gthmap, _meta in valdata.generator(batch_size, 2, sigma=3, is_shuffle=False, with_meta=True):

            count += batch_size
            if count > valdata.get_dataset_size():
                break

            out = self.model.predict(_img)

            suc, bad = cal_heatmap_acc(out[-1], _meta, threshold)

            total_suc += suc
            total_fail += bad

        acc = total_suc * 1.0 / (total_fail + total_suc)

        print('Eval Accuray ', acc, '@ Epoch ', epoch)

        with open(os.path.join(self.get_folder_path(), 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

    def on_epoch_end(self, epoch, logs=None):
        # This is a walkaround to sovle model.save() issue
        # in which large network can't be saved due to size.

        # save model to json
        if epoch == 0:
            jsonfile = os.path.join(self.foldpath, "net_arch.json")
            with open(jsonfile, 'w') as f:
                f.write(self.model.to_json())

        # save weights
        modelName = os.path.join(self.foldpath, "weights_epoch" + str(epoch) + ".h5")
        self.model.save_weights(modelName)

        print("Saving model to ", modelName)

        self.run_eval(epoch)
