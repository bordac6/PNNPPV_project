import keras
import os
import datetime
import config_reader
from time import time
from nyuhand_datagen import NYUHandDataGen
from eval_heatmap import cal_heatmap_acc
import cv2
import numpy as np
from heatmap_process import post_process_heatmap


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath, inres, outres):
        self.foldpath = foldpath
        self.inres = inres
        self.outres = outres
        self.acc_history = []

    def get_folder_path(self):
        return self.foldpath

    def run_eval(self, epoch):
        # dataset_path = os.path.join('D:\\', 'nyu_croped')
        # dataset_path = '/home/tomas_bordac/nyu_croped'
        dataset_path = config_reader.load_path()
        valdata = NYUHandDataGen('joint_data.mat', dataset_path, inres=self.inres, outres=self.outres, is_train=False, is_testtrain=False)

        total_suc, total_fail = 0, 0
        total_arr_mean, total_arr_med = [], []
        threshold = 0.2

        count = 0
        batch_size = 8
        for _img, _gthmap, _meta in valdata.generator(batch_size, 2, sigma=3, is_shuffle=False, with_meta=True):

            count += batch_size
            if count > valdata.get_dataset_size():
                break

            out = self.model.predict(_img)
            kp = _meta[0]['tpts']

            if count+batch_size > valdata.get_dataset_size():
                for i in range(1):
                    orig_image = cv2.resize(_img[i], dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
                    pred_kps = post_process_heatmap(out[-1][i])
                    pred_kps = np.array(pred_kps)
                    
                    imgs = tuple()
                    for j in range(out[-1].shape[-1]):
                        print_image = np.zeros(shape=(64,64,3))
                        print_image[:,:,0] = out[-1][i,:,:,j]
                        print_image[:,:,1] = out[-1][i,:,:,j]
                        print_image[:,:,2] = out[-1][i,:,:,j]
                        cv2.circle(print_image, (int(kp[j,0]/7.5), int(kp[j,1]/7.5)), 5, (0,0,255), 2)
                        cv2.circle(print_image, (int(pred_kps[j,0]), int(pred_kps[j,1])), 5, (255,0,0), 2)
                        imgs += (print_image,)
                        cv2.circle(orig_image, (int(kp[j,0]), int(kp[j,1])), 5, (0,0,255), 2)
                        cv2.circle(orig_image, (int(pred_kps[j,0]*7.5), int(pred_kps[j,1]*7.5)), 5, (255,0,0), 2)

                    all_images = np.hstack(imgs)
                    cv2.imshow('Predicted htamps', all_images)
                    
                    cv2.waitKey(1000)

            suc, bad, mean, med = cal_heatmap_acc(out[-1], _meta, threshold)

            total_suc += suc
            total_fail += bad
            total_arr_mean.append(mean)
            total_arr_med.append(med)

        acc = total_suc * 1.0 / (total_fail + total_suc)
        self.acc_history.append(acc)

        print('Eval Accuray ', acc, '@ Epoch ', epoch)
        print('mean distance {}; median distance {}'.format(np.mean(total_arr_mean), np.median(total_arr_med)))

#       # print best acc predicted keypoints to original image
        # if np.argmax(self.acc_history) == len(self.acc_history)-1:
        #     orig_image = cv2.resize(_img[i], dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
        #     for j in range(kp.shape[0]):
        #         cv2.circle(orig_image, (int(kp[j,0]), int(kp[j,1])), 5, (0,0,255), 2)
        #         cv2.circle(orig_image, (int(pred_kps[j,0]*7.5), int(pred_kps[j,1]*7.5)), 5, (255,0,0), 2)
        #     cv2.imshow('orig with heatmaps in {}. epoch'.format(epoch), orig_image)
            
        with open(os.path.join('./', 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + ':' + str(np.mean(total_arr_mean)) + ':' + str(np.median(total_arr_med)) + '\n')

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
