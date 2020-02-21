import keras
import os
import datetime
import config_reader
from time import time
from nyuhand_datagen import NYUHandDataGen
from rhd_datagen import RhdDataGen
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
        self.acc2_history = []

    def get_folder_path(self):
        return self.foldpath

    def run_eval(self, epoch):
        # dataset_path = os.path.join('D:\\', 'nyu_croped')
        # dataset_path = '/home/tomas_bordac/nyu_croped'
        dataset_path = config_reader.load_path()
        valdata = RhdDataGen('anno_training.mat', dataset_path, inres=self.inres, outres=self.outres, is_train=False, is_testtrain=False)

        total_suc, total_fail, total_suc_bigger, total_fail_bigger = 0, 0, 0, 0
        total_arr_mean, total_arr_med = [], []
        threshold = 0.2

        count = 0
        batch_size = 1
        for _img, _gthmap, _meta in valdata.generator(batch_size, 8, sigma=3, is_shuffle=False, with_meta=True):

            count += batch_size
            if count > valdata.get_dataset_size():
                break

            out = self.model.predict(_img)

            if count+batch_size > valdata.get_dataset_size():
                for i in range(1):
                    kp = _meta[i]['tpts']
                    orig_image = cv2.resize(_img[i], dsize=(self.inres[0], self.inres[1]), interpolation=cv2.INTER_CUBIC)
                    pred_kps = post_process_heatmap(out[-1][i])
                    pred_kps = np.array(pred_kps)
                    
                    imgs = tuple()
                    for j in range(out[-1].shape[-1]):
                        print_image = np.zeros(shape=(self.outres[0],self.outres[1],3))
                        print_image[:,:,0] = out[-1][i,:,:,j]
                        print_image[:,:,1] = out[-1][i,:,:,j]
                        print_image[:,:,2] = out[-1][i,:,:,j]
                        cv2.circle(print_image, (int(kp[j,0]/4), int(kp[j,1]/4)), 5, (0,0,255), 2)
                        cv2.circle(print_image, (int(pred_kps[j,0]), int(pred_kps[j,1])), 5, (255,0,0), 2)
                        imgs += (print_image,)
                        cv2.circle(orig_image, (int(kp[j,0]), int(kp[j,1])), 5, (0,0,255), 2)
                        cv2.circle(orig_image, (int(pred_kps[j,0]*4), int(pred_kps[j,1]*4)), 5, (255,0,0), 2)

                    all_images = np.hstack(imgs)

                    cv2.imshow('Original htmaps {}'.format(i), np.array(_gthmap)[-1][0][:,:,i])
                    cv2.imshow('Predicted htamps {}'.format(i), all_images)
                    cv2.imshow('Image with predicted joints {}'.format(i), orig_image)
                    
            for k in range(8):
                layers = tuple()
                for i in range(out[-1].shape[-1]):
                    layers += (out[k][0,:,:,i],)
                cv2.imshow('Predicted heatmap output[{}] HG'.format(k), np.hstack(layers))
            
            cv2.waitKey(100)
            
            suc, bad, between_thresholds, mean, med = cal_heatmap_acc(out[-1], _meta, threshold)

            total_suc += suc
            total_fail += (bad + between_thresholds)
            total_suc_bigger += (between_thresholds + suc)
            total_fail_bigger += bad
            total_arr_mean.append(mean)
            total_arr_med.append(med)

        acc = total_suc * 1.0 / (total_fail + total_suc)
        acc_2 = total_suc_bigger * 1.0 / (total_fail_bigger + total_suc_bigger)
        self.acc_history.append(acc)
        self.acc2_history.append(acc_2)

        print('Eval Accuray [0.2] ', acc, '@ Epoch ', epoch)
        print('Eval Accuray [0.5] ', acc_2, '@ Epoch ', epoch)
        print('mean distance {}; median distance {}'.format(np.mean(total_arr_mean), np.median(total_arr_med)))
        print('AVG max distance {}; min distance {}'.format(np.max(total_arr_mean), np.min(total_arr_mean)))

#       # print best acc predicted keypoints to original image
        # if np.argmax(self.acc_history) == len(self.acc_history)-1:
        #     orig_image = cv2.resize(_img[i], dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
        #     for j in range(kp.shape[0]):
        #         cv2.circle(orig_image, (int(kp[j,0]), int(kp[j,1])), 5, (0,0,255), 2)
        #         cv2.circle(orig_image, (int(pred_kps[j,0]*7.5), int(pred_kps[j,1]*7.5)), 5, (255,0,0), 2)
        #     cv2.imshow('orig with heatmaps in {}. epoch'.format(epoch), orig_image)
            
        with open(os.path.join('./', 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + ':' + 
            str(np.mean(total_arr_mean)) + ':' + str(np.median(total_arr_med)) + ':' + 
            str(np.max(total_arr_mean)) + ':' + str(np.min(total_arr_mean)) + ':' + 
            str(np.max(total_arr_med)) + ':' + str(np.min(total_arr_med)) + ':' + 
            str(acc_2) + '\n')

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
