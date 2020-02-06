import sys
import argparse
import numpy as np
import cv2
import os
import glob

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../tools/")

import config_reader
from nyuhand_datagen import NYUHandDataGen
from heatmap_process import post_process_heatmap
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error

def cal_kp_distance(pre_kp, gt_kp, norm, threshold):
    # print('prediction: {}'.format(pre_kp))
    # print('ground true: {}'.format(gt_kp))
    # print('Euklidean distance: {}'.format(np.linalg.norm(gt_kp[0:2] - pre_kp[0:2])))
    # print('Euklidean div by norm: {}'.format(np.linalg.norm(gt_kp[0:2] - pre_kp[0:2])/norm))
    
    if gt_kp[0] > 1 and gt_kp[1] > 1:
        dif = np.linalg.norm(gt_kp[0:2] - pre_kp[0:2]) / norm
        if dif < threshold:
            # good prediction
            return 1, dif
        else:  # failed
            return 0, dif
    else:
        return -1, dif

def heatmap_accuracy(predhmap, meta, norm, threshold):
    pred_kps = post_process_heatmap(predhmap)
    pred_kps = np.array(pred_kps)

    gt_kps = meta['tpts']

    good_pred_count = 0
    failed_pred_count = 0
    arr_dif = []
    for i in range(gt_kps.shape[0]):
        dis, dif = cal_kp_distance(pred_kps[i, :], gt_kps[i, :] / 7.5, norm, threshold)
        if dis == 0:
            failed_pred_count += 1
        elif dis == 1:
            good_pred_count += 1
        arr_dif.append(dif)

    return good_pred_count, failed_pred_count, arr_dif

def cal_heatmap_acc(prehmap, metainfo, threshold):
    sum_good, sum_fail = 0, 0
    arr_mean, arr_med = [], []
    for i in range(prehmap.shape[0]):
        _prehmap = prehmap[i, :, :, :]
        good, bad, arr_dif = heatmap_accuracy(_prehmap, metainfo[i], norm=4.5, threshold=threshold) #norm fitted on gtmap

        sum_good += good
        sum_fail += bad
        arr_mean.append(np.mean(arr_dif))
        arr_med.append(np.median(arr_dif))

    return sum_good, sum_fail, np.mean(arr_mean), np.median(arr_med)

def load_model(modeljson, modelfile):
    with open(modeljson) as f:
        model = model_from_json(f.read())
    model.load_weights(modelfile)
    return model

def run_eval(model_json, model_weights, epoch, show_outputs=False):
    model = load_model(model_json, model_weights)
    model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])

    # dataset_path = '/home/tomas_bordac/nyu_croped'
    # dataset_path = '..\\..\\data\\nyu_croped'
    # dataset_path = os.path.join('D:\\', 'nyu_croped')
    dataset_path = config_reader.load_path()
    valdata = NYUHandDataGen('joint_data.mat', dataset_path, inres=(256, 256), outres=(64, 64), is_train=False, is_testtrain=False)

    total_suc, total_fail = 0, 0
    total_arr_mean, total_arr_med = [], []
    threshold = 1

    count = 0
    batch_size = 1
    for _img, _gthmap, _meta in valdata.generator(batch_size, 2, sigma=3, is_shuffle=False, with_meta=True):

        count += batch_size
        if count > valdata.get_dataset_size():
            break

        out = model.predict(_img)
        pred_map_batch = out[-1]

        if show_outputs:
            for i in range(batch_size):
                orig_image = cv2.resize(_img[i], dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
                kpanno = _meta[i]['tpts']
                pred_heatmaps = pred_map_batch[i]

                pred_kps = post_process_heatmap(pred_heatmaps)
                pred_kps = np.array(pred_kps)
                hmaps_to_print = tuple()
                for j in range(kpanno.shape[0]):
                    hmap = np.zeros(shape=(64,64,3))
                    hmap[:,:,0] = out[-1][i,:,:,j]
                    hmap[:,:,1] = out[-1][i,:,:,j]
                    hmap[:,:,2] = out[-1][i,:,:,j]
                    #to heatmaps
                    cv2.circle(hmap, (int(kpanno[j,0]/7.5), int(kpanno[j,1]/7.5)), 5, (0,0,255), 2)
                    cv2.circle(hmap, (int(pred_kps[j,0]), int(pred_kps[j,1])), 5, (255,0,0), 2)
                    hmaps_to_print += (hmap,)
                    #to original image
                    cv2.circle(orig_image, (int(kpanno[j,0]), int(kpanno[j,1])), 5, (0,0,255), 2)
                    cv2.circle(orig_image, (int(pred_kps[j,0]*7.5), int(pred_kps[j,1]*7.5)), 5, (255,0,0), 2)

                cv2.imshow('pred heatmaps with gt and pred kps', np.hstack(hmaps_to_print))
                cv2.imshow('orig image with gt and pred kps', orig_image)
                cv2.waitKey(0)

        suc, bad, mean, med = cal_heatmap_acc(out[-1], _meta, threshold)

        total_suc += suc
        total_fail += bad
        total_arr_mean.append(mean)
        total_arr_med.append(med)

    acc = total_suc * 1.0 / (total_fail + total_suc)

    print('Eval Accuray ', acc, '@ Epoch ', epoch)
    print('mean distance {}; median distance {}'.format(np.mean(total_arr_mean), np.median(total_arr_med)))

    # with open(os.path.join('./', 'val.txt'), 'a+') as xfile:
    #     xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + ':' + str(np.mean(total_arr_mean)) + ':' + str(np.median(total_arr_med)) '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_outputs", default=False, help="boolean to show predicted/gt heatmaps and points in test image")
    parser.add_argument("--all", default=False, help="boolean to validate all saved models in folder")
    # parser.add_argument("--resume_model", help="start point to retrain")
    # parser.add_argument("--resume_model_json", help="model json")

    args = parser.parse_args()
    if args.all:
        weights_paths = glob.glob("..\\..\\trained_models\\hg_nyu_102\\*.h5")
        for path in weights_paths:
            run_eval("..\\..\\trained_models\\hg_nyu_102\\net_arch.json", path, 1, args.show_outputs)
    else:
        # run_eval(args.resume_model_json, args.resume_model, 1)
        run_eval("..\\..\\trained_models\\hg_nyu_102\\net_arch.json", "..\\..\\trained_models\\hg_nyu_102\\weights_epoch10.h5", 1, args.show_outputs)
