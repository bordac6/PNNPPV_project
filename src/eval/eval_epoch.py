import sys
import argparse
import numpy as np
import cv2
import os

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../tools/")

import config_reader
from nyuhand_datagen import NYUHandDataGen
from heatmap_process import post_process_heatmap
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error

def cal_kp_distance(pre_kp, gt_kp, norm, threshold):
    print('prediction: {}'.format(pre_kp))
    print('ground true: {}'.format(gt_kp))
    print('Euklidean distance: {}'.format(np.linalg.norm(gt_kp[0:2] - pre_kp[0:2])))
    print('Euklidean div by norm: {}'.format(np.linalg.norm(gt_kp[0:2] - pre_kp[0:2])/norm))
    
    if gt_kp[0] > 1 and gt_kp[1] > 1:
        dif = np.linalg.norm(gt_kp[0:2] - pre_kp[0:2]) / norm
        if dif < threshold:
            # good prediction
            return 1
        else:  # failed
            return 0
    else:
        return -1

def heatmap_accuracy(predhmap, meta, norm, threshold):
    pred_kps = post_process_heatmap(predhmap)
    pred_kps = np.array(pred_kps)

    gt_kps = meta['tpts']

    good_pred_count = 0
    failed_pred_count = 0
    for i in range(gt_kps.shape[0]):
        dis = cal_kp_distance(pred_kps[i, :], gt_kps[i, :] / 7.5, norm, threshold)
        if dis == 0:
            failed_pred_count += 1
        elif dis == 1:
            good_pred_count += 1

    return good_pred_count, failed_pred_count

def cal_heatmap_acc(prehmap, metainfo, threshold):
    sum_good, sum_fail = 0, 0
    for i in range(prehmap.shape[0]):
        _prehmap = prehmap[i, :, :, :]
        good, bad = heatmap_accuracy(_prehmap, metainfo[i], norm=4.5, threshold=threshold)

        sum_good += good
        sum_fail += bad

    return sum_good, sum_fail

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

                for j in range(kpanno.shape[0]):
                    x = kpanno[j, 0]
                    y = kpanno[j, 1]
                    orig_image = cv2.circle(orig_image, (int(x), int(y)), 5, (0,0,255), 2)

                for j in range(pred_kps.shape[0]):
                    x = pred_kps[j, 0] * 4
                    y = pred_kps[j, 1] * 4
                    _img[i] = cv2.circle(_img[i], (int(x), int(y)), 5, (0,0,255), 2)

                cv2.imshow('orig with heatmaps', orig_image)
                cv2.imshow('gt heatmap', np.sum(_gthmap[-1][i], axis=-1))

                cv2.imshow('pred with heatmaps', _img[i])
                cv2.imshow('pred heatmaps', cv2.resize(np.sum(pred_heatmaps, axis=-1), dsize=(256, 256), interpolation=cv2.INTER_CUBIC))
                cv2.imshow('pred heatmap', cv2.resize(pred_heatmaps[0], dsize=(256, 256), interpolation=cv2.INTER_CUBIC))
                
                cv2.waitKey(0)

        suc, bad = cal_heatmap_acc(out[-1], _meta, threshold)

        total_suc += suc
        total_fail += bad

        print('success: {}'.format(suc))
        print('bad: {}'.format(bad))
        if bad > 0:
            input()

    acc = total_suc * 1.0 / (total_fail + total_suc)

    print('Eval Accuray ', acc, '@ Epoch ', epoch)

    # with open(os.path.join('./', 'val.txt'), 'a+') as xfile:
    #     xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_outputs", default=False, help="boolean to show predicted/gt heatmaps and points in test image")
    # parser.add_argument("--resume_model", help="start point to retrain")
    # parser.add_argument("--resume_model_json", help="model json")

    args = parser.parse_args()

    # run_eval(args.resume_model_json, args.resume_model, 1)
    run_eval('..\\..\\trained_models\\hg_nyu_101\\net_arch.json', '..\\..\\trained_models\\hg_nyu_101\\weights_epoch190.h5', 1, args.show_outputs)
