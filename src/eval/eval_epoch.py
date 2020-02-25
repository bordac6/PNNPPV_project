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
    if gt_kp[0] > 1 and gt_kp[1] > 1:
        dif = np.linalg.norm(gt_kp[0:2] - pre_kp[0:2]) / norm
        if dif < threshold:
            # good prediction
            return 1, dif
        elif dif < threshold + 0.3:
            return 2, dif
        else:  # failed
            return 0, dif
    else:
        print('WRONG')
        print(gt_kp)
        return -1, 255

def heatmap_accuracy(predhmap, meta, norm, threshold):
    pred_kps = post_process_heatmap(predhmap)
    pred_kps = np.array(pred_kps)

    gt_kps = meta['tpts']
    scale = meta['scale']

    good_pred_count = 0
    failed_pred_count = 0
    almost_pred_count = 0
    arr_dif = []
    for i in range(gt_kps.shape[0]):
        dis, dif = cal_kp_distance(pred_kps[i, :], gt_kps[i, :] / scale, norm, threshold)
        if dis == 0:
            failed_pred_count += 1
        elif dis == 1:
            good_pred_count += 1
        elif dis == 2:
            almost_pred_count += 1
        if dif < 255:
            arr_dif.append(dif)

    return good_pred_count, failed_pred_count, almost_pred_count, arr_dif

def cal_heatmap_acc(prehmap, metainfo, threshold):
    sum_good, sum_fail, sum_almost = 0, 0, 0
    arr_mean, arr_med = [], []
    for i in range(prehmap.shape[0]):
        _prehmap = prehmap[i, :, :, :]
        good, bad, almost, arr_dif = heatmap_accuracy(_prehmap, metainfo[i], norm=4.5, threshold=threshold) #norm fitted on gtmap

        sum_good += good
        sum_fail += bad
        sum_almost += almost
        arr_mean.append(np.mean(arr_dif))
        arr_med.append(np.median(arr_dif))

    return sum_good, sum_fail, sum_almost, np.mean(arr_mean), np.median(arr_med)

def load_model(modeljson, modelfile):
    with open(modeljson) as f:
        model = model_from_json(f.read())
    model.load_weights(modelfile)
    return model

def run_eval(model_json, model_weights, epoch, show_outputs=False, acc_history=[], acc2_history=[]):
    model = load_model(model_json, model_weights)
    model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])
    _inres = (256, 256)
    _outres = (64, 64)
    orig_size = 480

    # dataset_path = os.path.join('D:\\', 'nyu_croped')
    # dataset_path = '/home/tomas_bordac/nyu_croped'
    dataset_path = config_reader.load_path()
    valdata = NYUHandDataGen('joint_data.mat', dataset_path, inres=_inres, outres=_outres, is_train=False, is_testtrain=False)

    total_suc, total_fail, total_suc_bigger, total_fail_bigger = 0, 0, 0, 0
    total_arr_mean, total_arr_med = [], []
    threshold = 0.2
    n_stacked = 2

    count = 0
    batch_size = 1
    for _img, _gthmap, _meta in valdata.generator(batch_size, n_stacked, sigma=3, is_shuffle=False, with_meta=True):

        count += batch_size
        if count > valdata.get_dataset_size():
            break

        out = model.predict(_img)

        if count+batch_size > valdata.get_dataset_size():
            for i in range(1):
                kp = _meta[i]['tpts']
                scale = _meta[i]['scale']
                orig_image = cv2.resize(_img[i], dsize=(orig_size, orig_size), interpolation=cv2.INTER_CUBIC)
                pred_kps = post_process_heatmap(out[-1][i])
                pred_kps = np.array(pred_kps)
                
                imgs = tuple()
                for j in range(out[-1].shape[-1]):
                    print_image = np.zeros(shape=(_outres[0], _outres[1],3))
                    print_image[:,:,0] = out[-1][i,:,:,j]
                    print_image[:,:,1] = out[-1][i,:,:,j]
                    print_image[:,:,2] = out[-1][i,:,:,j]
                    cv2.circle(print_image, (int(kp[j,0]/scale), int(kp[j,1]/scale)), 5, (0,0,255), 2)
                    cv2.circle(print_image, (int(pred_kps[j,0]), int(pred_kps[j,1])), 5, (255,0,0), 2)
                    imgs += (print_image,)
                    cv2.circle(orig_image, (int(kp[j,0]), int(kp[j,1])), 5, (0,0,255), 2)
                    cv2.circle(orig_image, (int(pred_kps[j,0]*scale), int(pred_kps[j,1]*scale)), 5, (255,0,0), 2)

                all_images = np.hstack(imgs)

                cv2.imshow('Original htmaps {}'.format(i), np.array(_gthmap)[-1][0][:,:,i])
                cv2.imshow('Predicted htamps {}'.format(i), all_images)
                cv2.imshow('Image with predicted joints {}'.format(i), orig_image)
                
        for k in range(n_stacked):
            layers = tuple()
            for i in range(out[-1].shape[-1]):
                layers += (out[k][0,:,:,i],)
            cv2.imshow('Predicted heatmap output[{}] HG'.format(k), np.hstack(layers))
        
        cv2.waitKey(0)
        
        suc, bad, between_thresholds, mean, med = cal_heatmap_acc(out[-1], _meta, threshold)

        total_suc += suc
        total_fail += (bad + between_thresholds)
        total_suc_bigger += (between_thresholds + suc)
        total_fail_bigger += bad
        total_arr_mean.append(mean)
        total_arr_med.append(med)

    acc = total_suc * 1.0 / (total_fail + total_suc)
    acc_2 = total_suc_bigger * 1.0 / (total_fail_bigger + total_suc_bigger)
    acc_history.append(acc)
    acc2_history.append(acc_2)

    print('Eval Accuray [0.2] ', acc, '@ Epoch ', epoch)
    print('Eval Accuray [0.5] ', acc_2, '@ Epoch ', epoch)
    print('mean distance {}; median distance {}'.format(np.mean(total_arr_mean), np.median(total_arr_med)))
    print('AVG max distance {}; min distance {}'.format(np.max(total_arr_mean), np.min(total_arr_mean)))

    with open(os.path.join('./', 'val.txt'), 'a+') as xfile:
        xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + ':' + 
        str(np.mean(total_arr_mean)) + ':' + str(np.median(total_arr_med)) + ':' + 
        str(np.max(total_arr_mean)) + ':' + str(np.min(total_arr_mean)) + ':' + 
        str(np.max(total_arr_med)) + ':' + str(np.min(total_arr_med)) + ':' + 
        str(acc_2) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_outputs", default=False, help="boolean to show predicted/gt heatmaps and points in test image")
    parser.add_argument("--all", default=False, help="boolean to validate all saved models in folder")
    parser.add_argument("--resume_model", help="start point to retrain")
    parser.add_argument("--resume_model_json", help="model json")

    args = parser.parse_args()
    if args.all:
        weights_paths = glob.glob("..\\..\\trained_models\\hg_nyu_102\\*.h5")
        for path in weights_paths:
            run_eval("..\\..\\trained_models\\hg_nyu_102\\net_arch.json", path, 1, args.show_outputs)
    else:
        # run_eval(args.resume_model_json, args.resume_model, 1, True)
        run_eval("..\\..\\trained_models_toshiba\\nyu\\net_arch.json", "..\\..\\trained_models_toshiba\\nyu\\weights_epoch0.h5", 1, args.show_outputs)
