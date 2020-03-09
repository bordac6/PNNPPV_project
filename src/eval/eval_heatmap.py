from heatmap_process import post_process_heatmap
import data_process
import numpy as np
import copy


def get_predicted_kp_from_htmap(heatmap, meta, outres):
    # nms to get location
    kplst = post_process_heatmap(heatmap)
    kps = np.array(kplst)

    # use meta information to transform back to original image
    mkps = copy.copy(kps)
    for i in range(kps.shape[0]):
        mkps[i, 0:2] *= 7.5 
    #     mkps[i, 0:2] = data_process.transform(kps[i], meta['center'], meta['scale'], res=outres, invert=1, rot=0)

    return mkps


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


def heatmap_accuracy(predhmap, meta, norm, threshold, joints_acc):
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
            joints_acc[i] += 1
        elif dis == 2:
            almost_pred_count += 1
            joints_acc[i] += 1
        if dif < 255:
            arr_dif.append(dif)

    return good_pred_count, failed_pred_count, almost_pred_count, arr_dif


def cal_heatmap_acc(prehmap, metainfo, threshold, joints_acc):
    sum_good, sum_fail, sum_almost = 0, 0, 0
    arr_mean, arr_med = [], []
    for i in range(prehmap.shape[0]):
        _prehmap = prehmap[i, :, :, :]
        good, bad, almost, arr_dif = heatmap_accuracy(_prehmap, metainfo[i], norm=4.5, threshold=threshold, joints_acc=joints_acc) #norm fitted on gtmap NYUhand

        sum_good += good
        sum_fail += bad
        sum_almost += almost
        arr_mean.append(np.mean(arr_dif))
        arr_med.append(np.median(arr_dif))

    return sum_good, sum_fail, sum_almost, np.mean(arr_mean), np.median(arr_med)
