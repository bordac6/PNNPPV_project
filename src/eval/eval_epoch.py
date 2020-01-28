import sys
import argparse
import numpy as np

sys.path.insert(0, "../data_gen/")
from nyuhand_datagen import NYUHandDataGen
from heatmap_process import post_process_heatmap
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error

def cal_kp_distance(pre_kp, gt_kp, norm, threshold):
    print('pre_kp: {}'.format(pre_kp))
    print('gt_kp: {}'.format(gt_kp))

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
        dis = cal_kp_distance(pred_kps[i, :] * 7.5, gt_kps[i, :], norm, threshold)
        if dis == 0:
            failed_pred_count += 1
        elif dis == 1:
            good_pred_count += 1

    return good_pred_count, failed_pred_count

def cal_heatmap_acc(prehmap, metainfo, threshold):
    sum_good, sum_fail = 0, 0
    for i in range(prehmap.shape[0]):
        _prehmap = prehmap[i, :, :, :]
        good, bad = heatmap_accuracy(_prehmap, metainfo[i], norm=6.4, threshold=threshold)

        sum_good += good
        sum_fail += bad

    return sum_good, sum_fail

def load_model(modeljson, modelfile):
    with open(modeljson) as f:
        model = model_from_json(f.read())
    model.load_weights(modelfile)
    return model

def run_eval(model_json, model_weights, epoch):
    model = load_model(model_json, model_weights)
    model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])

    dataset_path = '/home/tomas_bordac/nyu_croped'
    valdata = NYUHandDataGen('joint_data.mat', dataset_path, inres=(256, 256), outres=(64, 64), is_train=False)

    total_suc, total_fail = 0, 0
    threshold = 0.5

    count = 0
    batch_size = 8
    for _img, _gthmap, _meta in valdata.generator(batch_size, 2, sigma=2, is_shuffle=False, with_meta=True):

        count += batch_size
        if count > valdata.get_dataset_size():
            break

        out = model.predict(_img)

        suc, bad = cal_heatmap_acc(out[-1], _meta, threshold)

        total_suc += suc
        total_fail += bad

        print('success: {}'.format(suc))
        print('bad: {}'.format(bad))
        input()

    acc = total_suc * 1.0 / (total_fail + total_suc)

    print('Eval Accuray ', acc, '@ Epoch ', epoch)

    with open(os.path.join('./', 'val.txt'), 'a+') as xfile:
        xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_model", help="start point to retrain")
    parser.add_argument("--resume_model_json", help="model json")

    args = parser.parse_args()

    run_eval(args.resume_model_json, args.resume_model, 1)
