import json
import os
import random
from argparse import ArgumentParser

import numpy as np

from .common import voc_ap, viou


def eval_detection_scores(gt_actions, pred_actions, viou_threshold):
    pred_actions = sorted(pred_actions, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_actions),), dtype=bool)
    hit_scores = np.ones((len(pred_actions))) * -np.inf
    for pred_idx, pred_action in enumerate(pred_actions):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_action in enumerate(gt_actions):
            if not gt_detected[gt_idx] \
                    and pred_action['id'] == gt_action['id']:
                ov = viou(pred_action['trajectory'], pred_action['duration'],
                          gt_action['trajectory'], gt_action['duration'])
                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            hit_scores[pred_idx] = pred_action['score']
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_actions), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def evaluate(prediction, groundtruth, viou_threshold=0.5):
    """
    evaluate visual action detection and visual action tagging.
    :param groundtruth: the dir of gt, e.g. groundtruth="/home/daivd/PycharmProjects/VORD/validation/"
    :param prediction: the path of pred, e.g. prediction="test/task2/3598080384_fks.json"
    :param viou_threshold:
    :param det_nreturns:
    :param tag_nreturns:
    :return:
    """
    gt_classes = set()
    for tracks in groundtruth.values():
        for traj in tracks:
            gt_classes.add(traj['category'])
    gt_class_num = len(gt_classes)

    prediction_actions = dict()
    for vid, tracks in prediction['results'].items():
        for traj in tracks:
            pred_action = {
                "id": vid,
                "score": traj['score'],
                "duration": traj['duration'],
                "trajectory": traj['trajectory']
            }
            if traj['category'] not in prediction_actions.keys():
                prediction_actions[traj['category']] = [pred_action]
            else:
                prediction_actions[traj['category']].append(pred_action)

    ap_class = dict()
    print('Computing average precision AP over {} classes...'.format(gt_class_num))

    for each_action in gt_classes:
        if each_action not in prediction_actions.keys():
            ap_class[each_action] = 0.
            continue

        groundtruth_actions = dict()
        for each_vid in groundtruth:
            # get groundtruth actions
            for each_gt_traj in groundtruth[each_vid]:
                if each_gt_traj['category'] == each_action:
                    gt_action = {
                        "id": each_vid,
                        "duration": each_gt_traj['duration'],
                        "trajectory": each_gt_traj['trajectory']
                    }
                    if each_action not in groundtruth_actions.keys():
                        groundtruth_actions[each_action] = [gt_action]
                    else:
                        groundtruth_actions[each_action].append(gt_action)

        pred_actions = prediction_actions[each_action]
        gt_actions = groundtruth_actions[each_action]

        det_prec, det_rec, det_scores = eval_detection_scores(
            gt_actions, pred_actions, viou_threshold)

        ap_class[each_action] = voc_ap(det_rec, det_prec)

    # compute mean ap and print
    print('=' * 25)
    ap_class_l = sorted(ap_class.items(), key=lambda ap_class: ap_class[0])
    for i, (cls, ap) in enumerate(ap_class_l):
        print('{}.{}\t{:.4f}'.format(i, cls, ap))
    mAP = np.mean(list(ap_class.values()))
    print('=' * 25)
    print('mAP\t{:.4f}'.format(mAP))

    return mAP, ap_class


def trans_gt_2_subm_4mat(groundtruth_json_path):
    with open(groundtruth_json_path, 'r') as gt_json_f:
        gt_json = json.load(gt_json_f)

    gt_actions = []
    for each_ins in gt_json['relation_instances']:

        begin_fid = each_ins['begin_fid']
        end_fid = each_ins['end_fid']

        each_ins_trajectory = []
        # end_fid += 1
        for each_traj in gt_json['trajectories'][begin_fid:end_fid]:
            for each_traj_obj in each_traj:
                if each_traj_obj['tid'] == each_ins['subject_tid']:
                    each_traj_frame = [
                        each_traj_obj['bbox']['xmin'],
                        each_traj_obj['bbox']['ymin'],
                        each_traj_obj['bbox']['xmax'],
                        each_traj_obj['bbox']['ymax']
                    ]
                    each_ins_trajectory.append(each_traj_frame)

        each_ins_action = {
            "category": each_ins['predicate'],
            "score": random.random(),
            "duration": [begin_fid, end_fid],
            "trajectory": each_ins_trajectory
        }

        gt_actions.append(each_ins_action)

    submiss_4mat = {
        "version": "VERSION 1.0",
        "results": {
            gt_json['video_id']: gt_actions
        },
        "external_data": {
            "used": True,
            "details": "First fully-connected layer from VGG-16 pre-trained on ILSVRC-2012 training set"
        }
    }
    with open(groundtruth_json_path[:-5] + '_sub.json', 'w+') as out_f:
        out_f.write(json.dumps(submiss_4mat))


def merge_gt_sub_4mat(gt_sub_4mat_path_list):
    results = {}
    for each_gt_path in gt_sub_4mat_path_list:
        with open(each_gt_path, 'r') as each_gt_f:
            each_gt_json = json.load(each_gt_f)
            video_id = each_gt_path.split('/')[-1][:-9]
            results[video_id] = each_gt_json['results'][video_id]
    # print(results.keys())
    merge_json = {
        "version": "VERSION 1.0",
        "results": results,
        "external_data": {
            "used": True,
            "details": "First fully-connected layer from VGG-16 pre-trained on ILSVRC-2012 training set"
        }
    }
    with open(gt_sub_4mat_path_list[0][:-5] + '_merge.json', 'w+') as out_f:
        out_f.write(json.dumps(merge_json))


def gen_vid_path(gt_dir):
    if os.path.isfile(os.path.join(gt_dir, 'gt_vid_path.json')):
        print('The video_id_path already exists! ')
    else:
        files_num = 0
        vid_path_dict = {}
        for root, dirs, files in os.walk(gt_dir):
            # for each_d in dirs:
            #     print(os.path.join(root, each_d))
            for each_f in files:
                files_num += 1
                # print(os.path.join(root, each_f))
                with open(os.path.join(root, each_f), 'r') as in_f:
                    each_json = json.load(in_f)
                    vid_path_dict[each_json['video_id']] = each_json['video_path'].replace('mp4', 'json')
        # print(files_num)
        print(gt_dir)
        with open(os.path.join(gt_dir, 'gt_vid_path.json'), 'w+') as out_f:
            out_f.write(json.dumps(vid_path_dict))
            print("The video_id_path is saved to {}".format(os.path.join(gt_dir, 'gt_vid_path.json')))


if __name__ == '__main__':
    """
    E.g.,
    python action_detection.py --groundtruth val_groundtruth.json --prediction val_prediction.json

    """
    parser = ArgumentParser(description='Video action detection evaluation.')
    parser.add_argument('--groundtruth', dest='gt_file', type=str, required=True,
                        help=('Groundtruth json file (please generate the file yourself',
                              ' referring to ../dataset/dataset.py:get_object_insts())'))
    parser.add_argument('--prediction', dest='pred_file', type=str, required=True, help='prediction file')
    args = parser.parse_args()

    print('Loading ground truth from ' + args.gt_file)
    assert os.path.exists(args.gt_file), args.gt_file + ' not found'
    with open(args.gt_file, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos: {}'.format(len(gt)))

    print('Loading prediction...')
    assert os.path.exists(args.pred_file), args.pred_file + ' not found'
    with open(args.pred_file, 'r') as fp:
        pred = json.load(fp)

    mean_ap, ap_class = evaluate(pred, gt)
