import argparse
import json
import os
import sys
from collections import defaultdict
import random

import numpy as np


def viou(traj_1, duration_1, traj_2, duration_2):
    """ compute the voluminal Intersection over Union
    for two trajectories, each of which is represented
    by a duration [fstart, fend) and a list of bounding
    boxes (i.e. traj) within the duration.
    """
    exit_flag = False
    if duration_1[1] - duration_1[0] != len(traj_1):
        print("The duration of duration1({}) doesnt match the traj1 shape: ({})"
              .format(duration_1, len(traj_1)))
        exit_flag = True
    if duration_2[1] - duration_2[0] != len(traj_2):
        print("The duration of duration2({}) doesnt match the traj2 shape: ({})"
              .format(duration_2, len(traj_2)))
        exit_flag = True

    if exit_flag is True:
        sys.exit(0)

    if duration_1[0] >= duration_2[1] or duration_1[1] <= duration_2[0]:
        return 0.
    elif duration_1[0] <= duration_2[0]:
        head_1 = duration_2[0] - duration_1[0]
        head_2 = 0
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    else:
        head_1 = 0
        head_2 = duration_1[0] - duration_2[0]
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    v_overlap = 0
    for i in range(tail_1 - head_1):
        roi_1 = traj_1[head_1 + i]
        roi_2 = traj_2[head_2 + i]
        left = max(roi_1[0], roi_2[0])
        top = max(roi_1[1], roi_2[1])
        right = min(roi_1[2], roi_2[2])
        bottom = min(roi_1[3], roi_2[3])
        v_overlap += (right - left + 1) * (bottom - top + 1)
    v1 = 0
    for i in range(len(traj_1)):
        v1 += (traj_1[i][2] - traj_1[i][0] + 1) * (traj_1[i][3] - traj_1[i][1] + 1)
    v2 = 0
    for i in range(len(traj_2)):
        v2 += (traj_2[i][2] - traj_2[i][0] + 1) * (traj_2[i][3] - traj_2[i][1] + 1)
    return float(v_overlap) / (v1 + v2 - v_overlap)


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC ap given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct ap calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_detection_scores(gt_actions, pred_actions, viou_threshold):
    # print(pred_actions.keys())
    pred_actions = sorted(pred_actions, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_actions),), dtype=bool)
    hit_scores = np.ones((len(pred_actions))) * -np.inf
    for pred_idx, pred_action in enumerate(pred_actions):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_action in enumerate(gt_actions):
            if not gt_detected[gt_idx] \
                    and pred_action['category'] == gt_action['category']:
                # print(gt_action.keys())
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
    # print(prec, rec, hit_scores)
    return prec, rec, hit_scores


def eval_tagging_scores(gt_actions, pred_actions):
    pred_actions = sorted(pred_actions, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(r['category'] for r in gt_actions)
    pred_triplets = []
    hit_scores = []
    for r in pred_actions:
        # triplet = tuple(r['triplet'])
        triplet = r['category']
        if triplet not in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if t not in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    # print(prec, rec, hit_scores)
    return prec, rec, hit_scores


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


def eval_visual_action(groundtruth, prediction, viou_threshold=0.5,
                       det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """
    evaluate visual action detection and visual action tagging.
    :param groundtruth: the dir of gt, e.g. groundtruth="/home/daivd/PycharmProjects/VORD/validation/"
    :param prediction: the path of pred, e.g. prediction="test/task2/3598080384_fks.json"
    :param viou_threshold:
    :param det_nreturns:
    :param tag_nreturns:
    :return:
    """

    print('evaluating...')

    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_actions = 0

    with open(prediction, 'r') as pred_f:
        prediction = json.load(pred_f)

    # find the groundtruth json file
    with open(os.path.join(groundtruth, 'gt_vid_path.json'), 'r') as dict_in_f:
        gt_vid_path_json = json.load(dict_in_f)

    process_index = 0
    all_videos_num = len(prediction['results'].keys())
    for each_vid_id in prediction['results'].keys():
        process_index += 1
        print("Now is evaluating: {}, process: {}/{}".format(each_vid_id, process_index, all_videos_num))

        # get groundtruth
        with open(os.path.join(groundtruth, gt_vid_path_json[each_vid_id]), 'r') as gt_f:
            groundtruth_json = json.load(gt_f)

        gt_version = groundtruth_json['version']
        pred_version = prediction['version']

        if gt_version != pred_version:
            print("The version of groundtruth({}) and prediction({}) are different!"
                  .format(gt_version, pred_version))
            sys.exit(0)

        gt_actions = []
        for each_ins in groundtruth_json['relation_instances']:

            begin_fid = each_ins['begin_fid']
            end_fid = each_ins['end_fid']

            each_ins_trajectory = []
            # end_fid += 1
            for each_traj in groundtruth_json['trajectories'][begin_fid:end_fid]:
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
                "duration": [begin_fid, end_fid],
                "trajectory": each_ins_trajectory
            }

            gt_actions.append(each_ins_action)

        predict_actions = prediction['results'][each_vid_id]

        det_prec, det_rec, det_scores = eval_detection_scores(
            gt_actions, predict_actions, viou_threshold)
        tag_prec, _, _ = eval_tagging_scores(gt_actions, predict_actions)

        # record per video evaluation results
        video_ap[each_vid_id] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            prec_at_n[nre].append(tag_prec[cut_off - 1])
        tot_gt_actions += len(gt_actions)

    # calculate mean ap for detection
    mAP = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_actions, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]

    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # print scores
    print('detection mAP (used in challenge): {}'.format(mAP))
    print('detection recall@50: {}'.format(rec_at_n[50]))
    print('detection recall@100: {}'.format(rec_at_n[100]))
    print('tagging precision@1: {}'.format(mprec_at_n[1]))
    print('tagging precision@5: {}'.format(mprec_at_n[5]))
    print('tagging precision@10: {}'.format(mprec_at_n[10]))
    return mAP, rec_at_n, mprec_at_n


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


if __name__ == '__main__':
    # trans_gt_2_subm_4mat('/home/daivd/PycharmProjects/VidVRD-helper/evaluation/test/task2/11566930393.json')
    merge_gt_sub_4mat(['/home/daivd/PycharmProjects/VidVRD-helper/evaluation/test/task2/2793806282_sub.json',
                       '/home/daivd/PycharmProjects/VidVRD-helper/evaluation/test/task2/3598080384_fks.json',
                       '/home/daivd/PycharmProjects/VidVRD-helper/evaluation/test/task2/11566930393_sub.json'])

    # parser = argparse.ArgumentParser(description='Video visual action evaluation.')
    # parser.add_argument('groundtruth_dir', type=str, help='Groundtruth json files diretory, e.g. '
    #                                                       '\'/home/daivd/PycharmProjects/VORD/validation/\'')
    # parser.add_argument('prediction_file', type=str, help='Prediction json file (submission format)')
    # args = parser.parse_args()
    #
    # gen_vid_path(args.groundtruth_dir)
    #
    # mAP, rec_at_n, mprec_at_n = eval_visual_action(args.groundtruth_dir, args.prediction_file)

    gen_vid_path('/home/daivd/PycharmProjects/VORD/validation')
    eval_visual_action('/home/daivd/PycharmProjects/VORD/validation/',
                       '/home/daivd/PycharmProjects/VidVRD-helper/evaluation/test/task2/2793806282_sub_merge.json')

    # with open('/home/daivd/PycharmProjects/VidVRD-helper/evaluation/test/task2/2793806282_sub_merge.json', 'r') as f:
    #     json_s = json.load(f)
    #     print(json_s['results']['2793806282'])

