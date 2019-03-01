import argparse
import json
import os
import sys
import random
from collections import defaultdict

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
                    and pred_action['category'] == gt_action['category']:
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


def evaluate(groundtruth, prediction, viou_threshold=0.5):
    """
    evaluate visual action detection and visual action tagging.
    :param groundtruth: the dir of gt, e.g. groundtruth="/home/daivd/PycharmProjects/VORD/validation/"
    :param prediction: the path of pred, e.g. prediction="test/task2/3598080384_fks.json"
    :param viou_threshold:
    :param det_nreturns:
    :param tag_nreturns:
    :return:
    """
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_actions = 0

    process_index = 0
    all_videos_num = len(prediction['results'].keys())
    for each_vid_id in prediction['results'].keys():
        process_index += 1
        print("Now is evaluating: {}, process: {}/{}".format(each_vid_id, process_index, all_videos_num))

        gt_actions = groundtruth[each_vid_id]
        predict_actions = prediction['results'][each_vid_id]

        det_prec, det_rec, det_scores = eval_detection_scores(
            gt_actions, predict_actions, viou_threshold)
        tag_prec, _, _ = eval_tagging_scores(gt_actions, predict_actions)

        # record per video evaluation results
        video_ap[each_vid_id] = voc_ap(det_rec, det_prec)
        # tp = np.isfinite(det_scores)
        # for nre in det_nreturns:
        #     cut_off = min(nre, det_scores.size)
        #     tot_scores[nre].append(det_scores[:cut_off])
        #     tot_tp[nre].append(tp[:cut_off])
        # for nre in tag_nreturns:
        #     cut_off = min(nre, tag_prec.size)
        #     prec_at_n[nre].append(tag_prec[cut_off - 1])
        # tot_gt_actions += len(gt_actions)

    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))
    print('detection mAP (used in challenge): {}'.format(mean_ap))
    return mean_ap, video_ap


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
    # find the groundtruth json file
    with open(os.path.join(groundtruth, 'gt_vid_path.json'), 'r') as dict_in_f:
        gt_vid_path_json = json.load(dict_in_f)
    groundtruth = dict()
    for each_vid_id in gt_vid_path_json:
        # get groundtruth
        with open(os.path.join(groundtruth, gt_vid_path_json[each_vid_id]), 'r') as gt_f:
            groundtruth_json = json.load(gt_f)
        # get action instances
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
        groundtruth[each_vid_id] = gt_actions

    with open('/home/daivd/PycharmProjects/VidVRD-helper/evaluation/test/task2/2793806282_sub_merge.json', 'r') as pred_f:
        prediction = json.load(pred_f)

    evaluate('/home/daivd/PycharmProjects/VORD/validation/', prediction)

    # with open('/home/daivd/PycharmProjects/VidVRD-helper/evaluation/test/task2/2793806282_sub_merge.json', 'r') as f:
    #     json_s = json.load(f)
    #     print(json_s['results']['2793806282'])

