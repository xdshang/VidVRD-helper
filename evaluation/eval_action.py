import argparse
import json
import sys

import numpy as np


def viou(traj_1, duration_1, traj_2, duration_2):
    """ compute the voluminal Intersection over Union
    for two trajectories, each of which is represented
    by a duration [fstart, fend) and a list of bounding
    boxes (i.e. traj) within the duration.
    """
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
    # print(type(gt_actions), gt_actions)
    # print(type(pred_actions), pred_actions)

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
    print(prec, rec, hit_scores)
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
    print(prec, rec, hit_scores)
    return prec, rec, hit_scores


def eval_visual_action(groundtruth, prediction, viou_threshold=0.5,
                       det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual action detection and visual
    action tagging.
    """
    print('evaluating...')

    gt_version = groundtruth['version']
    pred_version = prediction['version']
    if gt_version != pred_version:
        print("The version of groundtruth({}) and prediction({}) are different!"
              .format(gt_version, pred_version))
        sys.exit(0)

    gt_video_id = groundtruth['video_id']
    pred_video_id = list(prediction['results'].keys())[0]
    if gt_video_id != pred_video_id:
        print("The video id of groundtruth({}) and prediction({}) are different!"
              .format(gt_video_id, pred_video_id))
        sys.exit(0)

    # get groundtruth
    gt_actions = []
    for each_ins in groundtruth['relation_instances']:

        begin_fid = each_ins['begin_fid']
        end_fid = each_ins['end_fid']

        each_ins_trajectory = []
        # end_fid += 1
        for each_traj in groundtruth['trajectories'][begin_fid:end_fid]:
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

    # print(json.dumps({"a": actions_instances}))

    pred_actions = prediction['results'][pred_video_id]

    eval_tagging_scores(gt_actions, pred_actions)

    eval_detection_scores(gt_actions, pred_actions, viou_threshold)

    # video_ap = dict()
    # tot_scores = defaultdict(list)
    # tot_tp = defaultdict(list)
    # prec_at_n = defaultdict(list)
    # tot_gt_actions = 0
    # for vid, gt_actions in groundtruth.items():
    #     # predict_actions = prediction[vid]
    #     # print(vid, predict_actions)
    #     if vid == 'version':
    #         continue
    #     predict_actions = prediction['results'][gt_actions]
    #     gt_actions = groundtruth['relation_instances']
    #     det_prec, det_rec, det_scores = eval_detection_scores(
    #         gt_actions, predict_actions, viou_threshold)
    #     tag_prec, _, _ = eval_tagging_scores(gt_actions, predict_actions)
    #     # record per video evaluation results
    #     video_ap[vid] = voc_ap(det_rec, det_prec)
    #     tp = np.isfinite(det_scores)
    #     for nre in det_nreturns:
    #         cut_off = min(nre, det_scores.size)
    #         tot_scores[nre].append(det_scores[:cut_off])
    #         tot_tp[nre].append(tp[:cut_off])
    #     for nre in tag_nreturns:
    #         cut_off = min(nre, tag_prec.size)
    #         prec_at_n[nre].append(tag_prec[cut_off - 1])
    #     tot_gt_actions += len(gt_actions)
    # # calculate mean ap for detection
    # mAP = np.mean(video_ap.values())
    # # calculate recall for detection
    # rec_at_n = dict()
    # for nre in det_nreturns:
    #     scores = np.concatenate(tot_scores[nre])
    #     tps = np.concatenate(tot_tp[nre])
    #     sort_indices = np.argsort(scores)[::-1]
    #     tps = tps[sort_indices]
    #     cum_tp = np.cumsum(tps).astype(np.float32)
    #     rec = cum_tp / np.maximum(tot_gt_actions, np.finfo(np.float32).eps)
    #     rec_at_n[nre] = rec[-1]
    # # calculate mean precision for tagging
    # mprec_at_n = dict()
    # for nre in tag_nreturns:
    #     mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # # print scores
    # print('detection mAP (used in challenge): {}'.format(mAP))
    # print('detection recall@50: {}'.format(rec_at_n[50]))
    # print('detection recall@100: {}'.format(rec_at_n[100]))
    # print('tagging precision@1: {}'.format(mprec_at_n[1]))
    # print('tagging precision@5: {}'.format(mprec_at_n[5]))
    # print('tagging precision@10: {}'.format(mprec_at_n[10]))
    # return mAP, rec_at_n, mprec_at_n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video visual action evaluation.')
    parser.add_argument('groundtruth_file', type=str, help=('Groundtruth json file (please'
                                                            ' generate the file yourself or use the api provided in evaluate.py in the'
                                                            ' repository https://github.com/xdshang/VidVRD-helper)'))
    parser.add_argument('prediction_file', type=str, help='Prediction json file')
    args = parser.parse_args()

    with open(args.groundtruth_file, 'r') as fin:
        groundtruth_json = json.load(fin)
    with open(args.prediction_file, 'r') as fin:
        prediction_json = json.load(fin)

    # with open('test/task2/3598080384.json', 'r') as gt_f:
    #     groundtruth_json = json.load(gt_f)
    # with open('test/task2/task2.json', 'r') as gt_f:
    #     prediction_json = json.load(gt_f)

    mAP, rec_at_n, mprec_at_n = eval_visual_action(groundtruth_json, prediction_json)
