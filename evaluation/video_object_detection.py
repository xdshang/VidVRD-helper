# ====================================================
# @Time    : 2/25/19 9:11 AM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : video_object_detection.py
# ====================================================
import os
import os.path as osp
import json
import numpy as np
from argparse import ArgumentParser

from common import voc_ap


def IoU(bbox_1, bbox_2):
    """
    Get IoU value of two bboxes
    :param bbox_1:
    :param bbox_2:
    :return: IoU
    """
    w_1 = bbox_1[2] - bbox_1[0] + 1
    h_1 = bbox_1[3] - bbox_1[1] + 1
    w_2 = bbox_2[2] - bbox_2[0] + 1
    h_2 = bbox_2[3] - bbox_2[1] + 1
    area_1 = w_1 * h_1
    area_2 = w_2 * h_2

    overlap_bbox = (max(bbox_1[0], bbox_2[0]), max(bbox_1[1], bbox_2[1]),
                    min(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3]))
    overlap_w = max(0, (overlap_bbox[2] - overlap_bbox[0] + 1))
    overlap_h = max(0, (overlap_bbox[3] - overlap_bbox[1] + 1))

    overlap_area = overlap_w * overlap_h
    union_area = area_1 + area_2 - overlap_area
    IoU = overlap_area * 1.0 / union_area
    return IoU


def trajectory_overlap(gt_trajs, pred_traj):
    """
    Calculate overlap among trajectories
    :param gt_trajs:
    :param pred_traj:
    :param thresh_s:
    :return:
    """
    max_overlap = 0
    max_index = 0
    thresh_s = [0.5, 0.7, 0.9]
    for t, gt_traj in enumerate(gt_trajs):
        top1, top2, top3 = 0, 0, 0
        total = len(set(gt_traj.keys()) | set(pred_traj.keys()))
        for i, fid in enumerate(gt_traj):
            if fid not in pred_traj:
                continue
            sIoU = IoU(gt_traj[fid], pred_traj[fid])
            if sIoU >= thresh_s[0]:
                top1 += 1
                if sIoU >= thresh_s[1]:
                    top2 += 1
                    if sIoU >= thresh_s[2]:
                        top3 += 1

        tIoU = (top1 + top2 + top3) * 1.0 / (3 * total)

        if tIoU > max_overlap:
            max_overlap = tIoU
            max_index = t

    return max_overlap, max_index


def evaluate(pred, gt, use_07_metric=True, thresh_t=0.5):
    """
    Evaluate the predictions
    """
    gt_classes = set()
    for tracks in gt:
        for traj in tracks:
            gt_classes.add(traj['category'])
    gt_class_num = len(gt_classes)

    result_class = dict()
    results = pred['results']
    for vid, tracks in results.items():
        for traj in tracks:
            if traj['category'] not in result_class:
                result_class[traj['category']] = [[vid, traj['score'], traj['trajectory']]]
            else:
                result_class[traj['category']].append([vid, traj['score'], traj['trajectory']])

    ap_class = dict()
    print('Computing average precision AP over {} classes...'.format(gt_class_num))
    for c in gt_classes:
        if c not in result_class: 
            ap_class[c] = 0.
            continue
        npos = 0
        class_recs = {}

        for vid in gt:
            #print(vid)
            gt_trajs = [trk['trajectory'] for trk in gt[vid] if trk['category'] == c]
            det = [False] * len(gt_trajs)
            npos += len(gt_trajs)
            class_recs[vid] = {'trajectories': gt_trajs, 'det': det}

        trajs = result_class[c]
        vids = [trj[0] for trj in trajs]
        scores = np.array([trj[1] for trj in trajs])
        trajectories = [trj[2] for trj in trajs]

        nd = len(vids)
        fp = np.zeros(nd)
        tp = np.zeros(nd)

        sorted_inds = np.argsort(-scores)
        sorted_vids = [vids[id] for id in sorted_inds]
        sorted_traj = [trajectories[id] for id in sorted_inds]

        for d in range(nd):
            R = class_recs[sorted_vids[d]]
            gt_trajs = R['trajectories']
            pred_traj = sorted_traj[d]
            max_overlap, max_index = trajectory_overlap(gt_trajs, pred_traj)

            if max_overlap >= thresh_t:
                if not R['det'][max_index]:
                    tp[d] = 1.
                    R['det'][max_index] = True
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        ap_class[c] = ap
    
    # compute mean ap and print
    print('=' * 25)
    ap_class = sorted(ap_class.items(), key=lambda ap_class: ap_class[0])
    total_ap = 0.
    for i, (cls, ap) in enumerate(ap_class):
        print('{}.{}\t{:.4f}'.format(i, cls, ap))
        total_ap += ap
    mean_ap = total_ap / gt_class_num 
    print('=' * 25)
    print('mAP\t{:.4f}'.format(mean_ap))

    return mean_ap, ap_class


if __name__ == "__main__":
    """

    E.g.,
    python video_object_detection.py --groundtruth val_groundtruth.json --prediction val_prediction.json

    """
    parser = ArgumentParser(description='Video object detection evaluation.')
    parser.add_argument('--groundtruth', dest='gt_file', type=str, required=True,
            help=('Groundtruth json file (please generate the file yourself',
                ' referring to ../dataset/dataset.py:get_object_insts())'))
    parser.add_argument('--prediction', dest='pred_file', type=str, required=True, help='prediction file')
    args = parser.parse_args()
    
    print('Loading ground truth from ' + args.gt_file)
    assert osp.exists(args.gt_file), args.gt_file + ' not found'
    with open(args.gt_file, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos: {}'.format(len(gt)))

    print('Loading prediction...')
    assert osp.exists(args.pred_file), args.pred_file + ' not found'
    with open(args.pred_file, 'r') as fp:
        pred = json.load(fp)

    mean_ap, ap_class = evaluate(pred, gt)
