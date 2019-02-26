# ====================================================
# @Time    : 2/25/19 9:11 AM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : evaluate_vid.py
# ====================================================
import os
import os.path as osp
import json
import numpy as np
from  argparse import ArgumentParser


def IoU(bbox_1, bbox_2):
    """
    get IoU value of two bboxes
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
    overlap_w = max(0, (overlap_bbox[2] - overlap_bbox[0]+1))
    overlap_h = max(0, (overlap_bbox[3] - overlap_bbox[1]+1))

    overlap_area = overlap_w * overlap_h
    union_area = area_1 + area_2 - overlap_area

    IoU = overlap_area*1.0/union_area


    return IoU


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).

  Adopted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
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
    # correct AP calculation
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


def trajectory_overlap(gt_trajs, pred_traj):
    """
    calculate overlap among trajectories
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
        total = len(set(gt_traj.keys()|set(pred_traj.keys())))
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

        tIoU = (top1+top2+top3)*1.0 / (3*total)

        if tIoU > max_overlap:
            max_overlap = tIoU
            max_index = t

    return max_overlap, max_index


def evaluate(pred_file, gt_file, use_07_metric=True, thresh_t = 0.5):
    """
    Evaluate the predictions
    :param pred_file:
    :param gt_file:
    :param classname_file:
    :param cache_file:
    :return:
    """
    print('Loading prediction and ground truth file...')
    assert osp.exists(pred_file), pred_file+' not found'
    assert osp.exists(gt_file), gt_file + ' not found'

    with open(pred_file, 'r') as fp:
        pred = json.load(fp)
    with open(gt_file, 'r') as fp:
        gt = json.load(fp)

    print('Parsing prediction...')
    result_class = dict()
    results = pred['results']
    for vid, tracks in results.items():
        for traj in tracks:
            if traj['category'] not in result_class:
                result_class[traj['category']] = [[vid, traj['score'], traj['trajectory']]]
            else:
                result_class[traj['category']].append([vid, traj['score'], traj['trajectory']])

    ap_class = dict()
    print('Computing average precision AP...')
    for c in result_class:
        # if c != 'adult': continue
        npos = 0
        class_recs = {}
        gt_track = gt['results']
        for vid in gt_track:
            gt_trajs = [trk['trajectory'] for trk in gt_track[vid] if trk['category'] == c]
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

    return ap_class


def main(args):

    pred_file = osp.join(args.data_dir, args.pred_file)
    gt_file = osp.join(args.data_dir, args.gt_file)
    ap_class = evaluate(pred_file, gt_file, True)
    ap_class = sorted(ap_class.items(), key=lambda ap_class:ap_class[0])

    total_ap = 0.
    print('='*25)

    for i,(cls, ap) in enumerate(ap_class):
        print('{}.{}\t{:.4f}'.format(i, cls, ap))
        total_ap += ap
    print('=' * 25)
    print('mAP\t{:.4f}'.format(total_ap/len(ap_class)))


if __name__ == "__main__":
    """
    
    E.g.,
    python evaluate_vid.py --data_dir data --groundtruth val_gt.json --prediction val_prediction.json
    (Make sure both groundtruth file and prediction file are under the folder 'data/')

    """
    parser = ArgumentParser(description='Video object detection evaluation.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data', required=True, help='data path')
    parser.add_argument('--groundtruth', dest='gt_file', type=str, default='val_gt.json', required=True, help='groundtruth file')
    parser.add_argument('--prediction', dest='pred_file', type=str, default='val_prediction.json', required=True,  help='prediction file')
    args = parser.parse_args()
    main(args)
    
