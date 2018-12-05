from dlib import drectangle, correlation_tracker
import numpy as np

import os
import json
import cPickle as pkl
from collections import defaultdict, deque

from baseline import *


class Trajectory():
    """
    Object trajectory class that holds the bounding box trajectory and appearance feature (classeme)
    """
    def __init__(self, pstart, pend, rois, score, category, classeme, vsig=None, gt_trackid=-1):
        """
        bbox: drectangle
        """
        assert len(rois)==pend-pstart
        self.pstart = pstart
        self.pend = pend
        self.rois = deque(drectangle(*roi) for roi in rois)
        self.score = score
        self.category = category
        self.classeme = classeme
        # video signature
        self.vsig = vsig
        self.gt_trackid = gt_trackid

    def __lt__(self, other):
        return self.score < other.score

    def head(self):
        return self.rois[0]

    def tail(self):
        return self.rois[-1]

    def at(self, i):
        """
        Return the i_th bounding box
        Support fancy indexing
        """
        return self.rois[i]

    def roi_at(self, p):
        """
        Return the bounding box at frame p
        """
        return self.rois[p - self.pstart]

    def bbox_at(self, p):
        """
        return bbox in cv2 format
        """
        roi = self.rois[p - self.pstart]
        return (roi.left(), roi.top(), roi.width(), roi.height())

    def length(self):
        return self.pend - self.pstart

    def predict(self, roi, reverse=False):
        if reverse:
            self.rois.appendleft(roi)
            self.pstart -= 1
        else:
            self.rois.append(roi)
            self.pend += 1
        return roi

    def serialize(self):
        obj = dict()
        obj['pstart'] = int(self.pstart)
        obj['pend'] = int(self.pend)
        obj['rois'] = [(bbox.left(), bbox.top(), bbox.right(), bbox.bottom()) for bbox in self.rois]
        obj['score'] = float(self.score)
        obj['category'] = int(self.category)
        obj['classeme'] = [float(x) for x in self.classeme]
        obj['vsig'] = self.vsig
        obj['gt_trackid'] = self.gt_trackid
        return obj


def _intersect(bboxes1, bboxes2):
    """
    bboxes: t x n x 4
    """
    assert bboxes1.shape[0] == bboxes2.shape[0]
    t = bboxes1.shape[0]
    inters = np.zeros((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    _min = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    _max = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    w = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    h = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    for i in range(t):
        np.maximum.outer(bboxes1[i, :, 0], bboxes2[i, :, 0], out = _min)
        np.minimum.outer(bboxes1[i, :, 2], bboxes2[i, :, 2], out = _max)
        np.subtract(_max + 1, _min, out = w)
        w.clip(min = 0, out = w)
        np.maximum.outer(bboxes1[i, :, 1], bboxes2[i, :, 1], out = _min)
        np.minimum.outer(bboxes1[i, :, 3], bboxes2[i, :, 3], out = _max)
        np.subtract(_max + 1, _min, out = h)
        h.clip(min = 0, out = h)
        np.multiply(w, h, out = w)
        inters += w
    return inters


def _union(bboxes1, bboxes2):
    if id(bboxes1) == id(bboxes2):
        w = bboxes1[:, :, 2] - bboxes1[:, :, 0] + 1
        h = bboxes1[:, :, 3] - bboxes1[:, :, 1] + 1
        area = np.sum(w * h, axis = 0)
        unions = np.add.outer(area, area)
    else:
        w = bboxes1[:, :, 2] - bboxes1[:, :, 0] + 1
        h = bboxes1[:, :, 3] - bboxes1[:, :, 1] + 1
        area1 = np.sum(w * h, axis = 0)
        w = bboxes2[:, :, 2] - bboxes2[:, :, 0] + 1
        h = bboxes2[:, :, 3] - bboxes2[:, :, 1] + 1
        area2 = np.sum(w * h, axis = 0)
        unions = np.add.outer(area1, area2)
    return unions


def cubic_iou(bboxes1, bboxes2):
    # bboxes: n x t x 4 (left, top, right, bottom)
    if id(bboxes1) == id(bboxes2):
        bboxes1 = bboxes1.transpose((1, 0, 2))
        bboxes2 = bboxes1
    else:
        bboxes1 = bboxes1.transpose((1, 0, 2))
        bboxes2 = bboxes2.transpose((1, 0, 2))
    # compute cubic-IoU
    # bboxes: t x n x 4
    iou = _intersect(bboxes1, bboxes2)
    union = _union(bboxes1, bboxes2)
    np.subtract(union, iou, out = union)
    np.divide(iou, union, out = iou)
    return iou


def traj_iou(trajs1, trajs2):
    """
    Compute the pairwise trajectory IoU in trajs1 and trajs2.
    Assumuing all trajectories in trajs1 and trajs2 start at same frame and
    end at same frame.
    """
    bboxes1 = np.asarray([[[roi.left(), roi.top(), roi.right(), roi.bottom()] 
            for roi in traj.rois] for traj in trajs1])
    if id(trajs1) == id(trajs2):
        bboxes2 = bboxes1
    else:
        bboxes2 = np.asarray([[[roi.left(), roi.top(), roi.right(), roi.bottom()] 
                for roi in traj.rois] for traj in trajs2])
    iou = cubic_iou(bboxes1, bboxes2)
    return iou


def object_trajectory_proposal(dataset, vid, fstart, fend, gt=False, verbose=False):
    """
    Set gt=True for providing groundtruth bounding box trajectories and
    predicting classme feature only
    """
    vsig = get_segment_signature(vid, fstart, fend)
    name = 'traj_cls_gt' if gt else 'traj_cls'
    path = get_feature_path(name, vid)
    path = os.path.join(path, '{}-{}.json'.format(vsig, name))
    if os.path.exists(path):
        if verbose:
            print('loading object {} proposal for video segment {}'.format(name, vsig))
        with open(path, 'r') as fin:
            trajs = json.load(fin)
        trajs = [Trajectory(**traj) for traj in trajs]
    else:
        if verbose:
            print('no object {} proposal for video segment {}'.format(name, vsig))
        trajs = []
    return trajs