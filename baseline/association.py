import os
import cPickle
import argparse
import itertools
import time
from collections import deque, defaultdict
from copy import deepcopy

from dlib import drectangle
import numpy as np

from baseline import *
from trajectory import traj_iou, object_trajectory_proposal


def _merge_trajs(traj_1, traj_2):
    try:
        assert traj_1.pend > traj_2.pstart and traj_1.pstart < traj_2.pend
    except AssertionError:
        print('{}-{} {}-{}'.format(traj_1.pstart, traj_1.pend, traj_2.pstart, traj_2.pend))
    overlap_length = max(traj_1.pend - traj_2.pstart, 0)
    for i in range(overlap_length):
        roi_1 = traj_1.rois[traj_1.length() - overlap_length + i]
        roi_2 = traj_2.rois[i]
        left = (roi_1.left() + roi_2.left()) / 2
        top = (roi_1.top() + roi_2.top()) / 2
        right = (roi_1.right() + roi_2.right()) / 2
        bottom = (roi_1.bottom() + roi_2.bottom()) / 2
        traj_1.rois[traj_1.length() - overlap_length + i] = drectangle(left, top, right, bottom)
    for i in range(overlap_length, traj_2.length()):
        traj_1.predict(traj_2.rois[i])
    return traj_1


def _traj_iou(traj_1, traj_2):
    if traj_1.pend <= traj_2.pstart or traj_2.pend <= traj_1.pstart: # no overlap
        return 0
    if traj_1.pstart <= traj_2.pstart:
        t1 = deepcopy(traj_1)
        t2 = deepcopy(traj_2)
    else:
        t1 = deepcopy(traj_2)
        t2 = deepcopy(traj_1)
    overlap_length = t1.pend - t2.pstart
    t1.rois = deque(itertools.islice(t1.rois, t2.pstart-t1.pstart, t1.pend-t1.pstart))
    t2.rois = deque(itertools.islice(t2.rois, 0, t1.pend-t2.pstart))
    iou = traj_iou([t1], [t2])
    return iou[0,0]


class VideoRelation(object):
    '''
    Represent video level visual relation instances
    ----------
    Properties:
        vid - video name
        s_cid - object class id for subject
        pid - predicate id
        o_cid - object class id for object
        straj - merged trajectory of subject
        otraj - merged trajectory of object
        confs_list - list of confident score
    '''
    def __init__(self, vid, s_cid, pid, o_cid, straj, otraj, confs=1):
        self.vid = vid
        self.s_cid = s_cid
        self.pid = pid
        self.o_cid = o_cid
        self.straj = straj
        self.otraj = otraj
        self.confs_list = [confs]
        self.fstart = straj.pstart
        self.fend = straj.pend
    
    def __repr__(self):
        return '<VideoRelation {}[{:04d}-{:04d}] {}-{}-{}>'.format(
            self.vid, self.fstart, self.fend, self.s_cid, self.pid, self.o_cid)

    def triplet(self):
        return (self.s_cid, self.pid, self.o_cid)
    
    def mean_confs(self):
        return np.mean(self.confs_list)
    
    def both_overlap(self, straj, otraj, iou_thr=0.5):
        s_iou = _traj_iou(self.straj, straj)
        o_iou = _traj_iou(self.otraj, otraj)
        if s_iou >= iou_thr and o_iou >= iou_thr:
            return True
        else:
            return False

    def extend(self, straj, otraj, confs):
        self.straj = _merge_trajs(self.straj, straj)
        self.otraj = _merge_trajs(self.otraj, otraj)
        self.confs_list.append(confs)
        self.fstart = self.straj.pstart
        self.fend = self.otraj.pend

    def serialize(self, dataset):
        obj = dict()
        obj['triplet'] = [
            dataset.get_object_name(self.s_cid),
            dataset.get_predicate_name(self.pid),
            dataset.get_object_name(self.o_cid)
        ]
        obj['score'] = float(self.mean_confs())
        obj['duration'] = [
            int(self.fstart),
            int(self.fend)
        ]
        obj['sub_traj'] = self.straj.serialize()['rois']
        obj['obj_traj'] = self.otraj.serialize()['rois']
        return obj


def greedy_relational_association(dataset, short_term_relations, max_traj_num_in_clip=100):
    short_term_relations.sort(key=lambda x: int(x[0][1]))
    video_relation_list = []
    last_modify_rel_list = []
    for i, (index, prediction) in enumerate(short_term_relations):
        vid, fstart, fend = index
        # load prediction data
        pred_list, iou, trackid = prediction
        sorted_pred_list = sorted(pred_list, key=lambda x: x[0], reverse=True)
        if len(sorted_pred_list) > max_traj_num_in_clip:
            sorted_pred_list = sorted_pred_list[0:max_traj_num_in_clip]
        # load predict trajectory data
        trajs = object_trajectory_proposal(dataset, vid, fstart, fend)
        for traj in trajs:
            traj.pstart = fstart
            traj.pend = fend
            traj.vsig = get_segment_signature(vid, fstart, fend)
        # merge
        started_at = time.time()
        cur_modify_rel_list = []
        if i == 0:
            for pred_idx, pred in enumerate(sorted_pred_list):
                conf_score = pred[0]
                s_cid, pid, o_cid = pred[1]
                s_tididx, o_tididx = pred[2]
                straj = trajs[s_tididx]
                otraj = trajs[o_tididx]
                r = VideoRelation(vid, s_cid, pid, o_cid, straj, otraj, confs=conf_score)
                video_relation_list.append(r)
                cur_modify_rel_list.append(r)
        else:
            for pred_idx, pred in enumerate(sorted_pred_list):
                conf_score = pred[0]
                s_cid, pid, o_cid = pred[1]
                s_tididx, o_tididx = pred[2]
                straj = trajs[s_tididx]
                otraj = trajs[o_tididx]
                last_modify_rel_list.sort(key=lambda r: r.mean_confs(), reverse=True)
                is_merged = False
                for r in last_modify_rel_list:
                    if pred[1] == r.triplet():
                        if (straj.pstart < r.fend and otraj.pstart < r.fend) \
                                and r.both_overlap(straj,otraj):
                            r.extend(straj, otraj, conf_score)
                            last_modify_rel_list.remove(r)
                            cur_modify_rel_list.append(r)
                            is_merged = True
                            break
                if not is_merged:
                    r = VideoRelation(vid, s_cid, pid, o_cid, straj, otraj)
                    video_relation_list.append(r)
                    cur_modify_rel_list.append(r)
        last_modify_rel_list = cur_modify_rel_list
    
    return [rel.serialize(dataset) for rel in video_relation_list]
