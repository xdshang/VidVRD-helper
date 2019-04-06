import os
import json
from collections import defaultdict

from tqdm import tqdm


class Dataset(object):
    """
    Dataset base class with Json annotations without the "version" field.
    It helps maintaining the mapping between category id and category name,
    and parsing the annotations to get instances of object, action and visual relation.
    """

    def __init__(self, anno_rpath, video_rpath, splits):
        """
        anno_rpath: the root path of annotations
        video_rpath: the root path of videos
        splits: a list of splits in the dataset to load
        """
        self.anno_rpath = anno_rpath
        self.video_rpath = video_rpath
        self._load_annotations(splits)

    def _load_annotations(self, splits):
        print('loading annotations...')
        so = set()
        pred = set()
        self.split_index = defaultdict(list)
        self.annos = dict()
        for split in splits:
            anno_files = self._get_anno_files(split)
            annos = dict()
            for path in tqdm(anno_files):
                with open(path, 'r') as fin:
                    anno = json.load(fin)
                    anno = self._check_anno(anno)
                annos[anno['video_id']] = anno
            for vid, anno in annos.items():
                self.split_index[split].append(vid)
                for obj in anno['subject/objects']:
                    so.add(obj['category'])
                for rel in anno['relation_instances']:
                    pred.add(rel['predicate'])
            self.annos.update(annos)

        # build index for subject/object and predicate
        so = sorted(so)
        pred = sorted(pred)
        self.soid2so = dict()
        self.so2soid = dict()
        self.pid2pred = dict()
        self.pred2pid = dict()
        for i, name in enumerate(so):
            self.soid2so[i] = name
            self.so2soid[name] = i
        for i, name in enumerate(pred):
            self.pid2pred[i] = name
            self.pred2pid[name] = i
    
    def _check_anno(self, anno):
        assert 'version' not in anno
        return anno

    def _get_anno_files(self, split):
        raise NotImplementedError
    
    def get_video_path(self, vid):
        raise NotImplementedError
    
    def _get_action_predicates(self):
        raise NotImplementedError

    def get_object_num(self):
        return len(self.soid2so)

    def get_object_name(self, cid):
        return self.soid2so[cid]

    def get_object_id(self, name):
        return self.so2soid[name]

    def get_predicate_num(self):
        return len(self.pid2pred)

    def get_predicate_name(self, pid):
        return self.pid2pred[pid]

    def get_predicate_id(self, name):
        return self.pred2pid[name]

    def get_triplets(self, split):
        triplets = set()
        for vid in self.get_index(split):
            insts = self.get_relation_insts(vid, no_traj=True)
            triplets.update(inst['triplet'] for inst in insts)
        return triplets

    def get_index(self, split):
        """
        get list of video IDs for a split
        """
        if split in self.split_index:
            return self.split_index[split]
        else:
            for s in self.split_index.keys():
                if split in s:
                    print('INFO: infer the split name \'{}\' in this dataset from \'{}\''.format(s, split))
                    return self.split_index[s]
            else:
                raise Exception('Unknown split "{}" in the loaded dataset'.format(split))

    def get_anno(self, vid):
        """
        get raw annotation for a video
        """
        return self.annos[vid]

    def get_object_insts(self, vid):
        """
        get the object instances (trajectories) labeled in a video
        """
        anno = self.get_anno(vid)
        object_insts = []
        tid2cls = dict()
        for item in anno['subject/objects']:
            tid2cls[item['tid']] = item['category']
        traj = defaultdict(dict)
        for fid, frame in enumerate(anno['trajectories']):
            for roi in frame:
                traj[roi['tid']][str(fid)] = (roi['bbox']['xmin'],
                                            roi['bbox']['ymin'],
                                            roi['bbox']['xmax'],
                                            roi['bbox']['ymax'])
        for tid in traj:
            object_insts.append({
                'tid': tid,
                'category': tid2cls[tid],
                'trajectory': traj[tid]
            })
        return object_insts

    def get_action_insts(self, vid):
        """
        get the action instances labeled in a video
        """
        anno = self.get_anno(vid)
        action_insts = []
        actions = self._get_action_predicates()
        for each_ins in anno['relation_instances']:
            if each_ins['predicate'] in actions:
                begin_fid = each_ins['begin_fid']
                end_fid = each_ins['end_fid']
                each_ins_trajectory = []
                for each_traj in anno['trajectories'][begin_fid:end_fid]:
                    for each_traj_obj in each_traj:
                        if each_traj_obj['tid'] == each_ins['subject_tid']:
                            each_traj_frame = (
                                each_traj_obj['bbox']['xmin'],
                                each_traj_obj['bbox']['ymin'],
                                each_traj_obj['bbox']['xmax'],
                                each_traj_obj['bbox']['ymax']
                            )
                            each_ins_trajectory.append(each_traj_frame)
                each_ins_action = {
                    "category": each_ins['predicate'],
                    "duration": (begin_fid, end_fid),
                    "trajectory": each_ins_trajectory
                }
                action_insts.append(each_ins_action)
        return action_insts

    def get_relation_insts(self, vid, no_traj=False):
        """
        get the visual relation instances labeled in a video,
        no_traj=True will not include trajectories, which is
        faster.
        """
        anno = self.get_anno(vid)
        sub_objs = dict()
        for so in anno['subject/objects']:
            sub_objs[so['tid']] = so['category']
        if not no_traj:
            trajs = []
            for frame in anno['trajectories']:
                bboxes = dict()
                for bbox in frame:
                    bboxes[bbox['tid']] = (bbox['bbox']['xmin'],
                                        bbox['bbox']['ymin'],
                                        bbox['bbox']['xmax'],
                                        bbox['bbox']['ymax'])
                trajs.append(bboxes)
        relation_insts = []
        for anno_inst in anno['relation_instances']:
            inst = dict()
            inst['triplet'] = (sub_objs[anno_inst['subject_tid']],
                            anno_inst['predicate'],
                            sub_objs[anno_inst['object_tid']])
            inst['subject_tid'] = anno_inst['subject_tid']
            inst['object_tid'] = anno_inst['object_tid']
            inst['duration'] = (anno_inst['begin_fid'], anno_inst['end_fid'])
            if not no_traj:
                inst['sub_traj'] = [bboxes[anno_inst['subject_tid']] for bboxes in
                        trajs[inst['duration'][0]: inst['duration'][1]]]
                inst['obj_traj'] = [bboxes[anno_inst['object_tid']] for bboxes in
                        trajs[inst['duration'][0]: inst['duration'][1]]]
            relation_insts.append(inst)
        return relation_insts


class DatasetV1(Dataset):
    """
    Dataset base class with Json annotations in VERSION 1.0 format, supporting low memory mode
    It helps maintaining the mapping between category id and category name,
    and parsing the annotations to get instances of object, action and visual relation.
    """

    def __init__(self, anno_rpath, video_rpath, splits, low_memory):
        """
        anno_rpath: the root path of annotations
        video_rpath: the root path of videos
        splits: a list of splits in the dataset to load
        low_memory: if true, do not load memory-costly part 
                    of annotations (trajectories) into memory
        """
        self.anno_rpath = anno_rpath
        self.video_rpath = video_rpath
        self.low_memory = low_memory
        self._load_annotations(splits)

    def _check_anno(self, anno):
        assert 'version' in anno and anno['version']=='VERSION 1.0'
        if self.low_memory:
            del anno['trajectories']
        return anno

    def get_anno(self, vid):
        """
        get raw annotation for a video
        """
        if self.low_memory:
            for key, val in self.split_index.items():
                if vid in val:
                    split = key
                    break
            else:
                raise KeyError('{} not found in any split in the loaded dataset'.format(vid))

            anno_relative_path = self.annos[vid]['video_path'].replace('.mp4', '.json')
            with open(os.path.join(self.anno_rpath, split, anno_relative_path), 'r') as fin:
                anno = json.load(fin)
            return anno
        else:
            return self.annos[vid]