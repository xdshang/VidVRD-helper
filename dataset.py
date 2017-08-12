import glob
import json
import os


class Dataset():

  def __init__(self):
    self.rpath = '../VidVRD-dataset'
    self.data_root = os.path.join(self.rpath, 'videos')
    # load annotations for training set and test set
    print('loading dataset...')
    so = set()
    pred = set()
    self.train = []
    self.test = []
    self.annos = dict()
    for anno_path in glob.glob(os.path.join(self.rpath, 'train/*.json')):
      with open(anno_path, 'r') as fin:
        anno = json.load(fin)
      self.train.append(anno['video_id'])
      self.annos[anno['video_id']] = anno
      for obj in anno['subject/objects']:
        so.add(obj['category'])
      for rel in anno['relation_instances']:
        pred.add(rel['predicate'])
    for anno_path in glob.glob(os.path.join(self.rpath, 'test/*.json')):
      with open(anno_path, 'r') as fin:
        anno = json.load(fin)
      self.test.append(anno['video_id'])
      self.annos[anno['video_id']] = anno
      for obj in anno['subject/objects']:
        so.add(obj['category'])
      for rel in anno['relation_instances']:
        pred.add(rel['predicate'])
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

  def get_triplets(self, split='train'):
    triplets = set()
    for vid in self.get_index(split):
      insts = self.get_relation_insts(vid, no_traj=True)
      triplets.update(inst['triplet'] for inst in insts)
    return triplets

  def get_index(self, split='train'):
    """
    get list of video IDs for a split
    """
    if split == 'train':
      return self.train
    elif split == 'test':
      return self.test
    else:
      raise Exception('Unknown split {}'.format(split))

  def get_anno(self, vid):
    """
    get raw annotation for a video
    """
    return self.annos[vid]

  def get_relation_insts(self, vid, no_traj=False):
    """
    get the visual relation instances labeled in a video,
    no_traj=True will not include trajectories, which is
    faster.
    """
    anno = self.annos[vid]
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
      inst['duration'] = (anno_inst['begin_fid'], anno_inst['end_fid'])
      if not no_traj:
        inst['sub_traj'] = [bboxes[anno_inst['subject_tid']] for bboxes in
            trajs[inst['duration'][0]: inst['duration'][1]]]
        inst['obj_traj'] = [bboxes[anno_inst['object_tid']] for bboxes in
            trajs[inst['duration'][0]: inst['duration'][1]]]
      relation_insts.append(inst)
    return relation_insts

  def get_video_path(self, vid, imagenet_struture=False):
    """
    True if the directory videos uses imagenet struture
    """
    if imagenet_struture:
      if 'train' in vid:
        path = glob.glob(os.path.join(self.rpath, 'videos',
            'Data/VID/snippets/train/*/{}.mp4'.format(vid)))[0]
      elif 'val' in vid:
        path = os.path.join(self.rpath, 'videos',
            'Data/VID/snippets/val/{}.mp4'.format(vid))
      else:
        raise Exception('Unknown video ID {}'.format(vid))
    else:
      path = os.path.join(self.rpath, 'videos', '{}.mp4'.format(vid))
    return path


if __name__ == '__main__':
  dataset = Dataset()
  test_inds = dataset.get_index('test')
  insts = dataset.get_relation_insts(test_inds[111])