import os
import json
from time import strftime
from itertools import product, cycle
from collections import defaultdict, OrderedDict

import numpy as np
import h5py
from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.initializers import RandomNormal, RandomUniform
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras.optimizers import SGD, Adam
from keras import backend as K

from feature import FeatureExtractor
from baseline import *


_train_triplet_id = OrderedDict()


def feature_preprocess(feat):
    """ 
    Input feature is extracted according to Section 4.2 in the paper
    """
    # subject classeme + object classeme
    # feat[:, 0: 70]

    # subject TrajectoryShape + HoG + HoF + MBH motion feature
    # (since this feature is Bag-of-Word type, we l1-normalize it so that
    # each element represents the fraction instead of count)
    feat[:, 70: 1070] = np_utils.normalize(feat[:, 70: 1070], -1, 1)
    feat[:, 1070: 2070] = np_utils.normalize(feat[:, 1070: 2070], -1, 1)
    feat[:, 2070: 3070] = np_utils.normalize(feat[:, 2070: 3070], -1, 1)
    feat[:, 3070: 4070] = np_utils.normalize(feat[:, 3070: 4070], -1, 1)
    # object TrajectoryShape + HoG + HoF + MBH motion feature
    feat[:, 4070: 5070] = np_utils.normalize(feat[:, 4070: 5070], -1, 1)
    feat[:, 5070: 6070] = np_utils.normalize(feat[:, 5070: 6070], -1, 1)
    feat[:, 6070: 7070] = np_utils.normalize(feat[:, 6070: 7070], -1, 1)
    feat[:, 7070: 8070] = np_utils.normalize(feat[:, 7070: 8070], -1, 1)

    # relative posititon + size + motion feature
    # feat[:, 8070: 9070]
    # feat[:, 9070: 10070]
    # feat[:, 10070: 11070]
    return feat


class DataGenerator(FeatureExtractor):
    """
    Generate (incl. sample) relation features of (subject, object)s
    in the video segments that have multiple objects detected.
    """
    def __init__(self, dataset, param, prefetch_count=2):
        super(DataGenerator, self).__init__(dataset, prefetch_count)
        self.rng = np.random.RandomState(param['rng_seed'])
        self.batch_size = param['batch_size']
        self.max_sampling_in_batch = param['max_sampling_in_batch']
        assert self.max_sampling_in_batch <= self.batch_size
        self.phase = param['phase']

        print('preparing video segments for {}...'.format(self.phase))
        if self.phase == 'train':
            # initialize training triplet id
            _train_triplet_id.clear()
            triplets = dataset.get_triplets(split='train')
            for i, triplet in enumerate(triplets):
                sub_name, pred_name, obj_name = triplet
                sub_id = dataset.get_object_id(sub_name)
                pred_id = dataset.get_predicate_id(pred_name)
                obj_id = dataset.get_object_id(obj_name)
                _train_triplet_id[(sub_id, pred_id, obj_id)] = i

            self.short_rel_insts = defaultdict(list)
            video_indices = dataset.get_index(split='train')
            for vid in video_indices:
                for rel_inst in dataset.get_relation_insts(vid, no_traj=True):
                    segs = segment_video(*rel_inst['duration'])
                    for fstart, fend in segs:
                        # if multiple objects detected and the relation features extracted
                        # cache the corresponding groudtruth labels
                        if self.extract_feature(vid, fstart, fend, dry_run=True):
                            sub_name, pred_name, obj_name = rel_inst['triplet']
                            self.short_rel_insts[(vid, fstart, fend)].append((
                                rel_inst['subject_tid'],
                                rel_inst['object_tid'],
                                dataset.get_object_id(sub_name),
                                dataset.get_predicate_id(pred_name),
                                dataset.get_object_id(obj_name)
                            ))
            self.index = list(self.short_rel_insts.keys())
            self.ind_iter = cycle(range(len(self.index)))
        elif self.phase == 'test':
            self.index = []
            video_indices = dataset.get_index(split='test')
            for vid in video_indices:
                anno = dataset.get_anno(vid)
                segs = segment_video(0, anno['frame_count'])
                # enumerate all the possible segments
                for fstart, fend in segs:
                    # if multiple objects detected and the relation features extracted
                    if self.extract_feature(vid, fstart, fend, dry_run=True):
                        self.index.append((vid, fstart, fend))
            self.ind_iter = iter(range(len(self.index)))
        else:
            raise ValueError('Unknown phase: {}'.format(self.phase))

    def get_data(self):
        if self.phase == 'train':
            f = []
            r = []
            remaining_size = self.batch_size
            while remaining_size > 0:
                i = self.ind_iter.next()
                vid, fstart, fend = self.index[i]
                sample_num = np.minimum(remaining_size, self.max_sampling_in_batch)
                _f, _r = self._data_sampling(vid, fstart, fend, sample_num)
                remaining_size -= _f.shape[0]
                _f = feature_preprocess(_f)
                f.append(_f.astype(np.float32))
                r.append(_r.astype(np.float32))
            f = np.concatenate(f)
            r = np.concatenate(r)
            return f, r
        else:
            try:
                i = self.ind_iter.next()
            except StopIteration, e:
                return None
            index = self.index[i]
            pairs, feats, iou, trackid = self.extract_feature( *index)
            test_inds = [ind for ind, (traj1, traj2) in enumerate(pairs)
                    if trackid[traj1] < 0 and trackid[traj2] < 0]
            pairs = pairs[test_inds]
            feats = feature_preprocess(feats[test_inds])
            feats = feats.astype(np.float32)
            return index, pairs, feats, iou, trackid

    def _data_sampling(self, vid, fstart, fend, sample_num, iou_thres=0.5):
        pairs, feats, iou, trackid = self.extract_feature(vid, fstart, fend)
        feats = feats.astype(np.float32)
        pair_to_find = dict([((traj1, traj2), find)
                for find, (traj1, traj2) in enumerate(pairs)])
        tid_to_ind = dict([(tid, ind) for ind, tid in enumerate(trackid) if tid >= 0])

        pos = np.empty((0, 2), dtype = np.int32)
        for tid1, tid2, s, p, o in self.short_rel_insts[(vid, fstart, fend)]:
            if tid1 in tid_to_ind and tid2 in tid_to_ind:
                iou1 = iou[:, tid_to_ind[tid1]]
                iou2 = iou[:, tid_to_ind[tid2]]
                pos_inds1 = np.where(iou1 >= iou_thres)[0]
                pos_inds2 = np.where(iou2 >= iou_thres)[0]
                tmp = [(pair_to_find[(traj1, traj2)], _train_triplet_id[(s, p, o)])
                        for traj1, traj2 in product(pos_inds1, pos_inds2) if traj1 != traj2]
                if len(tmp) > 0:
                    pos = np.concatenate((pos, tmp))

        num_pos_in_this = np.minimum(pos.shape[0], sample_num)
        if pos.shape[0] > 0:
            pos = pos[np.random.choice(pos.shape[0], num_pos_in_this, replace=False)]

        return feats[pos[:, 0]], pos[:, 1]


class SelectionLayer(Layer):
    def __init__(self, sel_inds, **kwargs):
        self.sel_inds = sel_inds
        super(SelectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SelectionLayer, self).build(input_shape)

    def call(self, inputs):
        s = K.tf.gather(inputs[0], self.sel_inds[0], axis=1)
        p = K.tf.gather(inputs[1], self.sel_inds[1], axis=1)
        o = K.tf.gather(inputs[2], self.sel_inds[2], axis=1)
        return s*p*o

    def compute_output_shape(self, input_shape):
        return (None, self.sel_inds.shape[1])


def build_model(dataset, param):
    inp_f = Input(shape=(param['feature_dim'],), dtype='float32')
    prob_s = Input(shape=(param['object_num'],), dtype='float32')
    prob_o = Input(shape=(param['object_num'],), dtype='float32')

    p = Dense(units=param['predicate_num'])(inp_f)

    sel_inds = np.asarray(_train_triplet_id.keys(), dtype='int32').T
    r = SelectionLayer(sel_inds)([prob_s, p, prob_o])
    prob = Activation('softmax')(r)

    model = Model(inputs=[inp_f, prob_s, prob_o], outputs=[prob])
    model.summary()
    print('Trainable weights: {}'.format(model.trainable_weights))

    return model


def train(dataset, param):
    param['phase'] = 'train'
    param['object_num'] = dataset.get_object_num()
    param['predicate_num'] = dataset.get_predicate_num()

    data_generator = DataGenerator(dataset, param)
    # data_generator = DataGenerator(dataset, param, prefetch_count = 0)
    param['feature_dim'] = data_generator.get_data_shapes()[0][1]
    print('Feature dimension is {}'.format(param['feature_dim']))
    param['triplet_num'] = len(_train_triplet_id)
    print('Number of observed training triplets is {}'.format(param['triplet_num']))

    training_model = build_model(dataset, param)
    adam = Adam(lr=param['learning_rate'])
    training_model.compile(optimizer=adam, loss='categorical_crossentropy')

    for it in range(param['max_iter']):
        try:
            f, r = data_generator.get_prefected_data()
            prob_s = f[:, :35]
            prob_o = f[:, 35: 70]
            y = np_utils.to_categorical(r, num_classes=param['triplet_num'])
            loss = training_model.train_on_batch([f, prob_s, prob_o], [y])
            if it % param['display_freq'] == 0:
                print('{} - iter: {}/{} - loss: {:.4f}'.format(
                        strftime('%Y-%m-%d %H:%M:%S'), it, param['max_iter'], float(loss)))
            if it % param['save_freq'] == 0 and it > 0:
                param['model_dump_file'] = '{}_weights_iter_{}.h5'.format(param['model_name'], it)
                training_model.save_weights(os.path.join(get_model_path(), param['model_dump_file']))
        except KeyboardInterrupt:
            print('Early Stop.')
            break
    else:
        # save model
        param['model_dump_file'] = '{}_weights_iter_{}.h5'.format(param['model_name'], param['max_iter'])
        training_model.save_weights(os.path.join(get_model_path(), param['model_dump_file']))
    # save settings
    with open(os.path.join(get_model_path(), '{}_setting.json'.format(param['model_name'])), 'w') as fout:
        json.dump(param, fout, indent=4)


def predict(dataset, param):
    param['phase'] = 'test'
    data_generator = DataGenerator(dataset, param)
    # load model
    with h5py.File(os.path.join(get_model_path(), param['model_dump_file']), 'r') as fin:
        w = fin['/dense_1/dense_1/kernel:0'][:]
        b = fin['/dense_1/dense_1/bias:0'][:]

    print('predicting short-term visual relation...')
    pbar = tqdm(total=len(data_generator.index))
    short_term_relations = dict()
    # do not support prefetching mode in test phase
    data = data_generator.get_data()
    while data:
        # get all possible pairs and the respective features and annos
        index, pairs, feats, iou, trackid = data
        # make prediction
        p = feats.dot(w) + b
        s = feats[:, :35]
        o = feats[:, 35: 70]
        predictions = []
        for i in range(len(pairs)):
            top_s_ind = np.argsort(s[i])[-param['pair_topk']:]
            top_p_ind = np.argsort(p[i])[-param['pair_topk']:]
            top_o_ind = np.argsort(o[i])[-param['pair_topk']:]
            score = s[i][top_s_ind, None, None]*p[i][None, top_p_ind, None]*o[i][None, None, top_o_ind]
            top_flat_ind = np.argsort(score, axis = None)[-param['pair_topk']:]
            top_score = score.ravel()[top_flat_ind]
            top_s, top_p, top_o = np.unravel_index(top_flat_ind, score.shape)
            predictions.extend((
                    top_score[j], 
                    (top_s_ind[top_s[j]], top_p_ind[top_p[j]], top_o_ind[top_o[j]]), 
                    tuple(pairs[i])) 
                    for j in range(top_score.size))
        predictions = sorted(predictions, key=lambda x: x[0], reverse=True)[:param['seg_topk']]
        short_term_relations[index] = (predictions, iou, trackid)

        data = data_generator.get_data()
        pbar.update(1)

    pbar.close()
    return short_term_relations
