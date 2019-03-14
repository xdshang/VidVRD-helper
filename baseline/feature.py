import numpy as np
import h5py

import os
import time
import cPickle as pkl
import atexit, signal
from time import strftime, sleep
from itertools import cycle, product
from multiprocessing import Process, Queue, sharedctypes

from trajectory import traj_iou, object_trajectory_proposal
from baseline import *


class SharedArray(object):
    """Numpy array that uses sharedctypes to store data"""
    def __init__(self, shape, dtype=np.float32):
        # Compute total number of elements
        size = np.prod(shape)
        # Get the size of element
        if dtype == np.float32:
            typecode = 'f'
        elif dtype == np.float64:
            typecode = 'd'
        else:
            assert False, 'Unknown dtype.'
        self.data = sharedctypes.RawArray(typecode, size)
        self.shape = shape
        self.dtype = dtype

    def set_value(self, value):
        nparr = np.ctypeslib.as_array(self.data)
        nparr.shape = self.shape
        nparr[...] = value.astype(self.dtype, copy = False)

    def get_value(self, copy = True):
        nparr = np.ctypeslib.as_array(self.data)
        nparr.shape = self.shape
        if copy:
            return np.array(nparr)
        else:
            return nparr


class FeatureExtractor(Process):
    """
    Generate feature for a (vid, fstart, fend) every call
    Class for prefetching data in a separate process
    """
    def __init__(self, dataset, prefetch_count=2):
        super(FeatureExtractor, self).__init__()
        self.dataset = dataset
        self.prefetch_count = prefetch_count

    def _init_pool(self):
        prefetch_count = self.prefetch_count
        if prefetch_count > 0:
            self._blob_pool = [list() for i in range(prefetch_count)]
            self._free_queue = Queue(prefetch_count)
            self._full_queue = Queue(prefetch_count)

            shapes = self.get_data_shapes()
            for i, shape in enumerate(shapes):
                for j in range(prefetch_count):
                    self._blob_pool[j].append(SharedArray(shape, np.float32))

            for i in range(prefetch_count):
                self._free_queue.put(i)
            # Terminate the child process when the parent exists
            atexit.register(self._cleanup)
            self.start()
        else:
            print('Prefetching disabled.')

    def _cleanup(self):
        if self.prefetch_count > 0:
            print('Terminating DataFetcher')
            self.terminate()
            self.join()

    def run(self):
        # Pass SIGINT to the parent process
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print('DataFetcher started')
        while True:
            blobs = self.get_data()
            pool_ind = self._free_queue.get()
            for i, s_blob in enumerate(self._blob_pool[pool_ind]):
                s_blob.set_value(blobs[i])
            self._full_queue.put(pool_ind)

    def get_prefected_data(self):
        """
        Need get_data() to return a list of np.ndarray with float32 dtype
        WARNING: prefetching testing data may not work, because the shapes
        of data may vary.
        """
        if not hasattr(self, '_full_queue') and self.prefetch_count > 0:
            self._init_pool()
        blobs = []
        pool_ind = self._full_queue.get()
        for blob in self._blob_pool[pool_ind]:
            blobs.append(blob.get_value())
        self._free_queue.put(pool_ind)
        return blobs

    def get_data_shapes(self):
        if not hasattr(self, 'shapes'):
            print('Getting data to measure the shapes...')
            data = self.get_data()
            self.shapes = tuple(d.shape for d in data)
        return self.shapes

    def get_data(self):
        raise NotImplementedError

    def extract_feature(self, vid, fstart, fend, dry_run=False, verbose=False):
        vsig = get_segment_signature(vid, fstart, fend)
        path = get_feature_path('relation', vid)
        path = os.path.join(path, '{}-{}.h5'.format(vsig, 'relation'))
        if os.path.exists(path):
            if dry_run:
                return None, None, None, None
            else:
                if verbose:
                    print('loading relation feature for video segment {}...'.format(vsig))
                with h5py.File(path, 'r') as fin:
                    # N object trajectory proposals, whose trackids are all -1
                    # and M groundtruth object trajectories, whose trackids are provided by dataset
                    trackid = fin['trackid'][:]
                    # all possible pairs among N+M object trajectories
                    pairs = fin['pairs'][:]
                    # relation feature for each pair (in same order)
                    feats = fin['feats'][:]
                    # vIoU (traj_iou) for each pair (in same order)
                    iou = fin['iou'][:]
                return pairs, feats, iou, trackid
        else:
            if verbose:
                print('no relation feature for video segment  {}'.format(vsig))
            return None
