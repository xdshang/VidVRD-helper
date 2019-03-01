import os
import glob

from dataset import DatasetV1


class VidOR(DatasetV1):
    """
    The dataset used in ACM MM'19 Relation Understanding Challenge
    """

    def __init__(self, anno_rpath, video_rpath, splits, low_memory=True):
        """
        anno_rpath: the root path of annotations
        video_rpath: the root path of videos
        splits: a list of splits in the dataset to load
        low_memory: if true, do not load memory-costly part 
                    of annotations (trajectories) into memory
        """
        super(VidOR, self).__init__(anno_rpath, video_rpath, splits, low_memory)
        print('VidOR dataset loaded. {}'.format('(low memory mode enabled)' if low_memory else ''))

    def _get_anno_files(self, split):
        anno_files = glob.glob(os.path.join(self.anno_rpath, '{}/*/*.json'.format(split)))
        assert len(anno_files)>0, 'No annotation file found. Please check if the directory is correct.'
        return anno_files

    def get_video_path(self, vid):
        """
        True if the directory videos uses imagenet struture
        """
        sub_dir = self.annos[vid]['video_path']
        path = os.path.join(self.video_rpath, sub_dir, '{}.mp4'.format(vid))
        return path


if __name__ == '__main__':
    # to load the trainning set without low memory mode for faster processing, you need sufficient large RAM
    dataset = VidOR('../../vidor/annotation', '../../vidor/vidor', ['training', 'validation'], low_memory=True)
    inds = dataset.get_index('validation')
    print(dataset.get_object_insts(inds[111]))
