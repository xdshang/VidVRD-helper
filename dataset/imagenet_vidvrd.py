import os
import glob

from .dataset import Dataset


class ImagenetVidVRD(Dataset):
    """
    VidVRD dataset introduced in ACM MM'17
    """

    def __init__(self, anno_rpath, video_rpath, splits):
        """
        anno_rpath: the root path of annotations
        video_rpath: the root path of videos
        splits: a list of splits in the dataset to load
        """
        super(ImagenetVidVRD, self).__init__(anno_rpath, video_rpath, splits)
        print('ImageNet-VidVRD dataset loaded.')

    def _get_anno_files(self, split):
        anno_files = glob.glob(os.path.join(self.anno_rpath, '{}/*.json'.format(split)))
        assert len(anno_files)>0, 'No annotation file found. Please check if the directory is correct.'
        return anno_files

    def get_video_path(self, vid, imagenet_struture=False):
        """
        True if the directory videos uses imagenet struture
        """
        if imagenet_struture:
            if 'train' in vid:
                path = glob.glob(os.path.join(self.video_rpath,
                        'Data/VID/snippets/train/*/{}.mp4'.format(vid)))[0]
            elif 'val' in vid:
                path = os.path.join(self.video_rpath,
                        'Data/VID/snippets/val/{}.mp4'.format(vid))
            else:
                raise Exception('Unknown video ID {}'.format(vid))
        else:
            path = os.path.join(self.video_rpath, '{}.mp4'.format(vid))
        return path

