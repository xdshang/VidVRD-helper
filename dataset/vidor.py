import os
import glob

from .dataset import DatasetV1


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
        assert len(anno_files)>0, 'No annotation file found for \'{}\'. Please check if the directory is correct.'.format(split)
        return anno_files

    def _get_action_predicates(self):
        actions = [
            'watch','bite','kiss','lick','smell','caress','knock','pat',
            'point_to','squeeze','hold','press','touch','hit','kick',
            'lift','throw','wave','carry','grab','release','pull',
            'push','hug','lean_on','ride','chase','get_on','get_off',
            'hold_hand_of','shake_hand_with','wave_hand_to','speak_to','shout_at','feed',
            'open','close','use','cut','clean','drive','play(instrument)',
        ]
        for action in actions:
            assert action in self.pred2pid
        return actions

    def get_video_path(self, vid):
        return os.path.join(self.video_rpath, self.annos[vid]['video_path'])


if __name__ == '__main__':
    """
    To generate a single JSON groundtruth file for specific split and task,
    run this script from the parent directory, for example, 
    python -m dataset.vidor validation object ~/vidor_gt_val_object.json
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Generate a single JSON groundtruth file for VidOR')
    parser.add_argument('split', choices=['training', 'validation'], 
                        help='which dataset split the groundtruth generated for')
    parser.add_argument('task', choices=['object', 'action', 'relation'],
                        help='which task the groundtruth generated for')
    parser.add_argument('output', type=str, help='Output path')
    args = parser.parse_args()

    # to load the trainning set without low memory mode for faster processing, you need sufficient large RAM
    dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', ['training', 'validation'], low_memory=True)
    index = dataset.get_index(args.split)

    gts = dict()
    for ind in index:
        if args.task=='object':
            gt = dataset.get_object_insts(ind)
        elif args.task=='action':
            gt = dataset.get_action_insts(ind)
        elif args.task=='relation':
            gt = dataset.get_relation_insts(ind)
        gts[ind] = gt
    
    with open(args.output, 'w') as fout:
        json.dump(gts, fout, separators=(',', ':'))
