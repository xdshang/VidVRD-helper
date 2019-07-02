import json
import argparse

from dataset import VidVRD, VidOR
from evaluation import eval_video_object, eval_action, eval_visual_relation


def evaluate_object(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_object_insts(vid)
    mean_ap, ap_class = eval_video_object(groundtruth, prediction)


def evaluate_action(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_action_insts(vid)
    mean_ap, ap_class = eval_action(groundtruth, prediction)


def evaluate_relation(dataset, split, prediction, use_old_zeroshot_eval=False):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(groundtruth, prediction)
    # evaluate in zero-shot setting
    if use_old_zeroshot_eval:
        print('-- zero-shot setting (old)')
    else:
        print('-- zero-shot setting (new)')
    zeroshot_triplets = dataset.get_triplets(split).difference(
            dataset.get_triplets('train'))
    groundtruth = dict()
    zs_prediction = dict()
    for vid in dataset.get_index(split):
        gt_relations = dataset.get_relation_insts(vid)
        zs_gt_relations = []
        for r in gt_relations:
            if tuple(r['triplet']) in zeroshot_triplets:
                zs_gt_relations.append(r)
        if len(zs_gt_relations) > 0:
            groundtruth[vid] = zs_gt_relations
            if use_old_zeroshot_eval:
                # old zero-shot evaluation doesn't filter out non-zeroshot predictions
                # in a video, which will result in very low Average Precision 
                zs_prediction[vid] = prediction[vid]
            else:
                zs_prediction[vid] = []
                for r in prediction.get(vid, []):
                    if tuple(r['triplet']) in zeroshot_triplets:
                        zs_prediction[vid].append(r)
    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(groundtruth, zs_prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a set of tasks related to video relation understanding.')
    parser.add_argument('dataset', type=str, help='the dataset name for evaluation')
    parser.add_argument('split', type=str, help='the split name for evaluation')
    parser.add_argument('task', choices=['object', 'action', 'relation'], help='which task to evaluate')
    parser.add_argument('prediction', type=str, help='Corresponding prediction JSON file')
    args = parser.parse_args()

    if args.dataset=='vidvrd':
        if args.task=='relation':
            # load train set for zero-shot evaluation
            dataset = VidVRD('../vidvrd-dataset', '../vidvrd-dataset/videos', ['train', args.split])
        else:
            dataset = VidVRD('../vidvrd-dataset', '../vidvrd-dataset/videos', [args.split])
    elif args.dataset=='vidor':
        if args.task=='relation':
            # load train set for zero-shot evaluation
            dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', ['training', args.split], low_memory=True)
        else:
            dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', [args.split], low_memory=True)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    print('Loading prediction from {}'.format(args.prediction))
    with open(args.prediction, 'r') as fin:
        pred = json.load(fin)
    print('Number of videos in prediction: {}'.format(len(pred['results'])))

    if args.task=='object':
        evaluate_object(dataset, args.split, pred['results'])
    elif args.task=='action':
        evaluate_action(dataset, args.split, pred['results'])
    elif args.task=='relation':
        evaluate_relation(dataset, args.split, pred['results'])
