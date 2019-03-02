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


def evaluate_relation(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    mAP, rec_at_n, mprec_at_n = eval_visual_relation(groundtruth, prediction)
    # evaluate in zero-shot setting
    print('-- zero-shot')
    zeroshot_triplets = dataset.get_triplets(split).difference(
            dataset.get_triplets('train'))
    zeroshot_groundtruth = dict()
    for vid in dataset.get_index(split):
        gt_relations = dataset.get_relation_insts(vid)
        zs_gt_relations = []
        for r in gt_relations:
            if tuple(r['triplet']) in zeroshot_triplets:
                zs_gt_relations.append(r)
        if len(zs_gt_relations) > 0:
            zeroshot_groundtruth[vid] = zs_gt_relations
    mAP, rec_at_n, mprec_at_n = eval_visual_relation(
            zeroshot_groundtruth, prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a set of tasks related to video relation understanding.')
    parser.add_argument('dataset', type=str, help='the dataset name for evaluation')
    parser.add_argument('split', type=str, help='the split name for evaluation')
    parser.add_argument('task', choices=['object', 'action', 'relation'], help='which task to evaluate')
    parser.add_argument('prediction', type=str, help='Corresponding prediction JSON file')
    args = parser.parse_args()

    if args.dataset=='vidvrd':
        dataset = VidVRD('../vidvrd-dataset', '../vidvrd-dataset/videos', [args.split])
    elif args.dataset=='vidor':
        dataset = VidOR('../vidor/annotation', '../vidor/vidor', [args.split], low_memory=True)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    print('Loading prediction from {}'.format(args.prediction))
    with open(args.prediction, 'r') as fin:
        pred = json.load(fin)
    print('Number of videos in prediction: {}'.format(len(pred['results'])))

    if args.task=='object':
        evaluate_object(dataset, args.split, pred)
    elif args.task=='action':
        evaluate_action(dataset, args.split, pred)
    elif args.task=='relation':
        evaluate_relation(dataset, args.split, pred)