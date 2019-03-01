import json
import argparse

from dataset import VidVRD, VidOR
from evaluation import eval_video_object, eval_action, eval_visual_relation


def evaluate_object(dataset, split, prediction):
    print('- evaluating video objects')
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_object_insts(vid)
    mean_ap, ap_class = eval_video_object(prediction, groundtruth)


def evaluate_action(dataset, split, prediction):
    print('- evaluating actions')
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_action_insts(vid)
    mean_ap, ap_class = eval_action(prediction, groundtruth)


def evaluate_relation(dataset, split, prediction):
    print('- evaluating visual relations')
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
    parser = argparse.ArgumentParser(description='Video visual relation evaluation.')
    parser.add_argument('dataset', type=str, help='the dataset name for evaluation')
    parser.add_argument('split', type=str, help='the split name for evaluation')
    parser.add_argument('prediction', type=str, help='Prediction json file')
    parser.add_argument("--object", action="store_true", help="whether to evaluate video objects")
    parser.add_argument("--action", action="store_true", help="whether to evaluate actions")
    parser.add_argument("--relation", action="store_true", help="whether to evaluate visual relations")
    args = parser.parse_args()

    if args.dataset=='vidvrd':
        dataset = VidVRD('../vidvrd-dataset', '../vidvrd-dataset/videos', [args.split])
    elif args.dataset=='vidor':
        dataset = VidOR('../vidor/annotation', '../vidor/vidor', [args.split], low_memory=True)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    with open(args.prediction, 'r') as fin:
        prediction_json = json.load(fin)

    if args.object:
        evaluate_object(dataset, args.split, prediction_json)
    if args.action:
        evaluate_action(dataset, args.split, prediction_json)
    if args.relation:
        evaluate_relation(dataset, args.split, prediction_json)