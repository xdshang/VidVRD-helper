import json
import argparse
from dataset import Dataset


def eval_video_object(dataset, prediction):
    from evaluation import eval_video_object
    print('- evaluating video objects')
    raise NotImplementedError


def eval_action(dataset, prediction):
    from evaluation import eval_action
    print('- evaluating actions')
    raise NotImplementedError


def eval_visual_relation(dataset, prediction):
    from evaluation import eval_visual_relation
    # evaluate
    print('- evaluating visual relations')
    groundtruth = dict()
    for vid in dataset.get_index('test'):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    mAP, rec_at_n, mprec_at_n = eval_visual_relation(groundtruth, prediction)
    # evaluate in zero-shot setting
    print('-- zero-shot')
    zeroshot_triplets = dataset.get_triplets('test').difference(
            dataset.get_triplets('train'))
    zeroshot_groundtruth = dict()
    for vid in dataset.get_index('test'):
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
    parser.add_argument('prediction_file', type=str, help='Prediction json file')
    parser.add_argument("--object", action="store_true", help="whether to evaluate video objects")
    parser.add_argument("--action", action="store_true", help="whether to evaluate actions")
    parser.add_argument("--relation", action="store_true", help="whether to evaluate visual relations")
    args = parser.parse_args()

    dataset = Dataset()
    with open(args.prediction_file, 'r') as fin:
        prediction_json = json.load(fin)

    if args.action:
        eval_video_object(dataset, prediction_json)
    if args.object:
        eval_action(dataset, prediction_json)
    if args.relation:
        eval_visual_relation(dataset, prediction_json)