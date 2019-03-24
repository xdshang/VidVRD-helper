import ast
import os
import json
import argparse
import glob

import cv2
import numpy as np
from tqdm import trange

_colors = [(244, 67, 54), (255, 245, 157), (29, 233, 182), (118, 255, 3),
           (33, 150, 243), (179, 157, 219), (233, 30, 99), (205, 220, 57),
           (27, 94, 32), (255, 111, 0), (187, 222, 251), (24, 255, 255),
           (63, 81, 181), (156, 39, 176), (183, 28, 28), (130, 119, 23),
           (139, 195, 74), (0, 188, 212), (224, 64, 251), (96, 125, 139),
           (0, 150, 136), (121, 85, 72), (26, 35, 126), (129, 212, 250),
           (158, 158, 158), (225, 190, 231), (183, 28, 28), (230, 81, 0),
           (245, 127, 23), (27, 94, 32), (0, 96, 100), (13, 71, 161),
           (74, 20, 140), (198, 40, 40), (239, 108, 0), (249, 168, 37),
           (46, 125, 50), (0, 131, 143), (21, 101, 192), (106, 27, 154),
           (211, 47, 47), (245, 124, 0), (251, 192, 45), (56, 142, 60),
           (0, 151, 167), (25, 118, 210), (123, 31, 162), (229, 57, 53),
           (251, 140, 0), (253, 216, 53), (67, 160, 71), (0, 172, 193),
           (30, 136, 229), (142, 36, 170), (244, 67, 54), (255, 152, 0),
           (255, 235, 59), (76, 175, 80), (0, 188, 212), (33, 150, 243)]


def read_video(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise Exception('Cannot open {}'.format(path))
    video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    return video


def write_video(video, fps, size, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, cv2.CAP_FFMPEG, fourcc, fps, size)
    for frame in video:
        out.write(frame)
    out.release()


def visualiza_pred(pred_res_path, anno_rpath, video_rpath, dataset='vidor', splits=['training', 'validation']):
    """
    convert json result like (http://lms.comp.nus.edu.sg/research/video-relation-challenge/mm19-gdc/task3.html)
    2 (http://lms.comp.nus.edu.sg/research/dataset.html)
    """
    if dataset == 'vidor':
        from dataset.vidor import VidOR
        vid_dataset = VidOR(anno_rpath, video_rpath, splits)

    else:
        from dataset.vidvrd import VidVRD
        vid_dataset = VidVRD(anno_rpath, video_rpath, splits)

    with open(pred_res_path, 'r') as pred_res_in_f:
        pred_res_json = json.load(pred_res_in_f)

    pres_results = pred_res_json['results']
    for each_video_id in pres_results.keys():
        pred_rela_list = pres_results[each_video_id]
        gt_vid_anno = vid_dataset.get_anno(each_video_id)
        anno4mat_res = dict()

        for each_key in ['video_id', 'version', 'video_hash', 'video_path', 'frame_count', 'fps', 'width', 'height']:
            anno4mat_res[each_key] = gt_vid_anno[each_key]

        pred_res_objs = set()
        for each_rela_triplet in pred_rela_list:
            # deal with subject/objects
            pred_res_objs.add(each_rela_triplet['triplet'][0])
            pred_res_objs.add(each_rela_triplet['triplet'][2])
        pred_sub_obj_list = list()
        pred_sub_obj_dict = dict()
        for idx, obj in enumerate(pred_res_objs):
            pred_sub_obj_dict[obj] = idx
            pred_sub_obj_list.append(
                {
                    'tid': idx,
                    'category': obj
                }
            )
        anno4mat_res['subject/objects'] = pred_sub_obj_list

        # trajectories
        pred_trajectories_list = list()
        pred_rela_ins_list = list()
        for each_rela_triplet in pred_rela_list:
            # deal with trajectories
            pred_traj_list = list()
            sub, predicate, obj = each_rela_triplet['triplet']
            for each_obj in ['sub_traj', 'obj_traj']:
                for each_bbox in each_rela_triplet[each_obj]:
                    pred_traj_ins = dict()
                    if each_obj[:3] == 'sub':
                        pred_traj_ins['tid'] = pred_sub_obj_dict[sub]
                    else:
                        pred_traj_ins['tid'] = pred_sub_obj_dict[obj]
                    xmin, ymin, xmax, ymax = each_bbox
                    pred_traj_ins['bbox'] = {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    }
                    pred_traj_ins['generated'] = 1
                    pred_traj_ins['tracker'] = "none"
                    pred_traj_list.append(pred_traj_ins)
            pred_trajectories_list.append(pred_traj_list)

            # deal relation_instances
            each_pred_rela_ins = {
                'subject_tid': pred_sub_obj_dict[sub],
                'object_tid': pred_sub_obj_dict[obj],
                'predicate': predicate,
                'begin_fid': each_rela_triplet['duration'][0],
                'end_fid': each_rela_triplet['duration'][1],
                'score': each_rela_triplet['score']
            }
            pred_rela_ins_list.append(each_pred_rela_ins)
        anno4mat_res['trajectories'] = pred_trajectories_list
        anno4mat_res['relation_instances'] = pred_rela_ins_list

        vid_path_splits = gt_vid_anno['video_path'].split('/')
        predicate_anno_root_path = os.path.join(os.path.split(pred_res_path)[0],
                                                'predication', 'annotation', vid_path_splits[0])
        predicate_video_out_root_path = os.path.join(os.path.split(pred_res_path)[0],
                                                     'predication', 'videos', vid_path_splits[0])
        if not os.path.exists(predicate_anno_root_path):
            os.makedirs(predicate_anno_root_path)
        if not os.path.exists(predicate_video_out_root_path):
            os.makedirs(predicate_video_out_root_path)
        with open(os.path.join(predicate_anno_root_path, vid_path_splits[-1][:-4] + '.json'), 'w+') as pred_out_f:
            pred_out_json = json.dumps(anno4mat_res)
            pred_out_f.write(pred_out_json)

        for each_split in splits:
            origin_video_real_path = os.path.join(video_rpath, each_split, gt_vid_anno['video_path'])
            if os.path.exists(origin_video_real_path):
                vis_res_vid = os.path.join(predicate_video_out_root_path, vid_path_splits[-1])
                visualize(anno4mat_res, origin_video_real_path, vis_res_vid, pred=True)
                print("Generate visualize video: ", vis_res_vid)


def visualize(anno, video_path, out_path, pred=False):
    video = read_video(video_path)
    assert anno['frame_count'] == len(video), '{} : anno {} video {}'.format(anno['video_id'], anno['frame_count'], len(video))
    assert anno['width'] == video[0].shape[1] and anno['height'] == video[0].shape[0], \
        '{} : anno ({}, {}) video {}'.format(anno['video_id'], anno['height'], anno['width'], video[0].shape)
    # resize video to be 720p
    ratio = 720.0 / anno['height']
    boundary = 20
    size = int(round(anno['width'] * ratio)) + 2 * boundary, int(round(anno['height'] * ratio)) + 2 * boundary
    for i in range(anno['frame_count']):
        background = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        background[boundary:size[1] - boundary, boundary:size[0] - boundary] = cv2.resize(video[i], (
            size[0] - 2 * boundary, size[1] - 2 * boundary))
        video[i] = background
    # collect subject/objects
    subobj = dict()
    for x in anno['subject/objects']:
        subobj[x['tid']] = {
            'id': x['tid'] + 1,
            'name': x['category'],
            'color': _colors[x['tid'] % len(_colors)]
        }
    # collect related relations in each frame
    for i, f in enumerate(anno['trajectories']):
        for x in f:
            x['rels'] = []
            x['timestamp'] = -1
            for r in anno['relation_instances']:
                if r['subject_tid'] == x['tid'] and r['begin_fid'] <= i <= r['end_fid']:
                    x['rels'].append({
                        'timestamp': r['begin_fid'],
                        'predicate': r['predicate'],
                        'object_tid': r['object_tid']
                    })
                    if r['begin_fid'] > x['timestamp']:
                        x['timestamp'] = r['begin_fid']
    # draw frames
    max_timestamp = 1
    for i in range(len(anno['trajectories'])):
        f = anno['trajectories'][i]
        for x in sorted(f, key=lambda a: a['timestamp']):
            xmin = int(round(x['bbox']['xmin'] * ratio)) + boundary
            xmax = int(round(x['bbox']['xmax'] * ratio)) + boundary
            ymin = int(round(x['bbox']['ymin'] * ratio)) + boundary
            ymax = int(round(x['bbox']['ymax'] * ratio)) + boundary
            bbox_thickness = 1
            sub_name = '{}.{}'.format(x['tid'] + 1, subobj[x['tid']]['name'])
            sub_color = subobj[x['tid']]['color']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scalar = 0.5
            font_thickness = 1
            font_size, font_baseline = cv2.getTextSize(sub_name, font, font_scalar, font_thickness)
            h_font_scalar = 0.8
            h_font_thickness = 2
            h_font_size, h_font_baseline = cv2.getTextSize(sub_name, font, h_font_scalar, h_font_thickness)
            # draw subject
            cv2.rectangle(video[i], (xmin, ymin), (xmax, ymax), sub_color[::-1], bbox_thickness)
            cv2.rectangle(video[i], (xmin, ymin - font_size[1] - font_baseline), (xmin + font_size[0], ymin),
                          sub_color[::-1], -1)
            cv2.putText(video[i], sub_name, (xmin, ymin - font_baseline), font, font_scalar, (0, 0, 0), font_thickness,
                        cv2.LINE_AA)
            # draw relations
            if len(x['rels']) > 0:
                rels = sorted(x['rels'], key=lambda a: a['timestamp'], reverse=True)
                if rels[0]['timestamp'] > max_timestamp:
                    max_timestamp = rels[0]['timestamp']
                y = ymin + h_font_size[1]
                for r in rels:
                    obj_color = subobj[r['object_tid']]['color']
                    if pred:
                        rel_name = '{}_{}.{}, {}'.format(r['predicate'], r['object_tid'] + 1, subobj[r['object_tid']]['name'], r['score'])
                    else:
                        rel_name = '{}_{}.{}'.format(r['predicate'], r['object_tid'] + 1, subobj[r['object_tid']]['name'])
                    if r['timestamp'] == max_timestamp:
                        cv2.putText(video[i], rel_name, (xmin, y + font_baseline), font, h_font_scalar, obj_color[::-1],
                                    h_font_thickness, cv2.LINE_AA)
                        y += h_font_size[1] + h_font_baseline
                    else:
                        cv2.putText(video[i], rel_name, (xmin, y + font_baseline), font, font_scalar, obj_color[::-1],
                                    font_thickness, cv2.LINE_AA)
                        y += font_size[1] + font_baseline

    write_video(video, anno['fps'], size, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize annotation in video')
    parser.add_argument('-video', type=str, help='Root path of videos')
    parser.add_argument('-anno', type=str, help='A annotation json file or a directory of annotation jsons')
    parser.add_argument('-out', type=str, help='Root path of output videos')
    parser.add_argument('-pred', type=ast.literal_eval, default=False, help='If need 2 visualize pred, set this True')
    parser.add_argument('-pred_anno', type=str, help='Root path of predication json')

    args = parser.parse_args()

    if args.pred:
        # e.g.
        # python visualize.py \
        #   -video /home/daivd/PycharmProjects/vidor \
        #   -anno /home/daivd/PycharmProjects/vidor/annotation \
        #   -out /home/daivd/Desktop/testout \
        #   -pred True \
        #   -pred_anno /home/daivd/Desktop/test.json
        visualiza_pred(args.pred_anno, args.anno, args.video, 'vidor', ['validation'])
    else:
        if os.path.isdir(args.anno):
            anno_paths = glob.glob('{}/*.json'.format(args.anno))
            args.out = os.path.join(args.out, os.path.basename(os.path.normpath(args.anno)))
            if not os.path.exists(args.out):
                os.mkdir(args.out)
        else:
            anno_paths = [args.anno]

        for i in trange(len(anno_paths)):
            with open(anno_paths[i], 'r') as fin:
                anno = json.load(fin)
            if 'video_path' in anno:
                video_path = os.path.join(args.video, anno['video_path'])
            else:
                video_path = os.path.join(args.video, '{}.mp4'.format(anno['video_id']))
            out_path = os.path.join(args.out, '{}.mp4'.format(anno['video_id']))
            visualize(anno, video_path, out_path)
