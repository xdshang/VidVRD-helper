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


def visualize(anno, video_path, out_path):
    video = read_video(video_path)
    assert anno['frame_count']==len(video), '{} : anno {} video {}'.format(anno['video_id'], anno['frame_count'], len(video))
    assert anno['width']==video[0].shape[1] and anno['height']==video[0].shape[0],\
            '{} : anno ({}, {}) video {}'.format(anno['video_id'], anno['height'], anno['width'], video[0].shape)
    # resize video to be 720p
    ratio = 720.0/anno['height']
    boundary = 20
    size = int(round(anno['width']*ratio))+2*boundary, int(round(anno['height']*ratio))+2*boundary
    for i in range(anno['frame_count']):
        background = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        background[boundary:size[1]-boundary, boundary:size[0]-boundary] = cv2.resize(video[i], (size[0]-2*boundary, size[1]-2*boundary))
        video[i] = background
    # collect subject/objects
    subobj = dict()
    for x in anno['subject/objects']:
        subobj[x['tid']] = {
            'id': x['tid']+1,
            'name': x['category'],
            'color': _colors[x['tid']%len(_colors)]
        }
    # collect related relations in each frame
    for i, f in enumerate(anno['trajectories']):
        for x in f:
            x['rels'] = []
            x['timestamp'] = -1
            for r in anno['relation_instances']:
                if r['subject_tid']==x['tid'] and r['begin_fid']<=i<=r['end_fid']:
                    x['rels'].append({
                        'timestamp': r['begin_fid'],
                        'predicate': r['predicate'],
                        'object_tid': r['object_tid']
                    })
                    if r['begin_fid']>x['timestamp']:
                        x['timestamp'] = r['begin_fid']
    # draw frames
    max_timestamp = 1
    for i in range(len(anno['trajectories'])):
        f = anno['trajectories'][i]
        for x in sorted(f, key=lambda a: a['timestamp']):
            xmin = int(round(x['bbox']['xmin']*ratio))+boundary
            xmax = int(round(x['bbox']['xmax']*ratio))+boundary
            ymin = int(round(x['bbox']['ymin']*ratio))+boundary
            ymax = int(round(x['bbox']['ymax']*ratio))+boundary
            bbox_thickness = 1
            sub_name = '{}.{}'.format(x['tid']+1, subobj[x['tid']]['name'])
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
            cv2.rectangle(video[i], (xmin, ymin-font_size[1]-font_baseline), (xmin+font_size[0], ymin), sub_color[::-1], -1)
            cv2.putText(video[i], sub_name, (xmin, ymin-font_baseline), font, font_scalar, (0, 0, 0), font_thickness, cv2.LINE_AA)
            # draw relations
            if len(x['rels'])>0:
                rels = sorted(x['rels'], key=lambda a: a['timestamp'], reverse=True)
                if rels[0]['timestamp']>max_timestamp:
                    max_timestamp = rels[0]['timestamp']
                y = ymin+h_font_size[1]
                for r in rels:
                    obj_color = subobj[r['object_tid']]['color']
                    rel_name = '{}_{}.{}'.format(r['predicate'], r['object_tid']+1, subobj[r['object_tid']]['name'])
                    if r['timestamp']==max_timestamp:
                        cv2.putText(video[i], rel_name, (xmin, y+font_baseline), font, h_font_scalar, obj_color[::-1], h_font_thickness, cv2.LINE_AA)
                        y += h_font_size[1]+h_font_baseline
                    else:
                        cv2.putText(video[i], rel_name, (xmin, y+font_baseline), font, font_scalar, obj_color[::-1], font_thickness, cv2.LINE_AA)
                        y += font_size[1]+font_baseline

    write_video(video, anno['fps'], size, out_path)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize annotation in video')
    parser.add_argument('video', type=str, help='Root path of videos')
    parser.add_argument('anno', type=str, help='A annotation json file or a directory of annotation jsons')
    parser.add_argument('out', type=str, help='Root path of output videos')
    args = parser.parse_args()

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
