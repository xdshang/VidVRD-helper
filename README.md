# Video Visual Relation Detection Helpler

This repository contains the helper script for loading [VidVRD dataset](http://lms.comp.nus.edu.sg/research/VidVRD.html), and the script for evaluating the VidVRD task.
Please cite this paper if the dataset helps your research:
```
@inproceedings{shang2017video,
    author={Shang, Xindi and Ren, Tongwei and Guo, Jingfan and Zhang, Hanwang and Chua, Tat-Seng},
    title={Video Visual Relation Detection},
    booktitle={ACM International Conference on Multimedia},
    address={Mountain View, CA USA},
    month={October},
    year={2017}
}
```

#### Prediction JSON format
```python
{
  "ILSVRC2015_train_00010001": [                # video ID
    { 
      # a visual relation instance
      "triplet": ["person", "ride", "bicycle"]  # relation triplet
      "score": 0.9                              # confidence score
      "duration": [15, 150]                     # starting and ending frame IDs
      "sub_traj": [                             # subject trajectory
        [9.0, 10.0, 45.0, 20.0],                # bounding box at the starting frame
        ...                                     # [left, top, right, bottom]
      ]
      "obj_traj": [                             # object trajectory
        [22.0, 23.0, 67.0, 111.0],
        ...
      ]
    },
    ...                                         # other instances predicted for this video
  ],
  ...                                           # other videos
}
```
