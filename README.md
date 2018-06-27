# Video Visual Relation Detection Helpler

This repository contains some helper functions for the convenient usage of [VidVRD dataset](http://lms.comp.nus.edu.sg/research/VidVRD.html). It also has a script for evaluating the VidVRD task.
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

## Baseline Quick Start
1. Install the prerequisites
```
conda create -n vidvrd python=2.7 anaconda cmake tensorflow=1.8.0 keras tqdm
source activate vidvrd
pip install dlib==19.3.1 --isolated
``` 
2. Download precomputed features, model and detected relations from [here](http://dl.nextcenter.org/public/nuswide/VidVRD-baseline-precomputed.zip), and decompress the zipfile under `baseline` folder.
3. Run `python evaluation.py baseline/models/baseline_video_relations.json` to evaluate the precomputed detected relations. Since a few wrong labels in the dataset were corrected after paper submission, the result is slightly different from the one reported in the paper. Some qualitative results can be found [here](http://mm.zl.io).
4. Run `python baseline.py --detect` to detect video visual relations using the precomputed model.
5. Run `python baseline.py --train` to train a new model by adjusting the hyperparameters in the script, based on the precomputed features.
