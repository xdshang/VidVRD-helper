`dataset.py` provides the general methods to parse the annotations for two datasets,
while other specific scripts, such as `imagenet_vidvrd.py` and `vidor.py`, implements particular details
of the datasets, such as directory structure and label information.

It is recommended to place the downloaded datasets under the same folder as this repository, like:
```
├── vidor-dataset
│   ├── annotation
│   │   ├── training
│   │   └── validation
│   └── video
├── vidvrd-dataset
│   ├── test
│   ├── train
│   └── videos
└── VidVRD-helper
    ├── baseline
    ├── baseline.py
    ├── dataset
    ├── evaluate.py
    ├── evaluation
    ├── LICENSE
    ├── README.md
    └── visualize.py
```
Otherwise, you need to change the details in the spefic scripts.
