The baseline code for the VidVRD dataset introduced in the following paper.
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

## Baseline Quick Start
1. Install the prerequisites
```
conda create -n vidvrd python=2.7 anaconda cmake tensorflow=1.8.0 keras tqdm ffmpeg=3.4 py-opencv
export PYTHONNOUSERSITE=1 && source activate vidvrd
pip install dlib==19.3.1 --isolated
``` 
2. Download precomputed features, model and detected relations from [here](https://zdtnag7mmr.larksuite.com/file/boxusS8Z0kwEizoPPh5h7vx7Usf), and decompress the zipfile under the same folder as this repository.
3. Run `python evaluate.py vidvrd test relation ../vidvrd-baseline-output/models/baseline_relation_prediction.json` to evaluate the precomputed detected relations. Since a few wrong labels in the dataset were corrected after paper submission, the result is slightly different from the one reported in the paper. Some qualitative results can be found [here](http://mm.zl.io).
4. Run `python baseline.py --detect` to detect video visual relations using the precomputed model.
5. Run `python baseline.py --train` to train a new model by adjusting the hyperparameters in the script, based on the precomputed features.
