import mmcv
import torch
import sys
import os
import pandas as pd

from mmdet.apis import inference_detector, init_detector, show_result
config = './configs/OpenImages_configs/r50-FPN-1x_classsampling_TSD/r50-FPN-1x_classsampling_TSD.py'
ckpt = './data/checkpoints/r50-FPN-1x_classsampling_TSD.pth'

model = init_detector(config, ckpt, device=torch.device("cuda", 0))
model.CLASSES = pd.read_csv('./data/checkpoints/challenge2019/cls-label-description.csv', header=None).sort_values(2)[1].tolist()

import glob
files = glob.glob(sys.argv[1])
out = sys.argv[2] if sys.argv[2:] else 'result.pkl'

import pickle
if os.path.exists(out):
    with open(out, 'rb') as f:
        all_res = pickle.load(f)
else:
    all_res={}

for f in files:
    print(f)
    # out_file = "/".join(["result"] + f.split('/')[-2:])
    #if os.path.exists(out_file):
    #    print('skip')
    #    continue
    img = mmcv.imread(f)
    result = inference_detector(model, img)

    #img = show_result(img, result, model.CLASSES, score_thr=0.3, show=False)
    #img = mmcv.imrescale(img, 0.8)
    #mmcv.imwrite(img, out_file, auto_mkdir=True)
    all_res[f] = result

with open('result.pkl', 'wb') as f:
    pickle.dump(all_res, f)
