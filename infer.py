import mmcv
import torch
import sys
import os
import joblib
import pandas as pd

from mmdet.apis import inference_detector, init_detector, show_result
config = './configs/OpenImages_configs/r50-FPN-1x_classsampling_TSD/r50-FPN-1x_classsampling_TSD.py'
ckpt = './data/checkpoints/r50-FPN-1x_classsampling_TSD.pth'

model = init_detector(config, ckpt, device=torch.device("cuda", 0))
model.CLASSES = pd.read_csv('./data/checkpoints/challenge2019/cls-label-description.csv', header=None).sort_values(2)[1].tolist()

import glob
files = sorted(glob.glob(sys.argv[1]))
out = sys.argv[2] if sys.argv[2:] else 'result.pkl'

if os.path.exists(out):
    all_res = joblib.load(out)
else:
    all_res={}

def save():
    joblib.dump(all_res, out)
    print('save to', out)


for i, f in enumerate(files):
    print(i, f)
    if f in all_res:
        print('skip')
        continue
    img = mmcv.imread(f)
    result = inference_detector(model, img)
    #img = show_result(img, result, model.CLASSES, score_thr=0.3, show=False)
    #img = mmcv.imrescale(img, 0.8)
    #mmcv.imwrite(img, out_file, auto_mkdir=True)
    all_res[f] = result
    if (i+1) % 10000 == 0:
        save()

save()

