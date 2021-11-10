import mmcv
import torch
import time
from mmdet.apis import inference_detector, init_detector, show_result
config = './configs/OpenImages_configs/r50-FPN-1x_classsampling_TSD/r50-FPN-1x_classsampling_TSD.py'
ckpt = './data/checkpoints/r50-FPN-1x_classsampling_TSD.pth'

model = init_detector(config, ckpt, device=torch.device("cuda", 0))

img = mmcv.imread('./test.jpg')
start = time.time()
result = inference_detector(model, img)
print('time cost:', time.time()-start)

import pandas as pd
model.CLASSES = pd.read_csv('./data/checkpoints/challenge2019/cls-label-description.csv', header=None).sort_values(2)[1].tolist()
#show_result(img, result, model.CLASSES, score_thr=0.5, show=False, out_file="test2.jpg")

