"""
Extract features for UnsupPR.
"""
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F

import timm
import sys


model_name = sys.argv[1]
feature_name = sys.argv[2]
split = sys.argv[3]

"""
model names:
# vit_large_patch14_224_clip_laion2b
# eva_large_patch14_196.in22k_ft_in22k_in1k
# resnet50
# vit_large_patch16_224.augreg_in21k_ft_in1k
# resnet18
# vit_large_patch14_clip_224.laion2b_ft_in12k_in1k
# vit_base_patch16_224.dino

python tools/featextrater_folderwise_UnsupPR.py vit_large_patch14_clip_224.laion2b features_vit-laion2b val
python tools/calculate_similariity.py features_vit-laion2b val trn
"""
model = timm.create_model(model_name, pretrained=True)
model.eval()
model = model.cuda()


# load the image transformer
t = []
t.append(T.Resize(model.pretrained_cfg['input_size'][1], interpolation=Image.BICUBIC))
t.append(T.CenterCrop(model.pretrained_cfg['input_size'][1]))
t.append(T.ToTensor())
t.append(T.Normalize(model.pretrained_cfg['mean'], model.pretrained_cfg['std']))
center_crop = T.Compose(t)


# save_dir = f"/c1/kangsan/Painter/datasets/VOC2012/similarity/{feature_name}_{split}"
# image_root = f"/c1/kangsan/Painter/datasets/VOC2012/JPEGImages/{split}"
# examples = os.listdir(f"/c1/kangsan/Painter/datasets/VOC2012/SegmentationClass/{split}")
# examples = [os.path.join(image_root, example.strip()[:-4]+'.jpg') for example in examples]

save_dir = f"/c1/kangsan/Painter/datasets/ap10k/similarity/{feature_name}_{split}"
image_root = f"/c1/kangsan/Painter/datasets/ap10k/data_pair/{split}"
examples = os.listdir(f"/c1/kangsan/Painter/datasets/ap10k/data_pair/{split}")
examples = [os.path.join(image_root, example) for example in examples if example.endswith('image.png')]
    
imgs = []

global_features = torch.tensor([]).cuda()
for example in tqdm(examples):
    path = os.path.join(example)
    img = Image.open(path).convert("RGB")
    img = center_crop(img)
    imgs.append(img)

    if len(imgs) == 128:

        imgs = torch.stack(imgs).cuda()
        with torch.no_grad():
            features = model.forward_features(imgs)
            features = model.forward_head(features,pre_logits=True)
            if len(global_features) == 0:
                global_features = features
            else:
                global_features = torch.cat((global_features,features))

        imgs = []

imgs = torch.stack(imgs).cuda()
with torch.no_grad():
    features = model.forward_features(imgs)
    features = model.forward_head(features,pre_logits=True)
    if len(global_features) == 0:
        global_features = features
    else:
        global_features = torch.cat((global_features,features))

features = global_features.cpu().numpy().astype(np.float32)
np.savez(save_dir, examples=examples, features=features)