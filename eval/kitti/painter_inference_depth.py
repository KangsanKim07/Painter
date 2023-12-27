# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import sys
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')
import lora_models_painter


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1'):
    # build model
    model = getattr(lora_models_painter, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.load_state_dict(torch.load(args.lora_path)['model'], strict=False)
    model.eval()
    return model


def run_one_image(img, tgt, size, model, out_path, device):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)
    
    loss, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 80000, 0, 80000)
    output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    output = output.mean(-1).int()
    output = Image.fromarray(output.numpy())
    if not os.path.exists('/'.join(out_path.split('/')[:-1])):
        os.makedirs('/'.join(out_path.split('/')[:-1]))
    output.save(out_path)
    

def get_args_parser():
    parser = argparse.ArgumentParser('NYU Depth V2', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='study_room_0005b/rgb_00094')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--lora_path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    ckpt_path = args.ckpt_path

    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]

    model_painter = prepare_model(ckpt_path, args.model)
    print('Model loaded.')

    device = torch.device("cuda")
    model_painter.to(device)

    dst_dir = os.path.join('/c1/kangsan/Painter/models_inference',
                           "kitti_depth_inference/")
    print(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    img_src_dir = "/c1/kangsan/Painter/datasets/kitti"
    anno_path_list = glob.glob(img_src_dir + f"/annotations/val/*/*/*/*/*.png")
    img_path_list = []
    for anno_path in anno_path_list:
        date = anno_path.split('/')[8]
        date_fd = date[:10]
        img_fd = anno_path.split('/')[-2]
        img_name = anno_path.split('/')[-1]
        image_path = f"/c1/kangsan/Painter/datasets/kitti/images/{date_fd}/{date}/{img_fd}/data/{img_name}"
        img_path_list.append(image_path)

    img2_path = "/c1/kangsan/Painter/datasets/kitti/images/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png"
    tgt_path = "/c1/kangsan/Painter/datasets/kitti/annotations/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png"
    tgt2_path = tgt_path

    res, hres = args.input_size, args.input_size

    for img_path in tqdm.tqdm(img_path_list):
        date = img_path.split('/')[8]
        img_fd = img_path.split('/')[9]
        img_name = img_path.split('/')[-1]
        
        out_path = os.path.join(dst_dir, date, img_fd, img_name)
        img = Image.open(img_path).convert("RGB")
        size = img.size
        img = img.resize((res, hres))
        img = np.array(img) / 255.
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.
        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        tgt = Image.open(tgt_path)
        tgt = np.array(tgt) / 80000.
        tgt = tgt * 255
        tgt = Image.fromarray(tgt).convert("RGB")
        tgt = tgt.resize((res, hres))
        tgt = np.array(tgt) / 255.
        tgt2 = Image.open(tgt2_path)
        tgt2 = np.array(tgt2) / 80000.
        tgt2 = tgt2 * 255
        tgt2 = Image.fromarray(tgt2).convert("RGB")
        tgt2 = tgt2.resize((res, hres))
        tgt2 = np.array(tgt2) / 255.
        tgt = np.concatenate((tgt2, tgt), axis=0)

        assert tgt.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        torch.manual_seed(2)
        with torch.no_grad():
            run_one_image(img, tgt, size, model_painter, out_path, device)
