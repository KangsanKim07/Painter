# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os
import glob
import json
import tqdm
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('NYU Depth V2 preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['train', 'val'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='/c1/kangsan/Painter/datasets/kitti')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    output_dict = []
    save_path = os.path.join(args.output_dir, "kitti_{}_image_depth.json".format(args.split))
    anno_path_list = glob.glob(args.output_dir + f"/annotations/{args.split}/*/*/*/*/*.png")
    
    for anno_path in tqdm.tqdm(anno_path_list):
        date = anno_path.split('/')[8]
        date_fd = date[:10]
        img_fd = anno_path.split('/')[-2]
        img_name = anno_path.split('/')[-1]

        image_path = f"kitti/images/{date_fd}/{date}/{img_fd}/data/{img_name}"
        target_path = '/'.join(anno_path.split('/')[5:])

        pair_dict = {}
        pair_dict["image_path"] = image_path
        pair_dict["target_path"] = target_path
        pair_dict["type"] = "nyuv2_image2depth"
        output_dict.append(pair_dict)

    json.dump(output_dict, open(save_path, 'w'))
