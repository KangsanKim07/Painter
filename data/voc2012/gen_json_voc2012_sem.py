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
    parser = argparse.ArgumentParser('ADE20k semantic segmentation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['training', 'validation'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/VOC2012')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()
    data_root = '/c1/kangsan/Painter/'
    image_dir = os.path.join(data_root, "datasets/VOC2012/JPEGImages", args.split)
    annos_dir = os.path.join(data_root, "datasets/VOC2012/annotations_with_color", args.split)
    save_path = os.path.join(data_root, args.output_dir, "VOC2012_{}_image_semantic.json".format(args.split))

    output_dict = []

    image_path_list = glob.glob(os.path.join(image_dir, '*g'))
    for image_path in tqdm.tqdm(image_path_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        image_path = os.path.join(image_dir, image_name + '.jpg')
        panoptic_path = os.path.join(annos_dir, image_name + '.png')
        assert os.path.isfile(image_path)
        assert os.path.isfile(panoptic_path)
        pair_dict = {}
        pair_dict["image_path"] = os.path.join("VOC2012/JPEGImages/{}/".format(args.split), image_name + ".jpg")
        pair_dict["target_path"] = "VOC2012/annotations_with_color/{}/".format(args.split) + image_name + ".png"
        pair_dict["type"] = "voc2012_image2semantic"
        output_dict.append(pair_dict)
    json.dump(output_dict, open(save_path, 'w'))
