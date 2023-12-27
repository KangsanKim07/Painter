# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import random

from PIL import Image
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform


class PairDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        json_path_list: list,
        transform: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
        transform3: Optional[Callable] = None,
        transform_seccrop: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        masked_position_generator: Optional[Callable] = None,
        use_two_pairs: bool = True,
        half_mask_ratio:float = 0.,
        current_labeled_set = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.pairs = []
        self.weights = []
        type_weight_list = [1]
        for idx, json_path in enumerate(json_path_list):
            cur_pairs = json.load(open(json_path))
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
            self.weights.extend([type_weight_list[idx] * 1./cur_num]*cur_num)
            print(json_path, type_weight_list[idx])
        if current_labeled_set is not None:
            self.current_lab = json.load(open(current_labeled_set))
        self.use_two_pairs = use_two_pairs
        if self.use_two_pairs:
            self.pair_type_dict = {}
            for idx, pair in enumerate(self.pairs):
                if "type" in pair:
                    if pair["type"] not in self.pair_type_dict:
                        self.pair_type_dict[pair["type"]] = [idx]
                    else:
                        self.pair_type_dict[pair["type"]].append(idx)
            for t in self.pair_type_dict:
                print(t, len(self.pair_type_dict[t]))
        self.transforms = PairStandardTransform(transform, target_transform) if transform is not None else None
        self.transforms2 = PairStandardTransform(transform2, target_transform) if transform2 is not None else None
        self.transforms3 = PairStandardTransform(transform3, target_transform) if transform3 is not None else None
        self.transforms_seccrop = PairStandardTransform(transform_seccrop, target_transform) if transform_seccrop is not None else None
        self.masked_position_generator = masked_position_generator
        self.half_mask_ratio = half_mask_ratio
        
        with open('/c1/kangsan/Painter/datasets/VOC2012/similarity/train2train_top50-similarity.json', 'r') as f:
            self.t2tsimilarity = json.load(f)
        with open('/c1/kangsan/Painter/datasets/VOC2012/similarity/val2train_top50-similarity.json', 'r') as f:
            self.v2tsimilarity = json.load(f)

    def _load_image(self, path: str) -> Image.Image:
        while True:
            try:
                img = Image.open(os.path.join(self.root, path))
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break
        # process for nyuv2 depth: scale to 0~255
        if "sync_depth" in path:
            # nyuv2's depth range is 0~10m
            img = np.array(img) / 10000.
            img = img * 255
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return img

    def _combine_images(self, image, image2, interpolation='bicubic'):
        # image under image2
        h, w = image.shape[1], image.shape[2]
        dst = torch.cat([image, image2], dim=1)
        return dst

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]
        image_origin = self._load_image(pair['image_path'])
        target_origin = self._load_image(pair['target_path'])

        # decide mode for interpolation
        pair_type = pair['type']
        if "depth" in pair_type or "pose" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        elif "image2" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'nearest'
        elif "2image" in pair_type:
            interpolation1 = 'nearest'
            interpolation2 = 'bicubic'
        else:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
            
        # no aug for instance segmentation
        if "inst" in pair['type'] and self.transforms2 is not None:
            cur_transforms = self.transforms2
        elif "pose" in pair['type'] and self.transforms3 is not None:
            cur_transforms = self.transforms3
        else:
            cur_transforms = self.transforms

        image_origin, target_origin = cur_transforms(image_origin, target_origin, interpolation1, interpolation2)
        pseudo = [x[:-4] for x in os.listdir('/c1/kangsan/Painter/datasets/VOC2012/pseudo')]

        if self.transforms2 is not None:
            isTraining = True
        else:
            isTraining = False
            
        if self.use_two_pairs:
            name = os.path.basename(pair['image_path'])[:-4]
            current_lab = [os.path.basename(x['image_path'])[:-4] for x in self.pairs]
            bimages, btargets, bmasks, bvalids = [], [], [], []
            if isTraining: # training
                current_lab.remove(name)
                demo_names = [x for x in self.t2tsimilarity[name] if x in current_lab]
                if len(demo_names) == 0:
                    demo_names = [random.choice(current_lab)]
            else: # validation
                candidates = self.v2tsimilarity[name]
                current_lab = [os.path.basename(x['image_path'])[:-4] for x in self.current_lab]
                demo_names = [x for x in candidates if x in current_lab]
                if len(demo_names) == 0:
                    demo_names = [random.choice(current_lab)]
            
            for demo_name in demo_names[:5]:
                if isTraining:
                    image2 = self._load_image(pair['image_path'].replace(name, demo_name))
                    if not demo_name in pseudo:
                        target2 = self._load_image(f"VOC2012/annotations_with_color/training/{demo_name}.png")
                    else:
                        target2 = self._load_image(f"VOC2012/pseudo/{demo_name}.png")
                else:
                    image2 = self._load_image(pair['image_path'].replace(name, demo_name).replace("validation", "training"))
                    if demo_name not in pseudo:
                        target2 = self._load_image(f"VOC2012/annotations_with_color/training/{demo_name}.png")
                    else:
                        target2 = self._load_image(f"VOC2012/pseudo/{demo_name}.png")
                    
                image2, target2 = cur_transforms(image2, target2, interpolation1, interpolation2)
                image = self._combine_images(image_origin, image2, interpolation1)
                target = self._combine_images(target_origin, target2, interpolation2)

                use_half_mask = torch.rand(1)[0] < self.half_mask_ratio
                if (self.transforms_seccrop is None) or ("inst" in pair['type']) or ("pose" in pair['type']) or use_half_mask:
                    pass
                else:
                    image, target = self.transforms_seccrop(image, target, interpolation1, interpolation2)
                
                valid = torch.ones_like(target)
                imagenet_mean=torch.tensor([0.485, 0.456, 0.406])
                imagenet_std=torch.tensor([0.229, 0.224, 0.225])
                if "image2depth" in pair_type:
                    thres = torch.ones(3) * (1e-3 * 0.1)
                    thres = (thres - imagenet_mean) / imagenet_std
                    valid[target < thres[:, None, None]] = 0
                elif "ade20k_image2semantic" in pair_type:
                    thres = torch.ones(3) * (1e-5) # ignore black
                    thres = (thres - imagenet_mean) / imagenet_std
                    valid[target < thres[:, None, None]] = 0
                elif "image2panoptic_sem_seg" in pair_type:
                    thres = torch.ones(3) * (1e-5) # ignore black
                    thres = (thres - imagenet_mean) / imagenet_std
                    valid[target < thres[:, None, None]] = 0
                elif "image2pose" in pair_type:
                    thres = torch.ones(3) * (1e-5) # ignore black
                    thres = (thres - imagenet_mean) / imagenet_std
                    valid[target > thres[:, None, None]] = 10.0
                    fg = target > thres[:, None, None]
                    if fg.sum() < 100*3:
                        valid = valid * 0.
                elif "image2panoptic_inst" in pair_type:
                    thres = torch.ones(3) * (1e-5) # ignore black
                    thres = (thres - imagenet_mean) / imagenet_std
                    fg = target > thres[:, None, None]
                    if fg.sum() < 100*3:
                        valid = valid * 0.

                if use_half_mask:
                    num_patches = self.masked_position_generator.num_patches
                    mask = np.zeros(self.masked_position_generator.get_shape(), dtype=np.int32)
                    mask[mask.shape[0]//2:, :] = 1
                else:
                    mask = self.masked_position_generator()
                
                bimages.append(image)
                btargets.append(target)
                bmasks.append(torch.tensor(mask))
                bvalids.append(valid)
        
        bimages = torch.stack(bimages)
        btargets = torch.stack(btargets)
        bmasks = torch.stack(bmasks)
        bvalids = torch.stack(bvalids)
        
        return bimages, btargets, bmasks, bvalids

    def __len__(self) -> int:
        return len(self.pairs)


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target = self.transform(input, target, interpolation1, interpolation2)
        return input, target
