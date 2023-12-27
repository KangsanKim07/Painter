#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image
import cv2

if __name__ == "__main__":
    dataset_dir = "/c1/kangsan/Painter/datasets/livecell/"
    names = os.listdir(dataset_dir+'pano_sem_seg/panoptic_segm_val_with_color')
    for name in names:
        img = cv2.imread(dataset_dir+'pano_sem_seg/panoptic_segm_val_with_color/'+name)
        img = img // 255
        img = img + (img - 1)
        cv2.imwrite(dataset_dir+"panoptic_semseg_val/"+name, img)
    
