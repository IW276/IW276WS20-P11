import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path

DIR_DATASETS = '../datasets/'
DIR_PRETRAINED_MODELS = '../pretrained-models/'
PATH_TO_JSON = DIR_DATASETS + 'human_pose.json'
PATH_TO_DENSENET = DIR_PRETRAINED_MODELS + 'densenet121_baseline_att 256x256_B_epoch_160.pth'
#PATH_TO_RESNET = DIR_PRETRAINED_MODELS + 'resnet18_baseline_att_224x224_A_epoch_249.pth'

def main(video_path, json_path):
    json_data = load_json(PATH_TO_JSON)
    
def load_json(path):
    json_data = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r") as f:
            json_data.append(json.load((f)))
    return json_data