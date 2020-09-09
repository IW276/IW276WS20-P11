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
from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
import ipywidgets
from IPython.display import display
#import argparse
import os.path

DIR_DATASETS = '../datasets/'
DIR_PRETRAINED_MODELS = '../pretrained-models/'

DATASET = 'human_pose.json'

MODEL_RESNET18 = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL_RESNET18 = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

WIDTH = 224
HEIGHT = 224

with open(DIR_DATASETS + DATASET, 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

model.load_state_dict(torch.load(DIR_PRETRAINED_MODELS + MODEL_RESNET18))

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
torch.save(model_trt.state_dict(), DIR_PRETRAINED_MODELS + OPTIMIZED_MODEL_RESNET18)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(DIR_PRETRAINED_MODELS + OPTIMIZED_MODEL_RESNET18))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
# camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)

camera.running = True

image_w = ipywidgets.Image(format='jpeg')

display(image_w)

def execute(change):
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    image_w.value = bgr8_to_jpeg(image[:, ::-1, :])


#execute({'new': camera.value})

camera.observe(execute, names='value')

#camera.unobserve_all()

