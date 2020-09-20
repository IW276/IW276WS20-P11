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

import sys
import jetson.utils
import argparse
from os import path


DIR_DATASETS = '../datasets/'
DIR_PRETRAINED_MODELS = '../pretrained-models/'

DATASET = 'human_pose.json'

MODEL_RESNET18 = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL_RESNET18 = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

WIDTH = 224
HEIGHT = 224

parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--video', type=str, default='video.mp4', help="URI of the input stream")
parser.add_argument('--path', type=str, nargs='?', default='/videos/', help="URI of the output stream")
#args = parser.parse_args()
args = parser.parse_known_args()[0]

splited_video= args.video.split('.')
video_name = splited_video[0]

# create video sources & outputs
input = jetson.utils.videoSource(args.path + args.video, argv=sys.argv)
output = jetson.utils.videoOutput(args.path + video_name + '_demo_old.mp4', argv=sys.argv)


def preprocess(image):
    global device
    device = torch.device('cuda')
    #image = cv2.cvtColor(image, )
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def clean_up():
    cv2.destroyAllWindows()
    print('all released')


def execute(image, tm):
    img_data = preprocess(image)
    cmap, paf = model_trt(image)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    fps = 1.0 / (time.time() - tm)
    print("FPS:%f " % fps)
    draw_objects(image, counts, objects, peaks)
    cv2.putText(image, "FPS: %f" % fps, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


with open(DIR_DATASETS + DATASET, 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

if not path.exists(DIR_PRETRAINED_MODELS + OPTIMIZED_MODEL_RESNET18):
    model.load_state_dict(torch.load(DIR_PRETRAINED_MODELS + MODEL_RESNET18))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), DIR_PRETRAINED_MODELS + OPTIMIZED_MODEL_RESNET18)
    print('Model optimized')

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(DIR_PRETRAINED_MODELS + OPTIMIZED_MODEL_RESNET18))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

output.Open()
# capture frames until user exits
while output.IsStreaming():
    t = time.time()
    img = input.Capture(format='rgb8')
    output.Render(img)
    output.SetStatus("Video Viewer | {:d}x{:d} | {:.1f} FPS".format(img.width, img.height, output.GetFrameRate()))
    print('Started execution')
    execute(img, t)


clean_up()
print("Process finished")