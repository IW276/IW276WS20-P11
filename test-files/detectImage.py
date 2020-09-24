import torch
import json
import trt_pose.coco
import trt_pose.models
import PIL.Image
import cv2
import torchvision.transforms as transforms
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import ipywidgets
from IPython.display import display

DIR_DATASETS = '../datasets/'
DATASET = 'resnet18CrowdPose.json'

DIR_PRETRAINED_MODELS = '../pretrained-models/'
MODEL_RESNET18 = 'resnet18_crowdpose_224x224_epoch_129.pth'
OPTIMIZED_MODEL_RESNET18 = 'resnet18_crowdpose_224x224_epoch_129_trt.pth'

with open(DIR_DATASETS + DATASET, 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print(num_links)

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

model.load_state_dict(torch.load(DIR_PRETRAINED_MODELS + MODEL_RESNET18))

WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model.load_state_dict(torch.load(DIR_PRETRAINED_MODELS + MODEL_RESNET18))

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

def execute(image):
    data = preprocess(image)
    cmap, paf = model(data)
    print(cmap)
    print(paf)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)

src = cv2.imread("TestPic.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(src, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

execute(img)
cv2.imwrite("Ergebnis17.jpg", img)