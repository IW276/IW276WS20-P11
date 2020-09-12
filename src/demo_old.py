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


DIR_DATASETS = '../datasets/'
DIR_PRETRAINED_MODELS = '../pretrained-models/'

DATASET = 'human_pose.json'

MODEL_RESNET18 = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL_RESNET18 = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

WIDTH = 224
HEIGHT = 224

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
        else:
            peak = (j, None, None)
            kpoint.append(peak)
    return kpoint

parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--video', type=str, default='video.mp4')
parser.add_argument('--path', type=str, default='/videos/')
args = parser.parse_args()

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

print('Model optimized')

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

cap = cv2.VideoCapture(args.path+args.video)

print('Video captured')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter(args.path+'output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
X_compress = int(cap.get(3)) / WIDTH * 1.0
Y_compress = int(cap.get(4)) / HEIGHT * 1.0

print('out_video')


def execute(image, src, tm, out_vid):
    img_data = preprocess(image)
    cmap, paf = model_trt(img_data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - tm)
    print("FPS:%f " % fps)
    draw_objects(src, counts, objects, peaks)
    cv2.putText(src, "FPS: %f" % fps, (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    out_vid.write(src)


count = 0


while cap.isOpened():
    t = time.time()
    ret, frame = cap.read()

    if not ret:
        print("Video load Error.")
        break

    img = cv2.resize(frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    execute(img, frame, t, out_video)
    print('Started execution')

cv2.destroyAllWindows()
print('Windows destroyed')
out_video.release()
cap.release()
print('all released')

