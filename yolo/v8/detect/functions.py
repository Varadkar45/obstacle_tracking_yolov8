import cv2
import torch
import warnings
from PIL import ImageOps
import numpy as np
from collections import deque
import math
import json


## Variables ##

output_dir =r'out\third_eye_tracker.mp4'

images = []
database = {}

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml"
font = cv2.FONT_HERSHEY_DUPLEX
obstacles = ['car', 'person', 'motorcycle','truck','bicycle', 'parking meter', 'cow', 'dog']

obs_hist = [0,0,0]
warn_count = 0

## Functions ##

def send_state(data):
    '''
    Used to send the current obstacle state to obs_state.json which is used for alerting
    '''
    with open('obs_state.json', 'w') as f:
        json.dump({"state":data}, f)

def new_obs(label, warn_avg_size):
    '''
    Used to create a new dictionary element in the obstacle database to store the required variables of each obstacle
    '''
    return {
        "area_hist":deque(maxlen=warn_avg_size), 
        "angle_hist":deque(maxlen=warn_avg_size), 
        "large":False, 
        "label":label,
        "del_area":0,
        "del_angle":0,
        "warning":deque(maxlen=warn_avg_size)
    }

def find_angle(mid, height, xc, yc):
    '''
    Returns the angle made by the obstacle with the reference point of the user (bottom mid)
    '''
    return np.abs(math.atan((mid - xc) / ((height - yc) + 0.001)) * (180/math.pi))

def update_angle(database, id, xc, yc, frame):
    '''
    Updates the angle history of obstacles in 'angle_hist' and the change in angle in 'del_angle'
    '''
    height, width= frame.shape[:2] 
    mid = int(width / 2)

    database[id]['angle_hist'].append(find_angle(mid, height, xc, yc))
    del_angle= get_del(database[id]['angle_hist'])
    database[id]['del_angle'] = del_angle
    frame = cv2.putText(frame,str(format(database[id]['del_angle'],".2f")),(xc,yc),font,0.5,(255,0,255),1)
    return database, frame

def get_del(vals):
    '''
    Used to estimate the change in value for area and angle (rate of increase or decrease)
    '''
    if(len(vals)<=1):
        return 0
    delsum = 0
    for i in range(1, len(vals)):
        delsum += vals[i] - vals[i-1]
    return delsum / len(vals)

def draw_ROI(frame,roi, ext_roi):
    '''
    Adds ROI outline to the image output
    '''
    height, width= frame.shape[:2]  
    left, right = int(roi * width), int((1-roi) * width)
    mid = int(width / 2)
    ROI_region = [[(left,height),(left,0),(right,0),(right,height)]]
    ROI_region2 = [[(left,height),(left,0),(mid,0),(mid,height)]]
    cv2.rectangle(frame, ROI_region[0][1],ROI_region[0][3],(0,0,0),1)
    cv2.rectangle(frame, ROI_region2[0][1],ROI_region2[0][3],(0,0,0),1)
    cv2.rectangle(frame, (int(ext_roi*width),0),(int((1-ext_roi) * width), height),(0,0,0),1)
    return frame, left, right

def x1y1x2y2_to_xywh(x1y1x2y2):
    '''
    Performs co-ordinate transformation for Deepsort model
    '''
    xywhs = torch.zeros_like(x1y1x2y2)
    for i,xyxy in enumerate(x1y1x2y2):
        x1,y1,x2,y2 = xyxy[0],xyxy[1],xyxy[2],xyxy[3]
        w = x2 - x1
        h = y2 - y1
        xc = x1 + w/2
        yc = y1 + h/2
        xywhs[i,0] = xc
        xywhs[i,1] = yc
        xywhs[i,2] = w 
        xywhs[i,3] = h
    return xywhs

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return np.asarray(ImageOps.expand(img, padding))

def save_output(images, fps, output_dir = output_dir):
    size = images[0].shape
    out = cv2.VideoWriter(output_dir,cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1],size[0]))
    for i in range(len(images)):
        out.write(images[i])
    out.release()