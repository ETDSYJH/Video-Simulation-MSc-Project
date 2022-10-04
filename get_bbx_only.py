import pandas as pd
import numpy as np
import os

import cv2
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg

import random
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
seg_model.eval()
seg_model.to(device)



model = torch.load('C:/Users/YJH\Desktop/_/window tracking/mask-rcnn-window.h5')
model.eval()
model.to(device)






def get_prediction(img_path,confidence= 0.15):
    img = img_path
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    with torch.no_grad():
      pred = model([img])
      scores = list(pred[0]['scores'].detach().cpu().numpy())
      pred_bbox = list(pred[0]['boxes'].detach().cpu().numpy())
      output= []
      for i in range(len(scores)):
        if scores[i] > confidence:
          instance = {}
          instance['bbx_id'] = i
          instance['scores'] = scores[i]
          instance['bbox'] = list(pred_bbox[i])
          output.append(instance)
    torch.cuda.empty_cache()

    return output


def get_bbx(img_path, video_data = True):

    if video_data:
        video = cv2.VideoCapture(img_path)

        vid_pred = {}
        key = 0

        while True:
            grabbed,frame=video.read()
            if not grabbed:
                break

            outputs = get_prediction(frame)
            vid_pred[key]=outputs
            key+=1
            
    else:
        vid_pred = {}
        key = 0
        kitti_seq = sorted(os.listdir(img_path))

        for file in kitti_seq[2153:3835]:
            frame = Image.open(os.path.join(img_path, file))

            outputs = get_prediction(frame)
            vid_pred[key]=outputs
            key+=1


    return vid_pred