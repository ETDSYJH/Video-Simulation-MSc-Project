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





def get_coloured_mask(mask, building=False):

    r = np.zeros_like(mask)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1]=255
    if building:
      coloured_mask = np.stack([r, b, b], axis=2)
    else:
      coloured_mask = np.stack([b,r, b], axis=2)
    return coloured_mask

def get_prediction(img_path, ins_model, confidence):
    img = img_path
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    with torch.no_grad():
      pred = ins_model([img])
      scores = pred[0]['scores'].detach().cpu().numpy()


      pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'][scores>confidence].detach().cpu().numpy())]
      # dumped_bx = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'][scores<=confidence].detach().cpu().numpy())]


      pred_mask = (pred[0]['masks'][scores>confidence].detach().cpu().numpy())> 0.6
      # dumped_mask = (pred[0]['masks'][scores<=confidence].detach().cpu().numpy())> 0.6

    del pred
    torch.cuda.empty_cache()
    return pred_mask, pred_boxes, #dumped_mask, dumped_bx


def get_seg(img):
  inputs = feature_extractor(images=img, return_tensors="pt").to(device)
  with torch.no_grad():
    outputs = seg_model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

# First, rescale logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
      logits,
      size=(1070, 1920), # (height, width)
      mode='bilinear',
      align_corners=False
)
    del outputs
    torch.cuda.empty_cache()

# Second, apply argmax on the class dimension
  pred_seg = upsampled_logits.argmax(dim=1)[0]
  return pred_seg






def segment_video(img_path, confidence=0.1, rect_th=2, text_size=2, text_th=2):

  video = cv2.VideoCapture(img_path)

  img_array = []


  previous_bbx = []
  while True:
    grabbed,frame=video.read()
    if not grabbed:
      break

    masks, boxes = get_prediction(frame, confidence)
    
    # if len(previous_bbx) != 0:
    #   for i in range(len(dumped_bx)):
    #     box = dumped_bx[i]
    #     distance = np.absolute(np.array(box)-np.array(previous_bbx))
    #     list_sum = [diff.sum()<40 for diff in distance]

    #     if any(list_sum):
    #       masks = np.concatenate((masks, dumped_mask[i][None,:]))
    #       boxes.append(box)



    segformer_mask = get_seg(frame).cpu()
    building_mask = segformer_mask ==2
    non_building = segformer_mask !=2


    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
      if True:

        rgb_mask = get_coloured_mask(masks[i].squeeze())
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)

        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
      else: pass


    
    img = cv2.addWeighted(img, 1, get_coloured_mask(building_mask, True), 0.5, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_array.append(img)
    del masks, boxes,building_mask,segformer_mask, non_building, img
    # cv2_imshow(img)
    # break
  return img_array