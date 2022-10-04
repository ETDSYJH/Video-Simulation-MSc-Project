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



from filterpy.kalman import KalmanFilter

from seq_bbox import associate


def bbx_to_obs(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    h = bbox[0] + width/2.
    v = bbox[1] + height/2.
    a = width * height    #scale is just area
    ratio = width / float(h)
    return [np.array([h, v, a]).reshape((4, 1)), ratio]


def obs_to_bbx(obs, ratio):
    width = np.sqrt(obs[2]*ratio)
    height = obs[2]/width
    bbx = np.zeros((1,4))
    bbx[0] = obs[0]-width/2.
    bbx[1] = obs[1]-height/2.
    bbx[2] = obs[0]+width/2.
    bbx[3] = obs[1]+height/2.
    
    return bbx

class tubelet_kf():
    
    ID = 0
    def __init__(self, bbx):
        self.kfilter = KalmanFilter(dim_x = 6, dim__z = 3)
        self.kfilter.F = np.array([[1,0,0,1,0,0],
                                   [0,1,0,0,1,0],
                                   [0,0,1,0,0,1],
                                   [0,0,0,1,0,0],
                                   [0,0,0,0,1,0],
                                   [0,0,0,0,0,1]])
        self.kfilter.H = np.array([[1,0,0,0,0,0],
                                   [0,1,0,0,0,0],
                                   [0,0,1,0,0,0]])
        self.kfilter.R = np.array([[5,0,0,0,0,0],
                                   [0,5,0,0,0,0],
                                   [0,0,10,0,0,0],
                                   [0,0,0,10,0,0],
                                   [0,0,0,0,10,0],
                                   [0,0,0,0,0,10]])
        self.kfilter.P = np.array([[10,0,0,0,0,0],
                                   [0,10,0,0,0,0],
                                   [0,0,10,0,0,0],
                                   [0,0,0,8795,0,0],
                                   [0,0,0,0,8934,0],
                                   [0,0,0,0,0,2323]])
        self.kfilter.Q = np.array([[3,0,0,0,0,0],
                                   [0,3,0,0,0,0],
                                   [0,0,0.01,0,0,0],
                                   [0,0,0,0.01,0,0],
                                   [0,0,0,0,0.01,0],
                                   [0,0,0,0,0,0.01]])
        self.kfilter.x[:3] = bbx_to_obs(bbx)
        self.history = []
        self.tubeletsID = tubelet_kf.ID
        tubelet_kf.count += 1
        self.detect_count = 0
        self.undetected_frame = 0
        
        
        
        
    def update(self, bbx):
        self.kfilter.update(bbx_to_obs(bbx)[0])
        self.detect_count += 1
        self.undetected_frame = 0
        self.aspect_ratio = bbx_to_obs(bbx)[1]
        
    def predict(self,bbx):
        self.kfilter.predict()
        self.undetected_frame +=1
        self.history.append(self.kfilter.x)
        return obs_to_bbx(self.kfilter.x, self.aspect_ratio)
    
    def current_est(self):
        return obs_to_bbx(self.kfilter.x, self.aspect_ratio)
    

class kalman_tracking():
    
    
    def __init__(self, min_length, min_iou = 0.25):
        self.min_length = min_length
        self.min_iou = min_iou
        self.frfame = 0
        self.tubelets = []
        
    def process(self, bbx):
        self.frame += 1
        tube_preds = np.zeros((len(self.tubelets),4 ))

        for id, tubelet in enumerate(tube_preds):
            bbx_pred = self.tubelets[id].predict()[0]
            tubelet[:] = [bbx_pred[0],bbx_pred[1],bbx_pred[2],bbx_pred[3]]
        
        matched, unmatched_bbx = associate(bbx, tube_preds, self.min_iou)
        
        for ij in matched:
            self.tubelets[ij[1]].update(bbx[ij[0]])
            
            
        for i in unmatched_bbx:
            self.tubelets( tubelet_kf(bbx[i]))
        for j in reversed(len(self.tubelets)):
            
            tubelet = self.tubelets[j]
            if (tubelet.undetected_frame > 10):
                self.tubelets.pop(j)