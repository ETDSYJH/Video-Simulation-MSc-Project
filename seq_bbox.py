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





class seq_bbox():
    
    def __init__(self, batched_processing = True):
        self.batched_p = batched_processing
        
        
        
    def ifoverlap(self, bb1, bb2):
      if (bb1[0]>=bb2[2]) or (bb1[2]<= bb2[0]) or (bb1[3]<=bb2[1]) or (bb1[1]>=bb2[3]):
        return False
      else: return True
      
      
    def compute_dist(self, bbx1,bbx2):
        
        if self.ifoverlap(bbx1,bbx2):
            
            left = max(bbx1[0], bbx2[0])
            top = max(bbx1[1], bbx2[1])
            right = min(bbx1[2], bbx2[2])
            bottom = min(bbx1[3], bbx2[3])
            
            intersection = (right-left)*(bottom-top)
            
            bbx1_area = (bbx1[2] - bbx2[0])*(bbx1[3] - bbx1[1])
            bbx2_area = (bbx2[2] - bbx2[0])*(bbx2[3] - bbx2[1])
            
            iou = intersection / (bbx1_area + bbx2_area - intersection)
            distance_f = 1/iou
        else: distance_f = 10000.
        return distance_f
    
    def pairing(self, pred_frame):
        paired, unmatched = [],[]
        footage = list(pred_frame.keys())
        for i in range(len(pred_frame)):
            
            current_pairs = []
            frame1, frame2 = pred_frame[i], pred_frame[i+1]
            
            bbx_in1, bbx_in2 = len(frame1), len(frame2)
            if bbx_in1 != 0 and bbx_in2 != 0:
                dist_mtx = np.zeros((bbx_in1, bbx_in2))
                for i, bbox1 in enumerate(frame1):
                    for j, bbox2 in enumerate(frame2):
                        dist_mtx [i,j] = self.compute_dist(bbox1, bbox2)
                        
                current_pairs = self.greedy_assign(dist_mtx)
                
            matched_bbx2 = [ pair[1] for pair in current_pairs ]
            for i in range(bbx_in2):
                if i not in matched_bbx2:
                    unmatched.append(i)
            paired.append(current_pairs)
            return paired, unmatched
        
        
    def greedy_assign(self, dist_mtx):
        
        matched = []
        
        while dist_mtx.min() != 10000.:
            i,j = np.where(dist_mtx==dist_mtx.min())
            i,j = int(i), int(j)
            dist_mtx[i,:] = 10000.
            dist_mtx[:,j] = 10000.
            matched.append((i,j))
        
        return matched
    
    
    def create_tubelets(self, pred_frame, pairs):
        tubelets, t_index = [], 0
        
        current_frame = 0
        
        for frame_id in range(len(pred_frame)):
            if frame_id != len(pred_frame)-1 :
                searching = None
                
                
                
        
                for scan in range(frame_id, len(pred_frame)-1):
                    if searching == None:
                        if len(pairs[scan]) != 0:

                           tubelets.append([pred_frame[scan][pairs[scan][0][0]]])
                           searching = pairs[scan][0][1]
                           del pairs[scan]
                        
                        else: 
                            frame_id += 1
                            continue
                    
                    else:
                        first_bbx = [bbx[0] for bbx in pairs[scan]]
                        if searching in first_bbx:
                            index = first_bbx.index(searching)
                            tubelets[-1].append(pred_frame[scan][searching])
                            searching = pairs[scan][index][1]
                            del pairs[scan][index]
                            
                        else:
                            tubelets[-1].append(pred_frame[scan][searching])
                            searching = None
                            break
        
            if searching != None:
                tubelets[-1].append(pred_frame[frame_id][searching])
                searching = None
                
        return tubelets
    
    
def __call__(self, predictions):
    pairs,_ = self.pairing(predictions)
    tubelets = self.create_tubelets(predictions, pairs)
    
    return tubelets
        
                        
                           
                        