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



camera_mtx = np.array([[552.554261, 0.000000, 682.049453],
                       [0.000000, 552.554261, 238.769549],
                       [0.000000, 0.000000, 1.000000 ]])





def get_prediction(img_path):
    img = img_path
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    with torch.no_grad():
      pred = model([img])
      scores = pred[0]['scores'].detach().cpu().numpy()


      pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().cpu().numpy())]



      pred_mask = (pred[0]['masks'].detach().cpu().numpy())> 0.6

    del pred
    torch.cuda.empty_cache()
    return pred_mask, pred_boxes


def get_corners(mask, bbx_id):
  im = np.array(Image.fromarray(mask[bbx_id].reshape(376,1408)).convert('L'))
  corners = cv2.goodFeaturesToTrack(im, 4, 0.01, 10)
  corners = np.array(corners).reshape(4,-1)
  return process_corners(corners)



def ByFlann(kp1, kp2, des1, des2, flag="ORB"):

    if (flag == "SIFT" or flag == "sift"):

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        search_params = dict(check=50)
    else:

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(check=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(des1, des2)
    return matches



def RANSAC(kp1, kp2, matches):
    MIN_MATCH_COUNT = 10

    matchType = type(matches[0])
    good = []
    print(matchType)
    if isinstance(matches[0], cv2.DMatch):

        good = matches
    else:

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 变换矩阵.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # h, w = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        #
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print
        "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None
        
    return kp1, kp2, good, matchesMask





def coompute_Ematrix(im1, im2):
    points_1 = []
    points_2 = []
    
    # masks_1, _ = get_prediction(im1)
    # masks_2, _ = get_prediction(im2)


    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    flannmatch = ByFlann(kp1,kp2,des1,des2, 'SIFT')
    kp1,kp2,matches,mask = RANSAC(kp1,kp2,flannmatch)
    if mask == None:
        return None
    else:
        for i in range(len(matches)):
            if mask == 1:
                points_1.append(kp1[matches[i].queryIdx].pt)
                points_2.append(kp2[matches[i].queryIdx].pt)
        if len(points_2)<= 8:
            return None
        points_1 = np.array(points_1)
        points_2 = np.array(points_2)
        E, mask = cv2.findEssentialMat(points_1, points_2, focal=0, pp=(0, 0), method=cv2.RANSAC)
        points, R_est, t_est, mask_pose = cv2.recoverPose(E, points_1, points_2)
        return R_est, t_est
    
def compute_proj_mtx(R_est, t_est, camera_mtx = camera_mtx):
    id_0 = np.concatenate((np.eye(3), np.zeros((3,1))), axis = 1)
    id_1 = np.concatenate((R_est,t_est), axis =1)
    P_1 = np.dot(camera_mtx, id_0)
    P_2 = np.dot(camera_mtx, id_1)
    return P_1, P_2


def triangulate(proj1, proj2, corners1, corners2):
    camera_coord = []
    for i in range(len(corners1)):
        h_coord1 = np.append(corners1[i], 1.)
        h_coord2 = np.append(corners2[i], 1.)
        tri_pt = cv2.triangulatePoints(proj1, proj2, corners1[i].T, corners2[i].T)
        camera_coord.append(tri_pt[:3])
    
    return camera_coord

def width_across_frame(coords):
    width = []
    for i in range(len(coords)):
        x1 = coords[i][0]
        x2 = coords[i][2]
        width.append(np.linalg.norm(x1-x2))
    return width
        
    

    



# corners_1 = np.concatenate((get_corners(masks, 8), get_corners(masks, 19)), axis =0)


# frame = Image.open(os.path.join(img_path, kitti_seq[2153+595]))
# masks, boxes = get_prediction(frame)
# corners_2 = np.concatenate((get_corners(masks, 9), get_corners(masks, 27)), axis =0)



# # corners = np.int0(corners_2[2].reshape(-1,1))
  
# # # we iterate through each corner, 
# # # making a circle at each point that we think is a corner.
# # frame = cv2.imread(os.path.join(img_path, kitti_seq[2153+595]))
# # for i in corners:
# #     x, y = corners.ravel()
# #     cv2.circle(frame, (x, y), 3, 205, -1)
# # plt.figure(figsize=(20,30))
# # plt.imshow(frame), plt.show()