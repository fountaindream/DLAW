from __future__ import absolute_import

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torch import nn

def get_similarity(img_path,mask_path):
    similarities = []
    imgs = os.listdir(img_path)
    masks = os.listdir(mask_path)
    assert len(imgs) == len(masks)
    for i in range(len(imgs)):
        mask = np.load(os.path.join(mask_path,masks[i]))
        img = Image.open(os.path.join(img_path,imgs[i]))
        img_hist = get_hist(img)
        temp = []
        for i in range(1,7):
            img = np.array(img,dtype=np.uint8).transpose(2,0,1)
            part_img = img * (mask == i)
            part_img = part_img.transpose(1,2,0)
            part_img_hist = get_hist(part_img)
            match = cv2.compareHist(img_hist, part_img_hist, cv2.HISTCMP_BHATTACHARYYA)
            temp.append(match)
            temp = np.hstack(temp)
        similarities.append(temp)
    similarities = np.vstack(similarities)
    #part_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    return similarities

def g_similarity(image, image1, mask, mask1):
    dists = []
    image = np.array(image).astype('uint8')
    image1 = np.array(image1).astype('uint8')
    img_hist = get_hist(image)
    img_hist1 = get_hist(image1)        
    dist = cv2.compareHist(img_hist, img_hist1, cv2.HISTCMP_BHATTACHARYYA)
    similaritie = np.exp(-1*dist)
    return similaritie

def similarity(image, image1, mask, mask1):
    dists = []
    image = np.array(image).astype('uint8')
    image1 = np.array(image1).astype('uint8')    
    #img_hist = get_hist(image)
    for i in range(1,7):
        image = image.transpose(2,0,1)
        image1 = image1.transpose(2,0,1)        
        if not (image.shape[0] == 3):
            image = image.transpose(1,2,0)
        if not (image1.shape[0] == 3):        
            image1 = image1.transpose(1,2,0)
        part_img = image * (mask == i)
        part_img = part_img.transpose(1,2,0)
        part_img_hist = get_hist(part_img)
        part_img1 = image1 * (mask1 == i)
        part_img1 = part_img1.transpose(1,2,0)
        part_img_hist1 = get_hist(part_img1)
        dist = cv2.compareHist(part_img_hist, part_img_hist1, cv2.HISTCMP_BHATTACHARYYA)
        dists.append(dist)
    dists = np.hstack(dists)
    similarities = np.exp(-1*dists)
    return similarities

def get_hist(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return img_gray_hist

class DistributedErasing(object):
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, img, img1, mask, mask1):
        similarities = similarity(img, img1, mask, mask1)

        #print(similarities)

        img = np.array(img,dtype=np.uint8).transpose(2,0,1)
        if not (img.shape[0] == 3):
            img = img.transpose(1,2,0)        

        median_ = np.median(similarities)
        for i in range(1,7):
            part_img = img * (mask==i)
            p = similarities[i-1]
            if p >= median_:
                img = img - (p * np.array(part_img)).astype('uint8')            
            if p < median_:
                img = img + (p * np.array(part_img)).astype('uint8')

        
        # p = g_similarity(img, img1, mask, mask1)

        image = Image.fromarray(img.transpose(1,2,0))

        return image