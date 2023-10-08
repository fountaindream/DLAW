from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn
from .TripletLoss import TripletLoss

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def compute_similairity(features, part_features):
    x = features
    y = part_features
    dist = torch.mm(x, y.transpose(1,0))
    similairity = torch.exp(torch.neg(dist))

    return similairity


class MatchingVector(nn.Module):
    """
        Compute the adaptive weights in the feature level
    """
    def __init__(self, num_classes):
        super(MatchingVector, self).__init__()
        self.in_planes = 256
        self.base = ClothesDetector()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.part_num = 7
        self.num_classes = num_classes

        #part
        self.bottleneck_part = nn.BatchNorm1d(self.in_planes*(self.part_num))
        self.bottleneck_part.bias.requires_grad_(False)  # no shift

        self.classifier_part = nn.Linear(self.in_planes*(self.part_num), self.in_planes, bias=False)

        self.bottleneck_part.apply(weights_init_kaiming)
        self.classifier_part.apply(weights_init_classifier)
            
        #global
        self.bottleneck_global = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_global.bias.requires_grad_(False)  # no shift
        # self.classifier_global = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck_global.apply(weights_init_kaiming)
        # self.classifier_global.apply(weights_init_classifier)
        
        #fore
        self.bottleneck_fore = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_fore.bias.requires_grad_(False)  # no shift
        # self.classifier_fore = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck_fore.apply(weights_init_kaiming)
        # self.classifier_fore.apply(weights_init_classifier)

    def forward(self, x):
        
        y_part, y_global, y_fore, clustering_feat_map, part_pd_score = self.base(x)  # (b, 2048, 1, 1)

        y_global = y_global.view(y_global.shape[0], -1)
        y_fore = y_fore.view(y_fore.shape[0], -1)

        part_vector = []
        for i in range(self.part_num):
            y_part[i] = y_part[i].view(y_part[i].shape[0], -1)  # flatten to (bs, 2048)            
            similairity = compute_similairity(y_global,y_part[i])
            part_vector.append(similairity)
        part_vector = torch.stack(part_vector)
        matching_vector, _ = torch.max(part_vector,0)
        
        y_part = torch.cat(y_part, 1)
        y_part = y_part.view(y_part.shape[0], -1)
        feat_part = self.bottleneck_part(y_part)
        feat_global = self.bottleneck_global(y_global)
        feat_fore = self.bottleneck_fore(y_fore)

        return part_pd_score, matching_vector

