import torch
from torch import nn
import torch.nn.functional as F


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

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.conv.apply(weights_init_kaiming)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class SpatialAttn(nn.Module):
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(256, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # bilinear resizing
        x = F.upsample(x, (x.size(2)*2, x.size(3)*2), mode='bilinear', align_corners=True)
        # scaling conv
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x

class ClothesDetector(nn.Module):

    def __init__(self):
        super(ClothesDetector, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True)
        )        
        self.spatial_attn = SpatialAttn()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.part_num = 7
        self.part_cls_layer = nn.Conv2d(in_channels=256,
                                                out_channels=self.part_num,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

    def forward(self, x):
        x = self.cls_head(x)
        mask = self.spatial_attn(x) #simple spatial attention. can be removed
        x = x*mask
        
        N, f_h, f_w = x.size(0), x.size(2), x.size(3)
        part_cls_score = self.part_cls_layer(x)
        part_pred = F.softmax(part_cls_score, dim=1)
        
        y_part = []
        for p in range(self.part_num):
            y_part.append(self.gap(x*part_pred[:,p,:,:].view(N,1,f_h,f_w)))
        # y_part = torch.cat(y_part, 1)
        y_g = self.gap(x)#full
        y_fore = self.gap(x*torch.sum(part_pred[:,1:self.part_num,:,:], 1).view(N,1,f_h,f_w))#foreground
                
        return y_part, y_g, y_fore, x, part_cls_score

    def random_init(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def load_param(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        logger.info('=> loading pretrained model {}'.format(pretrained_path))
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            logger.info(
                '=> loading {} pretrained model {}'.format(k, pretrained_path))
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
