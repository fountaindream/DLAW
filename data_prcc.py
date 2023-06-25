from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from utils.DistributedErasing import DistributedErasing
from utils.aug_lib import TrivialAugment
from utils.transforms import get_affine_transform
from opt_prcc import opt
import os
import re
import cv2
import numpy as np
import random
import torch

class Data():
    def __init__(self):

        train_transform = transforms.Compose([
            #TrivialAugment(),
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = Market1501(train_transform, 'train', opt.data_path)
        self.testset = Market1501(test_transform, 'test', opt.data_path)
        self.queryset = Market1501(test_transform, 'query', opt.data_path)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True) #, drop_last=True
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,
                                                  pin_memory=True) #, drop_last=True

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path, crop_size=[128, 128],scale_factor=0.25,
                 rotation_factor=30):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path
        self.type = dtype

        if dtype == 'train':
            self.data_path +=  'new_train' #'/train'#'/cloth-change_train'#'/cloth-unchange_train'#'/cloth-change_train'  # 
        elif dtype == 'test':
            self.data_path +=  'new_A' #'/test' # '/cloth-change_test' #
        else:
            self.data_path +=   'new_B' #'/query'#'/cloth-change_query' # 'new_B' #
#A/C 14 A/B 16      
        
        self.parsing_path = data_path + 'new_train_parsing' #'/train_parsing' # 

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]
        self.parsings = [path for path in self.list_pictures(self.parsing_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
        self._cam2label = {_id: idx for idx, _id in enumerate(self.unique_cams)}
        self._clo2label = {_id: idx for idx, _id in enumerate(self.unique_clos)}  

    def __getitem__(self, index):
        path = self.imgs[index]
        parsing_path = self.parsings[index]
        try:
            path1 = self.imgs[index+1]
            parsing_path1 = self.parsings[index+1]        
        except:
            path1 = self.imgs[index-1]
            parsing_path1 = self.parsings[index-1]            
        target = self._id2label[self.id(path)]
        cams = self._cam2label[self.camera(path)]
        clos = self._clo2label[self.cloth(path)]

        img = self.loader(path)
        img1 = self.loader(path1)
        parsing_label, label_parsing1 = self.parsing(parsing_path)
        parsing_label_1, label_parsing1_1 = self.parsing(parsing_path1)        
        if self.type == 'train':
            img_erasing = DistributedErasing(probability=0.6)
            img = img_erasing(img, img1, label_parsing1, label_parsing1_1)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target, cams, clos, parsing_label

    def __len__(self):
        return len(self.imgs)

    def parsing(self, file_path):
        """
        :param file_path: unix style file path
        :return: parsing label
        """
        label_parsing1 = np.load(file_path)
        label_parsing = np.resize(label_parsing1,(24,8))

        return label_parsing, label_parsing1  

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def cloth(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[1])-1    

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1])-1

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def clothes(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.cloth(path) for path in self.imgs]

    @property
    def unique_clos(self):
        """
        :return: unique clothes ids in ascending order
        """
        return sorted(set(self.clothes))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @property
    def unique_cams(self):
        """
        :return: unique camera ids in ascending order
        """
        return sorted(set(self.cameras))    



    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])
