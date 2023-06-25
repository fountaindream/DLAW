import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
#torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)

class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.cuda()
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels, cams, clos, parsings) in enumerate(self.train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            clos = clos.cuda()
            parsings = parsings.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels, clos, parsings)
            loss.backward()
            self.optimizer.step()

    def evaluate(self):

        self.model.eval()

        def rank(qf,ql,qc,qh,gf,gl,gc,gh):
            query = qf.view(-1,1)
            # print(query.shape)
            score = torch.mm(gf,query)
            score = score.squeeze(1).cpu()
            score = score.numpy()
            # predict index
            index = np.argsort(score)  #from small to large
            index = index[::-1]
            # index = index[0:2000]
            # good index
            query_index = np.argwhere(gl==ql)
            camera_index = np.argwhere(gc==qc)
            cloth_index = np.argwhere(gh==qh)             
            junk_index = np.argwhere(gl==-1)

            ap_tmp, CMC_tmp = compute_mAP(index, query_index, junk_index)
            return ap_tmp, CMC_tmp

        def compute_mAP(index, good_index, junk_index):
            ap = 0
            cmc = torch.IntTensor(len(index)).zero_()
            if good_index.size==0:   # if empty
                cmc[0] = -1
                return ap,cmc

            # remove junk_index
            mask = np.in1d(index, junk_index, invert=True)
            index = index[mask]

            # find good_index index
            ngood = len(good_index)
            mask = np.in1d(index, good_index)
            rows_good = np.argwhere(mask==True)
            rows_good = rows_good.flatten()
            
            cmc[rows_good[0]:] = 1
            for i in range(ngood):
                d_recall = 1.0/ngood
                precision = (i+1)*1.0/(rows_good[i]+1)
                if rows_good[i]!=0:
                    old_precision = i*1.0/rows_good[i]
                else:
                    old_precision=1.0
                ap = ap + d_recall*(old_precision + precision)/2

            return ap, cmc

        query_feature, query_label, query_cam, query_cloth = extract_feature(self.model, tqdm(self.query_loader))
        gallery_feature, gallery_label, gallery_cam, gallery_cloth = extract_feature(self.model, tqdm(self.test_loader))        
        # query_feature = query_feature.cuda()
        # gallery_feature = gallery_feature.cuda()

        #print(query_feature.shape)      
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        #print(query_label)
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = rank(query_feature[i],query_label[i],query_cam[i],query_cloth[i],gallery_feature,gallery_label,gallery_cam,gallery_cloth)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC/len(query_label) #average CMC
        print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))       

    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')


if __name__ == '__main__':

    data = Data()
    model = MGN()
    model = torch.nn.DataParallel(model)
    # model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)
    loss = Loss()
    main = Main(model, loss, data)

    if opt.mode == 'train':

        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            main.train()
            if epoch % 50 == 0:
                print('\nstart evaluate')
                main.evaluate()
                os.makedirs('weights/PRCC/', exist_ok=True)
                torch.save(model.state_dict(), ('weights/PRCC/model_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()
