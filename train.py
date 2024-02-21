#coding: utf-8
import numpy as np
from tqdm import tqdm
import random
import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from sklearn.metrics import confusion_matrix
from Mydataset import Dataset_ETA
import utils as ut
import random
from Net import UNet_ETA
from Mydataset import Dataset_ETA
from loss import soft_dice_cldice, soft_cldice



def dataset(args):
    train_transform = ut.ExtCompose([ut.ExtRandomCrop(size=(256, 256)),
                                     ut.ExtRandomHorizontalFlip(),
                                     ut.ExtRandomVerticalFlip(),
                                     ut.ExtRandomRotation(degrees=90),
                                     ut.ExtToTensor(),
                                     ut.ExtNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                     ])
                                     
    if args.dataset=='DRIVE':
        test_transform = ut.TExtCompose([ut.TExtResize(size=(592, 592)),
                                        ut.TExtToTensor(),
                                        ut.TExtNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                        ])
                                        
    elif args.dataset=='CHASE':
        test_transform = ut.TExtCompose([ut.TExtResize(size=(1008, 1008)),
                                        ut.TExtToTensor(),
                                        ut.TExtNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                        ])


    data_train = Dataset_ETA(dataset=args.dataset, root = "Dataset", dataset_type='train', transform=train_transform)
    data_test = Dataset_ETA(dataset=args.dataset, root = "Dataset", dataset_type='test', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=True, drop_last=True, num_workers=4)


    return train_loader, test_loader
    
    

def fast_hist(label_true, label_pred):
    mask = (label_true >= 0) & (label_true < 2)
    hist = np.bincount(2 * label_true[mask].astype(int) + label_pred[mask], minlength=2 ** 2,
          ).reshape(2, 2)

    return hist



def IoU(output, target, label):
    output = np.array(output)
    target = np.array(target)

    # onehot
    confusion_matrix = np.zeros((2, 2))

    for lt, lp in zip(target, output):
        confusion_matrix += fast_hist(lt.flatten(), lp.flatten())

    diag = np.diag(confusion_matrix)
    iou_den = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - (diag+1e-7))
    iou = (diag+1e-7) / np.array(iou_den, dtype=np.float32)
 
    return iou


### training ###
def train(epoch):
    model.train()
    sum_loss = 0
    correct = 0
    total = 0
    c = 0
    for batch_idx, (inputs, targets1, targets2, targets3, targets) in enumerate(tqdm(train_loader, leave=False)):
        inputs = inputs.cuda(device)
        targets1 = targets1.unsqueeze(dim=1).cuda(device).float()
        targets2 = targets2.unsqueeze(dim=1).cuda(device).float()
        targets3 = targets3.unsqueeze(dim=1).cuda(device).float()
        targets = targets.unsqueeze(dim=1).cuda(device).float()
         
        output1, output2, output3, output4, output, _, _, _, _ = model(inputs)

        loss1 = criterion1(output1, targets1) + criterion1(output2, targets2) + criterion1(output3, targets3)  + criterion1(output4, targets) + criterion1(output, targets)
        loss2 = criterion2(output1, targets1) + criterion2(output2, targets2) + criterion2(output3, targets3)  + criterion2(output4, targets) + criterion2(output, targets)
        
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        
    return sum_loss/(batch_idx+1)


### test ###
def test(epoch):
    sum_loss = 0
    model.eval()
    predict = []
    answer = []
    labels = np.arange(2)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.cuda(device)
            targets = targets.unsqueeze(dim=1).cuda(device).float()

            output1, output2, output3, output4, output, _, _, _, _ = model(inputs)
            
            loss = criterion1(output, targets) + criterion2(output, targets)
            
            sum_loss += loss.item()
            output = torch.sigmoid(output)
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()
 
            for j in range(1):
                out = np.where(output[j]>=0.5, 1, 0)
                predict.append(out[0])
                answer.append(targets[j,0])

        iou = IoU(predict, answer, label=labels)

        miou = np.mean(iou)

    return sum_loss/(batch_idx+1), miou



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--Tbatchsize', '-t', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--num_epochs', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', '-i', type=str, default='DRIVE',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=str, default=-1,
                        help='Directory to output the result')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Directory to output the result')
    args = parser.parse_args()

    gpu_flag = args.gpu
    
    print('# GPU : {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.num_epochs))
    print('')
    

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    cudnn.benchmark = True

    ### seed ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    ### save dir ###
    if not os.path.exists("{}".format(args.out)):
          	os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "model")):
          	os.mkdir(os.path.join("{}".format(args.out), "model"))

    PATH_1 = "{}/trainloss.txt".format(args.out)
    PATH_2 = "{}/testloss.txt".format(args.out)
    PATH_3 = "{}/IoU.txt".format(args.out)
    
    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass



    ### dataset ###
    train_loader, test_loader = dataset(args)


    ### model ###
    model = UNet_ETA(output=1).cuda(device)


    ### criterion ###
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = soft_dice_cldice()


    ### optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    sample = 0
    sample_loss = 10000000
    iters = 0
    for epoch in range(args.num_epochs):
        loss_train = train(epoch)
        loss_test, mm = test(epoch)


        print("epoch %d / %d" % (epoch+1, args.num_epochs))
        print('train_Loss: %.4f' % loss_train)
        print('test_Loss : %.4f' % loss_test)
        print(" mIoU    : %.4f" % mm)
        print("")

        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_train))
        with open(PATH_2, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_test))
        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, mm))


        PATH_best ="{}/model/model_train.pth".format(args.out)
        torch.save(model.state_dict(), PATH_best)

        if mm >= sample:
           sample = mm
           PATH_best ="{}/model/model_bestiou.pth".format(args.out)
           torch.save(model.state_dict(), PATH_best)

        if loss_test < sample_loss:
           sample_loss = loss_test
           PATH_best ="{}/model/model_bestloss.pth".format(args.out)
           torch.save(model.state_dict(), PATH_best)

        if (epoch+1)%100 == 0:
           PATH ="{}/model/model_{}.pth".format(args.out, epoch+1)
           torch.save(model.state_dict(), PATH)
