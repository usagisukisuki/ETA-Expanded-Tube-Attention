#coding: utf-8
import numpy as np
import random
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


from Mydataset import Dataset_ETA_test
import utils as ut
from Net import UNet_ETA
from cldice_metric import clDice



def dataset(args):
                                     
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



    data_test = Dataset_ETA_test(dataset=args.dataset, root = "Dataset", dataset_type='test', transform=test_transform)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=True, num_workers=1)


    return test_loader



def Save_image(cll, imgs, ano, path, num):
    img = np.where(imgs>=0.5, 1, 0)
    cll = cll[0]
    img = img[0,0]
    imgs = imgs[0,0]
    ano = ano[0]

    
    #at1g = np.where(ano1 - at1 >=0, ano1 - at1, 0)
    #at2g = np.where(ano2 - at2 >=0, ano2 - at2, 0)
    #at3g = np.where(ano3 - at3 >=0, ano3 - at3, 0)
    #at4g = np.where(ano - at4 >=0, ano - at4, 0)
    
    #at1gn = np.where(ano1 - at1 <0, ano1 - at1, 0)
    #at2gn = np.where(ano2 - at2 <0, ano2 - at2, 0)
    #at3gn = np.where(ano3 - at3 <0, ano3 - at3, 0)
    #at4gn = np.where(ano - at4 <0, ano - at4, 0)
    
    cll = np.transpose(cll, (1,2,0))
    mean1 = np.ones((cll.shape[0], cll.shape[1],1))*0.4914
    mean2 = np.ones((cll.shape[0], cll.shape[1],1))*0.4822
    mean3 = np.ones((cll.shape[0], cll.shape[1],1))*0.4465
    std1 = np.ones((cll.shape[0], cll.shape[1],1))*0.2023
    std2 = np.ones((cll.shape[0], cll.shape[1],1))*0.1994
    std3 = np.ones((cll.shape[0], cll.shape[1],1))*0.2010
    mean = np.append(mean1, mean2, axis=2)
    mean = np.append(mean, mean3, axis=2)
    std = np.append(std1, std2, axis=2)
    std = np.append(std, std3, axis=2)
    cll = cll*std + mean


    #print(img_n)
    dst1 = np.zeros((img.shape[0], img.shape[1],3))
    dst2 = np.zeros((img.shape[0], img.shape[1],3))

    dst1[img==0] = [0.0, 0.0, 0.0]
    dst1[img==1] = [255.0, 255.0, 255.0]
    dst2[ano==0] = [0.0, 0.0, 0.0]
    dst2[ano==1] = [255.0, 255.0, 255.0]


    cll = Image.fromarray(np.uint8(cll*255.0))
    dst1 = Image.fromarray(np.uint8(dst1))
    dst2 = Image.fromarray(np.uint8(dst2))
    imgs = Image.fromarray(np.uint8(imgs*255.0))


    cll.save(path+"/input/{}.png".format(num), quality=95)
    dst1.save(path+"/seg/{}.png".format(num), quality=95)
    imgs.save(path+"/seg/{}_sig.png".format(num), quality=95)
    dst2.save(path+"/ano/{}.png".format(num), quality=95)
    
    


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

def Dice(output, target, label):
    output = np.array(output)
    target = np.array(target)
    seg = output.flatten()
    target = target.flatten()
    mat = confusion_matrix(target, seg, labels=label)
    sns.heatmap(mat, annot=True, fmt='.0f', cmap='jet')
    plt.savefig("{}/CM.png".format(args.out))
    plt.close()
    mat = np.array(mat).astype(np.float32)
    mat_all = mat.sum()
    diag_all = np.sum(np.diag(mat))
    fn_all = mat.sum(axis=1)
    fp_all = mat.sum(axis=0)
    tp_tn = np.diag(mat)
    fp_fn = np.sum(mat - (np.diag(mat.diagonal())), axis=1)
    #print(fp_fn)
    
    precision = np.zeros((2)).astype(np.float32)
    recall = np.zeros((2)).astype(np.float32)
    sensitivity = np.zeros((2)).astype(np.float32)
    specificity = np.zeros((2)).astype(np.float32)
    fpr = np.zeros((2)).astype(np.float32)
    f2 = np.zeros((2)).astype(np.float32)

    accuracy = tp_tn.sum() / mat_all
    for i in range(2):
        if (fp_all[i] != 0)and(fn_all[i] != 0):  
            precision[i] = float(tp_tn[i]) / float(fp_all[i])
            sensitivity[i] = float(tp_tn[i]) / float(fn_all[i])
            fpr[i] = float(fp_fn[i]) / float(fp_all[i])
            if (precision[i] != 0)and(sensitivity[i] != 0):  
                f2[i] = (2.0*precision[i]*sensitivity[i]) / (precision[i]+sensitivity[i])
            else:       
                f2[i] = 0.0
        else:
            precision[i] = 0.0
            sensitivity[i] = 0.0
            fpr[i] = 0.0
            
    specificity[0] = sensitivity[1]
    specificity[1] = sensitivity[0]
    return accuracy, precision, sensitivity, specificity, f2, fpr

def AUC(output, target):
    output = np.array(output)
    target = np.array(target)
    seg = output.flatten()
    target = target.flatten()
    fpr, tpr, thresholds = roc_curve(target, seg)
    auc = roc_auc_score(target, seg)

    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig("{}/AUC.png".format(args.out))
    plt.close()
    return auc



def test():
    predict = []
    answer = []
    model_path = "{}/model/model_bestiou.pth".format(args.out)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    labels = np.arange(2)
    iou_all = []
    prob = []
    with torch.no_grad():
        for batch_idx, (inputs, targets1, targets2, targets3, targets) in enumerate(test_loader):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device).long()
    
            output1, output2, output3, output4, output, at1, at2, at3, at4 = model(inputs)

            outputn = torch.sigmoid(output)
            inputs = inputs.cpu().numpy()
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()

            for j in range(outputn.shape[0]):
                out = np.where(outputn[j]>=0.5, 1, 0)
                predict.append(out[0])
                answer.append(targets[j])
                
            Save_image(inputs, outputn, targets, "{}/image".format(args.out), batch_idx+1)

        iou = IoU(predict, answer, label=labels)
        accuracy, precision, sensitivity, specificity, f2, fpr = Dice(predict, answer, label=[0,1])
        auc = AUC(prob, answer)
        cldice = clDice(predict, answer)
        miou = np.mean(iou)
        m0 = iou[0]
        m1 = iou[1]
        mm = miou

        dm = np.mean(f2)
        d0 = f2[0]
        d1 = f2[1]


        print("accuracy    = %.2f" % (accuracy*100.))
        print("precision   = %.2f" % (precision[1]*100.))
        print("sensitivity = %.2f" % (sensitivity[1]*100.))
        print("specificity = %.2f" % (specificity[1]*100.))
        print("FPR         = %.2f" % (fpr[0]*100.))
        print("IoU         = %.2f ; " % (m1*100.))
        print("Dice        = %.2f ;" % (d1*100.))
        print("clDice      = %.2f" % (cldice*100.))
        print("AUC         = %.2f" % (auc*100.))
        with open(PATH, mode = 'a') as f:
            f.write("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" % (accuracy*100., precision[1]*100., sensitivity[1]*100., specificity[1]*100., fpr[0]*100., m1*100., d1*100., cldice*100., auc*100.))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tokusei')
    parser.add_argument('--Tbatchsize', '-t', type=int, default=1,
                            help='Number of images in each mini-batch')
    parser.add_argument('--out', '-o', type=str, default='result',
                            help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=str, default=-1,
                            help='Directory to output the result')
    parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Directory to output the result')
    args = parser.parse_args()

    gpu_flag = args.gpu

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

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
    if not os.path.exists(os.path.join("{}".format(args.out), "image")):
            os.mkdir(os.path.join("{}".format(args.out), "image"))

    if not os.path.exists(os.path.join("{}".format(args.out), "image","input")):
         	os.mkdir(os.path.join("{}".format(args.out), "image","input"))
    if not os.path.exists(os.path.join("{}".format(args.out), "image","seg")):
       	os.mkdir(os.path.join("{}".format(args.out), "image","seg"))
    if not os.path.exists(os.path.join("{}".format(args.out), "image","ano")):
         	os.mkdir(os.path.join("{}".format(args.out), "image","ano"))

    PATH = "{}/predict.txt".format(args.out)
    with open(PATH, mode = 'w') as f:
        pass
    
    test_loader = dataset(args)

    model = UNet_ETA(output=1).cuda(device)

    test()



