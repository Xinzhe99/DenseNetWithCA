import per_weight
import DNWithCA
import myloss
import cv2
import time
import pandas as pd
import os
import argparse
import numpy as np
from tqdm import tqdm
import joblib
import glob
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from my_data_driven import GetDataset
def parse_args():  # ²ÎÊý½âÎöÆ÷
    parser = argparse.ArgumentParser()
    # Ôö¼ÓÊôÐÔ
    parser.add_argument('--name', default='DNwithCA', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=200, type=int)  # Ô­À´ÊÇ100
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)  # 5e-4
    # parser.add_argument('--weight', default=[0.16, 0.84], type=float)
    parser.add_argument('--gamma', default=0.9, type=float)  # ExponentialLR gamma
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)  # 5e-4
    parser.add_argument('--training_dir',default="pic_sequence_train/",type=str)
    parser.add_argument('--test_dir', default="pic_sequence_test/", type=str)

    args = parser.parse_args()  # ÊôÐÔ¸øÓëargsÊµÀý£ºadd_argument ·µ»Øµ½ args ×ÓÀàÊµÀý
    return args

def main():
    args = parse_args()

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)  # ´´½¨ÎÄ¼þ¼Ð±£´æÄ£ÐÍ
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))  # ´òÓ¡²ÎÊýÅäÖÃ
    print('------------')
    with open('models/%s/args.txt' % args.name, 'w') as f:  # Ð´Èë²ÎÊýÎÄ¼þ
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu=torch.cuda.is_available()
    if use_gpu:
        print('GPU Mode Acitavted')
    else:
        print('CPU Mode Acitavted')
    # ¶¨ÒåÎÄ¼þdataset
    folder_dataset_train = glob.glob(args.training_dir + "*.jpg")
    folder_dataset_test = glob.glob(args.test_dir + "*.jpg")
    # ¶¨ÒåÔ¤´¦Àí·½Ê½
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5), (0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5), (0.5))])
    # ¶¨ÒåÊý¾Ý¼¯
    dataset_train = GetDataset(imageFolderDataset=folder_dataset_train,
                               transform=transform_train)
    dataset_test = GetDataset(imageFolderDataset=folder_dataset_test,
                              transform=transform_test)

    # Êý¾Ý¼¯loader
    train_loader = DataLoader(dataset_train,
                              shuffle=True,
                              batch_size=args.batch_size)

    test_loader = DataLoader(dataset_test,
                             shuffle=True,
                             batch_size=args.batch_size)
    net = DNWithCA.DNWithCA(trainmode=True)
    if use_gpu:
        net=net.cuda()
        net.cuda()
    else:
        net=net
    # critertion_self = myloss.Gra_loss()
    critertion1 = nn.MSELoss()
    critertion2= myloss.SSIM()
    #Adam
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)  # Adam·¨ÓÅ»¯,filterÊÇÎªÁË¹Ì¶¨²¿·Ö²ÎÊý
    scheduler=lr_scheduler.ExponentialLR(optimizer,gamma=args.gamma)

    running_train_loss = 0
    running_val_loss = 0
    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'lr',
                                'train_loss',
                                'val_loss'])
    for epoch in range(args.epochs):
        #ÑµÁ·
        net.train()
        t1=time.time()
        for i,in_img in tqdm(enumerate(train_loader),total=len(train_loader)):
            if use_gpu:
                in_img=in_img.cuda()
            else:
                in_img = in_img
            net_out=net(in_img)
            loss1=critertion1(in_img,net_out)
            loss2=1-critertion2(in_img,net_out)
            w1=1
            w2=3
            loss=w1*loss1+w2*loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss+=loss.item()
            print("[epoch: %3d/%3d, progress: %5d/%5d] train loss: %8f " % (epoch + 1, args.epochs, (i + 1) * args.batch_size, len(dataset_train), loss.item()))
        print('finish train epoch: [{}/{}] costs:{}  avg_loss:{}'.format(epoch + 1 ,args.epochs,(time.time() - t1),(running_train_loss/len(train_loader))))
        scheduler.step()
        if (epoch + 1) % 20 == 0:#Ã¿20¸öepoch¼ÇÂ¼Ò»´Î
            torch.save(net.state_dict(), 'models/{}/model_{}.pth'.format(args.name,(epoch + 1)))
        train_log = OrderedDict([('train_loss', running_train_loss/len(train_loader))])
        running_train_loss = 0
        #ÑéÖ¤
        net.eval()
        with torch.no_grad():
            t1 = time.time()
            for i, in_img in tqdm(enumerate(train_loader), total=len(train_loader)):
                if use_gpu:
                    in_img = in_img.cuda()
                else:
                    in_img = in_img
                net_out = net(in_img)
                loss1 = critertion1(in_img, net_out)
                loss2 = 1-critertion2(in_img, net_out)
                w1 = 1
                w2 = 3
                loss = w1 * loss1 + w2 * loss2
                running_val_loss += loss.item()
                print("[epoch: %3d/%3d, batch: %5d/%5d] test loss: %8f " % (epoch + 1, args.epochs, (i + 1) * args.batch_size, len(dataset_test), loss.item()))
            val_log = OrderedDict([('val_loss', running_val_loss / len(test_loader))])
            print('finish val epoch: [{}/{}] costs:{}  avg_loss:{}'.format((epoch + 1),args.epochs,(time.time() - t1),(running_val_loss/len(test_loader))))
            running_train_loss = 0
        tmp = pd.Series([
            epoch + 1,
            scheduler.get_last_lr(),
            train_log['train_loss'],
            val_log['val_loss'],
        ], index=['epoch', 'lr', 'train_loss', 'val_loss'])  # Series´´½¨×Öµä
        log = pd.concat([log, tmp], ignore_index=True)  # ÐÂÐ´µÄ
        log.to_csv('models/%s/log.csv' % args.name, index=False)  # log:ÑµÁ·µÄÈÕÖ¾¼ÇÂ¼
if __name__ == '__main__':
    main()
