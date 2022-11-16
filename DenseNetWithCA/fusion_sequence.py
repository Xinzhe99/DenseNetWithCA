import torch
import argparse
from torchvision import transforms
import time
import numpy as np
import os
import DNWithCA
import glob
import random
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian
from PIL import Image

def fusion(args,img1,img2,width=1200,height=900):
    net = DNWithCA.DNWithCA(trainmode=False)
    use_gpu=torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
        net.cuda()
        net.load_state_dict(torch.load(args.dict_path))
    else:
        net = net
        net.load_state_dict(torch.load(args.dict_path, map_location=torch.device('cuda:0')))
    net.eval()
    # if isinstance(img1,str):
    #     img1_pil = Image.open(img1).resize((width,height))
    # else :
    #     img1_pil = Image.fromarray(img1, mode='RGB')
    # if isinstance(img2,str):
    #     img2_pil = Image.open(img2).resize((width, height))
    # else :
    #     img2_pil = Image.fromarray(img2, mode='RGB')
    # img1_numpy = np.array(img1_pil)
    # img2_numpy = np.array(img2_pil)
    if isinstance(img1,str):
        img1_pil = Image.open(img1).resize((width,height))
        img1_numpy = np.array(img1_pil)
        ndim = img1_numpy.shape[2]
    else :
        img1_pil = Image.fromarray(img1, mode='RGB')
        img1_numpy = np.array(img1_pil)
        img1_numpy = cv2.cvtColor(img1_numpy, cv2.COLOR_RGB2BGR)
        ndim = img1_numpy.shape[0]
    if isinstance(img2,str):
        img2_pil = Image.open(img2).resize((width, height))
        img2_numpy = np.array(img2_pil)
        ndim = img2_numpy.shape[2]
    else :
        img2_pil = Image.fromarray(img2, mode='RGB')
        img2_numpy = np.array(img2_pil)
        img2_numpy = cv2.cvtColor(img2_numpy,cv2.COLOR_RGB2BGR)
        ndim = img2_numpy.shape[0]
    if ndim == 3:
        img1_gray = img1_pil.convert('L')
        img2_gray = img2_pil.convert('L')
    else:
        img1_gray = img1_pil # pil
        img2_gray = img2_pil  # pil
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    img1_tensor = data_transforms(img1_gray).unsqueeze(0).to(device)  # tensor
    img2_tensor = data_transforms(img2_gray).unsqueeze(0).to(device)  # tensor
    dict_path = 'models/DNwithCA/model_200.pth'
    net = DNWithCA.DNWithCA(trainmode=False)
    net.eval()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
        net.cuda()
        net.load_state_dict(torch.load(dict_path))
    else:
        net = net
        net.load_state_dict(torch.load(dict_path, map_location=torch.device('cuda:0')))
    decision_1 = net(img1_tensor, img2_tensor)  # dm·µ»ØµÄÊÇÒ»¸ö1\0×é³ÉµÄnumpy
    decision_2 = (decision_1 * 255).astype(np.uint8)

    d_map_binary = np.expand_dims(decision_2, axis=2)
    d_map_binary = np.concatenate((d_map_binary, d_map_binary, d_map_binary), axis=-1)
    fused_image_2 = img1_numpy * d_map_binary + img2_numpy * (1 - d_map_binary)
    fused_image_2 = Image.fromarray(np.uint8(fused_image_2))  # img: float32->ui
    # FC-CRT
    img = np.array(fused_image_2)
    d_map_rgb = d_map_binary.astype(np.uint32)
    d_map_lbl = d_map_rgb[:, :, 0] + (d_map_rgb[:, :, 1] << 8) + (d_map_rgb[:, :, 2] << 16)
    colors, labels = np.unique(d_map_lbl, return_inverse=True)
    HAS_UNK = False
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    use_2d = True
    if use_2d:
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=7, srgb=7, rgbim=img,
                               compat=1,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7,
                              zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(15)
    MAP = np.argmax(Q, axis=0)
    MAP = colorize[MAP, :]
    MAP = MAP.reshape(img.shape)
    decision_4 = (MAP[:, :, 0] * 255).astype(np.uint8)

    # guided filter
    if ndim == 3:
        decision_4 = np.expand_dims(decision_4, axis=2)
    temp_fused = img1_numpy * decision_4 + img2_numpy * (1 - decision_4)
    decision_5 = guided_filter(temp_fused, decision_4, gf_radius, eps=0.1)
    fused = img1_numpy * 1.0 * decision_5 + img2_numpy * 1.0 * (1 - decision_5)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    fused = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)
    return fused

def box_filter(imgSrc, r):
    """
    Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
    :param imgSrc: np.array, image
    :param r: int, radius
    :return: imDst: np.array. result of calculation
    """
    if imgSrc.ndim == 2:
        h, w = imgSrc.shape[:2]
        imDst = np.zeros(imgSrc.shape[:2])

        # cumulative sum over h axis
        imCum = np.cumsum(imgSrc, axis=0)
        # difference over h axis
        imDst[0: r + 1] = imCum[r: 2 * r + 1]
        imDst[r + 1: h - r] = imCum[2 * r + 1: h] - imCum[0: h - 2 * r - 1]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        # cumulative sum over w axis
        imCum = np.cumsum(imDst, axis=1)

        # difference over w axis
        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r]) - \
                             imCum[:, w - 2 * r - 1: w - r - 1]
    else:
        h, w = imgSrc.shape[:2]
        imDst = np.zeros(imgSrc.shape)

        # cumulative sum over h axis
        imCum = np.cumsum(imgSrc, axis=0)
        # difference over h axis
        imDst[0: r + 1] = imCum[r: 2 * r + 1]
        imDst[r + 1: h - r, :] = imCum[2 * r + 1: h, :] - imCum[0: h - 2 * r - 1, :]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        # cumulative sum over w axis
        imCum = np.cumsum(imDst, axis=1)

        # difference over w axis
        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r, 1]) - \
                             imCum[:, w - 2 * r - 1: w - r - 1]
    return imDst
def guided_filter(I, p, r, eps=0.1):
    """
    Guided Filter
    :param I: np.array, guided image
    :param p: np.array, input image
    :param r: int, radius
    :param eps: float
    :return: np.array, filter result
    """
    h, w = I.shape[:2]
    if I.ndim == 2:
        N = box_filter(np.ones((h, w)), r)
    else:
        N = box_filter(np.ones((h, w, 1)), r)
    mean_I = box_filter(I, r) / N
    mean_p = box_filter(p, r) / N
    mean_Ip = box_filter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box_filter(I * I, r) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)

    if I.ndim == 2:
        b = mean_p - a * mean_I
        mean_a = box_filter(a, r) / N
        mean_b = box_filter(b, r) / N
        q = mean_a * I + mean_b
    else:
        b = mean_p - np.expand_dims(np.sum((a * mean_I), 2), 2)
        mean_a = box_filter(a, r) / N
        mean_b = box_filter(b, r) / N
        q = np.expand_dims(np.sum(mean_a * I, 2), 2) + mean_b
    return q

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='DNwithCA', help='model name: (default: arch+timestamp)')
    parser.add_argument('--type',default='jpg',type=str)
    parser.add_argument('--fuse_Data_dir', default="pic_sequence_all/",type=str)
    parser.add_argument('--dict_path',default='models/DNwithCA/model_200.pth',type=str)#Ä£ÐÍÑ¡Ôñ¸ÄÕâÀï
    return parser.parse_args()
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
         return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

if __name__ == '__main__':
    args = parse_args()
    eps = 0.1
    gf_radius = 4
    device = torch.device('cuda:0')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('GPU Mode Acitavted')
    else:
        print('CPU Mode Acitavted')
    if not os.path.exists('result/%s'% args.name+'sequence'):
        os.makedirs('result/%s' % args.name+'sequence')
    pic_sequence_list=glob.glob(args.fuse_Data_dir+'*.jpg')
    random.shuffle(pic_sequence_list)
    temp_pic_sequence_list=[None]*(len(pic_sequence_list)-1)

    for i,data in enumerate(temp_pic_sequence_list):
        if i ==0:
            t1=time.time()
            fuse=fusion(args,pic_sequence_list[i],pic_sequence_list[i+1])
            temp_pic_sequence_list[i]=fuse

            # cv2.imwrite("result/{}/fusion_{}.{}".format(args.name+'sequence',str(i),args.type),temp_pic_sequence_list[i])
            print('Complete the transition fusion{},cost:{}'.format(str(i),time.time()-t1))
        else:
            t1 = time.time()
            fuse = fusion(args, temp_pic_sequence_list[i-1], pic_sequence_list[i+1])
            temp_pic_sequence_list[i] = fuse
            # cv2.imwrite("result/{}/fusion_{}.{}".format(args.name + 'sequence', str(i), args.type),
            #             temp_pic_sequence_list[i])

            print('Complete the transition fusion{},cost:{}'.format(str(i),time.time()-t1))
    cv2.imwrite("result/{}/fusion_result.{}".format(args.name + 'sequence', args.type),
                temp_pic_sequence_list[-1])
    print('Finish fusion!!!')



