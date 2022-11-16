import PIL
import cv2
from PIL import Image
import DNWithCA
import torch
import torchvision.transforms as transforms
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
if torch.cuda.is_available():
    device='cuda:0'
else:
    device='cpu'
eps=0.1
gf_radius = 4

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

def fuse(img1_path, img2_path):
    img1=Image.open(img1_path).resize((800,600))#PIL
    img2=Image.open(img2_path).resize((800,600))#PIL
    img1_numpy=np.array(img1)
    img2_numpy = np.array(img2)
    ndim=cv2.imread(img1_path,1).shape[2]
    if ndim != 3:
        img1_gray = img1
        img2_gray = img2
    else:
        img1_gray = img1.convert('L')#pil
        img2_gray = img2.convert('L')#pil
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    img1_tensor = data_transforms(img1_gray).unsqueeze(0).to(device)#tensor
    img2_tensor = data_transforms(img2_gray).unsqueeze(0).to(device)#tensor
    dict_path='models/DNwithCA/model_200.pth'
    net = DNWithCA.DNWithCA(trainmode=False)
    net.eval()
    use_gpu=torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
        net.cuda()
        net.load_state_dict(torch.load(dict_path))
    else:
        net = net
        net.load_state_dict(torch.load(dict_path, map_location=torch.device('cuda:0')))
    decision_1 = net(img1_tensor, img2_tensor)#dm·µ»ØµÄÊÇÒ»¸ö1\0×é³ÉµÄnumpy
    decision_2 = (decision_1 * 255).astype(np.uint8)

    d_map_binary = np.expand_dims(decision_2, axis=2)
    d_map_binary = np.concatenate((d_map_binary, d_map_binary, d_map_binary), axis=-1)
    fused_image_2 = img1 * d_map_binary + img2 * (1 - d_map_binary)
    fused_image_2 = Image.fromarray(np.uint8(fused_image_2))  # img: float32->ui
    #FC-CRT
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
    temp_fused = img1 * decision_4 + img2 * (1 - decision_4)
    decision_5 = guided_filter(temp_fused,decision_4,gf_radius,eps=0.1)
    fused = img1_numpy * 1.0 * decision_5 + img2_numpy * 1.0 * (1 - decision_5)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused

img1='pic_sequence_train/00023.jpg'
img2='pic_sequence_train/00078.jpg'

a=fuse(img1,img2)
a=cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
cv2.imshow('test',a)
cv2.waitKey()
