import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

def rotate(image, s):
    if s == 0:
        image = image
    if s == 1:
        HF = transforms.RandomHorizontalFlip(p=1)  # 闅忔満姘村钩缈昏浆
        image = HF(image)
    if s == 2:
        VF = transforms.RandomVerticalFlip(p=1)  # 闅忔満鍨傜洿缈昏浆
        image = VF(image)
    return image

# def color2gray(image, s):
#     if s == 0:
#         image = image
#     if s == 1:
#         l = image.convert('L')
#         n = np.array(l)  # 杞寲鎴恘umpy鏁扮粍
#         image = np.expand_dims(n, axis=2)
#         image = np.concatenate((image, image, image), axis=-1)  # axis=-1灏辨槸鏈€鍚庝竴涓€氶亾
#         image = Image.fromarray(image).convert('RGB')
#     return image

class GetDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
    def __getitem__(self, index):
        img = self.imageFolderDataset[index]
        img = Image.open(img).resize((800,600)).convert('L')  # input color image pair
        # ------------------data enhancement--------------------------#
        j = np.random.randint(0, 3, size=1)  # 随机0-3之间的整数
        img = rotate(img, j)
        # ------------------To tensor------------------#
        if self.transform is not None:
            img = self.transform(img)

            return img
    def __len__(self):
        return len(self.imageFolderDataset)

