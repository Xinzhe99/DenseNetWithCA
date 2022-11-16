from PIL import Image
import cv2
import torchvision.transforms as transforms
trans = transforms.Compose([transforms.ToTensor()])
img = Image.open('00023.jpg').convert('L')
img = trans(img)
print(img.shape)
