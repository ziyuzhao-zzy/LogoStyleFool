import os
import cv2 as cv
import torchvision.transforms as transforms


class_folder1 = sorted(os.listdir('logos'))
transf = transforms.ToTensor()
logoset = []
for logo in class_folder1:
    if(logo[0] != '.'):
        im = 'logos/' + logo
        img = cv.imread(im)
        img_rgb=cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_tensor = transf(img_rgb)
        logoset.append(img_tensor)