import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


unloader = transforms.ToPILImage()

def tensor_to_video(data, output_file):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(output_file, fourcc, 8.0, (112, 112))
    data = np.uint8(data.permute(0, 2, 3, 1).cpu().numpy() * 255)
    for i in range(16):
        frame = cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()

def npy_to_video(data, output_file):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(output_file, fourcc, 8.0, (112, 112))
    data = np.uint8(data.transpose(0, 2, 3, 1) * 255)
    for i in range(16):
        frame = cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()

def npy_to_png(data):
    vid = torch.from_numpy(np.load(data))
    vid = vid.permute(0, 3, 1, 2) / 255
    for i in range(16):
        image = vid[i, :, :, :].cuda().clone() # clone the tensor
        image = image.squeeze(0) # remove the fake batch dimension
        image = unloader(image)
        image.save('test/' + str(i) + '.png')